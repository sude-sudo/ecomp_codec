"""CPU-only v2/v3 regression, explicit migration and per-view temporal wiring."""
from dataclasses import replace
import unittest
from unittest import mock

import torch
from torch.nn import functional as F

from compress.decoder import SpatioTemporalDecoder, with_temporal_reference_init
from compress.dual_view_codec import DualViewCodec
from compress.lightweight_encoder import ImageGeometry
from fixtures.decoder_v2 import SpatioTemporalDecoder as DecoderV2


def migrate_v2(state):
    """Explicit warm-start policy: copy only the newly introduced wrist weights."""
    migrated = state.copy()
    for source, target in (("temporal.", "temporal_wrist."),
                           ("temporal_project.", "temporal_project_wrist.")):
        if any(k.startswith(target) for k in state):
            raise ValueError("Expected a v2 checkpoint without wrist temporal weights")
        for name, value in state.items():
            if name.startswith(source):
                migrated[target + name[len(source):]] = value.clone()
    return with_temporal_reference_init(migrated)


class DualTemporalRegression(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(20261104)

    @torch.no_grad()
    def test_v2_migration_main_bitexact_and_first_frame_both_bitexact(self):
        old, new = DecoderV2().eval(), SpatioTemporalDecoder().eval()
        before = {k: v.clone() for k, v in old.state_dict().items()}
        with self.assertRaisesRegex(RuntimeError, "Missing key"):
            new.load_state_dict(before, strict=True)
        migrated = migrate_v2(before)
        new.load_state_dict(migrated, strict=True)
        with self.assertRaises(ValueError):
            migrate_v2(migrated)
        # Unrelated missing weights must still fail; migration cannot hide them.
        bad = before.copy()
        del bad['content.0.weight']
        with self.assertRaises(RuntimeError):
            new.load_state_dict(migrate_v2(bad), strict=True)
        new.load_state_dict(migrated, strict=True)
        for k, v in before.items():
            torch.testing.assert_close(new.state_dict()[k], v, atol=0, rtol=0)
        for a, b in zip(new.temporal.parameters(), new.temporal_wrist.parameters()):
            self.assertNotEqual(a.data_ptr(), b.data_ptr())
        old_state = new_state = None
        for i, (hm, wm, hw, ww) in enumerate(((4, 8, 8, 4), (8, 4, 4, 4), (4, 4, 4, 8), (4, 8, 4, 4))):
            main, wrist = torch.rand(1,64,hm,wm), torch.rand(1,64,hw,ww)
            gm = ImageGeometry((hm*16-3,wm*16-5),(hm*16,wm*16))
            gw = ImageGeometry((hw*16-1,ww*16-7),(hw*16,ww*16))
            expected = old(main,wrist,gm,gw,old_state,episode_id=5,frame_index=i)
            actual = new(main,wrist,gm,gw,new_state,episode_id=5,frame_index=i)
            torch.testing.assert_close(actual['main_hat'],expected['main_hat'],atol=0,rtol=0)
            if i == 0:
                torch.testing.assert_close(actual['wrist_hat'],expected['wrist_hat'],atol=0,rtol=0)
                with self.assertRaisesRegex(ValueError,'Incompatible decoder history'):
                    new(main,wrist,gm,gw,expected['state'],episode_id=5,frame_index=1)
            else:
                self.assertGreater((actual['wrist_hat']-expected['wrist_hat']).abs().max().item(),0)
            self.assertEqual(actual['state'].episode_id, 5)
            self.assertEqual(actual['state'].next_frame, i+1)
            self.assertEqual(len(actual['state'].history), min(i+1,2))
            for a,b in zip(actual['state'].history,expected['state'].history):
                torch.testing.assert_close(a.main_latent,b.latent,atol=0,rtol=0)
                torch.testing.assert_close(a.main_rgb,b.rgb,atol=0,rtol=0)
            old_state, new_state = expected['state'], actual['state']

    @torch.no_grad()
    def test_reference_features_and_injection_order_use_each_views_geometry(self):
        decoder = SpatioTemporalDecoder().eval()
        main, wrist = torch.rand(1,64,4,8), torch.rand(1,64,8,4)
        gm, gw = ImageGeometry((61,119),(64,128)),ImageGeometry((121,59),(128,64))
        first = decoder(main,wrist,gm,gw)
        # Different, variable-size histories for each view; paired frame ordering.
        next_main, next_wrist = torch.rand(1,64,8,8),torch.rand(1,64,4,4)
        gm2, gw2 = ImageGeometry((125,127),(128,128)),ImageGeometry((63,61),(64,64))
        trace, handles = [], []
        for name in ('atten_4','temporal','temporal_wrist','temporal_project','temporal_project_wrist','decoder_2'):
            handles.append(getattr(decoder,name).register_forward_hook(
                lambda m, inputs, out, name=name: trace.append((name, tuple(inputs[0].shape[-2:])))))
        try:
            with mock.patch.object(decoder.temporal,'forward',wraps=decoder.temporal.forward) as tm, \
                 mock.patch.object(decoder.temporal_wrist,'forward',wraps=decoder.temporal_wrist.forward) as tw:
                out = decoder(next_main,next_wrist,gm2,gw2,first['state'],frame_index=1)
            for call, latent, image in ((tm.call_args, main, first['main_hat']),
                                         (tw.call_args, wrist, first['wrist_hat'])):
                references = call.args[1]
                expected = decoder.content(latent) + decoder.history_image(F.adaptive_avg_pool2d(image,latent.shape[-2:]))
                self.assertEqual(len(references),1)
                torch.testing.assert_close(references[0],expected,atol=0,rtol=0)
            torch.testing.assert_close(tm.call_args.args[0],decoder.content(next_main),atol=0,rtol=0)
            torch.testing.assert_close(tw.call_args.args[0],decoder.content(next_wrist),atol=0,rtol=0)
            self.assertEqual(trace,[('atten_4',(32,32)),('temporal',(8,8)),
                ('temporal_project',(32,32)),('temporal_wrist',(4,4)),
                ('temporal_project_wrist',(16,16)),('decoder_2',(32,32)),('decoder_2',(16,16))])
            self.assertEqual(out['wrist_hat'].shape[-2:],gw2.original_hw)
        finally:
            for handle in handles:
                handle.remove()

    @torch.no_grad()
    def test_both_cached_frames_matter_and_history_size_limit(self):
        decoder = SpatioTemporalDecoder(history_size=2).eval()
        gm,gw = ImageGeometry((64,128),(64,128)),ImageGeometry((128,64),(128,64))
        main,wrist = torch.rand(1,64,4,8),torch.rand(1,64,8,4)
        state = None
        for i in range(4):
            state = decoder(main+i,wrist-i,gm,gw,state,frame_index=i)['state']
        expected = decoder(main,wrist,gm,gw,state,frame_index=4)
        for index in range(2):
            entry = state.history[index]
            for field in ('wrist_latent','wrist_rgb'):
                entries = list(state.history)
                entries[index] = replace(entry, **{field:getattr(entry,field)+3})
                out = decoder(main,wrist,gm,gw,replace(state,history=tuple(entries)),frame_index=4)
                self.assertGreater((out['wrist_hat']-expected['wrist_hat']).abs().max().item(),1e-8)
                torch.testing.assert_close(out['main_hat'],expected['main_hat'],atol=0,rtol=0)
        one = SpatioTemporalDecoder(history_size=1).eval()
        state = None
        for i in range(3):
            state = one(main,wrist,gm,gw,state,frame_index=i)['state']
            self.assertEqual(len(state.history),1)

    @torch.no_grad()
    def test_analysis_and_frame_bytes_unchanged_from_v2(self):
        new, old = DualViewCodec(decoder_config='baseline').eval(), DualViewCodec(decoder_config='baseline').eval()
        old.decoder = DecoderV2().eval()
        new.update(force=True)
        old.analysis.load_state_dict(new.analysis.state_dict(),strict=True)
        for frame in (0,1,2):
            main,wrist = torch.rand(1,3,95,113),torch.rand(1,3,65,81)
            kw = dict(episode_id=9,frame_index=frame,main_rate=(3,.5),wrist_rate=(6,.2))
            self.assertEqual(old.compress(main,wrist,**kw),new.compress(main,wrist,**kw))


if __name__ == '__main__':
    unittest.main(verbosity=2)
