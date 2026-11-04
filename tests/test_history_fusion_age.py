"""CPU acceptance of learnable modality fusion and relative history ages."""
import unittest
from unittest import mock

import torch
from torch import nn

from compress.decoder import (DecodedFrame, SpatioTemporalDecoder,
                              with_temporal_reference_init)
from compress.dual_view_codec import DualViewCodec
from compress.lightweight_encoder import ImageGeometry
from fixtures.decoder_v3 import SpatioTemporalDecoder as DecoderV3


def frame():
    return DecodedFrame(torch.randn(1,64,4,8),torch.rand(1,3,61,119),
                        torch.randn(1,64,8,4),torch.rand(1,3,121,59))


class HistoryFusionAge(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(20261104)
        self.decoder = SpatioTemporalDecoder()

    def test_concat_shapes_sum_initialization_modal_effect_and_gradients(self):
        fuse = self.decoder.history_fuse
        self.assertIsInstance(fuse,nn.Conv2d)
        self.assertEqual((fuse.in_channels,fuse.out_channels,fuse.kernel_size),(128,64,(1,1)))
        latent,rgb = torch.randn(2,64,4,7,requires_grad=True),torch.randn(2,64,4,7,requires_grad=True)
        combined = torch.cat((latent,rgb),dim=1)
        self.assertEqual(combined.shape,(2,128,4,7))
        output = fuse(combined)
        self.assertEqual(output.shape,latent.shape)
        torch.testing.assert_close(output,latent+rgb,atol=1e-6,rtol=1e-6)
        for inputs in ((latent+1,rgb),(latent,rgb+1)):
            self.assertGreater((fuse(torch.cat(inputs,dim=1))-output).abs().max().item(),0.5)
        output.square().mean().backward()
        self.assertGreater(fuse.weight.grad[:,:64].abs().sum().item(),0)
        self.assertGreater(fuse.weight.grad[:,64:].abs().sum().item(),0)
        self.assertGreater(latent.grad.abs().sum().item(),0)
        self.assertGreater(rgb.grad.abs().sum().item(),0)

    @torch.no_grad()
    def test_age_rows_oldest_to_newest_for_both_views_and_any_history_size(self):
        for size in (1,2,3):
            decoder = SpatioTemporalDecoder(history_size=size)
            self.assertTrue(torch.equal(decoder.temporal_age_embedding.weight,torch.zeros(size+1,64)))
            for age in range(1,size+1):
                decoder.temporal_age_embedding.weight[age].fill_(age*.25)
            history = tuple(frame() for _ in range(size))
            for view in ('main','wrist'):
                with mock.patch.object(decoder.temporal_age_embedding,'forward',wraps=decoder.temporal_age_embedding.forward) as embed:
                    refs = decoder._history_references(history,view)
                self.assertEqual([c.args[0].item() for c in embed.call_args_list],list(range(size,0,-1)))
                for index,(entry,ref) in enumerate(zip(history,refs)):
                    latent,rgb = getattr(entry,view+'_latent'),getattr(entry,view+'_rgb')
                    expected = decoder.content(latent)+decoder.history_image(nn.functional.adaptive_avg_pool2d(rgb,latent.shape[-2:]))
                    torch.testing.assert_close(ref,expected+(size-index)*.25,atol=1e-6,rtol=1e-6)
                # The same cached frame changes age when a newer frame arrives.
                alone = decoder._history_references((history[0],),view)[0]
                torch.testing.assert_close(refs[0]-alone,torch.full_like(alone,(size-1)*.25),atol=1e-6,rtol=1e-6)
            self.assertNotEqual(decoder.temporal_age_embedding.weight[0].data_ptr(),decoder.temporal_age_embedding.weight[1].data_ptr())
            if size > 1:
                self.assertNotEqual(decoder.temporal_age_embedding.weight[1].data_ptr(),decoder.temporal_age_embedding.weight[2].data_ptr())

    @torch.no_grad()
    def test_content_age_pairing_changes_attention_but_token_order_does_not(self):
        history = (frame(),frame())
        for view,attention in (('main',self.decoder.temporal),('wrist',self.decoder.temporal_wrist)):
            query = self.decoder.content(torch.randn_like(getattr(history[0],view+'_latent')))
            # Zero ages: swapping content merely permutes concatenated tokens.
            refs = self.decoder._history_references(history,view)
            changed_order = self.decoder._history_references(tuple(reversed(history)),view)
            torch.testing.assert_close(attention(query,refs),attention(query,changed_order),atol=1e-6,rtol=1e-6)
            self.decoder.temporal_age_embedding.weight[1].copy_(torch.linspace(-3,3,64))
            self.decoder.temporal_age_embedding.weight[2].copy_(torch.linspace(2,-2,64))
            refs = self.decoder._history_references(history,view)
            swapped_content_ages = self.decoder._history_references(tuple(reversed(history)),view)
            output = attention(query,refs)
            self.assertGreater((output-attention(query,swapped_content_ages)).abs().max().item(),1e-5)
            # Reordering already tagged references preserves content-age pairs.
            torch.testing.assert_close(output,attention(query,tuple(reversed(refs))),atol=1e-6,rtol=1e-6)
            self.decoder.temporal_age_embedding.weight.zero_()

    def test_age_and_fusion_gradients_for_both_view_losses(self):
        history = (frame(),frame())
        for view,attention in (('main',self.decoder.temporal),('wrist',self.decoder.temporal_wrist)):
            self.decoder.zero_grad(set_to_none=True)
            query = self.decoder.content(torch.randn_like(getattr(history[0],view+'_latent')))
            attention(query,self.decoder._history_references(history,view)).square().mean().backward()
            grad = self.decoder.temporal_age_embedding.weight.grad
            self.assertTrue(torch.isfinite(grad).all())
            self.assertEqual(grad[0].abs().sum().item(),0)
            for age in (1,2):
                self.assertGreater(grad[age].abs().sum().item(),0)
            weight_grad = self.decoder.history_fuse.weight.grad
            for part in (weight_grad[:,:64],weight_grad[:,64:]):
                self.assertTrue(torch.isfinite(part).all())
                self.assertGreater(part.abs().sum().item(),0)

    @torch.no_grad()
    def test_first_frame_skips_fusion_embedding_and_both_attentions(self):
        d = self.decoder
        gm,gw = ImageGeometry((64,128),(64,128)),ImageGeometry((128,64),(128,64))
        with mock.patch.object(d,'_history_references',side_effect=AssertionError('history used')), \
             mock.patch.object(d.temporal_age_embedding,'forward',side_effect=AssertionError('age used')), \
             mock.patch.object(d.temporal,'forward',side_effect=AssertionError('attention used')), \
             mock.patch.object(d.temporal_wrist,'forward',side_effect=AssertionError('wrist attention used')):
            out = d(torch.rand(1,64,4,8),torch.rand(1,64,8,4),gm,gw)
        self.assertEqual(out['history_used'],0)
        self.assertEqual(len(out['state'].history),1)

    @torch.no_grad()
    def test_explicit_migration_full_and_decoder_partial_keys_and_metadata(self):
        old = DecoderV3(history_size=3)
        new = SpatioTemporalDecoder(history_size=3)
        state = old.state_dict()
        with self.assertRaises(RuntimeError):
            new.load_state_dict(state,strict=True)
        migrated = with_temporal_reference_init(state,history_size=3)
        self.assertEqual(len(migrated),len(state)+3)
        self.assertEqual(migrated._metadata,state._metadata)
        self.assertNotIn('history_fuse.weight',state)
        new.load_state_dict(migrated,strict=True)
        for name,value in state.items():
            torch.testing.assert_close(new.state_dict()[name],value,atol=0,rtol=0)
        with self.assertRaises(RuntimeError):
            new.load_state_dict(with_temporal_reference_init(state,history_size=2),strict=True)
        for key in ('history_fuse.weight','history_fuse.bias','temporal_age_embedding.weight'):
            partial = state.copy()
            partial[key] = migrated[key]
            with self.assertRaisesRegex(ValueError,'Incomplete'):
                with_temporal_reference_init(partial)
        del migrated['content.0.weight']
        with self.assertRaises(RuntimeError):
            new.load_state_dict(with_temporal_reference_init(migrated,3),strict=True)
        state4 = self.decoder.state_dict()
        state4['history_fuse.weight'].add_(.1)
        state4['temporal_age_embedding.weight'][1].add_(.2)
        unchanged = with_temporal_reference_init(state4)
        for key in state4:
            torch.testing.assert_close(unchanged[key],state4[key],atol=0,rtol=0)
        old_codec,new_codec = DualViewCodec(levels=6).eval(),DualViewCodec(levels=6).eval()
        old_codec.decoder = DecoderV3().eval()
        old_codec.update(force=True)
        new_codec.load_state_dict(with_temporal_reference_init(old_codec.state_dict()),strict=True)
        self.assertEqual(new_codec.analysis.gain.levels,6)

    @torch.no_grad()
    def test_v3_migrated_random_sequence_numerical_equivalence(self):
        old,new = DecoderV3().eval(),self.decoder.eval()
        new.load_state_dict(with_temporal_reference_init(old.state_dict()),strict=True)
        os = ns = None
        max_error = 0.
        for i,(hm,wm,hw,ww) in enumerate(((4,8,8,4),(8,8,4,4),(4,4,4,8),(4,8,8,4))):
            main,wrist = torch.randn(2,64,hm,wm),torch.randn(2,64,hw,ww)
            gm,gw = ImageGeometry((hm*16-3,wm*16-5),(hm*16,wm*16)),ImageGeometry((hw*16-1,ww*16-7),(hw*16,ww*16))
            before = old(main,wrist,gm,gw,os,frame_index=i)
            after = new(main,wrist,gm,gw,ns,frame_index=i)
            for key in ('main_hat','wrist_hat'):
                error = (before[key]-after[key]).abs().max().item()
                max_error = max(max_error,error)
                torch.testing.assert_close(before[key],after[key],atol=1e-6,rtol=1e-6)
            os,ns = before['state'],after['state']
        print(f'\nv3/v4 migrated random decoder sequence max_abs_error={max_error:.9g}')

    @torch.no_grad()
    def test_nonzero_age_fusion_real_stream_receiver_reset_and_unchanged_bytes(self):
        sender,receiver,old = DualViewCodec().eval(),DualViewCodec().eval(),DualViewCodec().eval()
        old.decoder = DecoderV3().eval()
        sender.decoder.history_fuse.weight.add_(torch.randn_like(sender.decoder.history_fuse.weight)*.01)
        sender.decoder.history_fuse.bias.add_(.02)
        sender.decoder.temporal_age_embedding.weight[1].copy_(torch.linspace(-.2,.3,64))
        sender.decoder.temporal_age_embedding.weight[2].copy_(torch.linspace(.4,-.3,64))
        sender.update(force=True)
        receiver.load_state_dict(sender.state_dict(),strict=True)
        old.analysis.load_state_dict(sender.analysis.state_dict(),strict=True)
        del receiver.analysis.g_a
        del receiver.analysis.hyperprior.h_a
        fs = ds = None
        for i,(ep,fi) in enumerate(((9,0),(9,1),(9,2),(9,3),(10,0),(10,1))):
            main,wrist = torch.rand(1,3,65+i,71),torch.rand(1,3,31,49+i)
            kwargs = dict(episode_id=ep,frame_index=fi,main_rate=(3,.5),wrist_rate=(6,.2))
            expected = sender(main,wrist,state=fs,**kwargs)
            packet = sender.compress(main,wrist,**kwargs)
            self.assertEqual(packet,old.compress(main,wrist,**kwargs))
            actual = receiver.decompress(packet,state=ds)
            for key in ('main_hat','wrist_hat'):
                torch.testing.assert_close(actual[key],expected[key],atol=0,rtol=0)
            self.assertEqual(actual['actual_bpp'],8*len(packet)/(main.shape[-2]*main.shape[-1]+wrist.shape[-2]*wrist.shape[-1]))
            for a,b in zip(actual['state'].history,expected['state'].history):
                for field in ('main_latent','main_rgb','wrist_latent','wrist_rgb'):
                    torch.testing.assert_close(getattr(a,field),getattr(b,field),atol=0,rtol=0)
                    self.assertFalse(getattr(a,field).requires_grad)
            fs,ds = expected['state'],actual['state']


if __name__ == '__main__':
    unittest.main(verbosity=2)
