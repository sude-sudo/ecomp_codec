"""CPU two-thread functional, causal and real-stream acceptance for Decoder-v5."""
from dataclasses import replace
import unittest

import torch
from torch.nn import functional as F

from compress.decoder import SpatioTemporalDecoder as DecoderV4
from compress.decoder_v5 import (CONFIGS, DecoderState, SpatioTemporalDecoderV5,
                                ViewTemporalSpatialAttention, coordinates, migrate_checkpoint)
from compress.dual_view_codec import DualViewCodec
from compress.lightweight_encoder import ImageGeometry


class DecoderV5Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(502)

    def small(self, **kw):
        return SpatioTemporalDecoderV5(M=32,N=32,**kw)

    def pair(self, size=1):
        tensors=[torch.randn(1,64,size,size) for _ in range(2)]
        geometry=ImageGeometry((size*16,size*16),(size*16,size*16))
        return (*tensors,geometry,geometry)

    def test_six_presets_and_exact_legacy_strict_loading(self):
        for name,(M,N,T,temporal) in CONFIGS.items():
            codec=DualViewCodec(decoder_config=name)
            self.assertEqual(codec.decoder.history_size,T)
            if name=='baseline':
                old=DecoderV4()
                codec.decoder.load_state_dict(old.state_dict(),strict=True)
                args=self.pair()
                with torch.no_grad():
                    a=old(*args);b=codec.decoder(*args)
                self.assertTrue(torch.equal(a['main_hat'],b['main_hat']))
            else:
                self.assertEqual((codec.decoder.M,codec.decoder.N),(M,N))
                self.assertEqual(codec.decoder.temporal_kind,temporal)
            del codec

    @torch.no_grad()
    def test_default_widths_256_and_unequal_view_with_history(self):
        decoder=SpatioTemporalDecoderV5().eval()
        gm=ImageGeometry((256,256),(256,256));gw=ImageGeometry((128,192),(128,192))
        main=torch.randn(1,64,16,16);wrist=torch.randn(1,64,8,12)
        first=decoder(main,wrist,gm,gw)
        self.assertEqual(first['main_hat'].shape,(1,3,256,256))
        self.assertEqual(first['wrist_hat'].shape,(1,3,128,192))
        e=first['state'].history[-1]
        self.assertEqual(e.main_low.shape,(1,320,16,16))
        self.assertEqual(e.main_high.shape,(1,192,64,64))
        self.assertEqual(e.wrist_high.shape,(1,192,32,48))
        second=decoder(main,wrist,gm,gw,state=first['state'],frame_index=1)
        self.assertEqual(second['history_used'],1)
        self.assertTrue(torch.isfinite(second['main_hat']).all())
        # Equal-resolution 256 input required by the design contract as well.
        equal=decoder(main,main,gm,gm)
        self.assertEqual(equal['wrist_hat'].shape,(1,3,256,256))

    @torch.no_grad()
    def test_history_8_16_order_reset_age_and_detached_cache(self):
        for length in (8,16):
            decoder=self.small(history_size=length).eval();state=None
            args=self.pair()
            for i in range(18):
                out=decoder(*args,state=state,episode_id=17,frame_index=i)
                self.assertEqual(out['history_used'],min(i,length))
                state=out['state']
                self.assertEqual(len(state.history),min(i+1,length))
                self.assertTrue(all(not t.requires_grad and t.grad_fn is None
                                    for e in state.history for t in vars(e).values()))
            with self.assertRaises(ValueError):decoder(*args,state=state,episode_id=17,frame_index=19)
            with self.assertRaises(ValueError):decoder(*args,state=state,episode_id=18,frame_index=18)
            with self.assertRaises(ValueError):decoder(*args,state=replace(state,history=state.history[:-1]),episode_id=17,frame_index=18)
            reset=decoder(*args,state=state,episode_id=18,frame_index=0)
            fresh=decoder(*args,episode_id=18,frame_index=0)
            self.assertTrue(torch.equal(reset['main_hat'],fresh['main_hat']))
            bad=replace(state,signature=('old',))
            with self.assertRaises(ValueError):decoder(*args,state=bad,episode_id=17,frame_index=18)
            # Returned RGB mutation cannot modify cached history.
            original=reset['state'].history[-1].main_rgb.clone()
            reset['main_hat'].zero_()
            self.assertTrue(torch.equal(original,reset['state'].history[-1].main_rgb))

    def test_effective_gradients_for_both_scales_expansion_and_all_fusion_sources(self):
        decoder=self.small(history_size=2)
        args=self.pair(size=2)
        first=decoder(*args)
        second=decoder(*args,state=first['state'],frame_index=1)
        third=decoder(*args,state=second['state'],frame_index=2)
        (third['main_hat'].square().mean()+third['wrist_hat'].square().mean()).backward()
        for name,p in decoder.named_parameters():
            self.assertIsNotNone(p.grad,name)
            self.assertTrue(torch.isfinite(p.grad).all(),name)
            self.assertGreater(p.grad.abs().sum().item(),1e-12,name)

    @torch.no_grad()
    def test_either_history_view_latent_rgb_and_cached_features_affect_current_views(self):
        decoder=self.small().eval();args=self.pair(size=2)
        first=decoder(*args)
        state=first['state'];entry=state.history[0]
        normal=decoder(*args,state=state,frame_index=1)
        for field in vars(entry):
            altered=replace(entry,**{field:getattr(entry,field)+torch.randn_like(getattr(entry,field))*5})
            out=decoder(*args,state=replace(state,history=(altered,)),frame_index=1)
            for view in ('main','wrist'):
                self.assertGreater((normal[view+'_hat']-out[view+'_hat']).abs().max().item(),1e-8,(field,view))

    @torch.no_grad()
    def test_no_future_or_legacy_history_accepted(self):
        decoder=self.small();args=self.pair()
        legacy=DecoderV4()(*args)['state']
        with self.assertRaises(ValueError):decoder(*args,state=legacy,frame_index=1)
        first=decoder(*args)
        with self.assertRaises(ValueError):decoder(*args,state=first['state'],frame_index=2)
        # Current target/future RGB is not part of the decoder interface.
        with self.assertRaises(TypeError):decoder(*args,original_rgb=torch.rand(1,3,16,16))

    @torch.no_grad()
    def test_time_view_and_spatial_identity_are_effective(self):
        module=ViewTemporalSpatialAttention(32,history_size=8)
        q=torch.randn(1,32,3,4)
        a=torch.randn(1,32,3,4);b=torch.randn(1,32,2,3)
        refs=[(a,1,0),(b,2,1)]
        normal=module(q,refs,0)
        variants=[[(a,2,0),(b,1,1)],[(a,1,1),(b,2,0)],[(a.flip(-1),1,0),(b,2,1)]]
        for altered in variants:
            self.assertGreater((normal-module(q,altered,0)).abs().max().item(),1e-7)
        self.assertTrue(torch.equal(module(q,[],0),q))

    @torch.no_grad()
    def test_token_sdpa_matches_dense_query_specific_attention(self):
        module=ViewTemporalSpatialAttention(32,history_size=2,query_chunk=2)
        tokens=torch.randn(1,6,32);pos=coordinates((2,3),tokens.device,tokens.dtype)
        refs=[(torch.randn(1,32,2,2),1,0,None),(torch.randn(1,32,1,3),2,1,None)]
        projected=module._project_references(refs)
        k,v,kpos,kbias=projected
        q=module.query(module.query_norm(tokens+module.view.weight[0])).reshape(1,6,4,8).transpose(1,2)
        bias=(module.relative(pos[:,None]-kpos[None])+kbias[None]).permute(2,0,1)[None]
        weights=(q@k.transpose(-1,-2)/8**.5+bias).softmax(-1)
        dense=(weights@v).transpose(1,2).reshape(1,6,32)
        expected=module.output(dense)*module.gate.tanh()
        torch.testing.assert_close(module._attend(tokens,pos,0,projected),expected,atol=1e-7,rtol=1e-5)

    def test_high_scale_partial_windows_unequal_memory_backward(self):
        module=ViewTemporalSpatialAttention(32,history_size=2,window_size=4,memory_grid=2,query_chunk=8)
        q=torch.randn(2,32,7,9,requires_grad=True)
        refs=[(torch.randn(2,32,5,3),1,0),(torch.randn(2,32,4,8),1,1)]
        out=module(q,refs,0)
        self.assertEqual(q.shape,out.shape)
        out.square().mean().backward()
        self.assertGreater(module.relative[0].weight.grad.abs().sum().item(),0)

    def test_wide_efficient_temporal_long_history(self):
        decoder=self.small(history_size=16,temporal='efficient')
        args=self.pair();state=None
        for i in range(17):
            out=decoder(*args,state=state,frame_index=i);state=out['state']
        self.assertEqual(out['history_used'],16)
        out['main_hat'].mean().backward()
        self.assertGreater(decoder.temporal.keys.weight.grad.abs().sum().item(),0)

    @torch.no_grad()
    def test_explicit_migration_and_packet_bytes_independent_receiver(self):
        old=DualViewCodec(levels=3,decoder_config='baseline').eval();old.update(force=True)
        new=DualViewCodec(levels=3,decoder_config='vts8',M=32,N=32,history_size=2).eval()
        weights,report=migrate_checkpoint(old.state_dict(),new)
        new.load_state_dict(weights,strict=True)
        self.assertTrue(report['initialized_decoder'])
        for k,v in old.analysis.state_dict().items():self.assertTrue(torch.equal(v,new.analysis.state_dict()[k]),k)
        receiver=DualViewCodec(levels=3,decoder_config='vts8',M=32,N=32,history_size=2).eval()
        receiver.load_state_dict(new.state_dict(),strict=True)
        del receiver.analysis.g_a;del receiver.analysis.hyperprior.h_a
        state=real_state=None
        for i in range(4):
            main=torch.rand(1,3,64,64);wrist=torch.rand(1,3,64,128)
            rates=dict(main_rate=(1,.25),wrist_rate=(2,0.))
            packet=old.compress(main,wrist,episode_id=9,frame_index=i,**rates)
            self.assertEqual(packet,new.compress(main,wrist,episode_id=9,frame_index=i,**rates))
            out=new(main,wrist,state=state,episode_id=9,frame_index=i,**rates)
            real=receiver.decompress(packet,state=real_state)
            for view in ('main','wrist'):torch.testing.assert_close(out[view+'_hat'],real[view+'_hat'],atol=1e-6,rtol=1e-6)
            self.assertTrue(torch.equal(out['estimated_bpp'],old(main,wrist,state=None,frame_index=0,**rates)['estimated_bpp']))
            state,real_state=out['state'],real['state']
        broken=old.state_dict().copy();del broken['analysis.gain.log_gain']
        with self.assertRaises(RuntimeError):migrate_checkpoint(broken,new)

    def test_invalid_configuration_rejected(self):
        for kwargs in (dict(history_size=0),dict(M=65),dict(N=127),dict(temporal='unknown')):
            with self.assertRaises(ValueError):SpatioTemporalDecoderV5(**kwargs)
        with self.assertRaises(ValueError):DualViewCodec(decoder_config='unknown')


if __name__=='__main__':unittest.main()
