"""No-RGB Temporal-1 and independent-axis dual-reference-view acceptance."""
from dataclasses import fields, replace
import inspect
import math
import unittest
from unittest.mock import patch

import torch
from torch import nn
from compressai.layers import GDN

from compress.decoder import EfficientCrossAttention
from compress.decoder_v5 import (DualViewTimeAttention, PreviousFrameAttention,
                                TemporalDecodedFrame, TemporalOneDualViewDecoder,
                                build_decoder, migrate_checkpoint)
from compress.dual_view_codec import DualViewCodec
from compress.lightweight_encoder import ImageGeometry, LightweightEncoder


class TemporalOneTimeAttentionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(811)

    def small(self, **kw):
        return TemporalOneDualViewDecoder(M=32, N=32, **kw)

    def pair(self, batch=1, hw=(2, 3), wrist_hw=(3, 2)):
        sizes = (hw, wrist_hw)
        tensors = tuple(torch.randn(batch, 64, *s) for s in sizes)
        geometry = tuple(ImageGeometry(tuple(n*16 for n in s), tuple(n*16 for n in s)) for s in sizes)
        return (*tensors, *geometry)

    def history(self, decoder, frames=3, **pair_kw):
        args = self.pair(**pair_kw)
        state = None
        for frame in range(frames):
            state = decoder(*args, state=state, episode_id=7, frame_index=frame)['state']
        return args, state

    def test_hand_calculated_time_operator_and_t1_identity(self):
        q = torch.tensor([[[[1., 0.]], [[0., 2.]]]])
        k = torch.tensor([[[[1., 0.], [0., 1.]], [[0., 1.], [2., 0.]]]])
        v = torch.tensor([[[[2., 4.], [6., 8.]], [[3., 9.], [5., 7.]]]])
        out, weights = DualViewTimeAttention.attend_temporal(q, k, v)
        expected = torch.empty_like(out)
        for channel in range(2):
            scores = [sum(float(q[0,channel,0,s] * k[0,channel,t,s]) for s in range(2))/math.sqrt(2)
                      for t in range(2)]
            probabilities = [math.exp(score)/sum(math.exp(x) for x in scores) for score in scores]
            for s in range(2):
                expected[0,channel,s] = sum(probabilities[t] * float(v[0,channel,t,s]) for t in range(2))
        torch.testing.assert_close(out, expected)
        torch.testing.assert_close(weights.sum(-1), torch.ones(1,2))
        one, w1 = DualViewTimeAttention.attend_temporal(q, k[:,:,:1], v[:,:,:1])
        torch.testing.assert_close(one, v[:,:,0], atol=0, rtol=0)
        torch.testing.assert_close(w1, torch.ones_like(w1), atol=0, rtol=0)
        # T=1 is not a learned time selector: no Q/K gradient through softmax.
        q.requires_grad_(); k1 = k[:,:,:1].clone().requires_grad_()
        DualViewTimeAttention.attend_temporal(q, k1, v[:,:,:1])[0].sum().backward()
        self.assertEqual(q.grad.abs().sum().item(), 0)
        self.assertEqual(k1.grad.abs().sum().item(), 0)

    def test_time_permutation_invariance_without_encoding(self):
        module = DualViewTimeAttention(32)
        q = torch.randn(2,32,1,6)
        k, v = torch.randn(2,32,4,6), torch.randn(2,32,4,6)
        out, weights = module.attend_temporal(q,k,v)
        order = [2,0,3,1]
        changed, changed_weights = module.attend_temporal(q,k[:,:,order],v[:,:,order])
        torch.testing.assert_close(out,changed)
        torch.testing.assert_close(weights[:,:,order],changed_weights)
        with self.assertRaises(ValueError):module.attend_temporal(q,k,v,time_bias=torch.zeros(32,4))

    def test_qkv_shapes_and_both_reference_views_are_executed(self):
        decoder = self.small()
        _, state = self.history(decoder)
        for scale in ('low','high'):
            module = getattr(decoder,'vts_'+scale)
            refs = module.references(state.history,scale)
            for query_view in (0,1):
                current = torch.randn(2,32,3,4)
                # Repeat historical batch to match current query batch.
                batched = {view: tuple(x.repeat(2,1,1,1) for x in fs) for view,fs in refs.items()}
                calls = []
                original = module.attend_temporal
                def observe(q,k,v):
                    calls.append((q.shape,k.shape,v.shape))
                    return original(q,k,v)
                with patch.object(module,'attend_temporal',side_effect=observe):
                    out = module(current,batched,query_view)
                self.assertEqual(out.shape,current.shape)
                self.assertEqual(calls,[(torch.Size([2,32,1,12]),torch.Size([2,32,3,12]),torch.Size([2,32,3,12]))]*2)

    def test_other_view_history_changes_output_for_both_targets_and_scales(self):
        decoder = self.small()
        _, state = self.history(decoder)
        for scale in ('low','high'):
            module = getattr(decoder,'vts_'+scale)
            refs = module.references(state.history,scale)
            current = torch.randn(1,32,3,4)
            for target in (0,1):
                own = 'main' if target==0 else 'wrist'
                other = 'wrist' if target==0 else 'main'
                normal = module(current,refs,target)
                altered = dict(refs)
                altered[other] = tuple(x + torch.randn_like(x)*4 for x in refs[other])
                torch.testing.assert_close(refs[own][0],altered[own][0],atol=0,rtol=0)
                self.assertGreater((normal-module(current,altered,target)).abs().max().item(),1e-8)

    def test_temporal1_only_previous_own_view_latent_and_features(self):
        decoder = self.small()
        _, state = self.history(decoder,frames=4)
        for scale in ('low','high'):
            short = getattr(decoder,'short_'+scale)
            self.assertIsInstance(short.attention['main'],EfficientCrossAttention)
            query = torch.randn(1,32,3,4)
            history = state.history
            normal = short(query,history,'main',scale)
            older = replace(history[-2], main_latent=history[-2].main_latent + 10,
                            **{'main_'+scale:getattr(history[-2],'main_'+scale)+10})
            changed = (*history[:-2],older,history[-1])
            torch.testing.assert_close(normal,short(query,changed,'main',scale),atol=0,rtol=0)
            for field in ('main_latent','main_'+scale):
                entry = replace(history[-1],**{field:getattr(history[-1],field)+torch.randn_like(getattr(history[-1],field))*4})
                self.assertGreater((normal-short(query,(*history[:-1],entry),'main',scale)).abs().max().item(),0)
            wrong_view = replace(history[-1], wrist_latent=history[-1].wrist_latent+10,
                                 **{'wrist_'+scale:getattr(history[-1],'wrist_'+scale)+10})
            torch.testing.assert_close(normal,short(query,(*history[:-1],wrong_view),'main',scale),atol=0,rtol=0)

    def test_parallel_query_sources_concat_and_single_residual(self):
        decoder = self.small()
        _, state = self.history(decoder)
        for scale in ('low','high'):
            main,wrist = torch.randn(1,32,3,4),torch.randn(1,32,2,5)
            spatial = decoder.atten_3 if scale=='low' else decoder.atten_4
            short,long = getattr(decoder,'short_'+scale),getattr(decoder,'vts_'+scale)
            calls = {key:[] for key in ('spatial','short','long','fusion')}
            handles = []
            for key,module in [('spatial',spatial),('short',short),('long',long)]:
                handles.append(module.register_forward_pre_hook(lambda m,a,key=key:calls[key].append(a)))
            fusion = getattr(decoder,'fusion_'+scale)
            for view in ('main','wrist'):
                handles.append(fusion[view].register_forward_pre_hook(lambda m,a:calls['fusion'].append(a[0])))
            expected = decoder.branch_deltas(main,wrist,state.history,scale)
            for values in calls.values():values.clear()
            out = decoder._parallel(main,wrist,state.history,scale)
            self.assertIs(calls['spatial'][0][0],main)
            self.assertIs(calls['spatial'][0][1],wrist)
            for index,(view,current) in enumerate((('main',main),('wrist',wrist))):
                self.assertIs(calls['short'][index][0],current)
                self.assertIs(calls['long'][index][0],current)
                cat = torch.cat(expected[view],dim=1)
                self.assertEqual(cat.shape[1],96)
                torch.testing.assert_close(calls['fusion'][index],cat)
                torch.testing.assert_close(out[index],current+fusion[view](cat))
            for h in handles:h.remove()
        disabled = self.small(branch_mask=(False,False,False))
        out = disabled._parallel(main,wrist,state.history,'low')
        torch.testing.assert_close(out[0],main,atol=0,rtol=0)
        torch.testing.assert_close(out[1],wrist,atol=0,rtol=0)

    def test_no_rgb_modules_cache_and_first_frame_zero(self):
        decoder = self.small()
        args = self.pair()
        for name,_ in decoder.named_modules():
            self.assertNotIn('rgb',name.lower())
            self.assertFalse(any(word in name for word in ('age','relative','window','memory_grid')))
        for cls in (PreviousFrameAttention,DualViewTimeAttention):
            source = inspect.getsource(cls)
            self.assertNotIn("'_rgb'",source)
            self.assertNotIn('adaptive_avg_pool',source)
            self.assertNotIn('scaled_dot_product_attention',source)
        result = decoder(*args)
        entry = result['state'].history[0]
        self.assertIsInstance(entry,TemporalDecodedFrame)
        self.assertFalse(any('rgb' in f.name for f in fields(entry)))
        self.assertEqual(len([x for x in vars(entry).values() if isinstance(x,torch.Tensor)]),6)
        for scale in ('low','high'):
            current = torch.randn(1,32,3,4)
            deltas = decoder.branch_deltas(current,current,(),scale)
            for view in ('main','wrist'):
                for delta in deltas[view][1:]:self.assertEqual(torch.count_nonzero(delta).item(),0)
                self.assertIsNone(getattr(decoder,'fusion_'+scale)[view][0].bias)
        for key in ('main_hat','wrist_hat'):
            self.assertTrue(torch.isfinite(result[key]).all())
            self.assertGreaterEqual(result[key].min().item(),0)
            self.assertLessEqual(result[key].max().item(),1)

    def test_two_scales_all_branches_and_cross_view_qkv_gradients(self):
        decoder = self.small()
        args,state = self.history(decoder,frames=3,batch=2)
        out = decoder(*args,state=state,episode_id=7,frame_index=3)
        (out['main_hat'].square().mean()+out['wrist_hat'].square().mean()).backward()
        for prefix in ('atten_3','atten_4','short_low','short_high','vts_low','vts_high','fusion_low','fusion_high'):
            params=[p for n,p in decoder.named_parameters() if n.startswith(prefix)]
            self.assertTrue(params)
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in params),prefix)
            self.assertGreater(sum(p.grad.abs().sum().item() for p in params),0,prefix)
        for scale in ('low','high'):
            module=getattr(decoder,'vts_'+scale)
            for layer in (module.query,module.key,module.value,module.view_fusion[0]):
                self.assertGreater(layer.weight.grad.abs().sum().item(),0)
            # Both historical inputs themselves must carry gradients at T>=2.
            current=torch.randn(1,32,2,3,requires_grad=True)
            refs={view:tuple(torch.randn(1,32,2,3,requires_grad=True) for _ in range(3))
                  for view in ('main','wrist')}
            module(current,refs).square().mean().backward()
            for view in ('main','wrist'):
                self.assertGreater(sum(x.grad.abs().sum().item() for x in refs[view]),0,view)

    @torch.no_grad()
    def test_continuous_window_reset_and_strict_state_validation(self):
        decoder = self.small().eval()
        args = self.pair()
        state=None
        for frame in range(11):
            result=decoder(*args,state=state,episode_id=9,frame_index=frame)
            self.assertEqual(result['short_history_used'],min(frame,1))
            self.assertEqual(result['long_history_used'],min(frame,8))
            state=result['state']
            self.assertEqual([e.frame_index for e in state.history],list(range(max(0,frame-7),frame+1)))
            self.assertTrue(all(not t.requires_grad and t.grad_fn is None for e in state.history
                                for t in vars(e).values() if isinstance(t,torch.Tensor)))
        cached = state.history[-1].main_low.clone()
        state.history[-1].main_latent.zero_()
        self.assertFalse(torch.equal(args[0],state.history[-1].main_latent))
        torch.testing.assert_close(cached,state.history[-1].main_low,atol=0,rtol=0)
        for bad in (replace(state,history=state.history[::-1]),replace(state,history=state.history[:-1]),
                    replace(state,signature=('old',)),
                    replace(state,history=(*state.history[:-1],replace(state.history[-1],episode_id=10)))):
            with self.assertRaises(ValueError):decoder(*args,state=bad,episode_id=9,frame_index=11)
        for episode,frame in ((10,11),(9,12)):
            with self.assertRaises(ValueError):decoder(*args,state=state,episode_id=episode,frame_index=frame)
        normal=decoder(*args,episode_id=10)
        reset=decoder(*args,state=state,episode_id=10)
        torch.testing.assert_close(normal['main_hat'],reset['main_hat'],atol=0,rtol=0)
        old=build_decoder('tri_fusion8',M=32,N=32)(*args)['state']
        with self.assertRaises(ValueError):decoder(*args,state=old,frame_index=1)
        with self.assertRaises(TypeError):decoder(*args,original_rgb=torch.rand(1,3,32,48))

    @torch.no_grad()
    def test_temporal_window_clipping_and_no_time_permutation_claim(self):
        decoder=self.small()
        _,state=self.history(decoder,frames=9)
        module=decoder.vts_low
        reference=module.references(state.history,'low')
        extra=replace(state.history[0],main_low=state.history[0].main_low+100)
        clipped=module.references((extra,*state.history),'low')
        current=torch.randn(1,32,2,3)
        torch.testing.assert_close(module(current,reference),module(current,clipped),atol=0,rtol=0)
        # Reordering K/V together is mathematically invariant without time encoding,
        # while reordering state entries is independently rejected by the decoder.
        reversed_refs={view:features[::-1] for view,features in reference.items()}
        torch.testing.assert_close(module(current,reference),module(current,reversed_refs),atol=1e-7,rtol=1e-5)

    @torch.no_grad()
    def test_default_width_256_shapes_different_views_and_batches(self):
        decoder=build_decoder('tri_temporal1_t8').eval()
        self.assertEqual((decoder.M,decoder.N,decoder.history_size,decoder.short_history_size),(320,192,8,1))
        for batch in (1,2):
            args=self.pair(batch=batch,hw=(16,16),wrist_hw=(8,12))
            first=decoder(*args)
            second=decoder(*args,state=first['state'],frame_index=1)
            self.assertEqual(second['main_hat'].shape,(batch,3,256,256))
            self.assertEqual(second['wrist_hat'].shape,(batch,3,128,192))
            entry=second['state'].history[-1]
            self.assertEqual(entry.main_low.shape,(batch,320,16,16))
            self.assertEqual(entry.main_high.shape,(batch,192,64,64))
        equal=self.pair(hw=(16,16),wrist_hw=(16,16))
        self.assertEqual(decoder(*equal)['wrist_hat'].shape,(1,3,256,256))

    def test_activation_and_legacy_configuration_regression(self):
        encoder=LightweightEncoder()
        self.assertEqual([sum(isinstance(x,nn.GELU) for x in s) for s in encoder.stages],[1,1,1,0])
        self.assertIsInstance(encoder.stages[-1][-1],nn.Conv2d)
        decoder=self.small()
        self.assertFalse(any(isinstance(m,GDN) for m in decoder.modules()))
        for layer in (decoder.decoder_1[1],decoder.decoder_1[3],decoder.decoder_2[1]):self.assertIsInstance(layer,nn.GELU)
        self.assertTrue(any(isinstance(m,nn.LeakyReLU) for m in decoder.atten_3.modules()))
        for config in ('baseline','vts8','vts16','wide-ldmic'):
            old=build_decoder(config)
            self.assertEqual(sum(isinstance(m,GDN) for m in old.modules()),3)
        old=build_decoder('tri_fusion8',M=32,N=32)
        self.assertEqual(old.short_history_size,2)
        self.assertTrue(hasattr(old.vts_low,'rgb'))
        self.assertIsInstance(old.vts_low.relative[1],nn.SiLU)

    @torch.no_grad()
    def test_real_packet_10_frames_independent_receiver_and_explicit_migration(self):
        old=DualViewCodec(levels=3,decoder_config='tri_fusion8',M=32,N=32).eval()
        old.update(force=True)
        codec=DualViewCodec(levels=3,decoder_config='tri_temporal1_t8',M=32,N=32).eval()
        with self.assertRaises(RuntimeError):codec.load_state_dict(old.state_dict(),strict=True)
        weights,report=migrate_checkpoint(old.state_dict(),codec)
        codec.load_state_dict(weights,strict=True)
        self.assertTrue(report['rebuild_optimizer'])
        self.assertTrue(report['restart_history_from_frame_zero'])
        self.assertTrue(any('rgb' in k for k in report['discarded_source']))
        self.assertTrue(any('vts_low.query' in k for k in report['initialized_decoder']))
        self.assertIn('new architecture',report['decoder_semantics'])
        self.assertEqual(len(torch.optim.Adam(codec.parameters()).state),0)
        receiver=DualViewCodec(levels=3,decoder_config='tri_temporal1_t8',M=32,N=32).eval()
        receiver.load_state_dict(codec.state_dict(),strict=True)
        del receiver.analysis.g_a
        del receiver.analysis.hyperprior.h_a
        state=real_state=None
        for frame in range(10):
            main,wrist=torch.rand(1,3,64,64),torch.rand(1,3,64,128)
            rates=dict(main_rate=(1,.25),wrist_rate=(2,0.))
            packet=codec.compress(main,wrist,episode_id=4,frame_index=frame,**rates)
            self.assertEqual(packet,old.compress(main,wrist,episode_id=4,frame_index=frame,**rates))
            real=receiver.decompress(packet,state=real_state)
            normal=codec(main,wrist,state=state,episode_id=4,frame_index=frame,**rates)
            for view in ('main','wrist'):
                torch.testing.assert_close(real[view+'_hat'],normal[view+'_hat'],atol=1e-6,rtol=1e-6)
                self.assertTrue(torch.isfinite(real[view+'_hat']).all())
            self.assertTrue(packet.startswith(b'ECD1'))
            self.assertGreater(real['actual_bpp'],0)
            self.assertTrue(torch.isfinite(normal['estimated_bpp']))
            self.assertEqual(real['rates'],dict(main=dict(level=1,alpha=.25),wrist=dict(level=2,alpha=0.)))
            self.assertFalse(any('rgb' in f.name for f in fields(real['state'].history[-1])))
            state,real_state=normal['state'],real['state']

    def test_invalid_shapes_and_configuration_are_rejected(self):
        for kw in (dict(history_size=2),dict(short_history_size=2),dict(branch_mask=(1,1,1))):
            with self.assertRaises(ValueError):self.small(**kw)
        with self.assertRaises(ValueError):TemporalOneDualViewDecoder(M=65)
        for history in (0,9,True):
            with self.assertRaises(ValueError):DualViewTimeAttention(32,history)
        q=torch.randn(1,32,1,6)
        with self.assertRaises(ValueError):DualViewTimeAttention.attend_temporal(q,torch.randn(1,32,9,6),torch.randn(1,32,9,6))
        with self.assertRaises(ValueError):DualViewTimeAttention.attend_temporal(q,torch.randn(1,32,2,5),torch.randn(1,32,2,5))


if __name__=='__main__':unittest.main(verbosity=2)
