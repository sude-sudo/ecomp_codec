"""CPU acceptance of parallel query dependencies, causal memories and real coding."""
from dataclasses import replace
import unittest

import torch
from torch import nn
from compressai.layers import GDN

from compress.decoder import EfficientCrossAttention
from compress.decoder_tri_fusion import ThreeBranchDecoder
from compress.decoder_v5 import build_decoder, migrate_checkpoint
from compress.dual_view_codec import DualViewCodec
from compress.lightweight_encoder import ImageGeometry, LightweightEncoder


class ThreeBranchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(703)

    def small(self, **kw):
        return ThreeBranchDecoder(M=32, N=32, **kw)

    def pair(self, batch=1, hw=(2, 2), wrist_hw=None):
        wrist_hw = hw if wrist_hw is None else wrist_hw
        tensors = (torch.randn(batch, 64, *hw), torch.randn(batch, 64, *wrist_hw))
        geometries = tuple(ImageGeometry(tuple(n * 16 for n in size), tuple(n * 16 for n in size))
                           for size in (hw, wrist_hw))
        return (*tensors, *geometries)

    def test_activation_boundaries_and_legacy_decoder_regression(self):
        encoder = LightweightEncoder()
        self.assertEqual([sum(isinstance(m, nn.GELU) for m in s) for s in encoder.stages], [1, 1, 1, 0])
        self.assertIsInstance(encoder.stages[-1][-1], nn.Conv2d)
        decoder = build_decoder('tri_fusion8')
        self.assertEqual((decoder.M, decoder.N, decoder.history_size, decoder.short_history_size), (320, 192, 8, 2))
        self.assertFalse(any(isinstance(m, GDN) for m in decoder.modules()))
        for module in (decoder.decoder_1[1], decoder.decoder_1[3], decoder.decoder_2[1]):
            self.assertIsInstance(module, nn.GELU)
        self.assertTrue(any(isinstance(m, nn.LeakyReLU) for m in decoder.atten_3.modules()))
        self.assertIsInstance(decoder.vts_low.relative[1], nn.SiLU)
        self.assertIsInstance(decoder.vts_low.fuse[1], nn.GELU)
        self.assertEqual(decoder.vts_low.heads, 8)
        self.assertIsInstance(decoder.short_low.attention['main'], EfficientCrossAttention)
        for config in ('baseline', 'vts8', 'vts16', 'wide-ldmic'):
            old = build_decoder(config)
            self.assertEqual(sum(isinstance(m, GDN) for m in old.modules()), 3)
        self.assertNotEqual(decoder.signature, build_decoder('vts8').signature)

    @torch.no_grad()
    def test_exact_zero_temporal_and_bias_free_single_residual(self):
        decoder = self.small()
        for scale, channels in (('low', 32), ('high', 32)):
            main, wrist = [torch.randn(2, channels, 2, 3) for _ in range(2)]
            deltas = decoder.branch_deltas(main, wrist, (), scale)
            for view in ('main', 'wrist'):
                for delta in deltas[view][1:]:
                    self.assertEqual(torch.count_nonzero(delta).item(), 0)
            for f in getattr(decoder, 'fusion_' + scale).values():
                self.assertIsNone(f[0].bias)
                self.assertEqual(torch.count_nonzero(f(torch.zeros(2, 96, 2, 3))).item(), 0)
        disabled = self.small(branch_mask=(False, False, False))
        outputs = disabled._parallel(main, wrist, (), 'low')
        self.assertTrue(torch.equal(outputs[0], main))
        self.assertTrue(torch.equal(outputs[1], wrist))

    @torch.no_grad()
    def test_parallel_inputs_and_actual_channel_concat_at_both_scales(self):
        decoder = self.small()
        first = decoder(*self.pair())
        history = first['state'].history
        for scale in ('low', 'high'):
            main, wrist = torch.randn(1, 32, 2, 3), torch.randn(1, 32, 3, 2)
            spatial = decoder.atten_3 if scale == 'low' else decoder.atten_4
            short = getattr(decoder, 'short_' + scale)
            long = getattr(decoder, 'vts_' + scale)
            seen = {'spatial': [], 'short': [], 'long': [], 'fusion': []}
            handles = []
            for name, module in (('spatial', spatial), ('short', short), ('long', long)):
                handles.append(module.register_forward_pre_hook(lambda m, args, name=name: seen[name].append(args)))
            fusion = getattr(decoder, 'fusion_' + scale)
            for view in ('main', 'wrist'):
                handles.append(fusion[view].register_forward_pre_hook(lambda m, args: seen['fusion'].append(args[0])))
            expected = decoder.branch_deltas(main, wrist, history, scale)
            for values in seen.values():
                values.clear()
            out = decoder._parallel(main, wrist, history, scale)
            self.assertIs(seen['spatial'][0][0], main)
            self.assertIs(seen['spatial'][0][1], wrist)
            for index, (view, current) in enumerate((('main', main), ('wrist', wrist))):
                self.assertIs(seen['short'][index][0], current)
                self.assertIs(seen['long'][index][0], current)
                concatenated = torch.cat(expected[view], dim=1)
                self.assertEqual(concatenated.shape[1], 96)
                torch.testing.assert_close(seen['fusion'][index], concatenated)
                torch.testing.assert_close(out[index], current + fusion[view](concatenated))
            for handle in handles:
                handle.remove()

    @torch.no_grad()
    def test_continuous_frames_windows_reset_order_and_cloned_cache(self):
        decoder = self.small().eval()
        args = self.pair()
        state = None
        for frame in range(10):
            result = decoder(*args, state=state, episode_id=11, frame_index=frame)
            self.assertEqual(result['short_history_used'], min(frame, 2))
            self.assertEqual(result['long_history_used'], min(frame, 8))
            state = result['state']
            self.assertEqual([e.frame_index for e in state.history], list(range(max(0, frame - 7), frame + 1)))
            for entry in state.history:
                for tensor in vars(entry).values():
                    if isinstance(tensor, torch.Tensor):
                        self.assertFalse(tensor.requires_grad)
                        self.assertIsNone(tensor.grad_fn)
        original = state.history[-1].main_rgb.clone()
        result['main_hat'].zero_()
        self.assertTrue(torch.equal(state.history[-1].main_rgb, original))
        bad_states = [replace(state, history=state.history[::-1]),
                      replace(state, history=state.history[:-1]),
                      replace(state, signature=('vts8',)),
                      replace(state, history=(*state.history[:-1], replace(state.history[-1], episode_id=99)))]
        for bad in bad_states:
            with self.assertRaises(ValueError):
                decoder(*args, state=bad, episode_id=11, frame_index=10)
        for episode, frame in ((12, 10), (11, 11)):
            with self.assertRaises(ValueError):
                decoder(*args, state=state, episode_id=episode, frame_index=frame)
        fresh = decoder(*args, episode_id=12)
        reset = decoder(*args, state=state, episode_id=12)
        torch.testing.assert_close(fresh['main_hat'], reset['main_hat'], atol=0, rtol=0)
        with self.assertRaises(TypeError):
            decoder(*args, original_rgb=torch.rand(1, 3, 32, 32))

    def test_all_three_branches_and_fusion_receive_finite_nonzero_gradients(self):
        decoder = self.small()
        args = self.pair(batch=2, hw=(2, 3), wrist_hw=(3, 2))
        first = decoder(*args)
        second = decoder(*args, state=first['state'], frame_index=1)
        result = decoder(*args, state=second['state'], frame_index=2)
        (result['main_hat'].square().mean() + result['wrist_hat'].square().mean()).backward()
        prefixes = ('atten_3', 'atten_4', 'short_low', 'short_high', 'vts_low', 'vts_high', 'fusion_low', 'fusion_high')
        for prefix in prefixes:
            parameters = [(n, p) for n, p in decoder.named_parameters() if n.startswith(prefix)]
            self.assertTrue(parameters)
            for name, parameter in parameters:
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            self.assertGreater(sum(p.grad.abs().sum().item() for _, p in parameters), 0, prefix)
        for scale in ('low', 'high'):
            short = getattr(decoder, 'short_' + scale)
            long = getattr(decoder, 'vts_' + scale)
            for view in ('main', 'wrist'):
                for layer in (short.attention[view].queries, short.attention[view].keys, short.attention[view].values):
                    self.assertGreater(layer.weight.grad.abs().sum().item(), 0)
            for layer in (long.query, long.key, long.value, long.relative[0], long.age, long.fuse[0]):
                self.assertGreater(layer.weight.grad.abs().sum().item(), 0)

    @torch.no_grad()
    def test_short_own_view_only_last_two_and_long_all_sources(self):
        decoder = self.small()
        args = self.pair()
        state = None
        for frame in range(4):
            result = decoder(*args, state=state, frame_index=frame)
            state = result['state']
        history = state.history
        main, wrist = [torch.randn(1, 32, 2, 2) for _ in range(2)]
        base = decoder.branch_deltas(main, wrist, history, 'low')
        for field in ('main_latent', 'main_rgb', 'wrist_latent', 'wrist_rgb', 'main_low', 'wrist_low'):
            entry = history[-1]
            altered = replace(entry, **{field: getattr(entry, field) + torch.randn_like(getattr(entry, field)) * 3})
            changed = decoder.branch_deltas(main, wrist, (*history[:-1], altered), 'low')
            own = field.split('_')[0]
            other = 'wrist' if own == 'main' else 'main'
            torch.testing.assert_close(base[other][1], changed[other][1], atol=0, rtol=0)
            if not field.endswith('_low'):
                self.assertGreater((base[own][1] - changed[own][1]).abs().max().item(), 0)
            for view in ('main', 'wrist'):
                self.assertGreater((base[view][2] - changed[view][2]).abs().max().item(), 0, (field, view))
        older = replace(history[0], main_latent=history[0].main_latent + 10)
        changed = decoder.branch_deltas(main, wrist, (older, *history[1:]), 'low')
        torch.testing.assert_close(base['main'][1], changed['main'][1], atol=0, rtol=0)
        self.assertGreater((base['main'][2] - changed['main'][2]).abs().max().item(), 0)
        # High-scale decoder features are independent VTS inputs as well.
        base_high = decoder.branch_deltas(main, wrist, history, 'high')
        for field in ('main_high', 'wrist_high'):
            entry = history[-1]
            altered = replace(entry, **{field: getattr(entry, field) + torch.randn_like(getattr(entry, field)) * 3})
            changed = decoder.branch_deltas(main, wrist, (*history[:-1], altered), 'high')
            self.assertGreater((base_high['main'][2] - changed['main'][2]).abs().max().item(), 0)
        # Public helper also bounds long references to the most recent eight.
        extra = (replace(history[0], main_latent=history[0].main_latent + 100),) * 5
        bounded = decoder.branch_deltas(main, wrist, extra + history + history, 'low')
        last_eight = decoder.branch_deltas(main, wrist, history + history, 'low')
        for view in ('main', 'wrist'):
            for a, b in zip(bounded[view], last_eight[view]):
                torch.testing.assert_close(a, b, atol=0, rtol=0)

    @torch.no_grad()
    def test_ablation_masks_do_not_change_other_branch_outputs(self):
        decoder = self.small()
        state = decoder(*self.pair())['state']
        main, wrist = [torch.randn(1, 32, 2, 2) for _ in range(2)]
        for scale in ('low', 'high'):
            base = decoder.branch_deltas(main, wrist, state.history, scale)
            for disabled in range(3):
                mask = tuple(i != disabled for i in range(3))
                ablation = self.small(branch_mask=mask)
                ablation.load_state_dict(decoder.state_dict(), strict=True)
                changed = ablation.branch_deltas(main, wrist, state.history, scale)
                for view in ('main', 'wrist'):
                    self.assertEqual(torch.count_nonzero(changed[view][disabled]).item(), 0)
                    for index in range(3):
                        if index != disabled:
                            torch.testing.assert_close(base[view][index], changed[view][index], atol=0, rtol=0)
                out = ablation(*self.pair())
                next_out = ablation(*self.pair(), state=out['state'], frame_index=1)
                self.assertEqual(next_out['history_used'], 1)
        low_only = self.small(fusion_scales=('low',))
        first = low_only(*self.pair())
        second = low_only(*self.pair(), state=first['state'], frame_index=1)
        self.assertTrue(torch.isfinite(second['main_hat']).all())

    @torch.no_grad()
    def test_default_widths_256_both_batch_sizes_and_unequal_views(self):
        decoder = ThreeBranchDecoder().eval()
        for batch in (1, 2):
            # B=2 also exercises a 256px view; the other camera is smaller.
            args = self.pair(batch=batch, hw=(16, 16), wrist_hw=(8, 12))
            first = decoder(*args)
            second = decoder(*args, state=first['state'], frame_index=1)
            self.assertEqual(second['main_hat'].shape, (batch, 3, 256, 256))
            self.assertEqual(second['wrist_hat'].shape, (batch, 3, 128, 192))
            for view in ('main', 'wrist'):
                image = second[view + '_hat']
                self.assertTrue(torch.isfinite(image).all())
                self.assertGreaterEqual(image.min().item(), 0)
                self.assertLessEqual(image.max().item(), 1)
            entry = second['state'].history[-1]
            self.assertEqual(entry.main_low.shape, (batch, 320, 16, 16))
            self.assertEqual(entry.main_high.shape, (batch, 192, 64, 64))
        equal_args = self.pair(hw=(16, 16))
        self.assertEqual(decoder(*equal_args)['wrist_hat'].shape, (1, 3, 256, 256))

    @torch.no_grad()
    def test_explicit_migration_discarded_igdns_and_strict_roundtrip(self):
        old = DualViewCodec(levels=3, decoder_config='vts8', M=32, N=32).eval()
        old.update(force=True)
        new = DualViewCodec(levels=3, decoder_config='tri_fusion8', M=32, N=32).eval()
        weights, report = migrate_checkpoint(old.state_dict(), new)
        new.load_state_dict(weights, strict=True)
        self.assertTrue(report['rebuild_optimizer'])
        self.assertTrue(report['restart_history_from_frame_zero'])
        self.assertTrue(any('decoder_1.1.beta' in name for name in report['discarded_source']))
        self.assertTrue(any('short_low' in name for name in report['initialized_decoder']))
        self.assertTrue(any('fusion_low' in name for name in report['initialized_decoder']))
        self.assertTrue(report['copied_analysis'])
        self.assertTrue(report['copied_decoder'])
        self.assertIn('semantics', report['analysis_semantics'])
        with self.assertRaises(RuntimeError):
            new.load_state_dict(old.state_dict(), strict=True)
        fresh_optimizer = torch.optim.Adam(new.parameters())
        self.assertEqual(len(fresh_optimizer.state), 0)
        receiver = DualViewCodec(levels=3, decoder_config='tri_fusion8', M=32, N=32).eval()
        receiver.load_state_dict(new.state_dict(), strict=True)
        # Prove the receiver cannot use original images or analysis transforms.
        del receiver.analysis.g_a
        del receiver.analysis.hyperprior.h_a
        state = real_state = None
        for frame in range(10):
            main, wrist = torch.rand(1, 3, 64, 64), torch.rand(1, 3, 64, 128)
            rates = dict(main_rate=(1, .25), wrist_rate=(2, 0.))
            packet = new.compress(main, wrist, episode_id=9, frame_index=frame, **rates)
            self.assertTrue(packet.startswith(b'ECD1'))
            normal = new(main, wrist, state=state, episode_id=9, frame_index=frame, **rates)
            real = receiver.decompress(packet, state=real_state)
            for view in ('main', 'wrist'):
                torch.testing.assert_close(normal[view + '_hat'], real[view + '_hat'], atol=1e-6, rtol=1e-6)
            self.assertGreater(real['actual_bpp'], 0)
            self.assertTrue(torch.isfinite(torch.tensor(real['actual_bpp'])))
            self.assertEqual(real['rates']['main'], dict(level=1, alpha=.25))
            self.assertEqual(real['rates']['wrist'], dict(level=2, alpha=0.))
            state, real_state = normal['state'], real['state']
        broken = old.state_dict().copy()
        del broken['analysis.gain.log_gain']
        with self.assertRaises(RuntimeError):
            migrate_checkpoint(broken, new)

    def test_invalid_configs(self):
        for kw in (dict(history_size=2), dict(short_history_size=3), dict(branch_mask=(1, 1, 1)),
                   dict(fusion_scales=('high',)), dict(M=65)):
            with self.assertRaises(ValueError):
                self.small(**kw) if 'M' not in kw else ThreeBranchDecoder(**kw)


if __name__ == '__main__':
    unittest.main(verbosity=2)
