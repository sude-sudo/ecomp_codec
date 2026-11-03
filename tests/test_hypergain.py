"""HyperGain placement, learned interpolation, wire compatibility and CPU checks."""

import copy
import io
import struct
import unittest
from unittest import mock

import torch
from torch.nn import functional as F

from compress.dual_view_codec import DualViewCodec
from compress.gain import GainedEncoderHyperprior, LatentGain, with_identity_hypergains
from compress.hyperprior import EncoderHyperprior, LatentHyperprior
from compress.lightweight_encoder import prepare_rgb


POINTS = ((0, 0.), (3, .5), (5, .25), (6, 1.), (7, 0.))


def perturb(gain):
    """Nonunit, channel-dependent, independently learned-looking hyper tables."""
    with torch.no_grad():
        gain.log_hyper_gain.copy_(torch.linspace(.2, .7, 8)[:, None] + .1 * torch.randn(8, 64))
        gain.log_inverse_hyper_gain.copy_(torch.linspace(-.4, -.1, 8)[:, None] + .1 * torch.randn(8, 64))


def entropy_bits(packet):
    """Independently parse ECG1 -> ECH1, excluding all transport metadata."""
    magic, _, _ = struct.unpack_from(">4sId", packet)
    assert magic == b"ECG1"
    magic, batch, h, w = struct.unpack_from(">4sIII", packet, 16)
    assert magic == b"ECH1"
    offset, y_bytes, z_bytes = 32, 0, 0
    for _ in range(batch):
        ny, nz = struct.unpack_from(">II", packet, offset)
        offset += 8 + ny + nz
        y_bytes += ny
        z_bytes += nz
    assert offset == len(packet)
    return y_bytes * 8, z_bytes * 8, batch * h * w, 32 + 8 * batch


class HyperGainAcceptance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(20261103)

    def test_initialization_and_interpolation(self):
        gain = LatentGain()
        self.assertEqual(sum(p.numel() for p in gain.parameters()), 2048)
        for table in (gain.log_hyper_gain, gain.log_inverse_hyper_gain):
            self.assertEqual(tuple(table.shape), (8, 64))
            self.assertTrue(table.requires_grad)
        for point in POINTS:
            for factor in gain.hyper_factors(*point):
                self.assertEqual(tuple(factor.shape), (1, 64, 1, 1))
                torch.testing.assert_close(factor, torch.ones_like(factor), atol=0, rtol=0)
        y_before = [p.detach().clone() for p in gain.factors(3, .5)]
        perturb(gain)
        for point in POINTS:
            level, alpha = point
            neighbor = min(level + 1, 7)
            for table, factor in zip((gain.log_hyper_gain, gain.log_inverse_hyper_gain),
                                     gain.hyper_factors(*point)):
                expected = table[level].exp().pow(1-alpha) * table[neighbor].exp().pow(alpha)
                torch.testing.assert_close(factor.flatten(), expected)
                self.assertTrue(torch.isfinite(factor).all() and (factor > 0).all())
            hg, ihg = gain.hyper_factors(*point)
            self.assertFalse(torch.allclose(hg * ihg, torch.ones_like(hg)))
        for a, b in zip(gain.hyper_factors(6, 1.), gain.hyper_factors(7, 0.)):
            torch.testing.assert_close(a, b, atol=0, rtol=0)
        for a, b in zip(y_before, gain.factors(3, .5)):
            torch.testing.assert_close(a, b, atol=0, rtol=0)

    def test_forward_gain_placement_and_rate_gradients(self):
        prior, gain = LatentHyperprior(), LatentGain()
        perturb(gain)
        y = torch.randn(2, 64, 8, 12, requires_grad=True)
        for level, alpha in POINTS:
            gain.zero_grad(set_to_none=True)
            prior.zero_grad(set_to_none=True)
            yg = gain(y, level, alpha)
            hg, ihg = gain.hyper_factors(level, alpha)
            with mock.patch.object(prior.entropy_bottleneck, "forward", wraps=prior.entropy_bottleneck.forward) as eb, \
                 mock.patch.object(prior.h_s, "forward", wraps=prior.h_s.forward) as hs:
                out = prior(yg, hyper_gain=hg, inverse_hyper_gain=ihg)
            z = prior.h_a(yg.abs())
            torch.testing.assert_close(eb.call_args.args[0], z * hg, atol=0, rtol=0)
            # h_s must retain an autograd path through InverseHyperGain.
            self.assertTrue(hs.call_args.args[0].requires_grad)
            loss = -sum(p.log().mean() for p in out["likelihoods"].values())
            loss += gain.inverse(out["y_hat"], level, alpha).square().mean()
            loss.backward()
            active = {level} if alpha == 0 else {level + 1} if alpha == 1 else {level, level + 1}
            for name, table in gain.named_parameters():
                self.assertTrue(torch.isfinite(table.grad).all(), name)
                for row in range(8):
                    norm = table.grad[row].abs().sum().item()
                    self.assertGreater(norm, 0, (name, row)) if row in active else self.assertEqual(norm, 0)

    def test_real_coding_uses_decoded_z_and_identical_indexes(self):
        prior, gain = LatentHyperprior().eval(), LatentGain()
        perturb(gain)
        # Avoid a degenerate untrained prior whose entire sigma is lower-bounded:
        # nonzero hyperlatents and positive scale logits make omissions observable.
        with torch.no_grad():
            prior.h_a[-1].bias.add_(1.)
            prior.h_s[-2].bias.add_(.6)
        prior.update(force=True)
        receiver = LatentHyperprior().eval()
        receiver.load_state_dict(prior.state_dict())
        del receiver.h_a
        y = torch.randn(2, 64, 8, 12) * 3
        for point in POINTS:
            yg = gain(y, *point)
            hg, ihg = gain.hyper_factors(*point)
            with mock.patch.object(prior.entropy_bottleneck, "compress", wraps=prior.entropy_bottleneck.compress) as eb, \
                 mock.patch.object(prior.h_s, "forward", wraps=prior.h_s.forward) as sender_hs, \
                 mock.patch.object(receiver.h_s, "forward", wraps=receiver.h_s.forward) as receiver_hs:
                packet = prior.compress(yg, hyper_gain=hg, inverse_hyper_gain=ihg)
                restored = receiver.decompress(packet["strings"], packet["shape"], inverse_hyper_gain=ihg)
            torch.testing.assert_close(eb.call_args.args[0], prior.h_a(yg.abs()) * hg, atol=0, rtol=0)
            decoded_zg = receiver.entropy_bottleneck.decompress(packet["strings"][1], packet["shape"])
            self.assertGreater(decoded_zg.abs().sum().item(), 0)
            for hs in (sender_hs, receiver_hs):
                torch.testing.assert_close(hs.call_args.args[0], decoded_zg * ihg, atol=0, rtol=0)
            indexes_sender = prior.gaussian_conditional.build_indexes(prior.h_s(sender_hs.call_args.args[0]))
            indexes_receiver = receiver.gaussian_conditional.build_indexes(receiver.h_s(receiver_hs.call_args.args[0]))
            torch.testing.assert_close(indexes_sender, indexes_receiver, atol=0, rtol=0)
            out = prior(yg, hyper_gain=hg, inverse_hyper_gain=ihg)
            torch.testing.assert_close(restored, out["y_hat"], atol=0, rtol=0)
            torch.testing.assert_close(restored, yg.round(), atol=0, rtol=0)
            expected_z = prior.entropy_bottleneck(prior.h_a(yg.abs()) * hg)[0] * ihg
            torch.testing.assert_close(sender_hs.call_args.args[0], expected_z, atol=0, rtol=0)

    def test_single_view_fresh_receiver_payload_accounting(self):
        model = GainedEncoderHyperprior().eval()
        perturb(model.gain)
        model.update(force=True)
        checkpoint = io.BytesIO()
        torch.save(model.state_dict(), checkpoint)
        checkpoint.seek(0)
        receiver = GainedEncoderHyperprior().eval()
        receiver.load_state_dict(torch.load(checkpoint, weights_only=True))
        del receiver.g_a
        del receiver.hyperprior.h_a
        for i, point in enumerate(POINTS):
            h, w = ((256, 256), (255, 257), (17, 31), (1, 1), (64, 128))[i]
            rgb = torch.rand(2, 3, h, w)
            expected = model(rgb, *point)
            packet = model.compress(rgb, *point)
            actual = receiver.decompress(packet)
            torch.testing.assert_close(actual["y_hat"], expected["y_hat"], atol=0, rtol=0)
            ry, rz, pixels, metadata_bytes = entropy_bits(packet)
            self.assertEqual(ry + rz, 8 * (len(packet) - metadata_bytes))
            self.assertEqual(actual["payload_bpp"], (ry + rz) / pixels)
            self.assertEqual(actual["actual_bpp"], len(packet) * 8 / pixels)
            self.assertEqual(actual["rate"], {"level": point[0], "alpha": point[1]})
            print(f"\nHyperGain rate={point} RGB=2x{h}x{w}: Ry={ry} Rz={rz} "
                  f"Rz_ratio={rz/(ry+rz):.6f} actual_bpp={actual['actual_bpp']:.6f}")

    def test_full_codec_backward_all_modules_and_update(self):
        model = DualViewCodec()
        perturb(model.analysis.gain)
        main, wrist = torch.rand(1, 3, 65, 71), torch.rand(1, 3, 31, 49)
        first = model(main, wrist, main_rate=(3, .5), wrist_rate=(6, .2))
        out = model(main, wrist, state=first["state"], frame_index=1,
                    main_rate=(3, .5), wrist_rate=(6, .2))
        loss = out["estimated_bpp"] + 10 * (F.mse_loss(out["main_hat"], main) + F.mse_loss(out["wrist_hat"], wrist))
        loss.backward()
        quantiles = model.analysis.hyperprior.entropy_bottleneck.quantiles
        for name, p in model.named_parameters():
            if p is quantiles:
                continue
            self.assertIsNotNone(p.grad, name)
            self.assertTrue(torch.isfinite(p.grad).all(), name)
            self.assertGreater(p.grad.abs().sum().item(), 0, name)
        before = {name: p.detach().clone() for name, p in model.analysis.gain.named_parameters()}
        torch.optim.Adam([p for p in model.parameters() if p is not quantiles], lr=1e-4).step()
        for name, p in model.analysis.gain.named_parameters():
            self.assertFalse(torch.equal(p, before[name]), name)

    def test_dual_view_multiframe_independent_rates_and_reset(self):
        sender = DualViewCodec().eval()
        perturb(sender.analysis.gain)
        sender.update(force=True)
        receiver = DualViewCodec().eval()
        receiver.load_state_dict(sender.state_dict())
        del receiver.analysis.g_a
        del receiver.analysis.hyperprior.h_a
        forward_state = decoded_state = None
        with mock.patch.object(receiver.decoder.temporal, "forward", wraps=receiver.decoder.temporal.forward) as temporal:
            for i, (episode, frame) in enumerate(((9, 0), (9, 1), (9, 2), (10, 0), (10, 1))):
                main, wrist = torch.rand(1, 3, 65 + i, 71), torch.rand(1, 3, 31, 49 + i)
                rates = {"main_rate": (3, .5), "wrist_rate": (6, .2)}
                expected = sender(main, wrist, forward_state, episode, frame, **rates)
                packet = sender.compress(main, wrist, episode, frame, **rates)
                actual = receiver.decompress(packet, decoded_state)
                for key in ("main_hat", "wrist_hat"):
                    torch.testing.assert_close(actual[key], expected[key], atol=0, rtol=0)
                self.assertEqual(actual["history_used"], min(frame, 2))
                self.assertEqual(actual["rates"], {"main": {"level": 3, "alpha": .5},
                                                   "wrist": {"level": 6, "alpha": .2}})
                for a, b in zip(actual["state"].history, expected["state"].history):
                    torch.testing.assert_close(a.latent, b.latent, atol=0, rtol=0)
                    torch.testing.assert_close(a.rgb, b.rgb, atol=0, rtol=0)
                magic, ep, fi, nm, nw = struct.unpack_from(">4sQIII", packet)
                self.assertEqual((magic, ep, fi), (b"ECD1", episode, frame))
                view_stats = [entropy_bits(packet[24:24+nm]), entropy_bits(packet[24+nm:])]
                bits = sum(ry + rz for ry, rz, _, _ in view_stats)
                metadata = 24 + sum(n for _, _, _, n in view_stats)
                self.assertEqual(bits, 8 * (len(packet) - metadata))
                self.assertEqual(actual["actual_bpp"], 8 * len(packet) / sum(p for _, _, p, _ in view_stats))
                forward_state, decoded_state = expected["state"], actual["state"]
            self.assertEqual(temporal.call_count, 3)

    def test_legacy_checkpoint_explicit_migration_and_old_stream(self):
        for model in (LatentGain(), GainedEncoderHyperprior().eval(), DualViewCodec().eval()):
            full = model.state_dict()
            legacy = copy.copy(full)
            for name in tuple(legacy):
                if name.endswith(("log_hyper_gain", "log_inverse_hyper_gain")):
                    del legacy[name]
            with self.assertRaises(RuntimeError):
                model.load_state_dict(legacy, strict=True)
            upgraded = with_identity_hypergains(legacy)
            self.assertEqual(len(legacy), len(full) - 2)
            self.assertEqual(getattr(upgraded, "_metadata", None), getattr(legacy, "_metadata", None))
            model.load_state_dict(upgraded, strict=True)
            self.assertEqual(set(upgraded), set(full))
        model = GainedEncoderHyperprior().eval()
        model.update(force=True)
        for point in POINTS:
            rgb = torch.rand(1, 3, 65, 71)
            image, geometry = prepare_rgb(rgb)
            # Legacy y-only stream: same G_y, original prior called without z gains.
            yg = model.gain(model.g_a(image), *point)
            legacy_packet = struct.pack(">4sId", b"ECG1", *point) + EncoderHyperprior._compress_latent(model, yg, geometry)
            self.assertEqual(model.compress(rgb, *point), legacy_packet)
            expected = model.gain.inverse(model.hyperprior(yg)["y_hat"], *point)
            torch.testing.assert_close(model.decompress(legacy_packet)["y_hat"], expected, atol=0, rtol=0)
        perturb(model.gain)
        new = with_identity_hypergains(model.state_dict())
        torch.testing.assert_close(new["gain.log_hyper_gain"], model.gain.log_hyper_gain)
        del new["gain.log_inverse_hyper_gain"]
        with self.assertRaises(ValueError):
            with_identity_hypergains(new)

    def test_optional_gain_contract_and_no_gain_compatibility(self):
        prior = LatentHyperprior().eval()
        prior.update(force=True)
        y = torch.randn(1, 64, 4, 4)
        unit = torch.ones(1, 64, 1, 1)
        plain = prior.compress(y)
        self.assertEqual(plain, prior.compress(y, hyper_gain=unit, inverse_hyper_gain=unit))
        torch.testing.assert_close(prior(y)["y_hat"], prior(y, hyper_gain=unit, inverse_hyper_gain=unit)["y_hat"], atol=0, rtol=0)
        for kwargs in ({"hyper_gain": unit}, {"inverse_hyper_gain": unit},
                       {"hyper_gain": unit.flatten(), "inverse_hyper_gain": unit}):
            with self.assertRaises(ValueError):
                prior(y, **kwargs)
            with self.assertRaises(ValueError):
                prior.compress(y, **kwargs)


if __name__ == "__main__":
    unittest.main(verbosity=2)
