"""CPU acceptance of decoder, causal state, and real dual-view frame packets."""

import io
from dataclasses import replace
import struct
import unittest
from unittest import mock

import torch
from torch.nn import functional as F

from compress.decoder import DecoderState, EfficientCrossAttention, SpatioTemporalDecoder
from compress.dual_view_codec import DualViewCodec
from compress.lightweight_encoder import ImageGeometry


class DecoderAcceptance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(20261103)
        self.decoder = SpatioTemporalDecoder()

    def test_attention_matches_explicit_factorization(self):
        attention = EfficientCrossAttention()
        query = torch.randn(2, 64, 4, 8)
        references = (torch.randn(2, 64, 4, 4), torch.randn(2, 64, 8, 4))
        q = attention.query(query).reshape(2, 2, 4, 32).softmax(-2)
        k = torch.cat([attention.key(r).flatten(2) for r in references], -1).reshape(2, 2, 4, 48).softmax(-1)
        v = torch.cat([attention.value(r).flatten(2) for r in references], -1).reshape(2, 2, 8, 48)
        # NxN form only in this tiny test; implementation uses a 4x8 summary.
        dense = v @ (k.transpose(-1, -2) @ q)
        expected = attention.project(dense.reshape(2, 16, 4, 8))
        torch.testing.assert_close(attention(query, references), expected)

    def test_context_changes_outputs_and_no_cyclic_dependency(self):
        main = torch.randn(1, 64, 4, 8)
        wrist = torch.randn(1, 64, 4, 4)
        gm, gw = ImageGeometry((61, 101), (64, 128)), ImageGeometry((31, 47), (64, 64))
        first = self.decoder(main, wrist, gm, gw)
        changed_wrist = self.decoder(main, wrist + torch.randn_like(wrist) * 3, gm, gw)
        self.assertGreater((first["main_hat"] - changed_wrist["main_hat"]).abs().max().item(), 1e-8)
        self.assertGreater((first["wrist_hat"] - changed_wrist["wrist_hat"]).abs().max().item(), 1e-8)
        next_frame = self.decoder(main, wrist, gm, gw, first["state"], frame_index=1)
        entry = first["state"].history[0]
        for altered in (replace(entry, main_latent=entry.main_latent + 5),
                        replace(entry, main_rgb=entry.main_rgb + 3)):
            state = DecoderState(0, 1, (altered,))
            result = self.decoder(main, wrist, gm, gw, state, frame_index=1)
            self.assertGreater((result["main_hat"] - next_frame["main_hat"]).abs().max().item(), 1e-8)
            torch.testing.assert_close(result["wrist_hat"], next_frame["wrist_hat"], atol=0, rtol=0)
        for altered in (replace(entry, wrist_latent=entry.wrist_latent + 5),
                        replace(entry, wrist_rgb=entry.wrist_rgb + 3)):
            state = DecoderState(0, 1, (altered,))
            result = self.decoder(main, wrist, gm, gw, state, frame_index=1)
            self.assertGreater((result["wrist_hat"] - next_frame["wrist_hat"]).abs().max().item(), 1e-8)
            torch.testing.assert_close(result["main_hat"], next_frame["main_hat"], atol=0, rtol=0)

    def test_decoder_gradients_detached_history_and_copy(self):
        geometry = ImageGeometry((64, 64), (64, 64))
        old = torch.randn(1, 64, 4, 4, requires_grad=True)
        old_wrist = torch.randn_like(old, requires_grad=True)
        first = self.decoder(old, old_wrist, geometry, geometry)
        cached = first["state"].history[0]
        fields = ("main_latent", "main_rgb", "wrist_latent", "wrist_rgb")
        saved = {name: getattr(cached, name).clone() for name in fields}
        for name in fields:
            self.assertFalse(getattr(cached, name).requires_grad)
        with torch.no_grad():
            first["main_hat"].zero_()
            first["wrist_hat"].zero_()
            old.add_(7)
            old_wrist.add_(7)
        for name in fields:
            torch.testing.assert_close(getattr(cached, name), saved[name], atol=0, rtol=0)
        main = torch.randn_like(old, requires_grad=True)
        wrist = torch.randn_like(old, requires_grad=True)
        result = self.decoder(main, wrist, geometry, geometry, first["state"], frame_index=1)
        (result["main_hat"].square().mean() + result["wrist_hat"].square().mean()).backward()
        for name, p in self.decoder.named_parameters():
            self.assertIsNotNone(p.grad, name)
            self.assertTrue(torch.isfinite(p.grad).all(), name)
            # Official key bias is a spatially constant softmax offset: its true
            # gradient is zero. Keep it for source fidelity; still require finiteness.
            if not name.endswith("keys.bias"):
                self.assertGreater(p.grad.abs().sum().item(), 0, name)
        self.assertGreater(main.grad.abs().sum().item(), 0)
        self.assertGreater(wrist.grad.abs().sum().item(), 0)
        self.assertIsNone(old.grad)
        self.assertIsNone(old_wrist.grad)
        for main_branch, wrist_branch in ((self.decoder.temporal, self.decoder.temporal_wrist),
                                           (self.decoder.temporal_project, self.decoder.temporal_project_wrist)):
            self.assertEqual(list(main_branch.state_dict()), list(wrist_branch.state_dict()))
            for a, b in zip(main_branch.parameters(), wrist_branch.parameters()):
                self.assertNotEqual(a.data_ptr(), b.data_ptr())

    def test_each_temporal_gradient_depends_only_on_its_view_loss(self):
        geometry = ImageGeometry((64, 64), (64, 64))
        main, wrist = torch.randn(1, 64, 4, 4), torch.randn(1, 64, 4, 4)
        first = self.decoder(main, wrist, geometry, geometry)
        for key, active, inactive in (("main_hat", self.decoder.temporal, self.decoder.temporal_wrist),
                                      ("wrist_hat", self.decoder.temporal_wrist, self.decoder.temporal)):
            self.decoder.zero_grad(set_to_none=True)
            out = self.decoder(main, wrist, geometry, geometry, first["state"], frame_index=1)
            out[key].square().mean().backward()
            self.assertGreater(active.values.weight.grad.abs().sum().item(), 0)
            self.assertTrue(all(p.grad is None for p in inactive.parameters()))


class DualViewAcceptance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(20261103)
        self.model = DualViewCodec(decoder_config='baseline')

    def test_training_sequence_rd_backward_and_optimizer(self):
        state = None
        quantiles = self.model.analysis.hyperprior.entropy_bottleneck.quantiles
        for i in range(3):
            self.model.zero_grad(set_to_none=True)
            main, wrist = torch.rand(1, 3, 65, 71), torch.rand(1, 3, 31, 49)
            result = self.model(main, wrist, state, episode_id=7, frame_index=i,
                                main_rate=(3, .4), wrist_rate=(5, .6))
            self.assertEqual(result["main_hat"].shape, main.shape)
            self.assertEqual(result["wrist_hat"].shape, wrist.shape)
            loss = result["estimated_bpp"] + 10 * F.mse_loss(result["main_hat"], main)
            loss += 5 * F.mse_loss(result["wrist_hat"], wrist)
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            if i > 0:
                for name, parameter in self.model.named_parameters():
                    if parameter is quantiles:
                        continue
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                    if not name.endswith("keys.bias"):
                        self.assertGreater(parameter.grad.abs().sum().item(), 0, name)
            state = result["state"]
            self.assertEqual(len(state.history), min(i + 1, 2))
        parameters = [self.model.decoder.temporal.value.weight, self.model.decoder.temporal_wrist.value.weight]
        old = [p.detach().clone() for p in parameters]
        torch.optim.Adam([p for p in self.model.parameters() if p is not quantiles], lr=1e-4).step()
        for before, parameter in zip(old, parameters):
            self.assertFalse(torch.equal(before, parameter))
        self.model.zero_grad(set_to_none=True)
        self.model.aux_loss().backward()
        self.assertGreater(quantiles.grad.abs().sum().item(), 0)

    def test_multiframe_real_bytes_match_forward_cache_and_bpp(self):
        self.model.eval()
        self.model.update(force=True)
        checkpoint = io.BytesIO()
        torch.save(self.model.state_dict(), checkpoint)
        checkpoint.seek(0)
        receiver = DualViewCodec(decoder_config='baseline').eval()
        receiver.load_state_dict(torch.load(checkpoint, weights_only=True))
        del receiver.analysis.g_a  # Receiver cannot encode or access source images.
        forward_state = decoded_state = None
        total_bytes = total_pixels = 0
        cases = [(1, 256, 256, 256, 256), (1, 255, 257, 129, 127),
                 (1, 17, 31, 1, 1), (1, 65, 71, 31, 49), (2, 63, 65, 33, 47)]
        for i, (b, mh, mw, wh, ww) in enumerate(cases):
            # Batch change starts a new synchronized episode; earlier changes of
            # resolution remain within one episode and use variable-size history.
            episode, index = (42, i) if i < 4 else (43, 0)
            main, wrist = torch.rand(b, 3, mh, mw), torch.rand(b, 3, wh, ww)
            rates = {"main_rate": (i, .25), "wrist_rate": (7, 0.)}
            expected = self.model(main, wrist, forward_state, episode, index, **rates)
            packet = self.model.compress(main, wrist, episode, index, **rates)
            actual = receiver.decompress(packet, decoded_state)
            for key, shape in (("main_hat", main.shape), ("wrist_hat", wrist.shape)):
                self.assertEqual(actual[key].shape, shape)
                self.assertTrue(torch.isfinite(actual[key]).all())
                self.assertTrue(((actual[key] >= 0) & (actual[key] <= 1)).all())
                torch.testing.assert_close(actual[key], expected[key], atol=0, rtol=0)
            self.assertEqual(actual["history_used"], min(index, 2))
            self.assertEqual(actual["episode_id"], episode)
            self.assertEqual(actual["frame_index"], index)
            self.assertEqual(actual["rates"]["main"], {"level": i, "alpha": .25})
            self.assertEqual(actual["rates"]["wrist"], {"level": 7, "alpha": 0.})
            for a, e in zip(actual["state"].history, expected["state"].history):
                for field in ("main_latent", "main_rgb", "wrist_latent", "wrist_rgb"):
                    torch.testing.assert_close(getattr(a, field), getattr(e, field), atol=0, rtol=0)
                    self.assertFalse(getattr(a, field).requires_grad)
            torch.testing.assert_close(actual["state"].history[-1].main_rgb, actual["main_hat"], atol=0, rtol=0)
            torch.testing.assert_close(actual["state"].history[-1].wrist_rgb, actual["wrist_hat"], atol=0, rtol=0)
            pixels = b * (mh * mw + wh * ww)
            self.assertEqual(actual["actual_bpp"], len(packet) * 8 / pixels)
            self.assertEqual(actual["num_bytes"], len(packet))
            total_bytes += len(packet)
            total_pixels += pixels
            print(f"\nepisode={episode} frame={index} history={actual['history_used']} "
                  f"main={mh}x{mw} wrist={wh}x{ww} bytes={len(packet)} bpp={actual['actual_bpp']:.6f}")
            forward_state, decoded_state = expected["state"], actual["state"]
        self.assertGreater(total_bytes * 8 / total_pixels, 0)
        # Compression alone is stateless; it never builds an image/history cache.
        with mock.patch.object(self.model.decoder, "forward", side_effect=AssertionError("decoder called")):
            self.model.compress(main, wrist)

    def test_episode_reset_and_order_errors(self):
        self.model.eval()
        self.model.update(force=True)
        main, wrist = torch.rand(1, 3, 64, 64), torch.rand(1, 3, 64, 64)
        first = self.model.decompress(self.model.compress(main, wrist, episode_id=11))
        p1 = self.model.compress(main, wrist, episode_id=11, frame_index=1)
        with self.assertRaisesRegex(ValueError, "history"):
            self.model.decompress(p1)
        second = self.model.decompress(p1, first["state"])
        with self.assertRaisesRegex(ValueError, "history"):
            self.model.decompress(p1, DecoderState(11, 1, ()))
        for packet, state in ((p1, second["state"]),
                              (self.model.compress(main, wrist, 11, 3), first["state"]),
                              (self.model.compress(main, wrist, 12, 1), first["state"])):
            with self.assertRaisesRegex(ValueError, "history"):
                self.model.decompress(packet, state)
        self.assertEqual(first["state"].next_frame, 1)  # Failed calls did not mutate it.
        for episode in (11, 12):
            reset_packet = self.model.compress(main, wrist, episode_id=episode)
            clean = self.model.decompress(reset_packet)
            reset = self.model.decompress(reset_packet, second["state"])
            self.assertEqual(reset["history_used"], 0)
            self.assertEqual(len(reset["state"].history), 1)
            torch.testing.assert_close(reset["main_hat"], clean["main_hat"], atol=0, rtol=0)
            torch.testing.assert_close(reset["wrist_hat"], clean["wrist_hat"], atol=0, rtol=0)
        with self.assertRaisesRegex(ValueError, "batch"):
            self.model.decompress(self.model.compress(main.repeat(2, 1, 1, 1),
                                  wrist.repeat(2, 1, 1, 1), 11, 1), first["state"])

    def test_invalid_inputs_and_no_balle_decoder(self):
        self.model.eval()
        self.model.update(force=True)
        main, wrist = torch.rand(1, 3, 64, 64), torch.rand(1, 3, 64, 64)
        packet = self.model.compress(main, wrist)
        for bad in (b"", packet[:20], b"BAD!" + packet[4:], packet[:-1], packet + b"x",
                    struct.pack(">4sQIII", b"ECD1", 0, 0, 0, len(packet) - 24) + packet[24:]):
            with self.assertRaises(ValueError):
                self.model.decompress(bad)
        for episode, index in ((-1, 0), (0, -1), (2**64, 0), (0, 2**32), (True, 0)):
            with self.assertRaises(ValueError):
                self.model.compress(main, wrist, episode, index)
        for module in self.model.modules():
            self.assertFalse(hasattr(module, "g_s"))
        with self.assertRaises(ValueError):
            self.model(main, wrist.repeat(2, 1, 1, 1))


if __name__ == "__main__":
    unittest.main(verbosity=2)
