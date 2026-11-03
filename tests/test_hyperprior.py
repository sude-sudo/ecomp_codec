"""CPU tests for actual entropy streams; the synthesis probe is test-only."""

import io
import struct
import unittest

import torch
from torch import nn
from torch.nn import functional as F

from compress.hyperprior import EncoderHyperprior, LatentHyperprior
from compress.lightweight_encoder import prepare_rgb, rgb_to_ycocg, ycocg_to_rgb


class SynthesisProbe(nn.Module):
    """Temporary random image head, never exported to the codec implementation."""

    def __init__(self):
        super().__init__()
        self.project = nn.Conv2d(64, 3, 1)

    def forward(self, y_hat, geometry):
        color = F.interpolate(self.project(y_hat), size=geometry.padded_hw,
                              mode="bilinear", align_corners=False)
        return geometry.crop(ycocg_to_rgb(color))


class HyperpriorAcceptance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(20261103)
        self.model = EncoderHyperprior()

    def test_original_components_and_layout(self):
        from compressai.entropy_models import EntropyBottleneck, GaussianConditional
        from compressai.models.google import ScaleHyperprior

        original = ScaleHyperprior(N=64, M=64)
        prior = self.model.hyperprior
        self.assertIs(type(prior.entropy_bottleneck), EntropyBottleneck)
        self.assertIs(type(prior.gaussian_conditional), GaussianConditional)
        for name in ("h_a", "h_s"):
            self.assertEqual(repr(getattr(prior, name)), repr(getattr(original, name)))
        self.assertFalse(hasattr(self.model, "g_s"))
        self.assertFalse(hasattr(prior, "g_s"))

    def test_color_roundtrip_and_crop(self):
        corners = torch.tensor([[r, g, b] for r in (0., 1.) for g in (0., 1.)
                                for b in (0., 1.)]).T.reshape(1, 3, 2, 4)
        torch.testing.assert_close(ycocg_to_rgb(rgb_to_ycocg(corners)), corners,
                                   atol=0, rtol=0)
        for h, w in ((255, 257), (17, 31), (1, 1)):
            rgb = torch.rand(1, 3, h, w, requires_grad=True)
            color, geometry = prepare_rgb(rgb)
            restored = geometry.crop(ycocg_to_rgb(color))
            torch.testing.assert_close(restored, rgb, atol=2e-7, rtol=0)
            restored.sum().backward()
            torch.testing.assert_close(rgb.grad, torch.ones_like(rgb))

    def test_training_gradients_and_aux_optimizer(self):
        self.model.train()
        rgb = torch.rand(2, 3, 65, 71, requires_grad=True)
        out = self.model(rgb)
        probe = SynthesisProbe()
        reconstructed = probe(out["y_hat"], out["geometry"])
        loss = out["estimated_bpp"] + F.mse_loss(reconstructed, rgb)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        # Quantiles use the original separate auxiliary loss/optimizer path.
        quantiles = self.model.hyperprior.entropy_bottleneck.quantiles
        main_params = [p for p in self.model.parameters() if p is not quantiles]
        for name, parameter in self.model.named_parameters():
            if parameter is quantiles:
                continue
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            self.assertGreater(parameter.grad.abs().sum().item(), 0, name)
        self.assertGreater(rgb.grad.abs().sum().item(), 0)
        before = self.model.g_a.stages[0][0].depthwise.weight.detach().clone()
        torch.optim.Adam(main_params, lr=1e-4).step()
        self.assertFalse(torch.equal(before, self.model.g_a.stages[0][0].depthwise.weight))
        self.model.zero_grad(set_to_none=True)
        aux = self.model.aux_loss()
        aux.backward()
        self.assertTrue(torch.isfinite(quantiles.grad).all())
        self.assertGreater(quantiles.grad.abs().sum().item(), 0)
        torch.optim.Adam([quantiles], lr=1e-3).step()

    def test_nonzero_latent_real_entropy_roundtrip(self):
        from compressai.models.google import ScaleHyperprior

        prior = LatentHyperprior().eval()
        prior.update(force=True)
        # Include negative, positive, and out-of-CDF-range symbols; an untrained
        # encoder alone often rounds to zero, which is insufficient coverage.
        y = torch.randn(2, 64, 8, 12) * 20
        packet = prior.compress(y)
        self.assertEqual(packet["shape"], (2, 3))
        self.assertTrue(all(isinstance(s, bytes) for stream in packet["strings"] for s in stream))
        decoded = prior.decompress(packet["strings"], packet["shape"])
        torch.testing.assert_close(decoded, y.round(), atol=0, rtol=0)
        torch.testing.assert_close(decoded, prior(y)["y_hat"], atol=0, rtol=0)
        self.assertGreater((decoded != 0).sum().item(), 100)
        # With identical h_a/h_s, probability parameters and CDFs, the upstream
        # Ballé implementation must produce byte-for-byte identical streams.
        reference = ScaleHyperprior(N=64, M=64).eval()
        reference.g_a = nn.Identity()  # Supply the already computed test latent.
        del reference.g_s
        reference.load_state_dict(prior.state_dict())
        self.assertEqual(packet, reference.compress(y))

    def test_rgb_bytes_fresh_receiver_bpp_and_synthesis(self):
        self.model.eval()
        self.model.update(force=True)
        # Serialize and load into a fresh receiver, including variable-size CDF
        # buffers via the unchanged CompressionModel.load_state_dict implementation.
        checkpoint = io.BytesIO()
        torch.save(self.model.state_dict(), checkpoint)
        checkpoint.seek(0)
        receiver = EncoderHyperprior().eval()
        receiver.load_state_dict(torch.load(checkpoint, weights_only=True))
        del receiver.g_a  # A real receiving endpoint does not need the encoder.
        probe = SynthesisProbe().eval()
        for batch, h, w in ((1, 256, 256), (2, 255, 257), (1, 17, 31), (1, 1, 1)):
            rgb = torch.rand(batch, 3, h, w)
            expected = self.model(rgb)
            packet = self.model.compress(rgb)
            self.assertIsInstance(packet, bytes)
            self.assertEqual(packet, self.model.compress(rgb))
            actual = receiver.decompress(packet)  # bytes only; no encoder-side tensors
            torch.testing.assert_close(actual["y_hat"], expected["y_hat"], atol=0, rtol=0)
            self.assertEqual(actual["geometry"], expected["geometry"])
            self.assertEqual(actual["geometry"].original_hw, (h, w))
            reconstruction = probe(actual["y_hat"], actual["geometry"])
            self.assertEqual(tuple(reconstruction.shape), (batch, 3, h, w))
            torch.testing.assert_close(reconstruction,
                                       probe(expected["y_hat"], expected["geometry"]),
                                       atol=0, rtol=0)
            self.assertEqual(actual["actual_bpp"], len(packet) * 8 / (batch * h * w))
            # Independent byte accounting: 16-byte header + 8 length bytes/image.
            payload = len(packet) - 16 - batch * 8
            self.assertEqual(actual["payload_bpp"], payload * 8 / (batch * h * w))
            self.assertGreater(actual["actual_bpp"], actual["payload_bpp"])
            self.assertTrue(torch.isfinite(expected["estimated_bpp"]))
            print(f"\nB={batch} RGB={h}x{w} latent={tuple(actual['y_hat'].shape)} "
                  f"bytes={len(packet)} actual_bpp={actual['actual_bpp']:.6f} "
                  f"payload_bpp={actual['payload_bpp']:.6f} "
                  f"estimated_bpp={expected['estimated_bpp'].item():.6f}")
        # Both cameras share this model, but carry independent geometry/streams.
        packets = [self.model.compress(torch.rand(1, 3, h, w))
                   for h, w in ((127, 129), (65, 71))]
        outputs = [receiver.decompress(packet) for packet in packets]
        pixels = 127 * 129 + 65 * 71
        pair_bpp = 8 * sum(map(len, packets)) / pixels
        self.assertAlmostEqual(pair_bpp, sum(out["actual_bpp"] * h * w
                               for out, (h, w) in zip(outputs, ((127, 129), (65, 71)))) / pixels)

    def test_coding_preconditions_and_packet_geometry(self):
        with self.assertRaisesRegex(RuntimeError, "eval"):
            self.model.compress(torch.rand(1, 3, 64, 64))
        self.model.eval()
        with self.assertRaisesRegex(RuntimeError, "update"):
            self.model.compress(torch.rand(1, 3, 64, 64))
        self.model.update(force=True)
        packet = self.model.compress(torch.rand(1, 3, 64, 64))
        for bad in (b"", packet[:10], b"BAD!" + packet[4:], packet[:-1], packet + b"x",
                    struct.pack(">4sIII", b"ECH1", 0, 64, 64)):
            with self.assertRaises(ValueError):
                self.model.decompress(bad)
        with self.assertRaises(ValueError):
            self.model.hyperprior(torch.rand(1, 64, 5, 4))


if __name__ == "__main__":
    unittest.main(verbosity=2)
