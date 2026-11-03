"""CPU acceptance for learned gains, continuous rates and real entropy coding."""

import io
import struct
import unittest

import torch

from compress.gain import GainedEncoderHyperprior, LatentGain
from compress.hyperprior import EncoderHyperprior, LatentHyperprior
from compress.lightweight_encoder import prepare_rgb


def rate_points(levels):
    return ([(s, 0.) for s in range(levels)]
            + [(s, a) for s in range(levels - 1) for a in (.25, .5, .9, 1.)])


class GainAcceptance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(20261103)
        self.model = GainedEncoderHyperprior()

    def test_geometric_interpolation_and_endpoints(self):
        gain = self.model.gain
        # Exercise distinct, channel-dependent and independently learned pairs.
        with torch.no_grad():
            gain.log_gain.add_(torch.linspace(-.3, .3, 64))
            gain.log_inverse_gain.add_(torch.linspace(.1, .4, 64))
        for level, alpha in rate_points(gain.levels):
            factors = gain.factors(level, alpha)
            adjacent = min(level + 1, gain.levels - 1)
            for table, factor in zip((gain.log_gain, gain.log_inverse_gain), factors):
                expected = table[level].exp().pow(1-alpha) * table[adjacent].exp().pow(alpha)
                torch.testing.assert_close(factor.flatten(), expected)
                self.assertTrue(torch.isfinite(factor).all() and (factor > 0).all())
            if level < gain.levels - 1:
                for a, b in zip(gain.factors(level, 1.), gain.factors(level + 1, 0.)):
                    torch.testing.assert_close(a, b, atol=0, rtol=0)
        y = torch.randn(1, 64, 4, 4)
        independent = gain.inverse(gain(y, 3, .5), 3, .5)
        self.assertFalse(torch.allclose(independent, y))  # Not tied reciprocals.

    def test_all_rates_training_gradients_and_parameter_updates(self):
        rgb = torch.rand(2, 3, 65, 71, requires_grad=True)
        gain = self.model.gain
        for level, alpha in rate_points(gain.levels):
            self.model.zero_grad(set_to_none=True)
            out = self.model(rgb, level, alpha)
            self.assertEqual(tuple(out["y_hat"].shape), (2, 64, 8, 8))
            loss = out["estimated_bpp"] + out["y_hat"].square().mean()
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            active = {level} if alpha == 0 else {level + 1} if alpha == 1 else {level, level + 1}
            for table in (gain.log_gain, gain.log_inverse_gain):
                self.assertTrue(torch.isfinite(table.grad).all())
                for s in range(gain.levels):
                    grad = table.grad[s].abs().sum().item()
                    self.assertGreater(grad, 0) if s in active else self.assertEqual(grad, 0)
            for name, parameter in self.model.named_parameters():
                if name.endswith(".quantiles"):
                    continue
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                self.assertGreater(parameter.grad.abs().sum().item(), 0, name)
        before = [p.detach().clone() for p in gain.parameters()]
        torch.optim.Adam(gain.parameters(), lr=1e-3).step()
        for old, new in zip(before, gain.parameters()):
            self.assertFalse(torch.equal(old, new))
        self.assertGreater(rgb.grad.abs().sum().item(), 0)

    def test_nonzero_latent_roundtrip_at_every_rate(self):
        prior = LatentHyperprior().eval()
        prior.update(force=True)
        gain = self.model.gain
        y = torch.randn(2, 64, 4, 8) * 5
        for level, alpha in rate_points(gain.levels):
            gained = gain(y, level, alpha)
            packet = prior.compress(gained)
            decoded = prior.decompress(packet["strings"], packet["shape"])
            torch.testing.assert_close(decoded, gained.round(), atol=0, rtol=0)
            actual = gain.inverse(decoded, level, alpha)
            expected = gain.inverse(prior(gained)["y_hat"], level, alpha)
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_packet_rate_recovery_fresh_receiver_and_bpp(self):
        self.model.eval()
        # Simulate learned nonuniform parameters, not only their initial values.
        with torch.no_grad():
            self.model.gain.log_gain.add_(torch.linspace(-.2, .2, 64))
            self.model.gain.log_inverse_gain.add_(torch.linspace(.03, .07, 64))
        self.model.update(force=True)
        checkpoint = io.BytesIO()
        torch.save(self.model.state_dict(), checkpoint)
        checkpoint.seek(0)
        receiver = GainedEncoderHyperprior().eval()
        receiver.load_state_dict(torch.load(checkpoint, weights_only=True))
        del receiver.g_a
        # Include a non-binary fraction to catch transport precision differences.
        points = rate_points(self.model.gain.levels) + [(3, .123456789012345)]
        for i, (level, alpha) in enumerate(points):
            batch, h, w = ((1, 256, 256), (2, 255, 257), (1, 17, 31), (1, 1, 1))[i % 4]
            rgb = torch.rand(batch, 3, h, w)
            expected = self.model(rgb, level, alpha)
            packet = self.model.compress(rgb, level, alpha)
            actual = receiver.decompress(packet)  # No out-of-band rate arguments.
            self.assertEqual(actual["rate"], {"level": level, "alpha": alpha})
            self.assertEqual(actual["geometry"], expected["geometry"])
            for key in ("y_hat", "gained_y_hat"):
                torch.testing.assert_close(actual[key], expected[key], atol=0, rtol=0)
            self.assertEqual(actual["num_bytes"], len(packet))
            self.assertEqual(actual["actual_bpp"], 8 * len(packet) / (batch * h * w))
            payload = len(packet) - 16 - 16 - 8 * batch
            self.assertEqual(actual["payload_bpp"], 8 * payload / (batch * h * w))
        print(f"\nGain: {self.model.gain.levels} discrete levels, "
              f"{len(points)} rate/shape stream checks; exact forward/decode equality")

    def test_identity_rate_and_probability_input(self):
        self.model.eval()
        self.model.update(force=True)
        baseline = EncoderHyperprior().eval()
        baseline.load_state_dict({k: v for k, v in self.model.state_dict().items()
                                  if not k.startswith("gain.")})
        rgb = torch.rand(1, 3, 127, 129)
        # Rate zero starts at identity, preserving the prior entropy payload.
        self.assertEqual(self.model.compress(rgb, 0)[16:], baseline.compress(rgb))
        level, alpha = 5, .3
        prepared, _ = prepare_rgb(rgb)
        y = self.model.g_a(prepared)
        gained = self.model.gain(y, level, alpha)
        direct = self.model.hyperprior(gained)
        out = self.model(rgb, level, alpha)
        for key in ("y", "z"):
            torch.testing.assert_close(out["likelihoods"][key], direct["likelihoods"][key], atol=0, rtol=0)
        # The envelope contains the unchanged hyperprior's coding of gained y.
        direct_packet = self.model._compress_latent(gained, out["geometry"])
        self.assertEqual(self.model.compress(rgb, level, alpha)[16:], direct_packet)

    def test_invalid_rate_and_packet_rejected(self):
        for levels in (1, 0, -1, 2.5, True):
            with self.assertRaises(ValueError):
                LatentGain(levels)
        invalid = [(-1, 0), (8, 0), (1.5, 0), (True, 0), (0, -.1),
                   (0, 1.1), (0, float("nan")), (0, float("inf")), (7, .1), (0, True)]
        self.model.eval()
        self.model.update(force=True)
        rgb = torch.rand(1, 3, 64, 64)
        packet = self.model.compress(rgb)
        for level, alpha in invalid:
            with self.assertRaises(ValueError):
                self.model(rgb, level, alpha)
            with self.assertRaises(ValueError):
                self.model.compress(rgb, level, alpha)
        for bad in (b"", packet[:10], b"BAD!" + packet[4:], packet[:-1], packet + b"x",
                    struct.pack(">4sId", b"ECG1", 8, 0.) + packet[16:],
                    struct.pack(">4sId", b"ECG1", 0, float("nan")) + packet[16:]):
            with self.assertRaises(ValueError):
                self.model.decompress(bad)


if __name__ == "__main__":
    unittest.main(verbosity=2)
