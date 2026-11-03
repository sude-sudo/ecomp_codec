"""CPU acceptance checks: python -m unittest discover -s tests -p test_lightweight_encoder.py -v."""

import unittest

import torch
from torch import nn

from compress.lightweight_encoder import (
    DSConv, LightweightEncoder, prepare_rgb, rgb_to_ycocg, ycocg_to_rgb,
)


def profile(module, x):
    """Count convolution MACs including GDN's dense channel mixing.

    FLOPs = 2 * MACs; bias, activation, square/root/division, and padding
    operations are excluded. Parameters include every trainable parameter.
    """
    from compressai.layers import GDN

    rows, handles = [], []

    def record(name):
        def hook(layer, inputs, output):
            if isinstance(layer, nn.Conv2d):
                macs = output.numel() * (layer.in_channels // layer.groups)
                macs *= layer.kernel_size[0] * layer.kernel_size[1]
                description = (f"Conv {layer.in_channels}->{layer.out_channels} "
                               f"k={layer.kernel_size[0]} s={layer.stride[0]} g={layer.groups}")
            else:
                macs = output.numel() * output.shape[1]
                description = "GDN (channel mixing only)"
            params = sum(p.numel() for p in layer.parameters())
            rows.append((name, description, tuple(output.shape), params, macs))
        return hook

    for name, layer in module.named_modules():
        if isinstance(layer, (nn.Conv2d, GDN)):
            handles.append(layer.register_forward_hook(record(name)))
    try:
        with torch.no_grad():
            module(x)
    finally:
        for handle in handles:
            handle.remove()
    return rows


class EncoderAcceptance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(20261103)
        self.encoder = LightweightEncoder()

    def assert_gradients(self, module):
        for name, parameter in module.named_parameters():
            self.assertTrue(parameter.requires_grad, name)
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            self.assertGreater(parameter.grad.abs().sum().item(), 0, name)

    def test_structure_and_forward_backward(self):
        for i, stage in enumerate(self.encoder.stages):
            self.assertEqual(len(stage), 3)
            self.assertIsInstance(stage[0], DSConv)
            self.assertIsInstance(stage[2], nn.GELU)
            cin, cout = self.encoder.channels[i:i + 2]
            self.assertEqual(stage[0].depthwise.groups, cin)
            self.assertEqual(stage[0].depthwise.stride, (2, 2))
            self.assertEqual(stage[0].pointwise.kernel_size, (1, 1))
            self.assertEqual(stage[0].pointwise.out_channels, cout)
            self.assertEqual(stage[1].groups, cout)
            self.assertEqual(stage[1].stride, (1, 1))
        x = torch.rand(2, 3, 256, 256, requires_grad=True)
        image, _ = prepare_rgb(x)
        y = self.encoder(image)
        self.assertEqual(tuple(y.shape), (2, 64, 16, 16))
        self.assertTrue(y.requires_grad)
        y.square().mean().backward()
        self.assert_gradients(self.encoder)
        self.assertGreater(x.grad.abs().sum().item(), 0)

    def test_color_and_geometry(self):
        for h, w in ((256, 256), (255, 257), (17, 31), (1, 1), (64, 128)):
            rgb = torch.rand(1, 3, h, w)
            image, geometry = prepare_rgb(rgb)
            ph, pw = geometry.padded_hw
            self.assertEqual(ph % 64, 0)
            self.assertEqual(pw % 64, 0)
            self.assertEqual(geometry.original_hw, (h, w))
            y = self.encoder(image)
            self.assertEqual(tuple(y.shape), (1, 64, ph // 16, pw // 16))
            self.assertEqual(y.shape[-1] % 4, 0)
            self.assertEqual(y.shape[-2] % 4, 0)
            torch.testing.assert_close(geometry.crop(ycocg_to_rgb(image)), rgb)
        red = torch.tensor([1., 0., 0.]).reshape(1, 3, 1, 1)
        torch.testing.assert_close(rgb_to_ycocg(red).flatten(), torch.tensor([.25, .5, -.25]))
        with self.assertRaises(ValueError):
            geometry.crop(torch.zeros(1, 3, 2, 2))
        with self.assertRaises(ValueError):
            self.encoder(torch.rand(1, 3, 17, 31))
        with self.assertRaises(TypeError):
            self.encoder(torch.zeros(1, 3, 64, 64, dtype=torch.uint8))

    def test_pair_reuses_parameters_and_accumulates_gradients(self):
        main = torch.rand(1, 3, 65, 71)
        wrist = torch.rand(1, 3, 31, 49)
        ids = tuple(id(p) for p in self.encoder.parameters())
        (ym, yw), geometries = self.encoder.encode_pair(main, wrist)
        self.assertEqual(geometries[0].original_hw, (65, 71))
        self.assertEqual(geometries[1].original_hw, (31, 49))
        ym.square().mean().backward()
        main_grads = [p.grad.clone() for p in self.encoder.parameters()]
        self.encoder.zero_grad(set_to_none=True)
        yw.square().mean().backward()
        wrist_grads = [p.grad.clone() for p in self.encoder.parameters()]
        self.assert_gradients(self.encoder)
        self.encoder.zero_grad(set_to_none=True)
        (ym, yw), _ = self.encoder.encode_pair(main, wrist)
        (ym.square().mean() + yw.square().mean()).backward()
        self.assert_gradients(self.encoder)
        for p, gm, gw in zip(self.encoder.parameters(), main_grads, wrist_grads):
            torch.testing.assert_close(p.grad, gm + gw)
        self.assertEqual(ids, tuple(id(p) for p in self.encoder.parameters()))
        before = [p.detach().clone() for p in self.encoder.parameters()]
        torch.optim.Adam(self.encoder.parameters(), lr=1e-3).step()
        for old, new in zip(before, self.encoder.parameters()):
            self.assertFalse(torch.equal(old, new))

    def test_hyperprior_interface(self):
        # Compatibility probe only: no new entropy implementation or decoder call.
        from compressai.models.google import ScaleHyperprior

        prior = ScaleHyperprior(N=64, M=64)
        del prior.g_a
        del prior.g_s
        for h, w in ((256, 256), (255, 257), (1, 1)):
            image, _ = prepare_rgb(torch.rand(1, 3, h, w))
            y = self.encoder(image)
            z = prior.h_a(y.abs())
            zh, zl = prior.entropy_bottleneck(z)
            scales = prior.h_s(zh)
            self.assertEqual(scales.shape, y.shape)
            yh, yl = prior.gaussian_conditional(y, scales)
            self.assertEqual(yh.shape, y.shape)
            self.assertTrue(torch.isfinite(yl).all() and (yl > 0).all())
            loss = yh.square().mean() - yl.log().mean() - zl.log().mean()
            loss.backward()
            self.assert_gradients(self.encoder)
            self.encoder.zero_grad(set_to_none=True)
            prior.zero_grad(set_to_none=True)

    def test_profile_against_original_ga(self):
        from compressai.models.google import ScaleHyperprior

        baseline = ScaleHyperprior(N=128, M=192).g_a
        x = torch.rand(1, 3, 256, 256)
        baseline_params = sum(p.numel() for p in baseline.parameters())
        baseline_macs = sum(row[-1] for row in profile(baseline, x))
        for name, module in (("Ballé quality=1 g_a", baseline),
                             ("Lightweight g_a", self.encoder)):
            rows = profile(module, x)
            params = sum(p.numel() for p in module.parameters())
            macs = sum(row[-1] for row in rows)
            print(f"\n{name}: parameters={params:,}; MACs={macs:,}; FLOPs={2 * macs:,}")
            for row in rows:
                print(row)
            if module is self.encoder:
                self.assertLess(params, baseline_params * 0.01)
                self.assertLess(macs, baseline_macs * 0.01)


if __name__ == "__main__":
    unittest.main(verbosity=2)
