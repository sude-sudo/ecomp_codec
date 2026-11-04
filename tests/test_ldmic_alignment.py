"""Alignment to verbatim LDMIC 4cf7318 regular-forward oracle, CPU only."""

import unittest
from dataclasses import replace
from unittest import mock

import torch
from torch import nn
from compressai.layers import GDN, ResidualBlock
from compressai.models.utils import deconv

from compress.decoder import (DecoderState, EfficientAttention,
                              EfficientCrossAttention, JointContextTransfer, SpatioTemporalDecoder)
from compress.dual_view_codec import DualViewCodec
from compress.lightweight_encoder import ImageGeometry, ycocg_to_rgb
from fixtures.ldmic_official import EfficientAttention as OfficialAttention
from fixtures.ldmic_official import JointContextTransfer as OfficialJCT
from fixtures.decoder_v1 import SpatioTemporalDecoder as LegacyDecoder


class OfficialSpatialPath(nn.Module):
    """LDMIC decoder declarations at M=N=64; project output head kept explicit."""

    def __init__(self):
        super().__init__()
        self.atten_3 = OfficialJCT(64)
        self.decoder_1 = nn.Sequential(deconv(64, 64), GDN(64, inverse=True),
                                       deconv(64, 64), GDN(64, inverse=True))
        self.atten_4 = OfficialJCT(64)
        self.decoder_2 = nn.Sequential(deconv(64, 64), GDN(64, inverse=True), deconv(64, 3))

    def forward(self, main, wrist):
        main, wrist = self.atten_3(main, wrist)
        main, wrist = self.atten_4(self.decoder_1(main), self.decoder_1(wrist))
        return (ycocg_to_rgb(self.decoder_2(main)).sigmoid(),
                ycocg_to_rgb(self.decoder_2(wrist)).sigmoid())


class LDMICAlignment(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(20261103)

    def test_structure_shared_views_and_separate_scales(self):
        decoder = SpatioTemporalDecoder()
        self.assertIsNot(decoder.atten_3, decoder.atten_4)
        self.assertNotEqual(decoder.atten_3.rb1.conv1.weight.data_ptr(),
                            decoder.atten_4.rb1.conv1.weight.data_ptr())
        for jct in (decoder.atten_3, decoder.atten_4):
            blocks = [m for m in jct.modules() if isinstance(m, ResidualBlock)]
            self.assertEqual(len(blocks), 4)
            self.assertEqual([(m.conv1.in_channels, m.conv2.out_channels) for m in blocks],
                             [(64,64),(64,64),(128,64),(64,64)])
            self.assertIsNotNone(jct.refine[0].skip)
            self.assertEqual((jct.attn.key_channels, jct.attn.value_channels, jct.attn.head_count), (8,16,2))
            for projection in (jct.attn.keys, jct.attn.queries, jct.attn.values, jct.attn.reprojection):
                self.assertEqual(projection.kernel_size, (1,1))
                self.assertIsNotNone(projection.bias)
            with mock.patch.object(jct.rb1, "forward", wraps=jct.rb1.forward) as rb1, \
                 mock.patch.object(jct.rb2, "forward", wraps=jct.rb2.forward) as rb2, \
                 mock.patch.object(jct.attn, "forward", wraps=jct.attn.forward) as attn, \
                 mock.patch.object(jct.refine, "forward", wraps=jct.refine.forward) as refine:
                jct(torch.rand(1,64,4,8), torch.rand(1,64,8,4))
            self.assertEqual([m.call_count for m in (rb1,rb2,attn,refine)], [2,2,2,2])
        self.assertEqual([type(m) for m in decoder.decoder_1],
                         [nn.ConvTranspose2d,GDN,nn.ConvTranspose2d,GDN])
        self.assertEqual([type(m) for m in decoder.decoder_2], [nn.ConvTranspose2d,GDN,nn.ConvTranspose2d])
        for name in ("decoder_1", "decoder_2"):
            for layer in getattr(decoder,name):
                if isinstance(layer, GDN):
                    self.assertTrue(layer.inverse)
                else:
                    self.assertEqual(layer.kernel_size, (5,5))
                    self.assertEqual(layer.stride, (2,2))
                    self.assertEqual(layer.in_channels, 64)
                    self.assertEqual(layer.out_channels, 3 if layer is decoder.decoder_2[-1] else 64)

    def test_official_attention_and_jct_numerical_oracles(self):
        for dtype in (torch.float32, torch.float64):
            attention = EfficientAttention().to(dtype)
            official = OfficialAttention(64,64,key_channels=8,head_count=2,value_channels=16).to(dtype)
            official.load_state_dict(attention.state_dict(), strict=True)
            main, wrist = torch.randn(2,64,5,7,dtype=dtype), torch.randn(2,64,5,7,dtype=dtype)
            for a,b in ((main,wrist),(wrist,main)):
                torch.testing.assert_close(attention(a,b), official(a,b), atol=1e-6, rtol=1e-6)
            jct, reference = JointContextTransfer().to(dtype), OfficialJCT(64).to(dtype)
            reference.load_state_dict(jct.state_dict(), strict=True)
            for output, expected in zip(jct(main,wrist),reference(main,wrist)):
                torch.testing.assert_close(output,expected,atol=1e-6,rtol=1e-6)

    def test_unequal_attention_math_and_gradients(self):
        attention = EfficientAttention()
        query = torch.randn(2,64,4,8,requires_grad=True)
        reference = torch.randn(2,64,7,5,requires_grad=True)
        # Independent explicit per-head oracle, extending only the two spatial sizes.
        q = attention.queries(query).flatten(2)
        k = attention.keys(reference).flatten(2)
        v = attention.values(reference).flatten(2)
        heads = []
        for i in range(2):
            qs = q[:,i*4:(i+1)*4].softmax(dim=1)
            ks = k[:,i*4:(i+1)*4].softmax(dim=2)
            vs = v[:,i*8:(i+1)*8]
            heads.append(((ks @ vs.transpose(1,2)).transpose(1,2) @ qs).reshape(2,8,4,8))
        expected = attention.reprojection(torch.cat(heads,dim=1))
        output = attention(query,reference)
        torch.testing.assert_close(output,expected,atol=1e-6,rtol=1e-6)
        output.square().mean().backward()
        for x in (query,reference):
            self.assertTrue(torch.isfinite(x.grad).all())
            self.assertGreater(x.grad.abs().sum().item(),0)
        with self.assertRaisesRegex(ValueError,"batch"):
            attention(query,reference[:1])

    def test_key_bias_cancels_and_legacy_aliases(self):
        attention = EfficientCrossAttention()
        query, reference = torch.randn(1,64,4,4), torch.randn(1,64,4,8)
        self.assertIs(attention.key, attention.keys)
        self.assertIs(attention.query, attention.queries)
        self.assertIs(attention.value, attention.values)
        self.assertIs(attention.project, attention.reprojection)
        self.assertFalse(any(n.startswith("key.") for n in attention.state_dict()))
        before = attention(query,(reference,))
        with torch.no_grad():
            attention.keys.bias.add_(torch.arange(8).float())
        after = attention(query,(reference,))
        torch.testing.assert_close(before,after,atol=1e-6,rtol=1e-6)
        after.square().mean().backward()
        self.assertIsNotNone(attention.keys.bias.grad)
        self.assertTrue(torch.isfinite(attention.keys.bias.grad).all())

    def test_first_frame_official_spatial_path_and_execution_order(self):
        decoder = SpatioTemporalDecoder().eval()
        oracle = OfficialSpatialPath().eval()
        keys = ("atten_3.","atten_4.","decoder_1.","decoder_2.")
        oracle.load_state_dict({k:v for k,v in decoder.state_dict().items() if k.startswith(keys)},strict=True)
        main, wrist = torch.randn(1,64,16,16), torch.randn(1,64,16,16)
        geometry = ImageGeometry((256,256),(256,256))
        trace, handles = [], []
        for name in ("atten_3","decoder_1","atten_4","decoder_2"):
            handles.append(getattr(decoder,name).register_forward_hook(
                lambda module, inputs, output, tag=name: trace.append((tag, tuple(inputs[0].shape[-2:])))))
        with torch.no_grad(), \
             mock.patch.object(decoder.temporal,"forward",side_effect=AssertionError("temporal called")), \
             mock.patch.object(decoder.temporal_project,"forward",side_effect=AssertionError("residual called")), \
             mock.patch.object(decoder.temporal_wrist,"forward",side_effect=AssertionError("wrist temporal called")), \
             mock.patch.object(decoder.temporal_project_wrist,"forward",side_effect=AssertionError("wrist residual called")):
            actual = decoder(main,wrist,geometry,geometry)
            expected = oracle(main,wrist)
        for h in handles:
            h.remove()
        self.assertEqual(trace,[("atten_3",(16,16)),("decoder_1",(16,16)),("decoder_1",(16,16)),
                                ("atten_4",(64,64)),("decoder_2",(64,64)),("decoder_2",(64,64))])
        for key, output in zip(("main_hat","wrist_hat"),expected):
            torch.testing.assert_close(actual[key],output,atol=1e-6,rtol=1e-6)

    def test_bidirectional_spatial_effect_at_both_scales(self):
        for shape in ((4,8),(8,16)):
            jct = JointContextTransfer()
            main,wrist = torch.randn(1,64,*shape), torch.randn(1,64,shape[1],shape[0])
            before = jct(main,wrist)
            self.assertGreater((jct(main,wrist+1)[0]-before[0]).abs().max().item(),1e-8)
            self.assertGreater((jct(main+1,wrist)[1]-before[1]).abs().max().item(),1e-8)
        decoder = SpatioTemporalDecoder().eval()
        main,wrist = torch.randn(1,64,4,8),torch.randn(1,64,4,4)
        gm,gw = ImageGeometry((64,128),(64,128)),ImageGeometry((64,64),(64,64))
        with torch.no_grad():
            before = decoder(main,wrist,gm,gw)
            self.assertGreater((decoder(main,wrist+1,gm,gw)["main_hat"]-before["main_hat"]).abs().max().item(),1e-8)
            self.assertGreater((decoder(main+1,wrist,gm,gw)["wrist_hat"]-before["wrist_hat"]).abs().max().item(),1e-8)

    def test_cache_dtype_and_device_validation(self):
        decoder = SpatioTemporalDecoder()
        main,wrist = torch.rand(1,64,4,4),torch.rand(1,64,4,4)
        geometry = ImageGeometry((64,64),(64,64))
        first = decoder(main,wrist,geometry,geometry)
        old = first["state"].history[0]
        for field in ("main_latent", "main_rgb", "wrist_latent", "wrist_rgb"):
            original = getattr(old, field)
            for bad in (original.double(), torch.empty_like(original, device="meta"), original.repeat(2,1,1,1)):
                entry = replace(old, **{field: bad})
                with self.assertRaisesRegex(ValueError,"batch/device/dtype"):
                    decoder(main,wrist,geometry,geometry,DecoderState(0,1,(entry,)),frame_index=1)

    def test_analysis_migration_and_bitstream_unchanged(self):
        old = DualViewCodec().eval()
        old.decoder = LegacyDecoder().eval()
        with torch.no_grad():
            old.analysis.gain.log_hyper_gain.add_(torch.linspace(.1,.5,64))
            old.analysis.gain.log_inverse_hyper_gain.add_(torch.linspace(-.3,-.1,64))
        old.update(force=True)
        old_state = old.state_dict()
        with self.assertRaises(RuntimeError):
            SpatioTemporalDecoder().load_state_dict(old.decoder.state_dict(),strict=True)
        new = DualViewCodec().eval()
        analysis_state = {k[len("analysis."):]:v for k,v in old_state.items() if k.startswith("analysis.")}
        new.analysis.load_state_dict(analysis_state,strict=True)
        main,wrist = torch.rand(1,3,95,113),torch.rand(1,3,65,81)
        for i,(mr,wr) in enumerate((((0,0.),(7,0.)),((3,.5),(6,.2)))):
            kwargs = dict(episode_id=19,frame_index=i,main_rate=mr,wrist_rate=wr)
            with mock.patch.object(new.decoder,"forward",side_effect=AssertionError("decoder during compress")):
                self.assertEqual(old.compress(main,wrist,**kwargs),new.compress(main,wrist,**kwargs))
        for name,value in analysis_state.items():
            torch.testing.assert_close(new.analysis.state_dict()[name],value,atol=0,rtol=0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
