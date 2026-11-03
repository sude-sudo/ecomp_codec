"""Ballé scale hyperprior for the 64-channel lightweight analysis transform.

Uses CompressAI 1.2.8's unmodified entropy components and ScaleHyperprior's
h_a/h_s layer layout. No image decoder is constructed or called.
Call eval() and update(force=True) after training before actual coding; the
receiver needs the same model weights/CDFs. Forward remains differentiable.
"""

import struct

import torch
from torch import Tensor, nn

from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.models import CompressionModel
from compressai.models.utils import conv, deconv

from .lightweight_encoder import ImageGeometry, LightweightEncoder, prepare_rgb


class LatentHyperprior(CompressionModel):
    """64-channel y -> quantized y_hat, with the original scale-only prior."""

    def __init__(self):
        super().__init__()
        self.h_a = nn.Sequential(
            conv(64, 64, kernel_size=3, stride=1), nn.ReLU(inplace=True),
            conv(64, 64), nn.ReLU(inplace=True), conv(64, 64),
        )
        self.h_s = nn.Sequential(
            deconv(64, 64), nn.ReLU(inplace=True),
            deconv(64, 64), nn.ReLU(inplace=True),
            conv(64, 64, kernel_size=3, stride=1), nn.ReLU(inplace=True),
        )
        self.entropy_bottleneck = EntropyBottleneck(64)
        self.gaussian_conditional = GaussianConditional(None)

    @staticmethod
    def _check_latent(y: Tensor) -> None:
        if (y.ndim != 4 or y.shape[1] != 64 or min(y.shape) < 1
                or any(s % 4 for s in y.shape[-2:])):
            raise ValueError("Expected Bx64xHxW latent with H/W multiples of 4")
        if not y.is_floating_point():
            raise TypeError("Expected floating-point latent")

    def _check_coding(self) -> None:
        if self.training:
            raise RuntimeError("Call eval() before entropy coding")
        if any(m._quantized_cdf.numel() == 0 for m in
               (self.entropy_bottleneck, self.gaussian_conditional)):
            raise RuntimeError("Call update() to initialize entropy coding CDFs")

    @staticmethod
    def _check_hyper_gain(gain):
        if gain is not None and (not isinstance(gain, Tensor) or gain.shape != (1, 64, 1, 1)):
            raise ValueError("Hyper gains must have shape [1, 64, 1, 1]")

    def _check_hyper_pair(self, hyper_gain, inverse_hyper_gain):
        if (hyper_gain is None) != (inverse_hyper_gain is None):
            raise ValueError("Supply both hyper gains, or neither for the original prior")
        self._check_hyper_gain(hyper_gain)
        self._check_hyper_gain(inverse_hyper_gain)

    def forward(self, y: Tensor, *, hyper_gain=None, inverse_hyper_gain=None):
        self._check_latent(y)
        self._check_hyper_pair(hyper_gain, inverse_hyper_gain)
        z = self.h_a(y.abs())
        if hyper_gain is not None:
            z = z * hyper_gain
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        if inverse_hyper_gain is not None:
            z_hat = z_hat * inverse_hyper_gain
        scales = self.h_s(z_hat)
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales)
        return {"y_hat": y_hat,
                "likelihoods": {"y": y_likelihoods, "z": z_likelihoods}}

    @torch.no_grad()
    def compress(self, y: Tensor, *, hyper_gain=None, inverse_hyper_gain=None):
        self._check_latent(y)
        self._check_coding()
        self._check_hyper_pair(hyper_gain, inverse_hyper_gain)
        z = self.h_a(y.abs())
        if hyper_gain is not None:
            z = z * hyper_gain
        z_strings = self.entropy_bottleneck.compress(z)
        # Use only the decoded hyperlatent to predict the coding distribution.
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.shape[-2:])
        if inverse_hyper_gain is not None:
            z_hat = z_hat * inverse_hyper_gain
        indexes = self.gaussian_conditional.build_indexes(self.h_s(z_hat))
        y_strings = self.gaussian_conditional.compress(y, indexes)
        return {"strings": [y_strings, z_strings], "shape": tuple(z.shape[-2:])}

    @torch.no_grad()
    def decompress(self, strings, shape, *, inverse_hyper_gain=None):
        self._check_coding()
        self._check_hyper_gain(inverse_hyper_gain)
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        if inverse_hyper_gain is not None:
            z_hat = z_hat * inverse_hyper_gain
        indexes = self.gaussian_conditional.build_indexes(self.h_s(z_hat))
        return self.gaussian_conditional.decompress(strings[0], indexes, z_hat.dtype)


# Versioned test-stage transport: magic, batch count, original image H/W;
# then (y length, z length, y bytes, z bytes) for each image in the batch.
# Padded dimensions and z shape are derived from the fixed 64-pixel alignment.
# Model weights/CDFs are shared separately, as in CompressAI; they are not payload.
_HEADER = struct.Struct(">4sIII")
_LENGTHS = struct.Struct(">II")
_MAGIC = b"ECH1"


class EncoderHyperprior(CompressionModel):
    """RGB -> self-contained bytes -> y_hat, ready for a future custom decoder.

    Each camera calls this same instance. No source image, latent, or Python
    geometry object is needed by decompress(). BPP uses original pixel count,
    includes both entropy streams, and exposes transport-inclusive/payload rates.
    """

    def __init__(self):
        super().__init__()
        self.g_a = LightweightEncoder()
        self.hyperprior = LatentHyperprior()

    def forward(self, rgb: Tensor):
        image, geometry = prepare_rgb(rgb)
        result = self.hyperprior(self.g_a(image))
        num_pixels = rgb.shape[0] * rgb.shape[-2] * rgb.shape[-1]
        result["estimated_bpp"] = sum(-p.log2().sum() for p in
                                      result["likelihoods"].values()) / num_pixels
        result["geometry"] = geometry
        return result

    @torch.no_grad()
    def compress(self, rgb: Tensor) -> bytes:
        self.hyperprior._check_coding()
        image, geometry = prepare_rgb(rgb)
        return self._compress_latent(self.g_a(image), geometry)

    @torch.no_grad()
    def _compress_latent(self, y: Tensor, geometry: ImageGeometry, *,
                         hyper_gain=None, inverse_hyper_gain=None) -> bytes:
        """Shared transport for an analysis latent, optionally transformed by Gain."""
        encoded = self.hyperprior.compress(y, hyper_gain=hyper_gain,
                                           inverse_hyper_gain=inverse_hyper_gain)
        h, w = geometry.original_hw
        packet = bytearray(_HEADER.pack(_MAGIC, y.shape[0], h, w))
        for y_string, z_string in zip(*encoded["strings"]):
            packet.extend(_LENGTHS.pack(len(y_string), len(z_string)))
            packet.extend(y_string)
            packet.extend(z_string)
        return bytes(packet)

    @torch.no_grad()
    def decompress(self, packet: bytes, *, inverse_hyper_gain=None):
        self.hyperprior._check_coding()
        if not isinstance(packet, bytes) or len(packet) < _HEADER.size:
            raise ValueError("Expected a complete ECH1 bytes packet")
        magic, batch, h, w = _HEADER.unpack_from(packet)
        if magic != _MAGIC or min(batch, h, w) < 1:
            raise ValueError("Invalid ECH1 header")
        padded = ((h + 63) // 64 * 64, (w + 63) // 64 * 64)
        ys, zs, offset, payload_bytes = [], [], _HEADER.size, 0
        for _ in range(batch):
            if offset + _LENGTHS.size > len(packet):
                raise ValueError("Truncated stream lengths")
            ny, nz = _LENGTHS.unpack_from(packet, offset)
            offset += _LENGTHS.size
            if min(ny, nz) < 1 or offset + ny + nz > len(packet):
                raise ValueError("Truncated entropy stream")
            ys.append(packet[offset:offset + ny])
            zs.append(packet[offset + ny:offset + ny + nz])
            offset += ny + nz
            payload_bytes += ny + nz
        if offset != len(packet):
            raise ValueError("Unexpected trailing packet bytes")
        y_hat = self.hyperprior.decompress([ys, zs], (padded[0] // 64, padded[1] // 64),
                                          inverse_hyper_gain=inverse_hyper_gain)
        num_pixels = batch * h * w
        return {"y_hat": y_hat, "geometry": ImageGeometry((h, w), padded),
                "num_bytes": len(packet), "actual_bpp": 8 * len(packet) / num_pixels,
                "payload_bpp": 8 * payload_bytes / num_pixels}
