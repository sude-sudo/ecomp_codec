"""Trainable analysis transform replacing Ballé ScaleHyperprior.g_a.

The four stride-2 stages follow compressai.models.google.ScaleHyperprior's
analysis-transform layout, with new weights and at most 64 channels. No
quantization, gain, entropy coding, or synthesis transform lives here.
"""

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _check_image(x: Tensor) -> None:
    if x.ndim != 4 or x.shape[1] != 3 or min(x.shape) < 1:
        raise ValueError("Expected a nonempty Bx3xHxW tensor")
    if not x.is_floating_point():
        raise TypeError("Expected a floating-point image tensor")


def rgb_to_ycocg(rgb: Tensor) -> Tensor:
    """Convert RGB in [0, 1] to YCoCg, keeping signed chroma and gradients."""
    _check_image(rgb)
    r, g, b = rgb.unbind(1)
    return torch.stack(((r + 2 * g + b) / 4, (r - b) / 2,
                        (-r + 2 * g - b) / 4), dim=1)


def ycocg_to_rgb(image: Tensor) -> Tensor:
    """Invert the color transform without clipping training gradients."""
    _check_image(image)
    y, co, cg = image.unbind(1)
    return torch.stack((y + co - cg, y + cg, y - co - cg), dim=1)


@dataclass(frozen=True)
class ImageGeometry:
    """Geometry to transmit alongside a future bitstream for exact cropping."""

    original_hw: tuple[int, int]
    padded_hw: tuple[int, int]

    def crop(self, reconstructed: Tensor) -> Tensor:
        if tuple(reconstructed.shape[-2:]) != self.padded_hw:
            raise ValueError("Reconstruction must match padded image dimensions")
        h, w = self.original_hw
        return reconstructed[..., :h, :w]


def prepare_rgb(rgb: Tensor) -> tuple[Tensor, ImageGeometry]:
    """Color-convert and replicate-pad right/bottom to multiples of 64.

    Four encoder strides and two hyperprior strides require this alignment.
    Padding is outside g_a; its forward accepts already prepared YCoCg.
    """
    image = rgb_to_ycocg(rgb)
    h, w = image.shape[-2:]
    ph, pw = (-h) % 64, (-w) % 64
    if ph or pw:
        image = F.pad(image, (0, pw, 0, ph), mode="replicate")
    return image, ImageGeometry((h, w), (h + ph, w + pw))


class DSConv(nn.Module):
    """3x3 spatial depthwise convolution followed by 1x1 channel projection."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, 3, stride=2,
                                   padding=1, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        return self.pointwise(self.depthwise(x))


class LightweightEncoder(nn.Module):
    """Shared-view g_a: Bx3xHxW YCoCg -> continuous Bx64x(H/16)x(W/16)."""

    channels = (3, 32, 48, 64, 64)
    downsampling_factor = 16
    latent_channels = 64

    def __init__(self):
        super().__init__()
        self.stages = nn.Sequential(*(
            nn.Sequential(
                DSConv(cin, cout),
                nn.Conv2d(cout, cout, 3, padding=1, groups=cout),
                nn.GELU(),
            )
            for cin, cout in zip(self.channels[:-1], self.channels[1:])
        ))

    def forward(self, image: Tensor) -> Tensor:
        _check_image(image)
        if any(size % 16 for size in image.shape[-2:]):
            raise ValueError("g_a requires multiples of 16; use prepare_rgb for 64 alignment")
        return self.stages(image)

    def encode_pair(self, main_rgb: Tensor, wrist_rgb: Tensor):
        """Encode each view through this same instance; preserve both geometries."""
        main, main_geometry = prepare_rgb(main_rgb)
        wrist, wrist_geometry = prepare_rgb(wrist_rgb)
        return (self(main), self(wrist)), (main_geometry, wrist_geometry)
