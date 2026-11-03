"""Learnable channel gains and geometric interpolation for variable-rate coding.

Reference: Cui et al., CVPR 2021, Eq. (8),
https://openaccess.thecvf.com/content/CVPR2021/html/Cui_Asymmetric_Gained_Deep_Image_Compression_With_Continuous_Rate_Adaptation_CVPR_2021_paper.html
and mmSir/GainedVAE compressai/models/gain.py::GainedScaleHyperprior.
Both y and z have independent gain pairs; the Ballé entropy algorithm is unchanged.
"""

import copy
import math
from numbers import Integral, Real
import struct

import torch
from torch import Tensor, nn

from .hyperprior import EncoderHyperprior
from .lightweight_encoder import prepare_rgb


class LatentGain(nn.Module):
    """Four independent positive y/z gain tables, each [levels, 64].

    g(s, alpha) = exp((1-alpha)*log_g[s] + alpha*log_g[s+1]).
    alpha=0 selects s; alpha=1 selects s+1. InverseGain uses the same formula,
    with its own learned parameters (not constrained to the exact reciprocal).
    One rate point applies to an entire image batch, independently per camera.
    """

    def __init__(self, levels: int = 8):
        super().__init__()
        if isinstance(levels, bool) or not isinstance(levels, Integral) or levels < 2:
            raise ValueError("At least two integer gain levels are required")
        self.levels = int(levels)
        # Distinct positive starting scales; learned RD ordering is evaluated
        # after training. Reciprocal initialization does not tie the parameters.
        initial = torch.linspace(0, math.log(8), self.levels)[:, None].repeat(1, 64)
        self.log_gain = nn.Parameter(initial)
        self.log_inverse_gain = nn.Parameter(-initial.clone())
        self.log_hyper_gain = nn.Parameter(torch.zeros(self.levels, 64))
        self.log_inverse_hyper_gain = nn.Parameter(torch.zeros(self.levels, 64))

    def validate_rate(self, level: int, alpha: float) -> tuple[int, float]:
        if (isinstance(level, bool) or not isinstance(level, Integral)
                or not 0 <= level < self.levels):
            raise ValueError(f"level must be an integer in [0, {self.levels - 1}]")
        if (isinstance(alpha, bool) or not isinstance(alpha, Real)
                or not math.isfinite(alpha) or not 0 <= alpha <= 1):
            raise ValueError("alpha must be a finite real number in [0, 1]")
        if level == self.levels - 1 and alpha != 0:
            raise ValueError("The last discrete level requires alpha=0")
        return int(level), float(alpha)

    def factors(self, level: int = 0, alpha: float = 0.):
        return self._interpolate((self.log_gain, self.log_inverse_gain), level, alpha)

    def hyper_factors(self, level: int = 0, alpha: float = 0.):
        return self._interpolate((self.log_hyper_gain, self.log_inverse_hyper_gain), level, alpha)

    def _interpolate(self, tables, level, alpha):
        level, alpha = self.validate_rate(level, alpha)
        adjacent = min(level + 1, self.levels - 1)
        return tuple(((1 - alpha) * table[level] + alpha * table[adjacent])
                     .exp().reshape(1, 64, 1, 1)
                     for table in tables)

    def forward(self, y: Tensor, level: int = 0, alpha: float = 0.) -> Tensor:
        return y * self.factors(level, alpha)[0]

    def inverse(self, y_hat: Tensor, level: int = 0, alpha: float = 0.) -> Tensor:
        return y_hat * self.factors(level, alpha)[1]


def with_identity_hypergains(state_dict):
    """Explicitly migrate a y-only checkpoint without mutating its tensors.

    Supports LatentGain, GainedEncoderHyperprior and DualViewCodec key prefixes.
    Both missing hyper tables become zeros (unit factors). Already present pairs
    are preserved; partial pairs are rejected. Then use strict load_state_dict.
    This does not migrate optimizer state or change an old bitstream's format.
    """
    upgraded = copy.copy(state_dict)
    for key, value in state_dict.items():
        if key != "log_gain" and not key.endswith(".log_gain"):
            continue
        prefix = key[:-len("log_gain")]
        names = (prefix + "log_hyper_gain", prefix + "log_inverse_hyper_gain")
        present = tuple(name in state_dict for name in names)
        if any(present) and not all(present):
            raise ValueError("Checkpoint has only one of the two HyperGain tables")
        if not any(present):
            for name in names:
                upgraded[name] = torch.zeros_like(value)
    return upgraded


# ECG1 wraps the existing ECH1 image/entropy packet with 16 rate bytes:
# magic, uint32 level, IEEE-754 float64 alpha. Encoding uses exactly the same
# Python float serialized here, so no lower-precision transport mismatch occurs.
_RATE_HEADER = struct.Struct(">4sId")
_RATE_MAGIC = b"ECG1"


class GainedEncoderHyperprior(EncoderHyperprior):
    """RGB -> y Gain -> z HyperGain/prior -> y InverseGain -> decoder-ready y_hat.

    decompress(packet) recovers the rate solely from bytes. As before, model
    weights and entropy CDFs must be shared; eval()/update() precede coding.
    The ungained EncoderHyperprior API/packet format remains available.
    """

    def __init__(self, levels: int = 8):
        super().__init__()
        self.gain = LatentGain(levels)

    def forward(self, rgb: Tensor, level: int = 0, alpha: float = 0.):
        level, alpha = self.gain.validate_rate(level, alpha)
        image, geometry = prepare_rgb(rgb)
        # h_a and the probability model both see the gained representation.
        hyper_gain, inverse_hyper_gain = self.gain.hyper_factors(level, alpha)
        result = self.hyperprior(self.gain(self.g_a(image), level, alpha),
                                 hyper_gain=hyper_gain, inverse_hyper_gain=inverse_hyper_gain)
        result["gained_y_hat"] = result["y_hat"]
        result["y_hat"] = self.gain.inverse(result["y_hat"], level, alpha)
        pixels = rgb.shape[0] * rgb.shape[-2] * rgb.shape[-1]
        result["estimated_bpp"] = sum(-p.log2().sum() for p in
                                      result["likelihoods"].values()) / pixels
        result["geometry"] = geometry
        result["rate"] = {"level": level, "alpha": alpha}
        return result

    @torch.no_grad()
    def compress(self, rgb: Tensor, level: int = 0, alpha: float = 0.) -> bytes:
        self.hyperprior._check_coding()
        level, alpha = self.gain.validate_rate(level, alpha)
        image, geometry = prepare_rgb(rgb)
        y = self.gain(self.g_a(image), level, alpha)
        hyper_gain, inverse_hyper_gain = self.gain.hyper_factors(level, alpha)
        return (_RATE_HEADER.pack(_RATE_MAGIC, level, alpha)
                + self._compress_latent(y, geometry, hyper_gain=hyper_gain,
                                        inverse_hyper_gain=inverse_hyper_gain))

    @torch.no_grad()
    def decompress(self, packet: bytes):
        if not isinstance(packet, bytes) or len(packet) < _RATE_HEADER.size:
            raise ValueError("Expected a complete ECG1 bytes packet")
        magic, level, alpha = _RATE_HEADER.unpack_from(packet)
        if magic != _RATE_MAGIC:
            raise ValueError("Invalid ECG1 rate header")
        level, alpha = self.gain.validate_rate(level, alpha)
        inverse_hyper_gain = self.gain.hyper_factors(level, alpha)[1]
        result = super().decompress(packet[_RATE_HEADER.size:],
                                    inverse_hyper_gain=inverse_hyper_gain)
        result["gained_y_hat"] = result["y_hat"]
        result["y_hat"] = self.gain.inverse(result["y_hat"], level, alpha)
        h, w = result["geometry"].original_hw
        pixels = result["y_hat"].shape[0] * h * w
        result["num_bytes"] = len(packet)
        result["actual_bpp"] = 8 * len(packet) / pixels
        result["rate"] = {"level": level, "alpha": alpha}
        return result
