# Frozen ablation baseline: ecomp_codec commit 7db7e58, compress/decoder.py.
# Only the package-relative import is changed for placement under tests/fixtures.
"""Lightweight reconstruction with LDMIC-style decoded-feature context transfer.

Design reference: Xinjie-Q/LDMIC models/ldmic.py, JointContextTransfer and
EfficientAttention (https://github.com/Xinjie-Q/LDMIC). Attention is performed
only at H/16, with a small key/value channel summary rather than an NxN map.
No Ballé synthesis transform is instantiated or used.
"""

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from compress.lightweight_encoder import ImageGeometry, ycocg_to_rgb


@dataclass(frozen=True)
class DecodedFrame:
    latent: Tensor
    rgb: Tensor


@dataclass(frozen=True)
class DecoderState:
    """Bounded, detached, decoder-owned history; pass None to start afresh.

    A batch represents synchronized streams from the same episode, with stable
    batch ordering. Use separate states for independent episodes/receivers.
    Runtime history is deliberately not part of a model's state_dict.
    """

    episode_id: int
    next_frame: int
    history: tuple[DecodedFrame, ...]


class EfficientCrossAttention(nn.Module):
    """Two-head context transfer, supporting unequal query/reference resolutions."""

    def __init__(self):
        super().__init__()
        self.query = nn.Conv2d(64, 8, 1)
        self.key = nn.Conv2d(64, 8, 1, bias=False)
        self.value = nn.Conv2d(64, 16, 1)
        self.project = nn.Conv2d(16, 64, 1)

    def forward(self, query: Tensor, references: tuple[Tensor, ...]) -> Tensor:
        b, _, h, w = query.shape
        q = self.query(query).reshape(b, 2, 4, h * w).softmax(dim=-2)
        k = torch.cat([self.key(r).flatten(2) for r in references], dim=-1)
        v = torch.cat([self.value(r).flatten(2) for r in references], dim=-1)
        k = k.reshape(b, 2, 4, -1).softmax(dim=-1)
        v = v.reshape(b, 2, 8, -1)
        summary = k @ v.transpose(-1, -2)
        context = summary.transpose(-1, -2) @ q
        return self.project(context.reshape(b, 16, h, w))


class SpatioTemporalDecoder(nn.Module):
    """Decoded main/wrist features -> causal, dual-view RGB reconstructions."""

    def __init__(self, history_size: int = 2):
        super().__init__()
        if isinstance(history_size, bool) or not isinstance(history_size, int) or history_size < 1:
            raise ValueError("history_size must be a positive integer")
        self.history_size = history_size
        self.content = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1), nn.GELU())
        self.history_image = nn.Conv2d(3, 64, 1)
        self.spatial = EfficientCrossAttention()
        self.temporal = EfficientCrossAttention()
        self.fuse = nn.Sequential(nn.Conv2d(192, 64, 1), nn.GELU(),
                                  nn.Conv2d(64, 64, 3, padding=1), nn.GELU())
        channels = (64, 48, 32, 16, 3)
        self.upsample = nn.ModuleList(nn.Conv2d(cin, cout, 3, padding=1)
                                      for cin, cout in zip(channels[:-1], channels[1:]))

    def _reconstruct(self, feature: Tensor, geometry: ImageGeometry) -> Tensor:
        for i, layer in enumerate(self.upsample):
            feature = layer(F.interpolate(feature, scale_factor=2, mode="nearest"))
            if i < len(self.upsample) - 1:
                feature = F.gelu(feature)
        # The head predicts YCoCg-domain logits; convert to RGB before bounding.
        return geometry.crop(ycocg_to_rgb(feature).sigmoid())

    def forward(self, main: Tensor, wrist: Tensor,
                main_geometry: ImageGeometry, wrist_geometry: ImageGeometry,
                state: DecoderState | None = None, episode_id: int = 0, frame_index: int = 0):
        for y, geometry in ((main, main_geometry), (wrist, wrist_geometry)):
            if (y.ndim != 4 or y.shape[1] != 64
                    or tuple(s * 16 for s in y.shape[-2:]) != geometry.padded_hw):
                raise ValueError("Decoded latent must match its padded image geometry")
        if main.shape[0] != wrist.shape[0]:
            raise ValueError("Both camera batches must have the same size")
        if frame_index == 0:
            history = ()  # First frame always resets, even for a reused episode ID.
        elif (state is None or state.episode_id != episode_id
              or state.next_frame != frame_index):
            raise ValueError("Missing or out-of-order decoder history; restart at frame 0")
        else:
            if len(state.history) != min(frame_index, self.history_size):
                raise ValueError("Incomplete decoder history; restart at frame 0")
            history = state.history[-self.history_size:]
        for entry in history:
            if (entry.latent.shape[0] != main.shape[0] or entry.rgb.shape[0] != main.shape[0]
                    or entry.latent.device != main.device or entry.latent.dtype != main.dtype):
                raise ValueError("History batch/device/dtype changed; restart at frame 0")

        main_feature, wrist_feature = self.content(main), self.content(wrist)
        # Both contexts use base decoded features, so neither view waits on the
        # other's reconstruction. Wrist remains independently reconstructible.
        spatial_main = self.spatial(main_feature, (wrist_feature,))
        spatial_wrist = self.spatial(wrist_feature, (main_feature,))
        if history:
            references = tuple(self.content(entry.latent) + self.history_image(
                F.adaptive_avg_pool2d(entry.rgb, entry.latent.shape[-2:])) for entry in history)
            temporal = self.temporal(main_feature, references)
        else:
            temporal = torch.zeros_like(main_feature)
        main_fused = self.fuse(torch.cat((main_feature, spatial_main, temporal), dim=1))
        wrist_fused = self.fuse(torch.cat((wrist_feature, spatial_wrist,
                                         torch.zeros_like(wrist_feature)), dim=1))
        main_rgb = self._reconstruct(main_fused, main_geometry)
        wrist_rgb = self._reconstruct(wrist_fused, wrist_geometry)
        # Copy as well as detach: caller mutation of outputs cannot corrupt cache.
        entry = DecodedFrame(main.detach().clone(), main_rgb.detach().clone())
        next_state = DecoderState(episode_id, frame_index + 1,
                                  (*history, entry)[-self.history_size:])
        return {"main_hat": main_rgb, "wrist_hat": wrist_rgb,
                "state": next_state, "history_used": len(history)}
