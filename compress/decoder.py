"""decoder-v4: dual-view temporal attention with modality fusion and frame ages.

JCT structure, EfficientAttention math and decoder_1/decoder_2 are adapted from
Xinjie-Q/LDMIC, commit 4cf7318505a001c13d9946581f77e2b4af224246,
models/ldmic.py. Copyright (c) 2023 Xinjie-Q, MIT License (full notice below).
Necessary adaptations: M=N=64, separate query/reference spatial dimensions,
and the existing RGB-logits/YCoCg conversion, sigmoid and crop contract.
The detached causal history and per-view temporal residuals after atten_4 are
project extensions, not part of LDMIC. This is not a full paper reproduction.
"""

# MIT License
# Copyright (c) 2023 Xinjie-Q
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import copy
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from compressai.layers import GDN, ResidualBlock
from compressai.models.utils import deconv

from .lightweight_encoder import ImageGeometry, ycocg_to_rgb


def with_temporal_reference_init(state_dict, history_size: int = 2):
    """Explicit v3->v4 state_dict migration (full codec or standalone decoder).

    Add only sum-equivalent fusion and zero age embeddings. Do not overwrite
    learned v4 weights or hide partially missing new parameters. Callers must
    still load strictly and use the model's history_size and Gain level count.
    """
    if isinstance(history_size, bool) or not isinstance(history_size, int) or history_size < 1:
        raise ValueError("history_size must be a positive integer")
    prefix = "decoder." if any(k.startswith("decoder.") for k in state_dict) else ""
    names = tuple(prefix + k for k in ("history_fuse.weight", "history_fuse.bias",
                                       "temporal_age_embedding.weight"))
    present = [name in state_dict for name in names]
    if any(present) and not all(present):
        raise ValueError("Incomplete fusion/age weights; explicit repair is required")
    migrated = copy.copy(state_dict)
    if all(present):
        return migrated
    reference = state_dict[prefix + "temporal.queries.weight"]
    weight = reference.new_zeros(64, 128, 1, 1)
    identity = torch.eye(64, device=reference.device, dtype=reference.dtype)
    weight[:, :64, 0, 0] = identity
    weight[:, 64:, 0, 0] = identity
    migrated[names[0]] = weight
    migrated[names[1]] = reference.new_zeros(64)
    migrated[names[2]] = reference.new_zeros(history_size + 1, 64)
    return migrated


@dataclass(frozen=True)
class DecodedFrame:
    """One synchronized frame pair, copied from decoder-owned reconstructions."""

    main_latent: Tensor
    main_rgb: Tensor
    wrist_latent: Tensor
    wrist_rgb: Tensor


@dataclass(frozen=True)
class DecoderState:
    """Bounded, detached, decoder-owned history; pass None to start afresh.

    A batch represents synchronized streams from the same episode, with stable
    batch ordering. Use separate states for independent episodes/receivers.
    Each history entry contains both views at the same frame. Runtime history is
    deliberately not part of a model's state_dict. Pre-v3 states are incompatible;
    restart from frame_index=0 after upgrading.
    """

    episode_id: int
    next_frame: int
    history: tuple[DecodedFrame, ...]


class EfficientAttention(nn.Module):
    """LDMIC factorized attention with independent query/reference token counts."""

    def __init__(self, key_in_channels=64, query_in_channels=64,
                 key_channels=8, head_count=2, value_channels=16):
        super().__init__()
        self.in_channels = query_in_channels
        self.key_channels, self.head_count, self.value_channels = key_channels, head_count, value_channels
        self.keys = nn.Conv2d(key_in_channels, key_channels, 1)
        self.queries = nn.Conv2d(query_in_channels, key_channels, 1)
        self.values = nn.Conv2d(key_in_channels, value_channels, 1)
        self.reprojection = nn.Conv2d(value_channels, query_in_channels, 1)

    def _attend(self, query, references):
        b, _, h, w = query.shape
        if not references or any(r.shape[0] != b for r in references):
            raise ValueError("Attention needs nonempty references with the query batch size")
        heads = self.head_count
        q = self.queries(query).reshape(b, heads, self.key_channels // heads, h*w).softmax(dim=-2)
        k = torch.cat([self.keys(r).flatten(2) for r in references], dim=-1)
        v = torch.cat([self.values(r).flatten(2) for r in references], dim=-1)
        k = k.reshape(b, heads, self.key_channels // heads, -1).softmax(dim=-1)
        v = v.reshape(b, heads, self.value_channels // heads, -1)
        summary = k @ v.transpose(-1, -2)
        context = summary.transpose(-1, -2) @ q
        return self.reprojection(context.reshape(b, self.value_channels, h, w))

    def forward(self, target: Tensor, reference: Tensor) -> Tensor:
        return self._attend(target, (reference,))


class EfficientCrossAttention(EfficientAttention):
    """Retain the old tuple-of-references import/API for the temporal branch.

    Attribute aliases are properties, so state_dict contains no duplicate keys.
    Parameter names now follow LDMIC; old Decoder weights are not compatible.
    """

    query = property(lambda self: self.queries)
    key = property(lambda self: self.keys)
    value = property(lambda self: self.values)
    project = property(lambda self: self.reprojection)

    def forward(self, query: Tensor, references: tuple[Tensor, ...]) -> Tensor:
        return self._attend(query, references)


class JointContextTransfer(nn.Module):
    """Original LDMIC JCT block structure, shared in both view directions."""

    def __init__(self, channels=64):
        super().__init__()
        self.rb1 = ResidualBlock(channels, channels)
        self.rb2 = ResidualBlock(channels, channels)
        self.attn = EfficientAttention(channels, channels, channels//8, 2, channels//4)
        self.refine = nn.Sequential(ResidualBlock(channels*2, channels),
                                    ResidualBlock(channels, channels))

    def forward(self, main, wrist):
        identity_main, identity_wrist = main, wrist
        main_feature = self.rb2(self.rb1(main))
        wrist_feature = self.rb2(self.rb1(wrist))
        context_main = self.attn(main_feature, wrist_feature)
        context_wrist = self.attn(wrist_feature, main_feature)
        return (identity_main + self.refine(torch.cat((context_main, main_feature), dim=1)),
                identity_wrist + self.refine(torch.cat((context_wrist, wrist_feature), dim=1)))


class SpatioTemporalDecoder(nn.Module):
    """Decoded main/wrist features -> causal, dual-view RGB reconstructions."""

    def __init__(self, history_size: int = 2):
        super().__init__()
        if isinstance(history_size, bool) or not isinstance(history_size, int) or history_size < 1:
            raise ValueError("history_size must be a positive integer")
        self.history_size = history_size
        self.atten_3 = JointContextTransfer(64)
        self.decoder_1 = nn.Sequential(deconv(64, 64, kernel_size=5, stride=2), GDN(64, inverse=True),
                                       deconv(64, 64, kernel_size=5, stride=2), GDN(64, inverse=True))
        self.atten_4 = JointContextTransfer(64)
        self.decoder_2 = nn.Sequential(deconv(64, 64, kernel_size=5, stride=2), GDN(64, inverse=True),
                                       deconv(64, 3, kernel_size=5, stride=2))
        # Preserve H/16 history extraction; inject only AFTER the last spatial JCT.
        self.content = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1), nn.GELU())
        self.history_image = nn.Conv2d(3, 64, 1)
        self.temporal = EfficientCrossAttention()
        self.temporal_project = nn.Conv2d(64, 64, 1)
        self.temporal_wrist = EfficientCrossAttention()
        self.temporal_project_wrist = nn.Conv2d(64, 64, 1)
        self.history_fuse = nn.Conv2d(128, 64, 1)
        self.temporal_age_embedding = nn.Embedding(history_size + 1, 64, padding_idx=0)
        with torch.no_grad():
            self.history_fuse.weight.zero_()
            identity = torch.eye(64)
            self.history_fuse.weight[:, :64, 0, 0].copy_(identity)
            self.history_fuse.weight[:, 64:, 0, 0].copy_(identity)
            self.history_fuse.bias.zero_()
            self.temporal_age_embedding.weight.zero_()

    def _history_references(self, history: tuple[DecodedFrame, ...], view: str) -> tuple[Tensor, ...]:
        """Chronological references with newest age=1, independent of view H/W."""
        if view not in ("main", "wrist"):
            raise ValueError("History view must be main or wrist")
        references = []
        # Keep token order unchanged for numerical continuity with v3. Cache is
        # oldest->newest, so age counts down rather than treating index 0 as t-1.
        for index, entry in enumerate(history):
            latent, rgb = getattr(entry, view + "_latent"), getattr(entry, view + "_rgb")
            latent_feature = self.content(latent)
            rgb_feature = self.history_image(F.adaptive_avg_pool2d(rgb, latent.shape[-2:]))
            feature = self.history_fuse(torch.cat((latent_feature, rgb_feature), dim=1))
            age = torch.tensor(len(history) - index, device=latent.device, dtype=torch.long)
            age_embedding = self.temporal_age_embedding(age).reshape(1, 64, 1, 1)
            references.append(feature + age_embedding)
        return tuple(references)

    def _reconstruct(self, feature: Tensor, geometry: ImageGeometry) -> Tensor:
        feature = self.decoder_2(feature)
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
            if not isinstance(entry, DecodedFrame) or not all(hasattr(entry, name) for name in
                    ("main_latent", "main_rgb", "wrist_latent", "wrist_rgb")):
                raise ValueError("Incompatible decoder history; restart at frame 0")
            for latent, rgb, current in ((entry.main_latent, entry.main_rgb, main),
                                         (entry.wrist_latent, entry.wrist_rgb, wrist)):
                if (latent.shape[0] != current.shape[0] or rgb.shape[0] != current.shape[0]
                        or latent.device != current.device or latent.dtype != current.dtype
                        or rgb.device != current.device or rgb.dtype != current.dtype):
                    raise ValueError("History batch/device/dtype changed; restart at frame 0")

        main_feature, wrist_feature = self.atten_3(main, wrist)
        main_feature, wrist_feature = self.atten_4(self.decoder_1(main_feature),
                                                 self.decoder_1(wrist_feature))
        if history:
            references = self._history_references(history, "main")
            temporal = self.temporal(self.content(main), references)
            main_feature = main_feature + self.temporal_project(
                F.interpolate(temporal, size=main_feature.shape[-2:], mode="nearest"))
            references_wrist = self._history_references(history, "wrist")
            temporal_wrist = self.temporal_wrist(self.content(wrist), references_wrist)
            wrist_feature = wrist_feature + self.temporal_project_wrist(
                F.interpolate(temporal_wrist, size=wrist_feature.shape[-2:], mode="nearest"))
        main_rgb = self._reconstruct(main_feature, main_geometry)
        wrist_rgb = self._reconstruct(wrist_feature, wrist_geometry)
        # Copy as well as detach: caller mutation of outputs cannot corrupt cache.
        entry = DecodedFrame(main.detach().clone(), main_rgb.detach().clone(),
                             wrist.detach().clone(), wrist_rgb.detach().clone())
        next_state = DecoderState(episode_id, frame_index + 1,
                                  (*history, entry)[-self.history_size:])
        return {"main_hat": main_rgb, "wrist_hat": wrist_rgb,
                "state": next_state, "history_used": len(history)}
