"""Parallel spatial / two-frame efficient / eight-frame VTS decoder.

Legacy decoder implementations and attention mathematics remain untouched.
Each branch returns an enhancement of the same unmodified scale input; only
the fusion's outer residual adds that input back.
"""
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .decoder import EfficientCrossAttention
from .decoder_v5 import DecodedFrame, DecoderState, SpatioTemporalDecoderV5
from .lightweight_encoder import ycocg_to_rgb


@dataclass(frozen=True)
class FusionDecodedFrame(DecodedFrame):
    episode_id: int
    frame_index: int


class ShortTemporalAttention(nn.Module):
    """V4 factorized cross-attention, independently queried at each scale/view."""
    history_size = 2

    def __init__(self, channels):
        super().__init__()
        self.latent = nn.Sequential(nn.Conv2d(64, channels, 1),
                                    nn.Conv2d(channels, channels, 3, padding=1), nn.GELU())
        self.content = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1), nn.GELU())
        self.rgb = nn.Conv2d(3, channels, 1)
        self.fuse = nn.Conv2d(2 * channels, channels, 1)
        self.age = nn.Embedding(3, channels, padding_idx=0)
        nn.init.zeros_(self.age.weight)
        with torch.no_grad():
            self.fuse.weight.zero_()
            self.fuse.bias.zero_()
            identity = torch.eye(channels)
            self.fuse.weight[:, :channels, 0, 0].copy_(identity)
            self.fuse.weight[:, channels:, 0, 0].copy_(identity)
        self.attention = nn.ModuleDict({view: EfficientCrossAttention(
            channels, channels, channels // 8, 4 if channels >= 128 else 2, channels // 4)
            for view in ('main', 'wrist')})

    def references(self, history, view, size):
        refs = []
        history = history[-self.history_size:]
        for index, entry in enumerate(history):
            latent = F.interpolate(self.latent(getattr(entry, view + '_latent')),
                                   size=size, mode='nearest')
            rgb = self.rgb(F.adaptive_avg_pool2d(getattr(entry, view + '_rgb'), size))
            fused = self.fuse(torch.cat((latent, rgb), dim=1))
            refs.append(fused + self.age.weight[len(history) - index][None, :, None, None])
        return tuple(refs)

    def forward(self, current, history, view):
        if not history:
            return torch.zeros_like(current)
        return self.attention[view](self.content(current),
                                    self.references(history, view, current.shape[-2:]))


class ThreeBranchDecoder(SpatioTemporalDecoderV5):
    """Explicit opt-in configuration; legacy vts8/vts16 keep their IGDN/serial path.

    branch_mask is (spatial, short, long). fusion_scales=('low',) is the
    low-only ablation: the high scale retains spatial JCT without temporal paths.
    Both options form part of the state signature and must match at the receiver.
    """
    def __init__(self, history_size=8, M=320, N=192, *, short_history_size=2,
                 branch_mask=(True, True, True), fusion_scales=('low', 'high'),
                 window_size=8, memory_grid=4, query_chunk=64):
        if history_size != 8 or isinstance(history_size, bool) or short_history_size != 2:
            raise ValueError('tri_fusion8 requires long history 8 and short history 2')
        if len(branch_mask) != 3 or any(type(flag) is not bool for flag in branch_mask):
            raise ValueError('branch_mask must contain three booleans: spatial, short, long')
        if tuple(fusion_scales) not in (('low',), ('low', 'high')):
            raise ValueError('fusion_scales must be low-only or low/high')
        super().__init__(history_size, M, N, temporal='vts', window_size=window_size,
                         memory_grid=memory_grid, query_chunk=query_chunk)
        self.temporal_kind = 'tri_fusion'
        self.short_history_size = short_history_size
        self.branch_mask = tuple(branch_mask)
        self.fusion_scales = tuple(fusion_scales)
        self.signature = ('decoder-tri-fusion-v1', *self.signature[1:],
                          self.short_history_size, self.branch_mask, self.fusion_scales)
        # Only this new decoder replaces the three synthesis IGDNs.
        self.decoder_1[1] = nn.GELU()
        self.decoder_1[3] = nn.GELU()
        self.decoder_2[1] = nn.GELU()
        self.short_low = ShortTemporalAttention(M)
        self.short_high = ShortTemporalAttention(N)
        self.fusion_low = self._fusion(M)
        self.fusion_high = self._fusion(N)

    @staticmethod
    def _fusion(channels):
        modules = nn.ModuleDict()
        for view in ('main', 'wrist'):
            # No bias: zero branch inputs cannot introduce a constant update.
            projection = nn.Conv2d(3 * channels, channels, 1, bias=False)
            with torch.no_grad():
                projection.weight.mul_(.1)
            modules[view] = nn.Sequential(projection, nn.GELU())
        return modules

    def _history(self, state, episode_id, frame_index, main, wrist):
        history = super()._history(state, episode_id, frame_index, main, wrist)
        for expected, entry in enumerate(history, start=frame_index - len(history)):
            if (not isinstance(entry, FusionDecodedFrame) or entry.frame_index != expected
                    or entry.episode_id != episode_id):
                raise ValueError('Reordered, missing or cross-episode tri-fusion history')
        return history

    def branch_deltas(self, main, wrist, history, scale):
        """Inspectable pure enhancements; no branch reads another branch's output."""
        if scale not in ('low', 'high'):
            raise ValueError('Unknown fusion scale')
        spatial = self.atten_3 if scale == 'low' else self.atten_4
        short = self.short_low if scale == 'low' else self.short_high
        long = self.vts_low if scale == 'low' else self.vts_high
        history = history[-8:]
        if self.branch_mask[0]:
            spatial_m, spatial_w = spatial(main, wrist)
            spatial_deltas = (spatial_m - main, spatial_w - wrist)
        else:
            spatial_deltas = (torch.zeros_like(main), torch.zeros_like(wrist))
        refs = long.references(history, scale) if self.branch_mask[2] else []
        results = {}
        for view_index, (view, current) in enumerate((('main', main), ('wrist', wrist))):
            short_delta = (short(current, history, view) if self.branch_mask[1]
                           else torch.zeros_like(current))
            long_delta = (long(current, refs, view_index) - current if refs
                          else torch.zeros_like(current))
            results[view] = (spatial_deltas[view_index], short_delta, long_delta)
        return results

    def _parallel(self, main, wrist, history, scale):
        branches = self.branch_deltas(main, wrist, history, scale)
        fusion = self.fusion_low if scale == 'low' else self.fusion_high
        return tuple(current + fusion[view](torch.cat(branches[view], dim=1))
                     for view, current in (('main', main), ('wrist', wrist)))

    def forward(self, main, wrist, main_geometry, wrist_geometry,
                state=None, episode_id=0, frame_index=0):
        for latent, geometry in ((main, main_geometry), (wrist, wrist_geometry)):
            if (latent.ndim != 4 or latent.shape[1] != 64
                    or tuple(s * 16 for s in latent.shape[-2:]) != geometry.padded_hw):
                raise ValueError('Expected 64-channel latent matching padded geometry')
        if main.shape[0] != wrist.shape[0]:
            raise ValueError('Camera batch sizes differ')
        history = self._history(state, episode_id, frame_index, main, wrist)
        low_m, low_w = self._parallel(self.input_project(main), self.input_project(wrist), history, 'low')
        high_m, high_w = self.decoder_1(low_m), self.decoder_1(low_w)
        if 'high' in self.fusion_scales:
            high_m, high_w = self._parallel(high_m, high_w, history, 'high')
        elif self.branch_mask[0]:
            high_m, high_w = self.atten_4(high_m, high_w)
        rgb_m = main_geometry.crop(ycocg_to_rgb(self.decoder_2(high_m)).sigmoid())
        rgb_w = wrist_geometry.crop(ycocg_to_rgb(self.decoder_2(high_w)).sigmoid())
        entry = FusionDecodedFrame(*(x.detach().clone() for x in
            (main, rgb_m, wrist, rgb_w, low_m, low_w, high_m, high_w)), episode_id, frame_index)
        return dict(main_hat=rgb_m, wrist_hat=rgb_w, history_used=len(history),
                    short_history_used=min(len(history), 2), long_history_used=len(history),
                    state=DecoderState(episode_id, frame_index + 1,
                                       (*history, entry)[-8:], self.signature))
