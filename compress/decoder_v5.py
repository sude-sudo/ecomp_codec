"""Decoder-only wide two-scale View x Time x Space reconstruction.

The v4 implementation in decoder.py remains the exact legacy baseline.
Spatial JCT topology follows that implementation; token SDPA, mixed local/global
memory and relative coordinate biases here are new project designs.
"""
from dataclasses import dataclass
from numbers import Integral

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from compressai.layers import GDN, ResidualBlock
from compressai.models.utils import deconv

from .decoder import EfficientAttention, EfficientCrossAttention, SpatioTemporalDecoder
from .lightweight_encoder import ImageGeometry, ycocg_to_rgb


CONFIGS = {
    'baseline': (64, 64, 2, 'legacy'),
    'wide-ldmic': (192, 128, 2, 'efficient'),
    'wide-elic': (320, 192, 2, 'efficient'),
    'long-history': (320, 192, 8, 'efficient'),
    'vts8': (320, 192, 8, 'vts'),
    'vts16': (320, 192, 16, 'vts'),
    'tri_fusion8': (320, 192, 8, 'tri_fusion'),
    'tri_temporal1_t8': (320, 192, 8, 'dualview-time'),
}


@dataclass(frozen=True)
class DecodedFrame:
    main_latent: Tensor
    main_rgb: Tensor
    wrist_latent: Tensor
    wrist_rgb: Tensor
    main_low: Tensor
    wrist_low: Tensor
    main_high: Tensor
    wrist_high: Tensor


@dataclass(frozen=True)
class TemporalDecodedFrame:
    """Decoder-owned references for the no-RGB path, oldest to newest."""
    main_latent: Tensor
    wrist_latent: Tensor
    main_low: Tensor
    wrist_low: Tensor
    main_high: Tensor
    wrist_high: Tensor
    episode_id: int
    frame_index: int


@dataclass(frozen=True)
class DecoderState:
    episode_id: int
    next_frame: int
    history: tuple[DecodedFrame | TemporalDecodedFrame, ...]
    signature: tuple


def coordinates(hw, device, dtype):
    """Pixel centers in normalized own-view coordinates; supports unequal H/W."""
    h, w = hw
    y = (torch.arange(h, device=device, dtype=dtype) + .5) / h
    x = (torch.arange(w, device=device, dtype=dtype) + .5) / w
    yy, xx = torch.meshgrid(y, x, indexing='ij')
    return torch.stack((yy, xx), -1).reshape(-1, 2)


class ScaledJointContextTransfer(nn.Module):
    """Same residual/JCT topology as v4; Q/K/V scale with width, 2 or 4 heads."""
    def __init__(self, channels):
        super().__init__()
        self.rb1 = ResidualBlock(channels, channels)
        self.rb2 = ResidualBlock(channels, channels)
        self.attn = EfficientAttention(channels, channels, channels//8,
                                       4 if channels >= 128 else 2, channels//4)
        self.refine = nn.Sequential(ResidualBlock(channels*2, channels), ResidualBlock(channels, channels))

    def forward(self, main, wrist):
        m = self.rb2(self.rb1(main)); w = self.rb2(self.rb1(wrist))
        return (main + self.refine(torch.cat((self.attn(m, w), m), 1)),
                wrist + self.refine(torch.cat((self.attn(w, m), w), 1)))


class ViewTemporalSpatialAttention(nn.Module):
    """Per-query token SDPA with explicit age/view/relative-position score biases.

    Low scale queries all historical spatial tokens. High scale queries exact
    same-view local tokens PLUS globally pooled memories from BOTH views. Only
    same-view local memory is resized to current geometry; cross-view memory is
    globally searchable, never assumed pixel-aligned. Query chunks bound live
    score/bias allocation; no factorized global statistical matrix is used.
    """
    def __init__(self, channels, history_size=8, *, window_size=None, memory_grid=4, query_chunk=64):
        super().__init__()
        if channels % 8 or channels < 32:
            raise ValueError('VTS channels must be >=32 and divisible by 8')
        if min(history_size, memory_grid, query_chunk) < 1 or (window_size is not None and window_size < 1):
            raise ValueError('VTS history/window/memory/chunk sizes must be positive')
        self.channels = channels
        self.heads = 8 if channels >= 128 else 4
        self.history_size, self.window_size = history_size, window_size
        self.memory_grid, self.query_chunk = memory_grid, query_chunk
        self.latent = nn.Conv2d(64, channels, 1)
        self.rgb = nn.Conv2d(3, channels, 1)
        self.fuse = nn.Sequential(nn.Conv2d(channels*3, channels, 1), nn.GELU())
        self.age = nn.Embedding(history_size+1, channels, padding_idx=0)
        self.view = nn.Embedding(2, channels)
        self.age_bias = nn.Embedding(history_size+1, self.heads, padding_idx=0)
        self.view_bias = nn.Embedding(2, self.heads)
        # A constant score offset cancels in softmax; omit its unidentifiable bias.
        self.relative = nn.Sequential(nn.Linear(2, 32), nn.SiLU(), nn.Linear(32, self.heads, bias=False))
        self.query_norm = nn.LayerNorm(channels)
        self.reference_norm = nn.LayerNorm(channels)
        self.query = nn.Linear(channels, channels)
        self.key = nn.Linear(channels, channels, bias=False)
        self.value = nn.Linear(channels, channels)
        self.output = nn.Linear(channels, channels)
        # Small nonzero gate: identity-like start AND Q/K/V gradients on first use.
        self.gate = nn.Parameter(torch.full((channels,), .01))

    def references(self, history, scale):
        refs = []
        for index, entry in enumerate(history):
            age = len(history)-index  # oldest T, newest 1
            for view, name in enumerate(('main', 'wrist')):
                decoder = getattr(entry, name+'_'+scale)
                size = decoder.shape[-2:]
                latent = F.interpolate(self.latent(getattr(entry, name+'_latent')), size=size, mode='nearest')
                rgb = self.rgb(F.adaptive_avg_pool2d(getattr(entry, name+'_rgb'), size))
                fused = self.fuse(torch.cat((latent, rgb, decoder), 1))
                refs.append((fused, age, view))
        return refs

    def _project_references(self, refs):
        keys, values, positions, biases = [], [], [], []
        for feature, age, view, pos in refs:
            b, c, h, w = feature.shape
            if not 1 <= age <= self.history_size or view not in (0, 1):
                raise ValueError('Historical age/view outside configured range')
            tokens = feature.flatten(2).transpose(1, 2)
            enriched = tokens + self.age.weight[age] + self.view.weight[view]
            tokens = self.reference_norm(enriched)
            keys.append(self.key(tokens)); values.append(self.value(tokens))
            positions.append(coordinates((h,w),feature.device,feature.dtype) if pos is None else pos)
            biases.append((self.age_bias.weight[age]+self.view_bias.weight[view])[None].expand(h*w,-1))
        k = torch.cat(keys,1);v = torch.cat(values,1)
        b,l,c = k.shape
        return (k.reshape(b,l,self.heads,c//self.heads).transpose(1,2),
                v.reshape(b,l,self.heads,c//self.heads).transpose(1,2),
                torch.cat(positions),torch.cat(biases))

    def _attend(self, tokens, pos, query_view, projected):
        k,v,kpos,kbias = projected
        b,l,c = tokens.shape
        q = self.query(self.query_norm(tokens+self.view.weight[query_view]))
        q = q.reshape(b,l,self.heads,c//self.heads).transpose(1,2)
        pieces = []
        for start in range(0,l,self.query_chunk):
            end = min(start+self.query_chunk,l)
            delta = pos[start:end,None,:]-kpos[None,:,:]
            bias = (self.relative(delta)+kbias[None]).permute(2,0,1)[None]
            context = F.scaled_dot_product_attention(q[:,:,start:end],k,v,attn_mask=bias,dropout_p=0.)
            pieces.append(context.transpose(1,2).reshape(b,end-start,c))
        return self.output(torch.cat(pieces,1))*self.gate.tanh()

    def forward(self, current, references, query_view=0):
        if not references:
            return current
        if query_view not in (0,1):
            raise ValueError('Query view must be 0 or 1')
        b,c,h,w = current.shape
        pos = coordinates((h,w),current.device,current.dtype).reshape(h,w,2)
        if self.window_size is None:
            refs = [(f,a,v,None) for f,a,v in references]
            update = self._attend(current.flatten(2).transpose(1,2),pos.reshape(-1,2),query_view,
                                  self._project_references(refs))
            return current+update.transpose(1,2).reshape(b,c,h,w)
        # Project pooled global memories once per view query, reuse across windows.
        global_refs = []
        local_refs = []
        for feature,age,view in references:
            size = tuple(min(n,self.memory_grid) for n in feature.shape[-2:])
            global_refs.append((F.adaptive_avg_pool2d(feature,size),age,view,None))
            if view == query_view:
                local_refs.append((F.interpolate(feature,size=(h,w),mode='bilinear',align_corners=False),age,view))
        global_kv = self._project_references(global_refs)
        output_rows = []
        for y in range(0,h,self.window_size):
            output_columns = []
            for x in range(0,w,self.window_size):
                region = current[:,:,y:y+self.window_size,x:x+self.window_size]
                region_pos = pos[y:y+self.window_size,x:x+self.window_size].reshape(-1,2)
                local = [(f[:,:,y:y+self.window_size,x:x+self.window_size],a,v,region_pos) for f,a,v in local_refs]
                local_kv = self._project_references(local)
                combined = (torch.cat((global_kv[0],local_kv[0]),2),torch.cat((global_kv[1],local_kv[1]),2),
                            torch.cat((global_kv[2],local_kv[2])),torch.cat((global_kv[3],local_kv[3])))
                update = self._attend(region.flatten(2).transpose(1,2),region_pos,query_view,combined)
                output_columns.append(region+update.transpose(1,2).reshape_as(region))
            output_rows.append(torch.cat(output_columns,3))
        return torch.cat(output_rows,2)


class SpatioTemporalDecoderV5(nn.Module):
    def __init__(self, history_size=8, M=320, N=192, temporal='vts', window_size=8, memory_grid=4, query_chunk=64):
        super().__init__()
        if any(isinstance(v,bool) or not isinstance(v,Integral) or v<1 for v in (history_size,M,N)):
            raise ValueError('Positive integer history size and widths are required')
        if M%32 or N%32 or temporal not in ('vts','efficient'):
            raise ValueError('Widths must be multiples of 32; temporal must be vts/efficient')
        self.history_size,self.M,self.N,self.temporal_kind = history_size,M,N,temporal
        self.signature = ('decoder-v5',M,N,history_size,temporal,window_size,memory_grid,query_chunk)
        self.input_project = nn.Conv2d(64,M,1)
        self.atten_3 = ScaledJointContextTransfer(M)
        self.decoder_1 = nn.Sequential(deconv(M,N,5,2),GDN(N,inverse=True),deconv(N,N,5,2),GDN(N,inverse=True))
        self.atten_4 = ScaledJointContextTransfer(N)
        self.decoder_2 = nn.Sequential(deconv(N,N,5,2),GDN(N,inverse=True),deconv(N,3,5,2))
        if temporal == 'vts':
            self.vts_low = ViewTemporalSpatialAttention(M,history_size,query_chunk=query_chunk)
            self.vts_high = ViewTemporalSpatialAttention(N,history_size,window_size=window_size,
                                                        memory_grid=memory_grid,query_chunk=query_chunk)
        else:
            self.content = nn.Sequential(nn.Conv2d(M,M,3,padding=1),nn.GELU())
            self.history_image = nn.Conv2d(3,M,1)
            self.history_fuse = nn.Conv2d(M*2,M,1)
            with torch.no_grad():
                eye = torch.eye(M)
                self.history_fuse.weight.zero_();self.history_fuse.bias.zero_()
                self.history_fuse.weight[:,:M,0,0].copy_(eye);self.history_fuse.weight[:,M:,0,0].copy_(eye)
            self.temporal_age_embedding = nn.Embedding(history_size+1,M,padding_idx=0)
            nn.init.zeros_(self.temporal_age_embedding.weight)
            heads = 4 if M>=128 else 2
            self.temporal = EfficientCrossAttention(M,M,M//8,heads,M//4)
            self.temporal_wrist = EfficientCrossAttention(M,M,M//8,heads,M//4)
            self.temporal_project = nn.Conv2d(M,N,1)
            self.temporal_project_wrist = nn.Conv2d(M,N,1)

    def _history(self, state, episode_id, frame_index, main, wrist):
        for value in (episode_id,frame_index):
            if isinstance(value,bool) or not isinstance(value,Integral) or value<0:
                raise ValueError('Episode/frame must be nonnegative integers')
        if frame_index==0:
            return ()
        if (not isinstance(state,DecoderState) or state.signature!=self.signature or state.episode_id!=episode_id
                or state.next_frame!=frame_index or len(state.history)!=min(frame_index,self.history_size)):
            raise ValueError('Incompatible/missing/out-of-order V5 history; restart from frame 0')
        for entry in state.history:
            if not isinstance(entry,DecodedFrame):
                raise ValueError('Incompatible historical entry')
            for name,current in (('main',main),('wrist',wrist)):
                for suffix,channels in (('latent',64),('rgb',3),('low',self.M),('high',self.N)):
                    t = getattr(entry,name+'_'+suffix)
                    if (t.ndim!=4 or t.shape[:2]!=(current.shape[0],channels) or t.device!=current.device or t.dtype!=current.dtype):
                        raise ValueError('History batch/channels/device/dtype mismatch')
        return state.history

    def _efficient(self, current, feature, history, view):
        refs=[]
        for index,entry in enumerate(history):
            latent=getattr(entry,view+'_latent');rgb=getattr(entry,view+'_rgb')
            fused=self.history_fuse(torch.cat((self.content(self.input_project(latent)),
                                              self.history_image(F.adaptive_avg_pool2d(rgb,latent.shape[-2:]))),1))
            refs.append(fused+self.temporal_age_embedding.weight[len(history)-index][None,:,None,None])
        attention = self.temporal if view=='main' else self.temporal_wrist
        project = self.temporal_project if view=='main' else self.temporal_project_wrist
        temporal = attention(self.content(current),tuple(refs))
        return feature+project(F.interpolate(temporal,size=feature.shape[-2:],mode='nearest'))

    def forward(self, main, wrist, main_geometry, wrist_geometry, state=None, episode_id=0, frame_index=0):
        for latent,geometry in ((main,main_geometry),(wrist,wrist_geometry)):
            if (latent.ndim!=4 or latent.shape[1]!=64 or tuple(s*16 for s in latent.shape[-2:])!=geometry.padded_hw):
                raise ValueError('Expected 64-channel latent matching padded geometry')
        if main.shape[0]!=wrist.shape[0]:raise ValueError('Camera batch sizes differ')
        history=self._history(state,episode_id,frame_index,main,wrist)
        m,w=self.input_project(main),self.input_project(wrist)
        low_m,low_w=self.atten_3(m,w)
        if history and self.temporal_kind=='vts':
            refs=self.vts_low.references(history,'low')
            low_m,low_w=self.vts_low(low_m,refs,0),self.vts_low(low_w,refs,1)
        high_m,high_w=self.atten_4(self.decoder_1(low_m),self.decoder_1(low_w))
        if history:
            if self.temporal_kind=='vts':
                refs=self.vts_high.references(history,'high')
                high_m,high_w=self.vts_high(high_m,refs,0),self.vts_high(high_w,refs,1)
            else:
                high_m=self._efficient(m,high_m,history,'main');high_w=self._efficient(w,high_w,history,'wrist')
        rgb_m=main_geometry.crop(ycocg_to_rgb(self.decoder_2(high_m)).sigmoid())
        rgb_w=wrist_geometry.crop(ycocg_to_rgb(self.decoder_2(high_w)).sigmoid())
        entry=DecodedFrame(*(x.detach().clone() for x in (main,rgb_m,wrist,rgb_w,low_m,low_w,high_m,high_w)))
        return dict(main_hat=rgb_m,wrist_hat=rgb_w,history_used=len(history),
                    state=DecoderState(episode_id,frame_index+1,(*history,entry)[-self.history_size:],self.signature))


class PreviousFrameAttention(nn.Module):
    """Original spatial EfficientCrossAttention against ONLY the own-view t-1."""
    history_size = 1

    def __init__(self, channels):
        super().__init__()
        self.latent = nn.Conv2d(64, channels, 1)
        self.content = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1), nn.GELU())
        self.reference_fusion = nn.Sequential(nn.Conv2d(2 * channels, channels, 1), nn.GELU())
        self.attention = nn.ModuleDict({view: EfficientCrossAttention(
            channels, channels, channels // 8, 4 if channels >= 128 else 2, channels // 4)
            for view in ('main', 'wrist')})

    def previous_reference(self, history, view, scale, size):
        entry = history[-1]  # Deliberately do not iterate over older entries.
        latent = F.interpolate(self.latent(getattr(entry, view + '_latent')), size=size, mode='nearest')
        feature = F.interpolate(getattr(entry, view + '_' + scale), size=size,
                                mode='bilinear', align_corners=False)
        return self.reference_fusion(torch.cat((latent, feature), dim=1))

    def forward(self, current, history, view, scale):
        if not history:
            return torch.zeros_like(current)
        reference = self.previous_reference(history, view, scale, current.shape[-2:])
        return self.attention[view](self.content(current), (reference,))


class DualViewTimeAttention(nn.Module):
    """Channel-wise time attention, separately over BOTH reference views.

    Q=[B,C,1,S], K/V=[B,C,T,S]. Space is the feature dimension;
    softmax is over time, not T*S. Without temporal encoding this operator is
    invariant to synchronous K/V time permutations. It cannot match different
    spatial coordinates. Channel mixing is supplied by the 1x1 projections.
    """
    def __init__(self, channels, history_size=8):
        super().__init__()
        if any(isinstance(v, bool) or not isinstance(v, Integral) or v < 1
               for v in (channels, history_size)) or history_size > 8:
            raise ValueError('Positive channels and history_size in [1,8] required')
        self.channels, self.history_size = channels, history_size
        self.query = nn.Conv2d(channels, channels, 1)
        self.key = nn.Conv2d(channels, channels, 1, bias=False)
        self.value = nn.Conv2d(channels, channels, 1)
        self.view_fusion = nn.Sequential(nn.Conv2d(2 * channels, channels, 1, bias=False), nn.GELU())
        self.gate = nn.Parameter(torch.full((channels,), .01))

    def references(self, history, scale):
        if scale not in ('low', 'high'):
            raise ValueError('Time attention requires a low/high reference scale')
        return {view: tuple(getattr(entry, view + '_' + scale)
                            for entry in history[-self.history_size:])
                for view in ('main', 'wrist')}

    def project_history(self, features, size):
        # T remains an independent dimension throughout QK/AV. Projecting B*T
        # feature maps together is only a channel-convolution batching detail.
        aligned = [F.interpolate(feature, size=size, mode='bilinear', align_corners=False)
                   for feature in features]
        packed = torch.stack(aligned, dim=1)  # B,T,C,H,W
        b, t, c, h, w = packed.shape
        packed = packed.reshape(b * t, c, h, w)
        return tuple(projection(packed).reshape(b, t, c, h * w).permute(0, 2, 1, 3)
                     for projection in (self.key, self.value))

    @staticmethod
    def attend_temporal(query, key, value, *, time_bias=None):
        """Pure QK/AV with an explicit future time-bias ablation boundary."""
        if time_bias is not None:
            raise ValueError('This baseline has no time encoding; time_bias needs an explicit future configuration')
        if (query.ndim != 4 or key.ndim != 4 or value.shape != key.shape
                or query.shape[2] != 1 or not 1 <= key.shape[2] <= 8
                or query.shape[:2] != key.shape[:2] or query.shape[-1] != key.shape[-1]
                or query.shape[-1] < 1):
            raise ValueError('Expected Q[B,C,1,S], K/V[B,C,T,S], 1<=T<=8')
        scores = (query * key).sum(dim=-1) * query.shape[-1] ** -.5  # B,C,T
        weights = scores.softmax(dim=-1)
        return (weights.unsqueeze(-1) * value).sum(dim=2), weights  # B,C,S; B,C,T

    def forward(self, current, references, query_view=0):
        if query_view not in (0, 1):
            raise ValueError('Query view must be main=0 or wrist=1')
        own = 'main' if query_view == 0 else 'wrist'
        other = 'wrist' if query_view == 0 else 'main'
        if not references or not references.get(own):
            if references and references.get(other):
                raise ValueError('Both historical views are required')
            return torch.zeros_like(current)
        if len(references[own]) != len(references.get(other, ())):
            raise ValueError('Historical views must have the same time length')
        b, c, h, w = current.shape
        query = self.query(current).flatten(2).unsqueeze(2)
        outputs = []
        for view in (own, other):
            key, value = self.project_history(references[view][-self.history_size:], (h, w))
            output, _ = self.attend_temporal(query, key, value)
            outputs.append(output.reshape(b, c, h, w))
        return self.view_fusion(torch.cat(outputs, dim=1)) * self.gate.tanh()[None, :, None, None]


class TemporalOneDualViewDecoder(nn.Module):
    """Parallel JCT + Temporal-1 + dual-view VTS-T8, with no RGB cache."""
    def __init__(self, history_size=8, M=320, N=192, *, short_history_size=1,
                 branch_mask=(True, True, True)):
        super().__init__()
        if any(isinstance(v, bool) or not isinstance(v, Integral) or v < 1
               for v in (history_size, short_history_size, M, N)) or M % 32 or N % 32:
            raise ValueError('Positive integer sizes and widths divisible by 32 required')
        if history_size != 8 or short_history_size != 1:
            raise ValueError('tri_temporal1_t8 requires history_size=8 and short_history_size=1')
        if len(branch_mask) != 3 or any(type(flag) is not bool for flag in branch_mask):
            raise ValueError('branch_mask must contain spatial/short/long booleans')
        self.history_size, self.M, self.N = history_size, M, N
        self.temporal_kind = 'dualview-time'
        self.short_history_size = 1
        self.branch_mask = tuple(branch_mask)
        self.signature = ('decoder-temporal1-dualview-time-norgb-v1', M, N, 8, 1,
                          'no-time-encoding', self.branch_mask)
        self.input_project = nn.Conv2d(64, M, 1)
        self.atten_3 = ScaledJointContextTransfer(M)
        self.decoder_1 = nn.Sequential(deconv(M, N, 5, 2), nn.GELU(), deconv(N, N, 5, 2), nn.GELU())
        self.atten_4 = ScaledJointContextTransfer(N)
        self.decoder_2 = nn.Sequential(deconv(N, N, 5, 2), nn.GELU(), deconv(N, 3, 5, 2))
        # Independent opt-in construction: no RGB/age/window/global-pool
        # modules or parameters are constructed for this configuration.
        self.vts_low = DualViewTimeAttention(M)
        self.vts_high = DualViewTimeAttention(N)
        self.short_low = PreviousFrameAttention(M)
        self.short_high = PreviousFrameAttention(N)
        self.fusion_low = self._fusion(M)
        self.fusion_high = self._fusion(N)

    @staticmethod
    def _fusion(channels):
        fusion = nn.ModuleDict({view: nn.Sequential(
            nn.Conv2d(3 * channels, channels, 1, bias=False), nn.GELU())
            for view in ('main', 'wrist')})
        with torch.no_grad():
            for module in fusion.values():
                module[0].weight.mul_(.1)
        return fusion

    def _history(self, state, episode_id, frame_index, main, wrist):
        for value in (episode_id, frame_index):
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
                raise ValueError('Episode/frame must be nonnegative integers')
        if frame_index == 0:
            return ()
        if (not isinstance(state, DecoderState) or state.signature != self.signature
                or state.episode_id != episode_id or state.next_frame != frame_index
                or len(state.history) != min(frame_index, self.history_size)):
            raise ValueError('Incompatible/missing/out-of-order no-RGB history; restart at frame 0')
        for expected, entry in enumerate(state.history, start=frame_index - len(state.history)):
            if (not isinstance(entry, TemporalDecodedFrame) or entry.episode_id != episode_id
                    or entry.frame_index != expected):
                raise ValueError('Reordered, missing or cross-episode no-RGB history')
            for view, current in (('main', main), ('wrist', wrist)):
                for suffix, channels in (('latent', 64), ('low', self.M), ('high', self.N)):
                    tensor = getattr(entry, view + '_' + suffix)
                    if (tensor.ndim != 4 or tensor.shape[:2] != (current.shape[0], channels)
                            or min(tensor.shape[-2:]) < 1 or tensor.device != current.device
                            or tensor.dtype != current.dtype):
                        raise ValueError('History batch/channels/geometry/device/dtype mismatch')
        return state.history

    def branch_deltas(self, main, wrist, history, scale):
        if scale not in ('low', 'high'):
            raise ValueError('Unknown parallel fusion scale')
        spatial = self.atten_3 if scale == 'low' else self.atten_4
        short = self.short_low if scale == 'low' else self.short_high
        long = self.vts_low if scale == 'low' else self.vts_high
        history = history[-8:]
        enhanced = spatial(main, wrist) if self.branch_mask[0] else (main, wrist)
        references = long.references(history, scale) if self.branch_mask[2] else {}
        return {view: (enhanced[index] - current,
                       short(current, history, view, scale) if self.branch_mask[1] else torch.zeros_like(current),
                       long(current, references, index) if self.branch_mask[2] else torch.zeros_like(current))
                for index, (view, current) in enumerate((('main', main), ('wrist', wrist)))}

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
        high_m, high_w = self._parallel(self.decoder_1(low_m), self.decoder_1(low_w), history, 'high')
        outputs = tuple(geometry.crop(ycocg_to_rgb(self.decoder_2(feature)).sigmoid())
                        for feature, geometry in ((high_m, main_geometry), (high_w, wrist_geometry)))
        entry = TemporalDecodedFrame(*(tensor.detach().clone() for tensor in
            (main, wrist, low_m, low_w, high_m, high_w)), episode_id, frame_index)
        return dict(main_hat=outputs[0], wrist_hat=outputs[1], history_used=len(history),
                    short_history_used=min(len(history), 1), long_history_used=len(history),
                    state=DecoderState(episode_id, frame_index + 1,
                                       (*history, entry)[-8:], self.signature))


def build_decoder(config='vts8', history_size=None, **overrides):
    if config not in CONFIGS:raise ValueError(f'Unknown decoder config {config}; choose {tuple(CONFIGS)}')
    M,N,default_history,temporal=CONFIGS[config]
    history_size=default_history if history_size is None else history_size
    if config=='baseline':
        if overrides:raise ValueError('Legacy baseline does not accept V5 overrides')
        return SpatioTemporalDecoder(history_size)
    if config == 'tri_temporal1_t8':
        settings = dict(M=M, N=N, history_size=history_size)
        settings.update(overrides)
        return TemporalOneDualViewDecoder(**settings)
    if config == 'tri_fusion8':
        from .decoder_tri_fusion import ThreeBranchDecoder
        settings = dict(M=M, N=N, history_size=history_size)
        settings.update(overrides)
        return ThreeBranchDecoder(**settings)
    settings=dict(M=M,N=N,history_size=history_size,temporal=temporal)
    settings.update(overrides)
    return SpatioTemporalDecoderV5(**settings)


def migrate_checkpoint(state_dict, target, *, copy_compatible_decoder=True):
    """Explicit full-codec migration, never strict=False; returns weights/report.

    All analysis keys are required, strictly loaded (including dynamic coding
    buffers) and copied unchanged. Only same-name AND same-shape Decoder tensors
    may be copied; others retain explicit target initialization and are listed.
    This mutates target analysis buffer shapes. Rebuild the optimizer/state.
    """
    analysis={k[len('analysis.'):]:v for k,v in state_dict.items() if k.startswith('analysis.')}
    target.analysis.load_state_dict(analysis,strict=True)
    initialized=target.state_dict()
    report=dict(copied_analysis=[],copied_decoder=[],initialized_decoder=[],discarded_source=[])
    for key,value in initialized.items():
        if key.startswith('analysis.'):
            initialized[key]=state_dict[key].detach().clone();report['copied_analysis'].append(key)
        elif copy_compatible_decoder and key in state_dict and value.shape==state_dict[key].shape:
            initialized[key]=state_dict[key].detach().clone();report['copied_decoder'].append(key)
        else:report['initialized_decoder'].append(key)
    copied = set(report['copied_analysis'] + report['copied_decoder'])
    report['discarded_source']=[k for k in state_dict if k not in copied]
    report['restart_history_from_frame_zero']=True
    report['rebuild_optimizer']=True
    report['analysis_semantics']='Final encoder GELU removed; copied tensors do not preserve old latent/bitstream semantics.'
    report['decoder_semantics']='Same-name, same-shape tensors only; new branches/fusion initialized, removed IGDN discarded.'
    if getattr(target.decoder, 'temporal_kind', None) == 'dualview-time':
        report['decoder_semantics'] += ' Temporal-1 / dual-view time-axis attention / no-RGB state is a new architecture, not equivalent to old VTS.'
        report['target_decoder_signature'] = target.decoder.signature
    return initialized,report
