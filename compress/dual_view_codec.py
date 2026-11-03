"""Trainable dual-camera, multi-frame codec with a self-contained frame packet.

Use forward(main, wrist, state=previous_state, episode_id=..., frame_index=i)
for RD training. For actual coding use compress(..., frame_index=i) followed by
decompress(packet, state=previous_state). Only reconstructed history is cached.
Frame 0 resets history. Nonzero frames require the exact preceding state.
Weights/CDFs and history_size are shared model configuration, not frame payload.
"""

from numbers import Integral
import struct

import torch
from torch import Tensor
from compressai.models import CompressionModel

from .decoder import DecoderState, SpatioTemporalDecoder
from .gain import GainedEncoderHyperprior


_FRAME = struct.Struct(">4sQIII")  # magic, episode ID, frame index, main/wrist lengths
_MAGIC = b"ECD1"


class DualViewCodec(CompressionModel):
    def __init__(self, levels: int = 8, history_size: int = 2):
        super().__init__()
        self.analysis = GainedEncoderHyperprior(levels)
        self.decoder = SpatioTemporalDecoder(history_size)

    @staticmethod
    def _check_sequence(episode_id, frame_index):
        for value, limit in ((episode_id, 2**64), (frame_index, 2**32)):
            if isinstance(value, bool) or not isinstance(value, Integral) or not 0 <= value < limit:
                raise ValueError("Episode/frame IDs must be unsigned integers within wire limits")

    def _reconstruct(self, main, wrist, state, episode_id, frame_index):
        return self.decoder(main["y_hat"], wrist["y_hat"], main["geometry"], wrist["geometry"],
                            state=state, episode_id=episode_id, frame_index=frame_index)

    def forward(self, main_rgb: Tensor, wrist_rgb: Tensor,
                state: DecoderState | None = None, episode_id: int = 0, frame_index: int = 0,
                main_rate: tuple[int, float] = (0, 0.), wrist_rate: tuple[int, float] = (0, 0.)):
        self._check_sequence(episode_id, frame_index)
        main = self.analysis(main_rgb, *main_rate)
        wrist = self.analysis(wrist_rgb, *wrist_rate)
        result = self._reconstruct(main, wrist, state, episode_id, frame_index)
        main_pixels = main_rgb.shape[0] * main_rgb.shape[-2] * main_rgb.shape[-1]
        wrist_pixels = wrist_rgb.shape[0] * wrist_rgb.shape[-2] * wrist_rgb.shape[-1]
        result["estimated_bpp"] = (main["estimated_bpp"] * main_pixels
                                   + wrist["estimated_bpp"] * wrist_pixels) / (main_pixels + wrist_pixels)
        result["likelihoods"] = {"main": main["likelihoods"], "wrist": wrist["likelihoods"]}
        return result

    @torch.no_grad()
    def compress(self, main_rgb: Tensor, wrist_rgb: Tensor,
                 episode_id: int = 0, frame_index: int = 0,
                 main_rate: tuple[int, float] = (0, 0.), wrist_rate: tuple[int, float] = (0, 0.)) -> bytes:
        self._check_sequence(episode_id, frame_index)
        if main_rgb.shape[0] != wrist_rgb.shape[0]:
            raise ValueError("Both camera batches must have the same size")
        # Encoding is stateless: no original or reconstructed history crosses to
        # the receiver, and no encoder-side decoder cache is needed.
        main = self.analysis.compress(main_rgb, *main_rate)
        wrist = self.analysis.compress(wrist_rgb, *wrist_rate)
        return _FRAME.pack(_MAGIC, episode_id, frame_index, len(main), len(wrist)) + main + wrist

    @torch.no_grad()
    def decompress(self, packet: bytes, state: DecoderState | None = None):
        if not isinstance(packet, bytes) or len(packet) < _FRAME.size:
            raise ValueError("Expected a complete ECD1 frame packet")
        magic, episode_id, frame_index, nm, nw = _FRAME.unpack_from(packet)
        if magic != _MAGIC or min(nm, nw) < 1 or _FRAME.size + nm + nw != len(packet):
            raise ValueError("Invalid ECD1 frame header or stream lengths")
        main = self.analysis.decompress(packet[_FRAME.size:_FRAME.size + nm])
        wrist = self.analysis.decompress(packet[_FRAME.size + nm:])
        result = self._reconstruct(main, wrist, state, episode_id, frame_index)
        pixels = sum(out["y_hat"].shape[0] * out["geometry"].original_hw[0]
                     * out["geometry"].original_hw[1] for out in (main, wrist))
        result["num_bytes"] = len(packet)
        result["actual_bpp"] = 8 * len(packet) / pixels
        result["rates"] = {"main": main["rate"], "wrist": wrist["rate"]}
        result["view_bpp"] = {"main": main["actual_bpp"], "wrist": wrist["actual_bpp"]}
        result["episode_id"], result["frame_index"] = episode_id, frame_index
        return result
