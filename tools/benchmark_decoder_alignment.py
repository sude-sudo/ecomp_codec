"""Small CPU-only decoder-v1/v2 comparison; no training or external data."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import statistics
import sys
from time import perf_counter
from zoneinfo import ZoneInfo

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]
from compress.decoder import EfficientAttention, JointContextTransfer
from compress.dual_view_codec import DualViewCodec
from fixtures.decoder_v1 import SpatioTemporalDecoder as LegacyDecoder
from fixtures.decoder_v2 import SpatioTemporalDecoder as AlignedDecoder
from fixtures.ldmic_official import EfficientAttention as OfficialAttention
from fixtures.ldmic_official import JointContextTransfer as OfficialJCT


def parameters(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


@torch.no_grad()
def jct_macs(module, size):
    """Both view directions: dense Conv MACs + factorized attention matmuls.

    Bias additions, activation and softmax are excluded. This JCT has no GDN.
    No NxN attention matrix is allocated by the measured implementation.
    """
    counts = {"conv": 0, "attention_matmul": 0}
    handles = []

    def convolution(layer, inputs, output):
        counts["conv"] += (output.numel() * layer.in_channels // layer.groups
                           * layer.kernel_size[0] * layer.kernel_size[1])

    def attention(layer, inputs, output):
        q, reference = inputs
        nq, nr = q.shape[-2]*q.shape[-1], reference.shape[-2]*reference.shape[-1]
        counts["attention_matmul"] += (q.shape[0] * layer.head_count * (layer.key_channels//layer.head_count)
                                        * (layer.value_channels//layer.head_count) * (nq+nr))

    for layer in module.modules():
        if isinstance(layer, nn.Conv2d):
            handles.append(layer.register_forward_hook(convolution))
        if isinstance(layer, EfficientAttention):
            handles.append(layer.register_forward_hook(attention))
    x = torch.rand(1,64,size,size)
    module(x,x)
    for handle in handles:
        handle.remove()
    return {**counts, "total":sum(counts.values())}


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S_%f")
    output = args.output or ROOT / "outputs" / f"ldmic_decoder_v2_{stamp}"
    output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2)
    torch.manual_seed(20261103)
    new = DualViewCodec().eval()
    new.decoder = AlignedDecoder().eval()  # Fixed v2 baseline; current production decoder is v3.
    old = DualViewCodec().eval()
    old.decoder = LegacyDecoder().eval()
    new.update(force=True)
    old.analysis.load_state_dict(new.analysis.state_dict(),strict=True)
    main_rgb, wrist_rgb = torch.rand(1,3,256,256), torch.rand(1,3,256,256)
    rates = dict(main_rate=(3,.5),wrist_rate=(6,.2))
    # Compare actual bytes, including headers, at identical analysis weights/CDFs.
    for frame in (0,1):
        assert new.compress(main_rgb,wrist_rgb,frame_index=frame,**rates) == old.compress(main_rgb,wrist_rgb,frame_index=frame,**rates)
    metrics = {"seed":20261103,"torch":torch.__version__,"device":"cpu","threads":2,
               "input":"B=1, 256x256 main + 256x256 wrist", "optimizer_steps":0,
               "samples":3,"warmup":1,"statistic":"median milliseconds",
               "baseline_commit":"7db7e58fd891e8979283a51df6be656420ab5cde",
               "old_decoder_params":parameters(old.decoder),"new_decoder_params":parameters(new.decoder),
               "old_codec_params":parameters(old),"new_codec_params":parameters(new),
               "jct_H16_macs":jct_macs(new.decoder.atten_3,16),
               "jct_H4_macs":jct_macs(new.decoder.atten_4,64),
               "identical_compressed_bytes":True,"latency":[]}
    for name,module,official in (("attention",EfficientAttention(),OfficialAttention(64,64,8,2,16)),
                                  ("jct",JointContextTransfer(),OfficialJCT(64))):
        official.load_state_dict(module.state_dict(),strict=True)
        a,b = torch.randn(2,64,8,8),torch.randn(2,64,8,8)
        actual,expected = module(a,b),official(a,b)
        actual = actual if isinstance(actual,tuple) else (actual,)
        expected = expected if isinstance(expected,tuple) else (expected,)
        metrics[name+"_oracle_max_abs_error"] = max((x-y).abs().max().item() for x,y in zip(actual,expected))

    def timed(fn):
        fn()
        times=[]
        for _ in range(3):
            start=perf_counter()
            fn()
            times.append((perf_counter()-start)*1000)
        return {"median":statistics.median(times),"samples":times}

    main_latent = new.analysis(main_rgb,*rates["main_rate"])
    wrist_latent = new.analysis(wrist_rgb,*rates["wrist_rate"])
    for name,model in (("decoder-v1",old),("decoder-v2",new)):
        prior_state = model(main_rgb,wrist_rgb,**rates)["state"]
        for frame,state in ((0,None),(1,prior_state)):
            packet=model.compress(main_rgb,wrist_rgb,frame_index=frame,**rates)
            forward = lambda: model(main_rgb,wrist_rgb,state=state,frame_index=frame,**rates)
            decoder = lambda: model.decoder(main_latent["y_hat"],wrist_latent["y_hat"],
                main_latent["geometry"],wrist_latent["geometry"],state=state,frame_index=frame)
            decode = lambda: model.decompress(packet,state=state)
            def roundtrip():
                return model.decompress(model.compress(main_rgb,wrist_rgb,frame_index=frame,**rates),state=state)
            times = {key:timed(fn) for key,fn in (("forward_ms",forward),("decoder_ms",decoder),
                     ("decompress_ms",decode),("compress_decompress_ms",roundtrip))}
            metrics["latency"].append({"model":name,"frame":"first" if frame==0 else "history",
                **{k:v["median"] for k,v in times.items()},
                "samples_ms":{k:v["samples"] for k,v in times.items()}})
    (output / "metrics.json").write_text(json.dumps(metrics,indent=2)+'\n')
    print(str(output))
    print(json.dumps(metrics,indent=2))


if __name__ == "__main__":
    main()
