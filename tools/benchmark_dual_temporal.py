"""CPU two-thread decoder-v2/v3 parameter, nominal MAC and latency comparison.

No training, external data, model download or GPU access. Decoder-v2 is the
fixed eca6843 baseline in tests/fixtures/decoder_v2.py.
"""
import argparse
import json
from pathlib import Path
import platform
import statistics
import sys
from time import perf_counter

import torch
from torch import nn
from compressai.layers import GDN

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tests')]
from compress.decoder import EfficientAttention
from compress.dual_view_codec import DualViewCodec
from fixtures.decoder_v2 import SpatioTemporalDecoder as DecoderV2
from fixtures.decoder_v2 import EfficientAttention as AttentionV2
from test_dual_temporal import migrate_v2


def parameters(model):
    return sum(p.numel() for p in model.parameters())


def macs(decoder, fn):
    """Nominal MACs: convolution, transposed convolution, GDN gamma, attention.

    Transposed convolutions count input elements * output channels/groups * K^2
    (nominal padded operations). GDN counts its channel-mixing multiply-adds.
    Excludes bias, softmax, GELU, elementwise products/adds, pooling, interpolation,
    crop, sigmoid and entropy decoding. This is not a hardware instruction count.
    """
    counts = dict(conv=0, deconv=0, gdn=0, attention_matmul=0)
    handles = []

    def count(module, inputs, output):
        x = inputs[0]
        if isinstance(module, nn.ConvTranspose2d):
            counts['deconv'] += x.numel() * (module.out_channels//module.groups) * module.kernel_size[0]*module.kernel_size[1]
        elif isinstance(module, nn.Conv2d):
            counts['conv'] += output.numel() * (module.in_channels//module.groups) * module.kernel_size[0]*module.kernel_size[1]
        elif isinstance(module, GDN):
            counts['gdn'] += x.numel()*x.shape[1]
        else:
            refs = inputs[1] if isinstance(inputs[1], tuple) else (inputs[1],)
            nq = x.shape[-2]*x.shape[-1]
            nr = sum(r.shape[-2]*r.shape[-1] for r in refs)
            counts['attention_matmul'] += x.shape[0]*module.head_count*(module.key_channels//module.head_count)*(module.value_channels//module.head_count)*(nq+nr)

    for m in decoder.modules():
        if isinstance(m,(nn.Conv2d,nn.ConvTranspose2d,GDN,EfficientAttention,AttentionV2)):
            handles.append(m.register_forward_hook(count))
    try:
        fn()
    finally:
        for h in handles:
            h.remove()
    return dict(**counts,total=sum(counts.values()))


def measure(fn, samples, warmup):
    for _ in range(warmup):
        fn()
    timings = []
    for _ in range(samples):
        start = perf_counter()
        fn()
        timings.append((perf_counter()-start)*1000)
    return dict(median_ms=statistics.median(timings),samples_ms=timings)


def view_breakdown(decoder, fn, samples):
    """Per-view temporal branch and final head; spatial JCT is jointly decoded."""
    timings = {k:[] for k in ('main_temporal_ms','wrist_temporal_ms','main_rgb_head_ms','wrist_rgb_head_ms')}
    handles, clocks = [], {}
    index = [0]

    def start_temporal(module, inputs):
        clocks['temporal'] = perf_counter()

    def end_temporal(tag):
        def hook(module, inputs, output):
            timings[tag].append((perf_counter()-clocks['temporal'])*1000)
        return hook

    def start_head(module, inputs):
        clocks['head'] = perf_counter()

    def end_head(module, inputs, output):
        key = 'main_rgb_head_ms' if index[0] % 2 == 0 else 'wrist_rgb_head_ms'
        timings[key].append((perf_counter()-clocks['head'])*1000)
        index[0] += 1

    # Explicitly report attention + upsampling + projection only, by starting
    # at attention; shared content/history extraction is in paired decoder time.
    for attention, projection, tag in ((decoder.temporal,decoder.temporal_project,'main_temporal_ms'),):
        handles.extend((attention.register_forward_pre_hook(start_temporal),projection.register_forward_hook(end_temporal(tag))))
    if hasattr(decoder,'temporal_wrist'):
        handles.extend((decoder.temporal_wrist.register_forward_pre_hook(start_temporal),
                        decoder.temporal_project_wrist.register_forward_hook(end_temporal('wrist_temporal_ms'))))
    handles.extend((decoder.decoder_2.register_forward_pre_hook(start_head),decoder.decoder_2.register_forward_hook(end_head)))
    try:
        for _ in range(samples):
            fn()
    finally:
        for h in handles:
            h.remove()
    return {k:statistics.median(v) if v else 0. for k,v in timings.items()}


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--samples',type=int,default=10)
    parser.add_argument('--warmup',type=int,default=3)
    args = parser.parse_args()
    if args.samples < 1 or args.warmup < 0:
        parser.error('invalid samples/warmup')
    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2)
    torch.manual_seed(20261104)
    old,new = DualViewCodec().cpu().eval(),DualViewCodec().cpu().eval()
    old.decoder = DecoderV2().cpu().eval()
    new.decoder.load_state_dict(migrate_v2(old.decoder.state_dict()),strict=True)
    new.update(force=True)
    old.analysis.load_state_dict(new.analysis.state_dict(),strict=True)
    report = dict(torch=torch.__version__,device='cpu',threads=2,seed=20261104,
                  platform=platform.platform(),baseline_commit='eca6843',warmup=args.warmup,
                  samples=args.samples,optimizer_steps=0,models={},cases=[],
                  mac_convention=macs.__doc__,latency_note='Paired decoder and entropy-decode+decoder latency measured uninstrumented. Per-view hooks separately measure temporal attention+resize+projection and decoder_2 head; spatial JCT and content/history extraction are shared in paired timing. CPU timings depend on server load.')
    for label,model in (('v2',old),('v3',new)):
        report['models'][label] = dict(decoder_params=parameters(model.decoder),codec_params=parameters(model))
    rates = dict(main_rate=(3,.5),wrist_rate=(6,.2))
    for mh,mw,wh,ww in ((256,256,256,256),(256,256,128,192)):
        main_rgb,wrist_rgb = torch.rand(1,3,mh,mw),torch.rand(1,3,wh,ww)
        main,wrist = new.analysis(main_rgb,*rates['main_rate']),new.analysis(wrist_rgb,*rates['wrist_rate'])
        for label,model in (('v2',old),('v3',new)):
            state = None
            for frame in range(2):
                state = model(main_rgb,wrist_rgb,state=state,frame_index=frame,**rates)['state']
            for frame,prior in ((0,None),(2,state)):
                packet = model.compress(main_rgb,wrist_rgb,frame_index=frame,**rates)
                assert packet == new.compress(main_rgb,wrist_rgb,frame_index=frame,**rates)
                fn = lambda: model.decoder(main['y_hat'],wrist['y_hat'],main['geometry'],wrist['geometry'],state=prior,frame_index=frame)
                decode = lambda: model.decompress(packet,state=prior)
                report['cases'].append(dict(model=label,main_hw=[mh,mw],wrist_hw=[wh,ww],
                    history_frames=frame,macs=macs(model.decoder,fn),
                    paired_decoder=measure(fn,args.samples,args.warmup),
                    paired_real_decompress=measure(decode,args.samples,args.warmup),
                    per_view=view_breakdown(model.decoder,fn,args.samples)))
    report['identical_compressed_bytes'] = True
    (args.output/'metrics.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
