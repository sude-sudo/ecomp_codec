"""CPU two-thread per-configuration subprocess profiling, no GPU or training."""
import argparse
from dataclasses import fields,replace
import json
from pathlib import Path
import platform
import resource
import statistics
import subprocess
import sys
from time import perf_counter

import torch
from torch import nn
from compressai.layers import GDN

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from compress.decoder import EfficientAttention
from compress.decoder_v5 import CONFIGS,ViewTemporalSpatialAttention,DualViewTimeAttention
from compress.dual_view_codec import DualViewCodec
from compress.lightweight_encoder import ImageGeometry


def parameter_count(m):return sum(p.numel() for p in m.parameters())


def cache_bytes(state):
    return sum(t.numel()*t.element_size() for e in state.history for t in vars(e).values() if isinstance(t, torch.Tensor))


def count_macs(model,fn):
    counts=dict(convolution=0,transposed_convolution=0,gdn=0,linear=0,factorized_attention=0,sdpa=0,temporal_axis_attention=0)
    handles=[];originals=[]
    def hook(m,inputs,output):
        x=inputs[0]
        if isinstance(m,nn.ConvTranspose2d):counts['transposed_convolution']+=x.numel()*(m.out_channels//m.groups)*m.kernel_size[0]*m.kernel_size[1]
        elif isinstance(m,nn.Conv2d):counts['convolution']+=output.numel()*(m.in_channels//m.groups)*m.kernel_size[0]*m.kernel_size[1]
        elif isinstance(m,GDN):counts['gdn']+=x.numel()*x.shape[1]
        elif isinstance(m,nn.Linear):counts['linear']+=output.numel()*m.in_features
        elif isinstance(m,EfficientAttention):
            refs=inputs[1] if isinstance(inputs[1],tuple) else (inputs[1],)
            counts['factorized_attention']+=x.shape[0]*m.key_channels*m.value_channels//m.head_count*(x.shape[-2]*x.shape[-1]+sum(r.shape[-2]*r.shape[-1] for r in refs))
        elif isinstance(m,DualViewTimeAttention):
            references=inputs[1]
            if references and references.get('main'):
                # QK and AV for two reference views; C independent time groups.
                counts['temporal_axis_attention']+=4*x.numel()*min(len(references['main']),m.history_size)
    for module in model.modules():
        if isinstance(module,(nn.Conv2d,nn.ConvTranspose2d,GDN,nn.Linear,EfficientAttention,DualViewTimeAttention)):
            handles.append(module.register_forward_hook(hook))
        if isinstance(module,ViewTemporalSpatialAttention):
            original=module._attend
            def wrap(tokens,pos,view,projected,original=original):
                counts['sdpa']+=2*tokens.shape[0]*tokens.shape[1]*projected[0].shape[2]*tokens.shape[2]
                return original(tokens,pos,view,projected)
            originals.append((module,original));module._attend=wrap
    try:fn()
    finally:
        for h in handles:h.remove()
        for module,original in originals:module._attend=original
    return dict(**counts,total=sum(counts.values()))


@torch.inference_mode()
def case(config,samples):
    torch.set_num_threads(2);torch.set_num_interop_threads(2);torch.manual_seed(5)
    model=DualViewCodec(decoder_config=config).cpu().eval();decoder=model.decoder
    main,wrist=[torch.randn(1,64,16,16) for _ in range(2)]
    geometry=ImageGeometry((256,256),(256,256))
    first=decoder(main,wrist,geometry,geometry)
    entry=first['state'].history[-1]
    history=tuple(replace(entry,**{f.name:(getattr(entry,f.name).clone() if isinstance(getattr(entry,f.name),torch.Tensor)
                                         else index if f.name=='frame_index' else getattr(entry,f.name))
                                 for f in fields(entry)}) for index in range(decoder.history_size))
    state=replace(first['state'],next_frame=decoder.history_size,history=history)
    fn=lambda:decoder(main,wrist,geometry,geometry,state=state,frame_index=decoder.history_size)
    fn();timings=[]
    # Timings are uninstrumented. MAC hooks run only after latency samples.
    for _ in range(samples):
        start=perf_counter();fn();timings.append((perf_counter()-start)*1000)
    params=dict(codec=parameter_count(model),encoder=parameter_count(model.analysis.g_a),
                analysis=parameter_count(model.analysis),decoder=parameter_count(decoder),
                jct_low=parameter_count(decoder.atten_3),jct_high=parameter_count(decoder.atten_4))
    params['temporal']=sum(parameter_count(getattr(decoder,k)) for k in ('vts_low','vts_high')) if hasattr(decoder,'vts_low') else sum(p.numel() for n,p in decoder.named_parameters() if n.startswith(('temporal','content','history_')))
    if hasattr(decoder, 'short_low'):
        params['short_temporal'] = sum(parameter_count(getattr(decoder, k)) for k in ('short_low', 'short_high'))
        params['fusion'] = sum(parameter_count(getattr(decoder, k)) for k in ('fusion_low', 'fusion_high'))
        params['temporal'] += params['short_temporal']
    return dict(config=config,history=decoder.history_size,parameters=params,
                cache_bytes=cache_bytes(state),median_ms=statistics.median(timings),samples_ms=timings,
                nominal_macs=count_macs(decoder,fn),peak_cpu_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                gpu_peak_memory_bytes=None)


def main():
    p=argparse.ArgumentParser();p.add_argument('--case',choices=CONFIGS);p.add_argument('--samples',type=int,default=3)
    p.add_argument('--output',type=Path);args=p.parse_args()
    if args.case:
        print(json.dumps(case(args.case,args.samples)));return
    if args.output is None:p.error('--output is required')
    args.output.mkdir(parents=True,exist_ok=False)
    result=dict(device='cpu',threads=2,torch=torch.__version__,platform=platform.platform(),
                input='B=1; two 256x256 views; full configured history; random latents; independent cloned cached entries',
                latency='one warmup then uninstrumented median; CPU host load varies; includes both views and cache copy',
                memory='peak CPU subprocess RSS including interpreter/model/cache/temporaries; not GPU memory; GPU unmeasured to avoid active training',
                macs='nominal conv/deconv/GDN gamma/linear, factorized attention and SDPA QK+AV; excludes softmax, norm, elementwise activations, pooling/interpolation and coding',cases=[])
    for name in CONFIGS:
        raw=subprocess.check_output([sys.executable,__file__,'--case',name,'--samples',str(args.samples)],text=True)
        result['cases'].append(json.loads(raw));print(name,'done',flush=True)
        (args.output/'metrics.json').write_text(json.dumps(result,indent=2))


if __name__=='__main__':main()
