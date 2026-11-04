"""Latency-only CPU benchmark: a call jointly reconstructs both camera frames."""
import argparse
from dataclasses import fields, replace
import json
from pathlib import Path
import platform
import statistics
import sys
from time import perf_counter

import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from compress.dual_view_codec import DualViewCodec
from compress.lightweight_encoder import ImageGeometry


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--samples',type=int,default=5)
    parser.add_argument('--warmup',type=int,default=1)
    args=parser.parse_args()
    if args.samples<1 or args.warmup<0:parser.error('invalid sample/warmup count')
    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);torch.set_num_interop_threads(2);torch.manual_seed(5)
    result=dict(device='cpu',threads=2,interop_threads=2,batch=1,dtype='float32',torch=torch.__version__,
                platform=platform.platform(),samples=args.samples,warmup=args.warmup,
                cpu=next((s.split(':',1)[1].strip() for s in Path('/proc/cpuinfo').read_text().splitlines() if s.startswith('model name')),'unknown'),
                scope='one forward jointly reconstructs main+wrist from decoded latents, including history feature fusion and cache cloning; excludes encoder, gain, entropy decode, disk I/O, model loading and history preparation',
                history='full configured window uses independently cloned first-frame cached tensors; valid state and age identities; synthetic/random weights, not a quality evaluation',
                note='same shared server; no GPU; load/frequency/scheduling variation affects wall time; paired FPS means frame-pairs per second',cases=[])
    for size in (128,256):
        for config in ('baseline','vts8','vts16'):
            model=DualViewCodec(decoder_config=config).cpu().eval();decoder=model.decoder
            images=[torch.randn(1,64,size//16,size//16) for _ in range(2)]
            geometry=ImageGeometry((size,size),(size,size))
            first=decoder(*images,geometry,geometry)
            entry=first['state'].history[-1]
            history=tuple(replace(entry,**{f.name:getattr(entry,f.name).clone() for f in fields(entry)})
                          for _ in range(decoder.history_size))
            full=replace(first['state'],next_frame=decoder.history_size,history=history)
            for phase,state,index in (('first_frame',None,0),('full_history',full,decoder.history_size)):
                def run():return decoder(*images,geometry,geometry,state=state,frame_index=index)
                for _ in range(args.warmup):run()
                times=[]
                for _ in range(args.samples):
                    started=perf_counter();run();times.append((perf_counter()-started)*1000)
                ordered=sorted(times);median=statistics.median(times)
                row=dict(config=config,size=size,phase=phase,history_used=index,median_ms=median,
                         min_ms=min(times),max_ms=max(times),p90_ms=ordered[min(len(times)-1,int(.9*len(times)))],
                         paired_fps=1000/median,samples_ms=times)
                result['cases'].append(row)
                (args.output/'results.json').write_text(json.dumps(result,indent=2))
                print(json.dumps(row),flush=True)
            del model,decoder,first,entry,history,full,run,state
    lines=['# CPU Decoder 双路推理时间','',f"CPU：{result['cpu']}；PyTorch {torch.__version__}；float32、batch=1、CPU 双线程。",
           '',f"各组预热 {args.warmup} 次、测量 {args.samples} 次；下表为一次同时重建主视角与腕部视角的调用耗时。",
           '', '| 配置 | 两路各自分辨率 | 历史帧数 | 中位数 ms | 最小～最大 ms | 帧对/秒 |',
           '|---|---|---:|---:|---:|---:|']
    for r in result['cases']:
        lines.append(f"| {r['config']} | {r['size']}×{r['size']} | {r['history_used']} | {r['median_ms']:.2f} | {r['min_ms']:.2f}～{r['max_ms']:.2f} | {r['paired_fps']:.3f} |")
    lines+=['','计时仅包含 Decoder、历史融合和缓存更新，不包含 Encoder、Gain、熵解码、磁盘读取、模型加载或预先填充历史。',
            '', '满历史使用独立 clone 的合成历史，随机参数/latent，不是 PSNR 实验。共享服务器调度和频率波动较大，完整样本保存在 results.json；不能视作独占 CPU 或 GPU 性能。']
    (args.output/'report.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
