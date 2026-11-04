"""CPU two-thread v3/v4 reference-fusion and age-embedding comparison."""
import argparse
import json
from pathlib import Path
import platform
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT),str(ROOT/'tests')]
from compress.decoder import with_temporal_reference_init
from compress.dual_view_codec import DualViewCodec
from fixtures.decoder_v3 import SpatioTemporalDecoder as DecoderV3
from benchmark_dual_temporal import parameters, macs, measure, view_breakdown


def temporal_parameters(decoder):
    prefixes = ('content.','history_image.','temporal.','temporal_project.',
                'temporal_wrist.','temporal_project_wrist.',
                'history_fuse.','temporal_age_embedding.')
    return sum(p.numel() for name,p in decoder.named_parameters() if name.startswith(prefixes))


def cache_bytes(entry):
    return sum(getattr(entry,k).numel()*getattr(entry,k).element_size()
               for k in ('main_latent','main_rgb','wrist_latent','wrist_rgb'))


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
    old,new = DualViewCodec(decoder_config='baseline').cpu().eval(),DualViewCodec(decoder_config='baseline').cpu().eval()
    old.decoder = DecoderV3().cpu().eval()
    new.decoder.load_state_dict(with_temporal_reference_init(old.decoder.state_dict()),strict=True)
    new.update(force=True)
    old.analysis.load_state_dict(new.analysis.state_dict(),strict=True)
    report = dict(torch=torch.__version__,device='cpu',threads=2,seed=20261104,
                  baseline_commit='8d0455e606d9ac329337810de6e46b538023340b',
                  platform=platform.platform(),samples=args.samples,warmup=args.warmup,
                  optimizer_steps=0,models={},cases=[],equivalence={},
                  mac_convention=macs.__doc__,
                  latency_note='Paired times are uninstrumented medians; per-view times separately instrument attention+resize+projection and decoder_2, excluding shared spatial and history-feature construction. Fusion cost is reflected in paired timing. CPU timings vary with host load.')
    for label,model in (('v3',old),('v4',new)):
        report['models'][label] = dict(codec_params=parameters(model),decoder_params=parameters(model.decoder),
                                       temporal_params=temporal_parameters(model.decoder))
    # Random nontrivial decoded latents, different view resolutions and frame
    # resolutions, independent evolving states: not a zero-input identity probe.
    states = [None,None]
    errors = dict(main=0.,wrist=0.,reference_feature=0.)
    from compress.lightweight_encoder import ImageGeometry
    for i,(hm,wm,hw,ww) in enumerate(((4,8,8,4),(8,8,4,4),(4,4,4,8),(4,8,8,4))):
        main,wrist = torch.randn(2,64,hm,wm),torch.randn(2,64,hw,ww)
        gm,gw = ImageGeometry((hm*16-3,wm*16-5),(hm*16,wm*16)),ImageGeometry((hw*16-1,ww*16-7),(hw*16,ww*16))
        outputs = [model.decoder(main,wrist,gm,gw,state,frame_index=i) for model,state in zip((old,new),states)]
        for tag,key in (('main','main_hat'),('wrist','wrist_hat')):
            errors[tag] = max(errors[tag],(outputs[0][key]-outputs[1][key]).abs().max().item())
        states = [out['state'] for out in outputs]
        for view in ('main','wrist'):
            refs = new.decoder._history_references(states[1].history,view)
            for entry,ref in zip(states[1].history,refs):
                latent,rgb = getattr(entry,view+'_latent'),getattr(entry,view+'_rgb')
                expected = new.decoder.content(latent)+new.decoder.history_image(torch.nn.functional.adaptive_avg_pool2d(rgb,latent.shape[-2:]))
                errors['reference_feature'] = max(errors['reference_feature'],(ref-expected).abs().max().item())
    assert max(errors.values()) < 1e-6,errors
    report['equivalence'] = dict(max_abs_error=errors,frames=4,batch=2,
                                 inputs='random normal decoded latents, changing unequal RGB geometries')
    rates = dict(main_rate=(3,.5),wrist_rate=(6,.2))
    for mh,mw,wh,ww in ((256,256,256,256),(256,256,128,192)):
        main_rgb,wrist_rgb = torch.rand(1,3,mh,mw),torch.rand(1,3,wh,ww)
        main,wrist = new.analysis(main_rgb,*rates['main_rate']),new.analysis(wrist_rgb,*rates['wrist_rate'])
        for label,model in (('v3',old),('v4',new)):
            state = None
            for frame_index in range(2):
                state = model(main_rgb,wrist_rgb,state=state,frame_index=frame_index,**rates)['state']
            for frame_index,prior in ((0,None),(2,state)):
                packet = model.compress(main_rgb,wrist_rgb,frame_index=frame_index,**rates)
                assert packet == new.compress(main_rgb,wrist_rgb,frame_index=frame_index,**rates)
                fn = lambda: model.decoder(main['y_hat'],wrist['y_hat'],main['geometry'],wrist['geometry'],state=prior,frame_index=frame_index)
                decode = lambda: model.decompress(packet,state=prior)
                result = fn()
                reference_timings = {}
                for view in ('main','wrist'):
                    def build_refs():
                        if label == 'v4':
                            return model.decoder._history_references(prior.history,view)
                        return tuple(model.decoder.content(getattr(entry,view+'_latent')) +
                            model.decoder.history_image(torch.nn.functional.adaptive_avg_pool2d(
                                getattr(entry,view+'_rgb'),getattr(entry,view+'_latent').shape[-2:]))
                            for entry in prior.history)
                    reference_timings[view] = measure(build_refs,args.samples,args.warmup) if prior else None
                report['cases'].append(dict(model=label,main_hw=[mh,mw],wrist_hw=[wh,ww],history_frames=frame_index,
                    macs=macs(model.decoder,fn),paired_decoder=measure(fn,args.samples,args.warmup),
                    paired_real_decompress=measure(decode,args.samples,args.warmup),
                    per_view=view_breakdown(model.decoder,fn,args.samples),
                    history_reference_build=reference_timings,
                    one_paired_frame_cache_bytes=cache_bytes(result['state'].history[-1]),
                    total_history_cache_bytes=sum(cache_bytes(e) for e in result['state'].history)))
    report['identical_compressed_bytes'] = True
    (args.output/'metrics.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
