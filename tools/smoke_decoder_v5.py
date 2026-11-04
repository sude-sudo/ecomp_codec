"""CPU-only continuous sequence training example; never starts a dataset run."""
import argparse
import json
from pathlib import Path
import sys

import torch
from torch.nn import functional as F

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from compress.decoder_v5 import CONFIGS, migrate_checkpoint
from compress.dual_view_codec import DualViewCodec


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--decoder-config',choices=CONFIGS,default='vts8')
    p.add_argument('--history-size',type=int)
    p.add_argument('--frames',type=int,default=3)
    p.add_argument('--size',type=int,default=64)
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--output',type=Path)
    args=p.parse_args()
    if args.frames<2 or args.size<64 or args.size%64:p.error('frames>=2; size a positive multiple of 64')
    torch.set_num_threads(2);torch.set_num_interop_threads(2);torch.manual_seed(5)
    levels=8
    if args.checkpoint:
        saved=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
        levels=saved['model']['analysis.gain.log_gain'].shape[0]
    model=DualViewCodec(levels=levels,decoder_config=args.decoder_config,history_size=args.history_size).cpu()
    if args.checkpoint:
        weights,report=migrate_checkpoint(saved['model'],model)
        model.load_state_dict(weights,strict=True)
        print(json.dumps(dict(migration=report)),flush=True)
        if args.output:
            args.output.mkdir(parents=True,exist_ok=False)
            (args.output/'migration.json').write_text(json.dumps(report,indent=2))
    model.train()
    quantile=model.analysis.hyperprior.entropy_bottleneck.quantiles
    params=[p for p in model.parameters() if p is not quantile]
    optimizer=torch.optim.Adam(params,lr=1e-5)
    auxiliary=torch.optim.Adam([quantile],lr=1e-3)
    state=None
    for frame_index in range(args.frames):
        optimizer.zero_grad(set_to_none=True);auxiliary.zero_grad(set_to_none=True)
        images=[torch.rand(1,3,args.size,args.size) for _ in range(2)]
        level=frame_index%levels
        out=model(*images,state=state,episode_id=0,frame_index=frame_index,
                  main_rate=(level,0.),wrist_rate=(level,0.))
        # Original RGB mean distortion/RD normalization; beta is only an example.
        distortion=sum(F.mse_loss(out[v+'_hat'],im) for v,im in zip(('main','wrist'),images))/2
        loss=out['estimated_bpp']+.05*255**2*distortion
        loss.backward();torch.nn.utils.clip_grad_norm_(params,5.,error_if_nonfinite=True);optimizer.step()
        auxiliary.zero_grad(set_to_none=True);aux=model.aux_loss();aux.backward();auxiliary.step()
        state=out['state']
        print(json.dumps(dict(frame_index=frame_index,history_used=out['history_used'],loss=loss.item(),
                              estimated_bpp=out['estimated_bpp'].item())),flush=True)
    print('CPU two-thread causal training smoke passed; no formal training started.',flush=True)


if __name__=='__main__':main()
