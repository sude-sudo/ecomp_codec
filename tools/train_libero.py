"""Train the dual-view codec directly on LIBERO HDF5 trajectories."""
import argparse
import csv
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import time

import h5py
import torch
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from compress.dual_view_codec import DualViewCodec
from compress.decoder_v5 import CONFIGS, migrate_checkpoint

# Paper MSE multipliers, indexed here from low to high rate/quality.
PAPER_BETAS = (0.0003, 0.001, 0.003, 0.007, 0.03, 0.05)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--steps', type=int, default=None, help='Optional step limit; default is complete epochs')
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--device', default='cpu', help='Explicitly select an available CUDA device for GPU training')
    p.add_argument('--checkpoint-every', type=int, default=1000)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--log-every', type=int, default=10)
    initialization = p.add_mutually_exclusive_group()
    initialization.add_argument('--resume', type=Path, help='Resume a checkpoint with matching architecture and encoder semantics')
    initialization.add_argument('--migrate-from', type=Path, help='Explicit weight migration with report and fresh optimizers')
    p.add_argument('--decoder-config', default='vts8', choices=CONFIGS)
    p.add_argument('--history-size', type=int)
    p.add_argument('--objective', choices=('paper', 'legacy'), default='paper')
    p.add_argument('--distortion-weight', type=float, default=20.)
    p.add_argument('--bpp-target', type=float, default=0.3)
    p.add_argument('--bpp-penalty', type=float, default=0.)
    args = p.parse_args()
    if args.distortion_weight <= 0 or args.bpp_target <= 0 or args.bpp_penalty < 0:
        p.error('weights and target must be positive; penalty must be nonnegative')
    if (args.steps is not None and args.steps < 1) or min(args.epochs, args.threads, args.checkpoint_every, args.log_every) < 1:
        p.error('steps, epochs, threads and checkpoint-every must be positive')
    device = torch.device(args.device)
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is required for this run; no CPU fallback')
        torch.cuda.set_device(device)
        print(f'GPU: {torch.cuda.get_device_name(device)} device={device}', flush=True)
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(args.threads)
    torch.manual_seed(42)
    rng = random.Random(42)
    episodes = []
    for path in sorted(args.data.glob('*.hdf5')):
        with h5py.File(path, 'r') as h:
            for name, demo in h['data'].items():
                length = len(demo['actions'])
                for view in ('agentview_rgb', 'eye_in_hand_rgb'):
                    if demo['obs'][view].shape != (length, 128, 128, 3):
                        raise ValueError(f'Unexpected image shape: {path}/{name}/{view}')
                episodes.append((str(path), name, length))
    if not episodes:
        raise ValueError('No trajectories found')
    total_frames = sum(x[2] for x in episodes)
    args.steps = args.steps if args.steps is not None else args.epochs * total_frames
    paper = args.objective == 'paper'
    levels = len(PAPER_BETAS) if paper else 8
    config = dict(data=str(args.data), steps=args.steps, device=str(device),
                  objective=args.objective, levels=levels,
                  beta_by_level=list(PAPER_BETAS) if paper else None,
                  mse_scale=255**2 if paper else 1,
                  epochs_requested=args.epochs, checkpoint_every=args.checkpoint_every,
                  tensorboard=str(args.output / 'tensorboard'), log_every=args.log_every,
                  threads=args.threads, seed=42, episodes=len(episodes),
                  total_frames=total_frames, batch_size=1,
                  lr=1e-4, aux_lr=1e-3, lambda_main=args.distortion_weight, lambda_wrist=args.distortion_weight,
                  bpp_target=args.bpp_target, bpp_penalty=args.bpp_penalty,
                  loss='estimated_bpp + distortion_weight * (main_mse + wrist_mse) + bpp_penalty * relu(estimated_bpp-bpp_target)^2',
                  split='none; all trajectories available to training',
                  initialization=str(args.resume or args.migrate_from) if (args.resume or args.migrate_from) else 'random; ' + ('decoder-v4' if args.decoder_config=='baseline' else 'decoder-v5'),
                  encoder_output_activation='none',
                  model_version='hypergain-v1/' + ('decoder-tri-fusion-v1' if args.decoder_config=='tri_fusion8'
                                                 else 'decoder-v4' if args.decoder_config=='baseline' else 'decoder-v5'),
                  decoder_config=args.decoder_config,
                  history_size=args.history_size if args.history_size is not None else CONFIGS[args.decoder_config][2],
                  short_history_size=2 if args.decoder_config=='tri_fusion8' else None,
                  source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=Path(__file__).resolve().parents[1],text=True).strip(),
                  sequence='shuffle trajectories, chronological frames, detached causal history')
    if paper:
        config.update(lambda_main=None, lambda_wrist=None, bpp_target=None, bpp_penalty=0.,
                      loss='estimated_bpp + beta[level] * 255**2 * (main_mse + wrist_mse) / 2',
                      reference='https://arxiv.org/html/2003.02012v3; sections 3.2, 4.1',
                      adaptation='Existing dual-view temporal codec; equal-pixel mean distortion and mean rate. Six paper MSE beta values reordered ascending. Discrete alpha=0 training; exponential gain interpolation at inference. Not a full AG-VAE architecture reproduction.')
    (args.output / 'config.json').write_text(json.dumps(config, indent=2))
    print(json.dumps(config), flush=True)
    model = DualViewCodec(levels=levels, decoder_config=args.decoder_config, history_size=args.history_size).to(device).train()
    if args.migrate_from:
        saved = torch.load(args.migrate_from, map_location=device, weights_only=False)
        weights, report = migrate_checkpoint(saved['model'], model)
        model.load_state_dict(weights, strict=True)
        (args.output / 'migration.json').write_text(json.dumps(report, indent=2))
        print('MIGRATED weights; fresh optimizers; history starts at frame 0', flush=True)
        del saved, weights
    quantiles = model.analysis.hyperprior.entropy_bottleneck.quantiles
    params = [v for v in model.parameters() if v is not quantiles]
    optimizer = torch.optim.Adam(params, lr=config['lr'])
    aux_optimizer = torch.optim.Adam([quantiles], lr=config['aux_lr'])
    if args.resume:
        saved = torch.load(args.resume, map_location=device, weights_only=False)
        if (saved.get('config', {}).get('encoder_output_activation') != 'none'
                or saved.get('config', {}).get('decoder_config') != args.decoder_config):
            raise ValueError('Checkpoint architecture/encoder semantics differ; use --migrate-from with fresh optimizers')
        model.load_state_dict(saved['model'], strict=True)
        optimizer.load_state_dict(saved['optimizer'])
        aux_optimizer.load_state_dict(saved['aux_optimizer'])
        print(f'RESUMED checkpoint={args.resume} previous_steps={saved["step"]}; new epoch starts at frame 0', flush=True)
        del saved
    tracked = {name: value.detach().clone() for name, value in model.named_parameters()}
    step = 0
    epoch = 0
    started = time.monotonic()
    level_stats = [dict(count=0, bpp=0., main_mse=0., wrist_mse=0., loss=0.,
                        main_psnr=0., wrist_psnr=0.) for _ in range(levels)]

    def write_level_stats():
        results = []
        for level, stats in enumerate(level_stats):
            count = stats['count']
            row = dict(level=level, count=count, beta=PAPER_BETAS[level] if paper else None)
            if count:
                row.update({key: value/count for key, value in stats.items() if key != 'count'})
            results.append(row)
        (args.output / 'per_level_training.json').write_text(json.dumps(dict(step=step, note='Training-only cumulative statistics; different frames and evolving weights, not a fixed-checkpoint RD evaluation.', levels=results), indent=2))

    def checkpoint():
        payload = dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                       aux_optimizer=aux_optimizer.state_dict(), step=step, epoch=epoch, config=config,
                       torch_rng_state=torch.get_rng_state(), python_rng_state=rng.getstate(),
                       note='Restart causal history at frame 0 when reloading weights.')
        temporary = args.output / 'latest.tmp'
        torch.save(payload, temporary)
        temporary.replace(args.output / 'latest.pt')
        changes = {}
        for prefix in ('analysis.g_a', 'analysis.gain', 'analysis.hyperprior',
                       'decoder.atten_3', 'decoder.atten_4', 'decoder.decoder_1',
                       'decoder.decoder_2', 'decoder.temporal.', 'decoder.temporal_project.',
                       'decoder.temporal_wrist.', 'decoder.temporal_project_wrist.',
                       'decoder.history_fuse.', 'decoder.temporal_age_embedding.',
                       'decoder.short_low.', 'decoder.short_high.', 'decoder.vts_low.',
                       'decoder.vts_high.', 'decoder.fusion_low.', 'decoder.fusion_high.'):
            values = [(value.detach() - tracked[name]).abs().max().item()
                      for name, value in model.named_parameters() if name.startswith(prefix)]
            changes[prefix] = max(values) if values else None
        (args.output / 'parameter_updates.json').write_text(json.dumps(dict(step=step, max_abs_changes=changes), indent=2))
        write_level_stats()
        print(f'CHECKPOINT step={step} path={args.output / "latest.pt"}', flush=True)

    scalar_fields = ('loss', 'bpp', 'main_mse', 'wrist_mse', 'main_psnr', 'wrist_psnr', 'aux_loss', 'grad_norm', 'fusion_grad_norm', 'age_grad_norm', 'history_used')
    with SummaryWriter(log_dir=config['tensorboard']) as tensorboard, \
            (args.output / 'epochs.csv').open('w', buffering=1) as epoch_stream, \
            (args.output / 'metrics.csv').open('w', buffering=1) as stream:
        fields = ['step', 'epoch', 'level', 'beta', 'distortion_weight', 'loss', 'bpp', 'main_mse', 'wrist_mse', 'main_psnr', 'wrist_psnr', 'rate_penalty', 'aux_loss', 'grad_norm', 'fusion_grad_norm', 'age_grad_norm', 'history_used', 'elapsed_s']
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        epoch_writer = csv.DictWriter(epoch_stream, fieldnames=['epoch', 'step', 'frames', 'complete', *scalar_fields])
        epoch_writer.writeheader()
        while step < args.steps:
            epoch += 1
            epoch_frames = 0
            epoch_totals = dict.fromkeys(scalar_fields, 0.)
            rng.shuffle(episodes)
            for episode_id, (path, name, length) in enumerate(episodes):
                state = None
                print(f'TRAJECTORY step={step} task={Path(path).stem} demo={name} frames={length}', flush=True)
                with h5py.File(path, 'r') as h:
                    obs = h['data'][name]['obs']
                    for frame in range(length):
                        images = [torch.from_numpy(obs[key][frame]).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=torch.float32) / 255
                                  for key in ('agentview_rgb', 'eye_in_hand_rgb')]
                        optimizer.zero_grad(set_to_none=True)
                        aux_optimizer.zero_grad(set_to_none=True)
                        level = rng.randrange(levels)
                        rate = (level, 0.)
                        out = model(*images, state=state, episode_id=episode_id,
                                    frame_index=frame, main_rate=rate, wrist_rate=rate)
                        main_mse = F.mse_loss(out['main_hat'], images[0])
                        wrist_mse = F.mse_loss(out['wrist_hat'], images[1])
                        if paper:
                            weight = PAPER_BETAS[level] * 255**2
                            rate_penalty = out['estimated_bpp'].new_zeros(())
                            loss = out['estimated_bpp'] + weight * (main_mse + wrist_mse) / 2
                        else:
                            weight = args.distortion_weight
                            rate_penalty = args.bpp_penalty * F.relu(out['estimated_bpp'] - args.bpp_target).square()
                            loss = out['estimated_bpp'] + weight * (main_mse + wrist_mse) + rate_penalty
                        if not torch.isfinite(loss):
                            raise FloatingPointError(f'Nonfinite loss at step {step}')
                        loss.backward()
                        norm = torch.nn.utils.clip_grad_norm_(params, 5., error_if_nonfinite=True)
                        optimizer.step()
                        aux_optimizer.zero_grad(set_to_none=True)
                        aux = model.aux_loss()
                        if not torch.isfinite(aux):
                            raise FloatingPointError('Nonfinite auxiliary loss')
                        aux.backward()
                        torch.nn.utils.clip_grad_norm_([quantiles], 5., error_if_nonfinite=True)
                        aux_optimizer.step()
                        state = out['state']
                        step += 1
                        if hasattr(model.decoder, 'vts_low'):
                            fusion_grad = model.decoder.vts_low.fuse[0].weight.grad
                            age_grad = model.decoder.vts_low.age.weight.grad
                        else:
                            fusion_grad = model.decoder.history_fuse.weight.grad
                            age_grad = model.decoder.temporal_age_embedding.weight.grad
                        row = dict(step=step, epoch=epoch, level=level, beta=PAPER_BETAS[level] if paper else '', distortion_weight=weight,
                                   loss=loss.item(), bpp=out['estimated_bpp'].item(),
                                   main_mse=main_mse.item(), wrist_mse=wrist_mse.item(), rate_penalty=rate_penalty.item(),
                                   main_psnr=-10*math.log10(max(main_mse.item(), 1e-12)),
                                   wrist_psnr=-10*math.log10(max(wrist_mse.item(), 1e-12)),
                                   aux_loss=aux.item(), grad_norm=float(norm),
                                   fusion_grad_norm=fusion_grad.norm().item() if fusion_grad is not None else 0.,
                                   age_grad_norm=age_grad.norm().item() if age_grad is not None else 0.,
                                   history_used=out['history_used'], elapsed_s=time.monotonic()-started)
                        writer.writerow(row)
                        epoch_frames += 1
                        for key in scalar_fields:
                            epoch_totals[key] += row[key]
                        if step == 1 or step % args.log_every == 0 or step == args.steps:
                            for key in scalar_fields:
                                tensorboard.add_scalar(f'train/{key}', row[key], step)
                            tensorboard.add_scalar('train/epoch', epoch, step)
                            tensorboard.add_scalar('train/lr', optimizer.param_groups[0]['lr'], step)
                        level_stats[level]['count'] += 1
                        for key in level_stats[level]:
                            if key != 'count':
                                level_stats[level][key] += row[key]
                        if step == 1 or step % 10 == 0:
                            print(json.dumps(row), flush=True)
                        if step == 1 or step % args.checkpoint_every == 0 or step == args.steps:
                            checkpoint()
                        if step >= args.steps:
                            break
                if step >= args.steps:
                    break
            summary = dict(epoch=epoch, step=step, frames=epoch_frames, complete=epoch_frames == total_frames,
                           **{key: value / epoch_frames for key, value in epoch_totals.items()})
            epoch_writer.writerow(summary)
            for key in scalar_fields:
                tensorboard.add_scalar(f'epoch/{key}', summary[key], epoch)
            tensorboard.flush()
            checkpoint()
            print('EPOCH ' + json.dumps(summary), flush=True)
    (args.output / 'completed.json').write_text(json.dumps(dict(step=step, epochs=epoch, elapsed_s=time.monotonic()-started)))
    print('TRAINING COMPLETED', flush=True)


if __name__ == '__main__':
    main()
