# LIBERO training with the paper's discrete Gain objective

Reference: Cui et al., *Asymmetric Gained Deep Image Compression With Continuous
Rate Adaptation*, sections 3.2 (equation 6) and 4.1:
https://arxiv.org/html/2003.02012v3

This applies the paper's discrete rate-distortion training principle to the
existing dual-view temporal codec. It is not a reproduction of the paper's
entire architecture or training schedule.

Six gain levels bind to the paper MSE multiplier set, indexed here in ascending
order of intended rate/quality:

| Level | beta | beta * 255^2 |
| --- | --- | --- |
| 0 | 0.0003 | 19.5075 |
| 1 | 0.001 | 65.025 |
| 2 | 0.003 | 195.075 |
| 3 | 0.007 | 455.175 |
| 4 | 0.03 | 1950.75 |
| 5 | 0.05 | 3251.25 |

Each optimization step uniformly samples one level. Both views select that
level's gain/inverse-gain and hyper-gain/inverse-hyper-gain tables. The shared
network and selected tables are jointly optimized. Training uses alpha=0;
existing geometric gain interpolation remains available for inference.

Images are RGB float tensors in [0,1]. With equally sized views:

    D = (MSE(main_hat, main) + MSE(wrist_hat, wrist)) / 2
    R = (bits_main + bits_wrist) / (pixels_main + pixels_wrist)
    L = R + beta[level] * 255^2 * D

The 255^2 factor expresses normalized MSE on an 8-bit pixel scale, explicitly
following the referenced GainedVAE reproduction's convention. That repository
is not an official implementation. Entropy bottleneck quantiles retain their
separate auxiliary optimizer. There is no shared fixed distortion weight or
extra BPP threshold penalty in paper mode.

The six-level model starts from random initialization; old eight-level
checkpoints are retained separately and not silently remapped. The initial run
uses one full LIBERO-Spatial epoch: all 500 trajectories and 62,250 frame pairs,
without validation/test splits. Temporal state resets at trajectory boundaries;
frames are chronological with detached reconstructed history.

metrics.csv records level, beta, effective weight, MSE, estimated BPP and PSNR.
per_level_training.json reports cumulative per-level training averages. These
are not fixed-checkpoint comparisons because sampled frames and weights vary.
Different beta values target different tradeoffs, but do not enforce exact BPP,
strict ordering at every step, or a maximum of 0.3. The earlier user preference
for 0.3 must be assessed per level using actual coding after training; the paper
beta values are not calibrated BPP targets for this dataset/model.

Example:

    python tools/train_libero.py --data /path/to/libero_spatial \
      --output /new/run/directory --device cuda:5 --objective paper --epochs 1

When loading its checkpoint instantiate DualViewCodec(levels=6), using config's
levels field. The legacy objective remains explicitly selectable for old runs.
