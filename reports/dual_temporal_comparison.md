# Decoder-v2 → v3: independent temporal enhancement for both views

Baseline: `eca6843` on `codec`. The v2 fixture is the baseline decoder verbatim,
except its package-relative import is made absolute for standalone tests.
The implementation adds only wrist temporal attention/projection and paired
history; LDMIC spatial blocks, analysis modules and packet formats are unchanged.
Shared content/history feature extractors preserve the original main path.

## Architecture and compatibility

| Component | v2 | v3 |
| --- | --- | --- |
| Spatial path | JCT@H/16 → decoder_1 → JCT@H/4 → decoder_2 | unchanged |
| Main temporal | temporal + temporal_project after atten_4 | same names, weights and calculation |
| Wrist temporal | none | independent temporal_wrist + temporal_project_wrist after atten_4 |
| History entry | main latent/RGB | synchronized main latent/RGB + wrist latent/RGB |
| History length | latest 2, detached copies | same, for both views |
| Old state reuse | v2 state | incompatible; restart frame_index=0 |
| v2 checkpoint | strictly matches v2 | missing wrist weights; explicit warm-start then strict loading |
| ECH1/ECG1/ECD1 | existing layouts | identical layouts and bytes at fixed analysis weights/CDFs |

An explicit migration copies main temporal weights to independent wrist weights;
all pre-existing parameters remain unchanged and unrelated missing keys still
fail strict loading. See README for the full-model recipe. Rebuild optimizers.
New wrist weights require training before making any reconstruction quality claim.

## Parameters

| Metric | v2 | v3 | Increase |
| --- | ---: | ---: | ---: |
| Decoder parameters | 1,056,611 | 1,063,939 | 7,328 (0.69%) |
| Codec parameters (default 8 Gain levels) | 1,558,833 | 1,566,161 | 7,328 (0.47%) |

The additional 7,328 parameters are 3,168 in wrist attention and 4,160 in its
1×1 projection. The shared content/history extractors add no parameters.

## CPU MACs and latency

CPU only, two PyTorch/OMP/MKL/OpenBLAS threads, seed 20261104, batch 1, float32,
eval/no-grad, PyTorch 2.5.1+cu121. Three warmups and ten samples per case;
latencies are medians, measured sequentially on the same host, subject to server
load. Samples and environment are in `dual_temporal_metrics.json`.

MACs are nominal convolution + transposed convolution + GDN channel mixing +
factorized attention matrix products. ConvTranspose counts input elements ×
output channels/group × kernel area. Bias, activation, softmax, elementwise ops,
pooling, interpolation, crop and entropy decoding are excluded. These counts
are not measured hardware instructions; MAC delta is independent of timing noise.

| Main / wrist RGB | History frames | Version | Decoder MACs | Paired decoder ms | Real paired decompress ms |
| --- | ---: | --- | ---: | ---: | ---: |
| 256×256 / 256×256 | 0 | v2 | 4,421,386,240 | 94.726 | 125.746 |
| 256×256 / 256×256 | 2 | v2 | 4,467,802,112 | 97.139 | 127.779 |
| 256×256 / 256×256 | 0 | v3 | 4,421,386,240 | 94.516 | 125.762 |
| 256×256 / 256×256 | 2 | v3 | 4,514,217,984 | 99.379 | 130.084 |
| 256×256 / 128×192 | 0 | v2 | 3,039,703,040 | 63.642 | 91.501 |
| 256×256 / 128×192 | 2 | v2 | 3,086,118,912 | 66.335 | 94.590 |
| 256×256 / 128×192 | 0 | v3 | 3,039,703,040 | 64.097 | 92.508 |
| 256×256 / 128×192 | 2 | v3 | 3,103,524,864 | 68.128 | 96.678 |

With 256×256 views and two history frames, added MACs are 46,415,872 (+1.04%).
First-frame MACs are identical because both temporal residuals are bypassed.
Real decompress includes both views' entropy decoding, decoder and cache creation;
paired decoder timing excludes entropy decoding. Compression bytes are asserted
identical between v2/v3 in every benchmark case.

### Per-view latency breakdown (two history frames)

The spatial JCT jointly processes both views, so it has no independent per-view
end-to-end latency. The following separately instrumented medians report each
view's attention + nearest resize + projection, and its shared decoder_2 call.
Content/history feature construction, cache copies and shared spatial computation
are included only in paired totals above. Hooked timings are not added to or
subtracted from the uninstrumented paired measurements.

| Main / wrist RGB | Version | Main temporal ms | Wrist temporal ms | Main RGB head ms | Wrist RGB head ms |
| --- | --- | ---: | ---: | ---: | ---: |
| 256×256 / 256×256 | v2 | 1.422 | 0.000 | 18.825 | 19.304 |
| 256×256 / 256×256 | v3 | 1.248 | 1.215 | 18.023 | 19.093 |
| 256×256 / 128×192 | v2 | 1.243 | 0.000 | 15.994 | 5.512 |
| 256×256 / 128×192 | v3 | 1.239 | 0.721 | 17.346 | 6.328 |

## CPU acceptance

Baseline: **40/40 passed, 11.784 s**. v3: **45/45 passed, 12.398 s**.
Both full runs used CPU and two threads; no GPU was accessed and no training
script was started. CPU unit tests include only tiny synthetic backward/optimizer
checks needed to verify gradient/update behavior.

Coverage includes:

- Both temporal attentions and projections have finite nonzero gradients (the
  official attention key bias mathematically cancels under softmax).
- Per-view losses backpropagate only into their own temporal attention parameters;
  weights are independent storage, including after explicit copying migration.
- Changing either cached wrist latent or RGB, in either of two retained frames,
  changes wrist reconstruction while main reconstruction is bit-exact unchanged.
- Main-only history perturbations affect main and leave wrist unchanged.
- Both views' cache tensors are detached clones; caller mutations do not alias.
- Both first-frame branches are skipped; official spatial output oracle passes.
- Ordering, dropped/repeated frames, Episode reset, cache limits and per-view
  batch/device/dtype validation remain active, including history_size=1.
- Unequal resolutions, resolution changes within a sequence and batch resets.
- Independent real-codec receiver without the image encoder reconstructs both
  views bit-exact to eval forward, including history caches and HyperGain rates.
- Main sequence outputs are bit-exact to fixed v2 under migrated weights;
  first-frame outputs match v2 for both views. Old states are explicitly rejected.
- v2 weights fail default strict v3 loading; explicit copying migration succeeds,
  leaves old parameters intact and does not hide unrelated missing weights.
- Official LDMIC attention/JCT numerical oracles remain unchanged.

Reproduce the full suite:

```sh
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' \
  python -m unittest discover -s tests -p 'test_*.py' -v
```

Reproduce the performance comparison (use a fresh output directory):

```sh
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' \
  python tools/benchmark_dual_temporal.py --output outputs/dual_temporal_v3_comparison
```

No PSNR or BPP improvement has been evaluated or claimed for this change.
