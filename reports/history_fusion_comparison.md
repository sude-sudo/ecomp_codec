# History modality fusion and temporal age encoding (decoder-v4)

Baseline: codec commit `8d0455e606d9ac329337810de6e46b538023340b` (decoder-v3).
Its decoder is retained verbatim in `tests/fixtures/decoder_v3.py`, with only the
package-relative lightweight_encoder import made absolute. This is a synthetic
CPU acceptance/complexity comparison, not a trained rate-distortion experiment.

## Changed files

- `compress/decoder.py`: shared Concat fusion, relative frame ages, explicit migration.
- `tests/test_history_fusion_age.py`: eight new targeted acceptance tests.
- `tests/test_dual_temporal.py`: explicitly chain v2→v3→v4 migration in the existing regression.
- `tests/fixtures/decoder_v3.py`: fixed pre-change baseline for numerical and complexity comparisons.
- `tools/benchmark_history_fusion.py`: fixed v3/v4 comparison and raw numerical/cache/timing results.
- `tools/benchmark_dual_temporal.py`: keep the historical v2/v3 tool on frozen fixtures.
- `README.md`, `SOURCE_MANIFEST.json`, `THIRD_PARTY_NOTICES.md`: structure, compatibility and provenance.
- This report, `reports/history_fusion_metrics.json`, and `reports/history_fusion_test_results.txt`: measured acceptance evidence.

## Temporal data flow

| Stage | v3 | v4 |
| --- | --- | --- |
| Query | content(current latent) | unchanged |
| History modalities | content(latent) + history_image(pool(RGB)) | Concat of the same two 64-channel features, then shared linear Conv2d(128,64,1) |
| Frame identity | none | shared Embedding(history_size+1,64), newest age=1 |
| Main K/V | main-only history | main-only fused/tagged history |
| Wrist K/V | wrist-only history | wrist-only fused/tagged history |
| Attention | independent main/wrist weights | same names, formulas and independent weights |
| Residual position | after atten_4, before decoder_2 | unchanged, resize/project per view |
| Cache | paired main/wrist latent/RGB | identical public fields, detached copies and size bounds |

The only changed feature computation is in `_history_references(history, view)`:

```python
latent_feature = self.content(latent)  # B,64,H_i,W_i
rgb_feature = self.history_image(F.adaptive_avg_pool2d(rgb, latent.shape[-2:]))
combined = torch.cat((latent_feature, rgb_feature), dim=1)  # B,128,H_i,W_i
feature = self.history_fuse(combined)  # B,64,H_i,W_i
age = torch.tensor(len(history) - index, device=latent.device, dtype=torch.long)
reference = feature + self.temporal_age_embedding(age).reshape(1,64,1,1)
```

The chronological cache remains oldest→newest, so len(history)=2 gives ages
2→1; one reference always gets age=1. This works for history_size=1,2,3 and
larger configured values. Row 0 is unused/padding. No future frames are added.
First frames bypass reference construction, both attentions and projections.

Each view pools against each of its own cached latent sizes. The current query
and each historical reference can have different H/W; attention concatenates
reference tokens without resizing one view to the other. Actual tests exercise
main latent 4×8 / wrist 8×4, then main 8×8 / wrist 4×4, including variable-size
histories. The independent receiver also tests changing non-multiple RGB sizes.

## Initialization and explicit checkpoint migration

Fusion weight is `[I I]` and bias=0; there is no extra GELU. Both modality channel
groups can subsequently learn independent weights. Age weights start at zero.
The existing Attention/projection names and all pre-existing module declarations
are unchanged. State schemas, JCT and attention class ASTs were checked against
the baseline and are identical. Encoder, Gain, HyperGain, hyperprior and codec
packet source files are unchanged.

Old v3 weights must be explicitly passed to `with_temporal_reference_init` and
then strictly loaded. It adds only history_fuse.weight, history_fuse.bias and
temporal_age_embedding.weight, for a complete codec or standalone decoder.
It preserves the original mapping/metadata, never overwrites existing v4 values,
rejects partial new-module weights, and does not hide other missing keys or shape
mismatches. Construct matching Gain levels/history_size and rebuild optimizers.
The README contains a runnable full-codec recipe. Public history fields are
unchanged from v3; on deployment model changes, restart at frame 0 to avoid
mixing histories reconstructed by different weights.

Measured numerical equivalence after explicit migration:

| Output | Maximum absolute error |
| --- | ---: |
| main | 0 |
| wrist | 0 |
| reference_feature | 0 |

This compares independent evolving v3/v4 states for four frames, batch 2,
random-normal decoded latents and changing unequal view resolutions. It is not
a zero-input probe. Fusion references are compared with the original sums.
The unit test independently repeats this sequence and reports zero error.
Equality is measured for this CPU FP32 initialization; it is not a promise of
bit identity on every backend or after fusion/age parameters are trained.

## Complexity

Default history_size=2, 8 Gain levels, FP32:

| Metric | v3 | v4 | Increase |
| --- | ---: | ---: | ---: |
| Total codec parameters | 1,566,161 | 1,574,609 | 8,448 |
| Decoder parameters | 1,063,939 | 1,072,387 | 8,448 |
| Temporal feature/attention parameters | 51,840 | 60,288 | 8,448 |

Temporal parameter count includes content/history_image, both attentions and
projections, and (v4) fusion/age modules. Added parameters: 8,256 for fusion and
192 for the three-row age table (including the unused padding row). In general
the increase is 8,256 + 64×(history_size+1).

MAC convention: nominal convolution, transposed convolution, GDN channel mixing
and factorized attention matmuls. Bias/activation, elementwise adds, pooling,
interpolation and entropy decoding are excluded. Embedding lookup/broadcast has
no MACs under this convention. For each view with N retained references and
latent h×w, fusion adds N×h×w×128×64 MACs. The old modality sum is replaced by the
new age addition; scalar additions are excluded consistently on both sides.

| Main / wrist RGB | History frames | Version | Decoder MACs | Paired decoder ms | Real paired decompress ms |
| --- | ---: | --- | ---: | ---: | ---: |
| 256×256 / 256×256 | 0 | v3 | 4,421,386,240 | 99.510 | 125.547 |
| 256×256 / 256×256 | 2 | v3 | 4,514,217,984 | 97.223 | 127.658 |
| 256×256 / 256×256 | 0 | v4 | 4,421,386,240 | 96.229 | 128.458 |
| 256×256 / 256×256 | 2 | v4 | 4,522,606,592 | 98.787 | 131.598 |
| 256×256 / 128×192 | 0 | v3 | 3,039,703,040 | 63.978 | 91.891 |
| 256×256 / 128×192 | 2 | v3 | 3,103,524,864 | 68.090 | 96.165 |
| 256×256 / 128×192 | 0 | v4 | 3,039,703,040 | 64.134 | 91.330 |
| 256×256 / 128×192 | 2 | v4 | 3,109,292,032 | 68.616 | 96.149 |

With equal 256×256 views and two references, fusion adds 8,388,608 MACs (+0.186%).
With 256×256 / 128×192 views it adds 5,767,168 MACs. First-frame MACs are identical
because no temporal modules run. These are nominal operation counts, not a
hardware instruction measurement.

### Per-view latency and cache

Two PyTorch/OMP/MKL/OpenBLAS CPU threads, eval/no-grad, batch 1, seed 20261104,
PyTorch 2.5.1+cu121, three warmups and ten samples, reported as medians.
No GPU or formal training was used. Paired decoder excludes entropy decoding;
real paired decompress includes entropy decoding, reconstruction and cache.

The following history-feature timings measure each view's complete two-frame
reference construction, so the new Concat/Conv/age cost is visible separately:

| Main / wrist RGB | Version | Main history build ms | Wrist history build ms | Main attention+resize+projection ms | Wrist attention+resize+projection ms |
| --- | --- | ---: | ---: | ---: | ---: |
| 256×256 / 256×256 | v3 | 0.975 | 0.976 | 1.251 | 1.196 |
| 256×256 / 256×256 | v4 | 1.498 | 1.487 | 1.255 | 1.212 |
| 256×256 / 128×192 | v3 | 0.968 | 0.518 | 1.251 | 0.730 |
| 256×256 / 128×192 | v4 | 1.479 | 0.724 | 1.221 | 0.715 |

Each stage is timed separately; these medians are not additive. Both spatial
JCTs jointly decode the views and do not have independent per-view end-to-end
timings. CPU load/noise can make identical first-frame paths look faster/slower;
no first-frame speedup is attributed to this change. Raw samples and RGB-head
breakdowns are preserved in history_fusion_metrics.json.

| Main / wrist RGB | One paired-frame tensor cache, v3 | v4 | Two-frame tensor cache, both |
| --- | ---: | ---: | ---: |
| 256×256 / 256×256 | 1,703,936 bytes | 1,703,936 bytes | 3,407,872 bytes |
| 256×256 / 128×192 | 1,171,456 bytes | 1,171,456 bytes | 2,342,912 bytes |

Counts include the four cached FP32 tensors only, excluding Python/Tensor object
overhead. No fused features or embeddings are cached, so persistent cache size
is unchanged. Concat introduces a transient 128-channel tensor per reference;
this is not part of the stored state.

## Acceptance results

- Frozen v3 snapshot: **45/45 tests passed, 12.353 s**.
- Full current suite: **53/53 tests passed, 14.622 s** (45 existing + 8 new).
- Final targeted fusion/age suite: **8/8 passed, 2.066 s**. Full current and targeted logs are in `history_fusion_test_results.txt`.
- Fusion inputs/Concat/output dimensions, sum initialization, both modality
  sensitivities and nonzero finite gradients for both weight halves pass.
- Ages 1/2 use distinct parameter rows; chronological mapping passes for both
  views and history_size=1/2/3. A fixed cached frame changes its tag as it ages.
- Nonzero age embeddings make content↔age reassignment affect Attention;
  permuting already tagged references preserves the result. Zero embeddings
  restore order invariance. Both used rows receive finite nonzero gradients
  through either view; unused row 0 has zero gradient.
- First-frame skipping, isolated main/wrist histories and attentions, detached
  clone caches, bounds, missing/out-of-order history rejection and Episode resets
  remain covered by the full suite.
- Independent receiver without source encoders reconstructs bit-exact to eval
  forward with nonzero trained-like fusion/age values, different view resolutions
  and Episode changes. Cached tensors also match.
- Fixed analysis weights/CDFs give identical compressed bytes against v3 for
  every synthetic stream/benchmark case. ECH1/ECG1/ECD1 and actual BPP accounting
  are unchanged. Official LDMIC numerical oracles still pass.
- Strict v3 loading fails as expected; explicit migration succeeds for standalone
  Decoder and full six-level Codec, preserves existing parameters and rejects
  partial/incorrect weights.

Reproduce (CPU only):

```sh
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' \
  python -m unittest discover -s tests -p 'test_*.py' -v
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' \
  python tools/benchmark_history_fusion.py --output outputs/history_fusion_v4_comparison
```

Use a fresh output directory for each benchmark. No PSNR, BPP, MS-SSIM or task
success improvement is claimed; these require training and ablation experiments.
