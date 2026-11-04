# Third-party notices

- **CompressAI 1.2.8** — https://github.com/InterDigitalInc/CompressAI
  Copyright (c) 2021-2024, InterDigital Communications, Inc.
  The Ballé ScaleHyperprior layer layout/interfaces are adapted in `compress/hyperprior.py`;
  unmodified EntropyBottleneck, GaussianConditional and CDF/coding helpers are runtime dependencies.
  Original BSD-style license, including patent notice: `licenses/CompressAI-LICENSE`.
  No installed CompressAI package files or pretrained weights are vendored.
- **LDMIC** — https://github.com/Xinjie-Q/LDMIC
  Copyright (c) 2023 Xinjie-Q. MIT license: `licenses/LDMIC-LICENSE`.
  `compress/decoder.py` adapts the substantive JointContextTransfer / EfficientAttention
  and decoder_1/decoder_2 code/structure at commit 4cf7318505a001c13d9946581f77e2b4af224246.
  The decoder module retains the MIT notice. `tests/fixtures/ldmic_official.py` contains
  attributed regular-forward class excerpts as a numerical oracle, not a rewritten oracle.
  Changes: M=N=64, independent query/reference spatial shapes, project RGB output
  and separate per-view temporal attention/residuals. This is not a full LDMIC reproduction.
- **GainedVAE** — https://github.com/mmSir/GainedVAE
  Used as a third-party implementation reference; it is not the paper's official code.
  No GainedVAE source is vendored. `compress/gain.py` independently implements
  learned y/z channel gain pairs and the geometric interpolation formula from Cui et al.,
  *Asymmetric Gained Deep Image Compression With Continuous Rate Adaptation*, CVPR 2021.
  Gain and Inverse Gain are independent positive log-parameterized tables; both use
  the same exponential interpolation, avoiding inconsistent reference-code variants.
- **EmbodiedComp** — https://github.com/Jianbo-maker/EmbodiedComp
  Source workspace and original codec integration context. This export includes only
  the newly authored Codec snapshot, not its VLA, robosuite, policy client or other project files.
- PyTorch, torchvision, NumPy, Pillow and Matplotlib are installed as dependencies;
  their source packages and respective licenses remain with their distributions.
