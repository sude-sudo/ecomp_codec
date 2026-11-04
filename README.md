# ecomp_codec

用于源码审查的双相机时空可变码率 Codec。代码从 EmbodiedComp 的 `codec`
分支 `850a336` 独立导出，不包含原项目历史、VLA、数据、权重或实验输出。
当前默认是 **decoder-v5 / VTS-8：M=320、N=192、8 帧双视角历史**。
旧 decoder-v4 完整保留为 `decoder_config='baseline'`，可严格加载历史 V4 权重。
V5 已完成 CPU 功能、梯度和开销验证，并已启动 LIBERO 十轮训练；尚无最终评估结果，不声称 PSNR/BPP 提升。
完整结构、六组消融配置、迁移与实测见 [Decoder-v5 报告](reports/decoder_v5_architecture.md)。

## 目录与数据流

```text
compress/
  lightweight_encoder.py  RGB↔YCoCg、64 倍数补齐、共享的四级轻量 Encoder
  hyperprior.py           64 通道 Ballé Scale Hyperprior 与真实熵编码
  gain.py                 y/z 各自可学习 Gain/Inverse Gain、8 档与连续指数插值
  decoder.py              双尺度 LDMIC JCT、双路独立时间残差、Concat 融合与帧龄编码
  decoder_v5.py           可配置宽 Decoder、双尺度 View×Time×Space Attention、显式迁移
  dual_view_codec.py      完整双视角前向、帧码流封装与解码入口
 tests/                   Codec CPU 单元/集成测试及固定官方 oracle/旧基线
 tools/smoke_dual_view_codec.py  外部 RGB 图片的冒烟/小规模过拟合脚本
```

主/腕 RGB → YCoCg → 共享 Encoder (`3→32→48→64→64`) → Gain →
Hyperprior 编码 y/z → bytes → 熵解码 → Inverse Gain → 自建 Decoder → 两路 RGB。
超先验路径为 `h_a(abs(y_g)) → HyperGain → EntropyBottleneck → InverseHyperGain → h_s`；
训练、真实压缩和解码使用相同几何插值，每个相机四组增益共用该相机的 `(level, alpha)`。
256×256 输入的 latent 为 `B×64×16×16`。EntropyBottleneck、GaussianConditional
及 CDF 更新直接使用 CompressAI，未修改其底层算法；不构建或调用 Ballé `g_s`。

以下空间/时序结构说明描述显式 `baseline`，V5 默认结构见上述报告。
Baseline Decoder 空间主干按 LDMIC 固定提交 `4cf7318` 适配为 M=N=64：
`JCT@H/16 → Deconv+IGDN×2 → JCT@H/4 → Deconv+IGDN → Deconv(3)`。
每个 JCT 保留 rb1/rb2、两头 EfficientAttention、两块 refine ResidualBlock 和外层残差，
两视角共享权重；两级 JCT 参数独立。不等分辨率 Attention 保留各自 token 数，不 resize 视角。
每个历史条目同步保存主路/腕部的 latent 与重建 RGB，默认最近 2 帧。
两路分别只读取本视角历史，使用共享的 `content(latent)` 与 `history_image(pool(rgb))`
提取两组 64 通道特征，Concat 成 128 通道后经共享 `history_fuse` 1×1 卷积回到 64 通道，
再加共享帧龄嵌入（最新历史 age=1，次新 age=2），作为 K/V 参考。
Q 仍来自当前 latent 的 `content`，`temporal` / `temporal_wrist` Attention 参数独立。
两路都在 H/16 提取时间上下文，按各自分辨率上采样，再分别通过
`temporal_project` / `temporal_project_wrist` 加到第二级 JCT 后的 H/4 特征，
最后进入共享的 `decoder_2`。首帧两路均严格跳过时间残差。
输出保留项目特有的 `ycocg_to_rgb(logits).sigmoid()` 和 crop，不属于原版 LDMIC。
历史缓存复制并 detach；当前帧与所有网络模块仍可反向传播，默认不跨帧回传历史梯度。

## 安装与 CPU 测试

推荐 Python 3.10。按需创建虚拟环境后：

```sh
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m unittest discover -s tests -p 'test_*.py' -v
```

测试不下载权重/图片，不运行仿真，不需要 GPU。覆盖前后向、Gain 插值、真实码流、
非整倍数尺寸、双视角、多帧缓存、Episode reset 与丢帧/乱序拒绝。

## 主要调用入口

```python
import torch
from compress.dual_view_codec import DualViewCodec

# 这里只用随机输入演示接口，不启动训练。
torch.set_num_threads(2)
model = DualViewCodec().cpu().eval()
model.update(force=True)  # 训练后实际编码前更新 CDF；接收端需要相同权重/CDF。
state = None
for frame_index in range(3):
    main = torch.rand(1, 3, 95, 113)
    wrist = torch.rand(1, 3, 65, 81)
    packet = model.compress(main, wrist, episode_id=1, frame_index=frame_index,
                            main_rate=(3, 0.5), wrist_rate=(7, 0.0))
    out = model.decompress(packet, state=state)
    state = out['state']
    # out['main_hat'], out['wrist_hat'], out['actual_bpp'], out['num_bytes']
```

编码不依赖 Decoder 缓存。接收端仅使用码流、共享模型和合法解码历史。
每个新 Episode 必须发送 `frame_index=0`，该帧自动清空历史；后续帧必须严格有序。
独立流使用独立 state；一个 batch 内需保持 Episode 和样本顺序一致。
`state=None` 表示空缓存。history_size 是需同步的模型配置，不存入帧载荷；
V5 默认 history_size=8，也支持 16；每个尺度有独立帧龄/视角编码。
V4 baseline 的帧龄表形状为 `[history_size+1,64]`。配置及 history_size 必须与权重匹配。
V4 与 V5 的 DecoderState 不兼容，不能沿用历史；切换模型必须从 frame_index=0 重建。

Gain 的 `alpha=0` 选当前档，`alpha=1` 选下一档；最后一档只允许 alpha=0。
编码器和解码器使用相同指数插值，码流携带档位/系数。实际 BPP 为完整双路帧字节数
乘 8，除以两路原始像素总数，包含熵码流与所有包头；估计 BPP 仅来自 likelihood。

### 模型版本与旧 checkpoint / 码流

当前组合版本为 **hypergain-v1 / decoder-v5**，以下旧迁移示例针对显式 V4 baseline。
`7b807fb` 是仅含 y-Gain 的旧版本。此前新增
`log_hyper_gain` 和 `log_inverse_hyper_gain`，各为 `[8,64]` 零初始化表，初始因子为 1。
`LatentGain.factors()` 和无 Gain 的 Hyperprior 调用保持兼容，ECH1/ECG1/ECD1 字节结构不变。

默认严格加载旧 checkpoint 会报告缺少两个参数；不会静默填值。显式迁移平铺的 state_dict：

```python
from compress.gain import with_identity_hypergains
old_state = torch.load("old_checkpoint.pt", map_location="cpu", weights_only=True)
# 对完整旧 Codec 只迁移分析端：旧 decoder-v1 不能严格加载到 decoder-v2/v3/v4。
analysis = {k[len("analysis."):]: v for k, v in old_state.items() if k.startswith("analysis.")}
model.analysis.load_state_dict(with_identity_hypergains(analysis), strict=True)
```

辅助函数不修改原字典，仅在两张超增益表都缺失时补零；不覆盖已有表，缺一张则报错。
支持 LatentGain、GainedEncoderHyperprior、DualViewCodec 的键前缀；其余缺键/形状仍由严格加载检查。
同结构新 checkpoint 可正常严格加载；旧代码不能直接读取新增参数，恢复优化时也应重建 optimizer。

`7db7e58` 的旧 Decoder 结构、权重与缓存不能等价迁移到 decoder-v2/v3/v4；须重新初始化/训练
新 Decoder 并 reset 历史。分析端可按上例严格加载（已有 HyperGain 表不会被覆盖）。
固定相同分析权重/CDF/帧头时，新旧 Decoder 的 compress 字节完全一致，但重建图像不等价。
新模型部署时仍须让发送/接收端共享同一 Decoder 版本与权重。

只有 **相同旧权重/CDF + 恒等超增益** 才保证旧码流逐字节及重建兼容。学习到非恒等超增益后，
必须共享相同新 checkpoint；不能把旧流交给任意新模型，或把新流交给旧模型。
现有包头不携带模型版本/hash，无法自动检测这种错配，必须由调用方同步模型身份。
不需要另传 64 通道增益表或独立 z 档位。详细对照与验收数据见 `reports/hypergain_comparison.md`。
Decoder 逐层结构、官方数值 oracle、必要差异、参数/MAC/时延见 `reports/ldmic_decoder_alignment.md`。
该报告描述历史 decoder-v2；历史 v2→v3 对比见
[`reports/dual_temporal_comparison.md`](reports/dual_temporal_comparison.md)。

### decoder-v3 → decoder-v4：Concat 融合与帧龄编码

基线为 `8d0455e`，保留在 `tests/fixtures/decoder_v3.py`。
空间主干、两路时序 Attention、投影层、查询特征与码流均不变，历史参考从
`content(latent) + history_image(pool(rgb))` 改为：

```python
latent_feature = self.content(latent)                    # B,64,H,W
rgb_feature = self.history_image(pool(rgb, latent_hw))   # B,64,H,W
feature = self.history_fuse(torch.cat((latent_feature, rgb_feature), dim=1))
reference = feature + self.temporal_age_embedding(age).reshape(1, 64, 1, 1)
```

`history_fuse` 是无激活的 `Conv2d(128,64,1)`，权重初始化为 `[I I]`，bias=0。
`temporal_age_embedding` 为 `Embedding(history_size+1,64,padding_idx=0)`，全部零初始化，
row 0 不使用；默认 row 1/2 分别代表 t−1/t−2。保留缓存的旧→新顺序，
统一的 `_history_references` 用 `age=len(history)-index` 赋龄；仅一帧时总是 age=1。
每路分别按自身历史 latent H/W 池化 RGB，首帧跳过所有时序参考构建。
融合与帧龄表共享，两路 Attention 仍独立，时间残差仍位于 atten_4 后、decoder_2 前。

新增 **8,448 参数**：融合卷积 8,256、默认帧龄表 192。
默认 8 个 Gain 档位时总参数为 1,574,609，Decoder 为 1,072,387。
双路 256×256、两帧历史时额外 8,388,608 MACs；缓存不增加特征，仍只保存四个张量。
完整的实测延迟、缓存字节数及数值误差见
[`reports/history_fusion_comparison.md`](reports/history_fusion_comparison.md)。

旧 v3 checkpoint 直接 strict 加载会明确报告三个新增键缺失。
显式迁移只补求和等价卷积和零帧龄表，不覆盖已有权重，不接受部分缺失的新模块：

```python
from compress.decoder import with_temporal_reference_init

checkpoint = torch.load("decoder_v3.pt", map_location="cpu", weights_only=True)
old_state = checkpoint["model"] if "model" in checkpoint else checkpoint
model = DualViewCodec(levels=old_state["analysis.gain.log_gain"].shape[0], history_size=2, decoder_config="baseline")
migrated = with_temporal_reference_init(old_state, history_size=model.decoder.history_size)
model.load_state_dict(migrated, strict=True)
```

该函数也接受独立 Decoder 的 state_dict；必须传入模型实际 history_size。
其他缺键、多键和形状错误仍由 strict 加载拒绝；新 v4 checkpoint 不会被重新初始化。
训练时应重建主/辅助 optimizer，将新增参数纳入主 optimizer。
`DecodedFrame`、`DecoderState` 的公共字段与 v3 完全相同，缓存仍 detach().clone()；
切换部署模型时建议丢弃运行时历史，从 frame_index=0 重建，避免混用不同权重的重建历史。
零帧龄和求和初始化的随机权重多帧对照误差见报告；这不表示帧龄编码训练后的质量收益。

```sh
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' \
  python tools/benchmark_history_fusion.py --output outputs/history_fusion_v4_comparison
```

### decoder-v2 → decoder-v3：历史与权重迁移

**旧 DecoderState 不兼容。** `DecodedFrame(latent, rgb)` 已改为
`DecodedFrame(main_latent, main_rgb, wrist_latent, wrist_rgb)`，每个条目对应同一个
Episode 的同一帧双视角。`DecoderState` 保留 `episode_id / next_frame / history`，
但旧 history 没有腕部数据，不能复用、补零或从主路复制。
升级后两端必须丢弃旧 state，从 **frame_index=0** 重新建立历史。
乱序、丢帧、Episode 切换、batch/device/dtype 检查继续有效；缓存仍 `detach().clone()`。

v3 新增 `decoder.temporal_wrist.*` 和 `decoder.temporal_project_wrist.*`。
旧 v2 checkpoint 默认 `strict=True` 必然报告缺少这些键；不能用 `strict=False`
静默忽略。可选择明确重新初始化整个 Decoder；或者显式将旧主路权重复制为腕路初值，
随后严格加载（复制后参数存储独立，并非绑定权重）：

```python
checkpoint = torch.load("decoder_v2.pt", map_location="cpu", weights_only=True)
old_state = checkpoint["model"] if "model" in checkpoint else checkpoint
model = DualViewCodec(levels=old_state["analysis.gain.log_gain"].shape[0], decoder_config="baseline")
migrated = old_state.copy()
for source, target in (
    ("decoder.temporal.", "decoder.temporal_wrist."),
    ("decoder.temporal_project.", "decoder.temporal_project_wrist."),
):
    if any(key.startswith(target) for key in old_state):
        raise ValueError("Expected a v2 checkpoint without wrist temporal weights")
    for key, value in old_state.items():
        if key.startswith(source):
            migrated[target + key[len(source):]] = value.clone()
# 在当前 v4 上还需要显式补齐模态融合和零帧龄编码；历史 v3 可直接加载 migrated。
from compress.decoder import with_temporal_reference_init
model.load_state_dict(with_temporal_reference_init(migrated), strict=True)
state = None  # 下一帧必须从 frame_index=0 开始
```

此迁移保留所有旧主路/空间参数，腕部新分支仅作显式 warm start；应重建两个 optimizer，
不直接复用旧 optimizer 参数组。未来若训练 v3，需要另行验证重建质量。
固定相同分析端权重、CDF 和帧元数据时，压缩字节与 v2 相同；接收端仍需相同
v3 模型权重与新历史状态，首帧之后的腕部重建与 v2 不等价。
本次未训练，也不声称 PSNR 或 BPP 改善。

CPU 双线程新旧对比（B=1，256×256 双路及 256×256/128×192 异分辨率；
首帧和两帧历史，默认 3 次预热、10 次测量）：

```sh
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' \
  python tools/benchmark_dual_temporal.py --output outputs/dual_temporal_v3_comparison
```

小规模 CPU 结构/时延对照（不训练，默认各测项 1 次预热、3 次测量）：
`OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python tools/benchmark_decoder_alignment.py`。
该脚本固定比较 `tests/fixtures/decoder_v1.py` 和 `decoder_v2.py` 的历史基线；
生产入口始终使用 v4；v2→v3 与 v3→v4 分别使用对应的固定基线对比脚本。
`EfficientCrossAttention` 导入及旧投影属性别名仍可用；参数存储名改为官方风格，不能据此复用旧权重。

### 后续训练接口

LIBERO HDF5 多 epoch 训练（默认新结构 `vts8`、10 个 epoch）：

```sh
python tools/train_libero.py --data /path/to/libero_spatial_hdf5 \
  --output outputs/vts8_epochs10 --device cuda:0 --epochs 10
tensorboard --logdir outputs/vts8_epochs10/tensorboard --port 6006
```

浏览器打开 `http://localhost:6006`，`train/*` 显示 step 曲线，
`epoch/*` 显示每轮按帧平均的 Loss、BPP、双路 PSNR、辅助损失及梯度指标。
每轮遍历所有轨迹，轨迹顺序随机、内部帧顺序不变，轨迹之间清空历史。
输出还包括 `metrics.csv`、`epochs.csv` 和每轮结束更新的 `latest.pt`。
`--steps` 可覆盖总步数用于短测；截断轮在 `epochs.csv` 中标记 `complete=False`。
`--resume` 严格加载同结构、同档位权重与优化器，在新输出目录从新一轮开始，
步数和曲线重新计数；默认六档 paper objective 不兼容旧八档 checkpoint。
需要旧 V4 时显式指定 `--decoder-config baseline`。
输入目录直接包含 HDF5 文件，每个 demo 的两路 RGB 均为 128×128；
Parquet 数据须先转换成该格式。训练统计来自训练集，不是验证指标。

`model.train()` 后调用与上述相同参数的 `model(main, wrist, state=..., ...)`，
返回两路重建、`likelihoods`、`estimated_bpp` 与新的 state。可构造
`estimated_bpp + lambda_main*MSE(main_hat, main) + lambda_wrist*MSE(wrist_hat, wrist)`。
优化器应包含所有参数；熵瓶颈的 `analysis.hyperprior.entropy_bottleneck.quantiles`
使用独立 optimizer 与 `model.aux_loss()`，其余参数用主 optimizer。
`model.update(force=True)` 用于训练后生成编码 CDF，不放在梯度前向中。
完整反向和辅助优化的可执行示例见 `tests/test_dual_view_codec.py`。

### 可选图片冒烟测试

自行提供至少三组按文件名排序的 RGB 图片：
`*_agentview_image.png` 与对应的 `*_robot0_eye_in_hand_image.png`。
这些是稀疏有序截图的功能测试，并不要求或声称是真实连续训练序列。

```sh
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python tools/smoke_dual_view_codec.py \
  --source-dir /path/to/paired_rgb --steps 120 --threads 2
```

仅该命令会运行少量图片过拟合。它在全新 `outputs/codec_smoke_*` 下保存输入、
真实码流、重建/对比图、PSNR/BPP/耗时、Loss 曲线、参数更新记录与测试 checkpoint，
不会覆盖已有目录。这些运行产物不会提交 Git。

## 引用与许可

参考 [CompressAI](https://github.com/InterDigitalInc/CompressAI) 的 Ballé 2018
ScaleHyperprior、[GainedVAE](https://github.com/mmSir/GainedVAE) 与
[Cui et al., CVPR 2021, Eq. (8)](https://openaccess.thecvf.com/content/CVPR2021/html/Cui_Asymmetric_Gained_Deep_Image_Compression_With_Continuous_Rate_Adaptation_CVPR_2021_paper.html)、
[LDMIC 固定官方源码](https://github.com/Xinjie-Q/LDMIC/blob/4cf7318505a001c13d9946581f77e2b4af224246/models/ldmic.py)。
第三方依赖/设计参考及许可证见 `THIRD_PARTY_NOTICES.md` 与 `licenses/`。
导出文件校验信息见 `SOURCE_MANIFEST.json`。
