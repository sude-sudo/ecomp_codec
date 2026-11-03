# ecomp_codec

用于源码审查的双相机时空可变码率 Codec。代码从 EmbodiedComp 的 `codec`
分支 `850a336` 独立导出，不包含原项目历史、VLA、数据、权重或实验输出。
目前已完成功能与小规模过拟合验证，尚未正式训练，不代表已达到可用的压缩质量。

## 目录与数据流

```text
compress/
  lightweight_encoder.py  RGB↔YCoCg、64 倍数补齐、共享的四级轻量 Encoder
  hyperprior.py           64 通道 Ballé Scale Hyperprior 与真实熵编码
  gain.py                 y/z 各自可学习 Gain/Inverse Gain、8 档与连续指数插值
  decoder.py              空间/时间上下文融合、四级上采样、显式历史状态
  dual_view_codec.py      完整双视角前向、帧码流封装与解码入口
 tests/                   32 项 Codec CPU 单元/集成测试
 tools/smoke_dual_view_codec.py  外部 RGB 图片的冒烟/小规模过拟合脚本
```

主/腕 RGB → YCoCg → 共享 Encoder (`3→32→48→64→64`) → Gain →
Hyperprior 编码 y/z → bytes → 熵解码 → Inverse Gain → 自建 Decoder → 两路 RGB。
超先验路径为 `h_a(abs(y_g)) → HyperGain → EntropyBottleneck → InverseHyperGain → h_s`；
训练、真实压缩和解码使用相同几何插值，每个相机四组增益共用该相机的 `(level, alpha)`。
256×256 输入的 latent 为 `B×64×16×16`。EntropyBottleneck、GaussianConditional
及 CDF 更新直接使用 CompressAI，未修改其底层算法；不构建或调用 Ballé `g_s`。

Decoder 在 H/16 特征层执行 LDMIC 风格的两头 Efficient Cross-Attention。
当前两路已解码基础特征先做双向空间融合，不相互等待重建结果。
主路将当前特征、空间上下文、时间上下文 Concat 后重建；时间参考只来自最近
2 帧已解码主路 latent 与重建 RGB。两路共享上采样网络，输出原尺寸、[0,1] RGB。
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
`state=None` 表示空缓存。history_size 是需同步的模型配置，不存入权重或帧载荷。

Gain 的 `alpha=0` 选当前档，`alpha=1` 选下一档；最后一档只允许 alpha=0。
编码器和解码器使用相同指数插值，码流携带档位/系数。实际 BPP 为完整双路帧字节数
乘 8，除以两路原始像素总数，包含熵码流与所有包头；估计 BPP 仅来自 likelihood。

### 模型版本与旧 checkpoint / 码流

本轮模型为 **hypergain-v1**；`7b807fb` 是仅含 y-Gain 的旧版本。新增
`log_hyper_gain` 和 `log_inverse_hyper_gain`，各为 `[8,64]` 零初始化表，初始因子为 1。
`LatentGain.factors()` 和无 Gain 的 Hyperprior 调用保持兼容，ECH1/ECG1/ECD1 字节结构不变。

默认严格加载旧 checkpoint 会报告缺少两个参数；不会静默填值。显式迁移平铺的 state_dict：

```python
from compress.gain import with_identity_hypergains
old_state = torch.load("old_checkpoint.pt", map_location="cpu", weights_only=True)
model.load_state_dict(with_identity_hypergains(old_state), strict=True)
```

辅助函数不修改原字典，仅在两张超增益表都缺失时补零；不覆盖已有表，缺一张则报错。
支持 LatentGain、GainedEncoderHyperprior、DualViewCodec 的键前缀；其余缺键/形状仍由严格加载检查。
新 checkpoint 可正常严格加载；旧代码不能直接读取新增参数，恢复优化时也应重建 optimizer。

只有 **相同旧权重/CDF + 恒等超增益** 才保证旧码流逐字节及重建兼容。学习到非恒等超增益后，
必须共享相同新 checkpoint；不能把旧流交给任意新模型，或把新流交给旧模型。
现有包头不携带模型版本/hash，无法自动检测这种错配，必须由调用方同步模型身份。
不需要另传 64 通道增益表或独立 z 档位。详细对照与验收数据见 `reports/hypergain_comparison.md`。

### 后续训练接口

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
[LDMIC 官方源码](https://github.com/Xinjie-Q/LDMIC/blob/main/models/ldmic.py)。
第三方依赖/设计参考及许可证见 `THIRD_PARTY_NOTICES.md` 与 `licenses/`。
导出文件校验信息见 `SOURCE_MANIFEST.json`。
