# HyperGain 实现对照与 CPU 验收

基线：`7b807fb`（y-only Gain）。本轮模型版本：`hypergain-v1`。
参考：[GainedScaleHyperprior](https://github.com/mmSir/GainedVAE/blob/master/compressai/models/gain.py)、
[CompressAI v1.2.8 ScaleHyperprior](https://github.com/InterDigitalInc/CompressAI/blob/v1.2.8/compressai/models/google.py)。
只修改增益与 Hyperprior 接口；未运行正式训练或图片过拟合。

## 1–3. 增益位置与数据流是否一致

指定参考实现先对 y 施加 Gain，再经过 h_a；在 EntropyBottleneck 量化/编码之前
对 z 施加 HyperGain；解码或量化后的 z 先经过 InverseHyperGain，才送入 h_s 预测尺度。
其真实编码也先解码 z_strings，再以该结果预测 y 编码分布。

本实现遵守相同的四组增益位置及真实量化依赖：

```text
y = g_a(YCoCg)
y_g = y * Gy
z = h_a(abs(y_g))
z_g = z * Gz
z_hat_g = EntropyBottleneck(z_g)
z_hat = z_hat_g * IGz
scales = h_s(z_hat)
y_hat_g = GaussianConditional(y_g, scales)
y_hat = y_hat_g * IGy
```

forward 训练时由 CompressAI 添加量化噪声，eval 时由原组件执行量化；真实压缩时
将 z_g 编码为 bytes，然后从这些 bytes 解码 z_hat_g。接收端也从同一 z 码流解码，
再乘同档 IGz，双方预测相同索引。z 的 likelihood 对应 z_g，不是逆增益后的 z_hat。

这不是逐行复制：参考 GainedVAE 的 h_a 直接接收带符号 y_g；我们按任务要求保留
Ballé ScaleHyperprior 的 `h_a(abs(y_g))`。没有引入参考项目的 Decoder 或其他熵模型。

## 4. log-domain 与 abs 参数化

参考代码把自由参数取 abs；我们把自由参数取 exp，几何插值等价于对 log 参数加权后取 exp。
四张独立表都用同一函数，`alpha=0` 对应 s，`alpha=1` 对应 s+1，末档只接受 alpha=0。
不强制 Gain/InverseGain 成倒数。exp 在实数域中光滑且为正；abs 在零点不可导，且零处梯度
可能阻碍离开零值。两者的参数空间与梯度尺度不同，并非训练过程数值等价。
浮点极值仍可能溢出/下溢，本轮只验证正常有限范围，不宣称参数化消除了所有数值风险。

参考仓库 compress 使用几何插值，而 decompress 又覆盖成线性的逆增益。
本实现不沿用该差异；forward、compress、decompress 全部使用相同的指数插值。
现有 y 表及其初始化不变；两张新增表全部零初始化，使 Gz/IGz 初始均为 1。

## 5. 保留的 CompressAI 原生实现

EntropyBottleneck、GaussianConditional、训练噪声/量化、likelihood、CDF 更新、
build_indexes、真实熵编码/解码、辅助 quantiles loss 和 state_dict CDF buffer 恢复均未修改。
继续依赖 CompressAI 1.2.8；h_a/h_s 的层配置、通道、激活也与本轮前相同。

## 6. 适配内容与保持不变的模块

所有四张表均为 `[8,64]`，广播因子为 `[1,64,1,1]`，适配现有 64 通道 y/z。
主/腕共享同一增益模块，各自的四组因子使用各自的 `(level, alpha)`；没有独立 z 档位。
新增 `LatentGain.hyper_factors()` 与可选 Hyperprior 参数，旧 `factors()` 返回值不变。
Encoder、颜色变换、Spatial/Temporal Attention、上采样、DecoderState 和 DualViewCodec
文件均未修改。RGB 仍对齐到 64 倍数，因此 y/z 空间尺寸及合法缓存约定不变。

## 7–8. 参数量与理论计算量

| 项目 | 修改前 | 修改后 |
|---|---:|---:|
| 四组增益模块总参数 | 1,024 | 2,048 |
| 完整 DualViewCodec 参数 | 640,593 | 641,617 |
| LatentHyperprior 内部参数（含概率参数） | 487,616 | 487,616 |
| h_a + h_s 卷积 MACs，单路 256×256 RGB | 35,258,368 | 35,258,368 |

新增参数为两张 8×64 表，总计 **1,024**，FP32 参数本体增加 4 KiB，不含梯度/优化器状态。
表属于 LatentGain，因此 Hyperprior 的内部可学习参数没有增加。

卷积 MAC 按标准稠密核计数：Conv2d 使用输出元素数×输入通道/组×核面积，
ConvTranspose2d 使用输入元素数×输出通道/组×核面积（含边界核位置的名义计算）。
不含 bias、激活、熵概率估计及插值；2 FLOPs/MAC 时卷积为 70,516,736 FLOPs。
256×256 时 y 为 64×16×16，z 为 64×4×4；前向/编码额外有 2×64×4×4 = **2,048** 次
z 路逐元素乘法，解码额外有 1,024 次逆超增益乘法，另有小规模指数插值。
因此卷积 MAC 不变，但不能说整个路径的算术开销完全不变。

## 9. 接口与码流

DualViewCodec 的 main_rate/wrist_rate、forward/compress/decompress 均无需改调用方式。
不带 Gain 的 LatentHyperprior/EncoderHyperprior 默认跳过新增乘法，旧回归测试仍通过。
LatentHyperprior.forward/compress 接收成对可选超增益；decompress 只需可选逆超增益。
只传一个训练/编码增益或传入错误形状会报错，避免遗漏逆变换。

ECH1/ECG1/ECD1 布局、level/alpha 字段及 BPP 定义均未变化。没有把增益向量写入码流。
实际 BPP 仍含所有头部；estimated_bpp 仍由 y/z likelihood 对原像素数归一化。

## 10. checkpoint 与旧码流兼容性

旧 y-only checkpoint 默认严格加载会缺两张表。显式调用
`with_identity_hypergains(state_dict)` 再严格加载可补齐恒等表；辅助函数不修改旧字典、
不覆盖已有两张表，缺一张的新 checkpoint 会报错，其余缺键/形状仍严格检查。
新 checkpoint 正常严格加载；旧代码不能直接严格加载它。恢复训练需重建 optimizer。

只有保留同一旧权重/CDF 且超增益为恒等时，才保证旧码流与旧重建兼容；测试逐字节验证了
此条件。非恒等超增益的新流必须使用匹配的新模型。现有头部不包含模型版本/hash，
无法自动检测新旧或不同训练模型错配，调用端必须同步模型身份。

## CPU 验收

命令：`OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m unittest discover -s tests -p 'test_*.py' -v`。
**32 项通过**：原 24 项及新增 8 项，运行约 9.3 秒；未使用 GPU。

- 8 档恒等初始化与非均匀通道参数；覆盖 (0,0)、(3,.5)、(5,.25)、(6,1)、(7,0)。
- 四组增益、Encoder、h_a/h_s、Decoder 的有限非零梯度与一次优化更新。
- 非恒等 Gz/IGz 的真实 z/y 码流，编码/解码 h_s 输入与索引一致，eval latent 逐元素相同。
- 独立 state_dict 接收端删除 g_a 和 h_a 后仍能解码，不引用发送端 Tensor。
- 双相机不同档位、五帧两 Episode、历史缓存及 Temporal Attention/reset。
- 新旧 checkpoint、旧流恒等兼容、Ry/Rz 分离与含全部包头的 BPP 核对。

随机初始化、非恒等超增益的字节统计（Ry/Rz 单位 bit；输入 batch=2）：

| (level, alpha) | RGB H×W | Ry | Rz | Rz/(Ry+Rz) |
|---|---:|---:|---:|---:|
| (0,0) | 256×256 | 128 | 11,136 | 0.988636 |
| (3,.5) | 255×257 | 93,440 | 13,952 | 0.129917 |
| (5,.25) | 17×31 | 11,712 | 768 | 0.061538 |
| (6,1) | 1×1 | 20,160 | 768 | 0.036697 |
| (7,0) | 64×128 | 40,192 | 1,472 | 0.035330 |

各行均验证 Ry+Rz 等于剔除头部后的实际 payload 位数。输入、尺寸不同，模型未训练，
这些比例仅是功能计数，不用于推断 HyperGain 的压缩优势或 RD 排序。
