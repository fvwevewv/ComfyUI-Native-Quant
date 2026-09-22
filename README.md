# ComfyUI FP8 Weight Quantization & Native Mixed-Precision Loaders

ComfyUI 自定义节点包，用于对 UNet / DiT 中的 `Linear` 权重进行低精度量化，并通过 ComfyUI 原生 `comfy.quant_ops`、`comfy_kitchen` 和 MixedPrecision 路径执行推理。

已重点检查并适配的模型家族：`Flux 1`、`Flux 2`、`Qwen Image`、`Qwen Edit`、`Qwen 2.1`、`Krea`、`Krea 2`、`Lumina 2`、`Z-Image`、`Klein`，以及 `Noob1.1E` UNet、Anima。

当前版本只保留两个正式 loader：

- `FP8 Checkpoint Loader`
- `INT8 Checkpoint Loader`

节点不会量化文本编码器或 VAE。TE/VAE 选项只决定使用 checkpoint 内置组件，还是加载指定的外部文件。

## 特性概览

- ComfyUI 原生 MixedPrecision state-dict 路径。
- FP8 E4M3、NVFP4、MXFP8、BNB NF4。
- INT8 W8A8、INT8 ConvRot W8A8、ConvRot W4A4 INT4/INT8，以及 W4A8 `Asym_W4A8_Int8` 路径。
- INT8 loader 支持 Comfy Kitchen CUDA / Triton backend 选择。
- 敏感层保护、Anima 边界层保护和标准 UNet `channels_last` 优化。
- `torch.compile` 兼容模式。

## 节点

### 1. FP8 Checkpoint Loader

节点名：`Fp8CheckpointLoader`

支持的 `dtype`：

| 模式 | 路径 | 说明 |
|---|---|---|
| `float8_e4m3fn` | Comfy 原生 MixedPrecision | 标准 FP8 E4M3；RTX 40 系及以上支持硬件加速 |
| `nvfp4` | Comfy 原生 MixedPrecision | NVIDIA NVFP4 4-bit 浮点格式，使用 16-value micro-block 的 E4M3 scaling；RTX 50 系及以上支持原生硬件路径 |
| `mxfp8` | Comfy 原生 MixedPrecision | MXFP8 microscaling FP8 格式 |
| `bnb-nf4` | 加载后 BNB NF4 patch | 需要 bitsandbytes；主要用于显存压缩 |

### 2. INT8 Checkpoint Loader

节点名：`Int8CheckpointLoader`

支持的 `quant_mode`：

| 模式 | Backend | 说明 |
|---|---|---|
| `int8_tensorwise` | CUDA / Triton | INT8 Tensor-wise W8A8 |
| `int8_tensorwise_convrot` | CUDA / Triton | INT8 ConvRot W8A8；满足 256 对齐时启用 ConvRot |
| `convrot_w4a4_int4` | CUDA | ConvRot W4A4 INT4 MMA |
| `convrot_w4a4_int8` / `asym_w4a8_int8` | CUDA / Triton* | ConvRot W4A4 INT8 MMA；`asym_w4a8_int8` 使用原生 W4A8 MixedPrecision |

`*` W4A4 只使用 CUDA；`asym_w4a8_int8` 支持 CUDA / Triton。`asym_w4a8_int8` 使用 ComfyUI 原生 `AsymW4A8Int8Layout`。

## TE / VAE 选择

两个 loader 都有 `text_encoder` 和 `vae` 两个选项：

1. `无需加载（AIO）`：使用 checkpoint 内置的 TE/VAE。
2. 其他选项：从 ComfyUI 的 `text_encoders` 或 `vae` 文件夹选择外部文件。

这里的 `AIO` 是 loader 的组件选择，不是模型名称。

外部组件只负责加载，不进入本节点的 UNet/DiT Linear 权重量化范围。

## 安装

进入 ComfyUI 的 `custom_nodes` 目录：

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/fvwevewv/comfyui-fp8-weight-quantize.git
```

重启 ComfyUI。

### 依赖

- ComfyUI 0.35.0 或更新版本；具体格式能力取决于当前 `comfy.quant_ops` 和 `comfy_kitchen`。
- `comfy_kitchen`：FP8、FP4、INT8、ConvRot 和 W4A8 原生路径需要。
- `bitsandbytes`：仅使用 `bnb-nf4` 时需要。
- `triton`：仅使用 INT8 loader 的 Triton backend 时需要。

## 量化和路由说明

### FP8 / NVFP4 / MXFP8

节点在 checkpoint 的 `state_dict` 阶段完成量化，并写入 ComfyUI 当前使用的逐层 `comfy_quant` 和 scale 元数据。模型构建后由 ComfyUI MixedPrecisionOps 接管。

### INT8 W8A8 / ConvRot

INT8 权重会写入 ComfyUI 原生 `int8_tensorwise` 元数据。ConvRot 层额外写入：

```json
{
  "format": "int8_tensorwise",
  "convrot": true,
  "convrot_groupsize": 256
}
```

如果某个 Linear 的输入维度不满足 256 对齐，`int8_tensorwise_convrot` 会对该层退回普通 INT8，而不会强行使用错误的旋转尺寸。

### ConvRot W4A4

W4A4 使用 ComfyUI 原生 `convrot_w4a4` layout：

- INT4 使用默认 `linear_dtype=int4`。
- INT8 使用 `linear_dtype=int8`。
- packed weight 会保存原始形状信息，供 ComfyUI 模型识别阶段恢复。
- Anima DiT 的 W4A4 INT4/INT8 默认主动禁用，因为实测会产生严重雪花、糊化或结构失真。

### `Asym_W4A8_Int8`

W4A8 使用 ComfyUI 原生 MixedPrecision state-dict 路径：

```text
量化权重
  -> asym_w4a8_int8 comfy_quant
  -> packed INT4 weight
  -> weight_s_rel
  -> weight_s_channel
  -> AsymW4A8Int8Layout
  -> Comfy Kitchen CUDA / Triton
```

对于 ConvRot group 64/16，当前 CUDA 旋转 fused kernel 有 group 256 限制，因此量化准备阶段使用 Comfy Kitchen eager；推理仍然保留原生 MixedPrecision Tensor。group 256 使用 CUDA，Triton 路径使用 Triton。

## 敏感层保护

默认 `skip_sensitive=True`。节点会跳过或保护以下类型的层：

- AdaLN、modulation、`img_mod`、`txt_mod`。
- norm、embed、输入投影和输出投影。
- time、vector、guidance 等条件嵌入。
- Krea、Lumina、Z-Image 等架构中的已知敏感组件。
- Anima 的首尾 block、早期 MLP 和非目标 attention projection。

保护规则的目标是避免用量化速度换取不可接受的结构性质量损失。

## 性能测试

以下测试在 RTX 4070 Laptop GPU 上完成：

| 项目 | 值 |
|---|---|
| ComfyUI | 0.37.0 |
| PyTorch | 2.13.0+cu130 |
| comfy-kitchen | 0.2.35 |
| Triton | 3.7.1 |
| Attention | SageAttention |
| 分辨率 | 1024×1536 |
| Steps | 30 |
| CFG | 5.0 |
| Sampler | `dpmpp_3m_sde_gpu` |
| Scheduler | `sgm_uniform` |
| Seed | 424242 |

PSNR/SSIM 相对于同模型、同参数的 baseline PNG 计算。平均每步是 30 steps 中的 step 间隔平均值。

### Anima / DiT

Baseline：1455.36 ms/step。

| 模式 | Backend | 平均每步 | 相对速度 | PSNR | SSIM |
|---|---|---:|---:|---:|---:|
| FP8 E4M3 | CUDA | 1091.97 ms | +24.97% | 19.37 dB | 0.8170 |
| NVFP4 | CUDA | 1475.33 ms | -1.37% | 13.15 dB | 0.6645 |
| MXFP8 | CUDA | 1592.97 ms | -9.46% | 20.56 dB | 0.8457 |
| BNB NF4 | CUDA | 1473.99 ms | -1.28% | 15.45 dB | 0.7152 |
| INT8 Tensor-wise | CUDA | 1112.35 ms | +23.57% | 22.03 dB | 0.8604 |
| INT8 Tensor-wise | Triton | 1181.75 ms | +18.80% | 22.18 dB | 0.8474 |
| INT8 ConvRot | CUDA | 1034.54 ms | +28.91% | 23.02 dB | 0.8796 |
| INT8 ConvRot | Triton | 1084.51 ms | +25.48% | 23.90 dB | 0.9007 |
| `asym_w4a8_int8` | CUDA | 1017.59 ms | +30.08% | 12.01 dB | 0.6408 |
| `asym_w4a8_int8` | Triton | 1113.37 ms | +23.50% | 11.91 dB | 0.6445 |

Anima 上速度/质量平衡最好的路径是 INT8 ConvRot。W4A8 CUDA 速度最高，但图像质量明显低于普通 INT8 和 ConvRot。

### Noob1.1E / UNet

Baseline：684.65 ms/step。

| 模式 | Backend | 平均每步 | 相对速度 | PSNR | SSIM |
|---|---|---:|---:|---:|---:|
| FP8 E4M3 | CUDA | 524.81 ms | +23.35% | 11.27 dB | 0.4336 |
| NVFP4 | CUDA | 681.03 ms | +0.53% | 9.92 dB | 0.3730 |
| MXFP8 | CUDA | 798.43 ms | -16.62% | 11.29 dB | 0.4370 |
| BNB NF4 | CUDA | 1121.83 ms | -63.85% | 9.43 dB | 0.3719 |
| INT8 Tensor-wise | CUDA | 454.42 ms | +33.63% | 11.83 dB | 0.4697 |
| INT8 Tensor-wise | Triton | 460.05 ms | +32.81% | 11.42 dB | 0.4459 |
| INT8 ConvRot | CUDA | 465.30 ms | +32.04% | 11.44 dB | 0.4482 |
| INT8 ConvRot | Triton | 515.33 ms | +24.73% | 12.39 dB | 0.5000 |
| ConvRot W4A4 INT4 | CUDA | 434.36 ms | +36.56% | 9.02 dB | 0.2975 |
| ConvRot W4A4 INT8 | CUDA | 530.16 ms | +22.56% | 9.76 dB | 0.3414 |
| `asym_w4a8_int8` | CUDA | 716.28 ms | -4.62% | 10.44 dB | 0.3890 |
| `asym_w4a8_int8` | Triton | 538.97 ms | +21.28% | 10.42 dB | 0.3862 |

### 注意：

在本次固定环境下，INT8 ConvRot 的 CUDA 比 Triton 更快，但 Triton 的 PSNR/SSIM 略高：

| 模型 | CUDA | Triton | 观察 |
|---|---:|---:|---|
| Anima INT8 ConvRot | 1034.54 ms/step，23.02 dB，0.8796 | 1084.51 ms/step，23.90 dB，0.9007 | CUDA 快约 4.6%，Triton 质量指标略高 |
| Noob1.1E INT8 ConvRot | 465.30 ms/step，11.44 dB，0.4482 | 515.33 ms/step，12.39 dB，0.5000 | CUDA 快约 9.7%，Triton 质量指标略高 |

这不是 CUDA 或 Triton 的普遍理论结论。实际结果会受到模型架构、矩阵形状、显存调度、kernel 融合方式、累加顺序和量化舍入误差影响。选择 backend 时，建议使用自己的模型和工作负载进行实测。

## 测试图片

以下图片来自同一组固定参数测试。每个区块中的两张图片并排展示，便于比较模型架构和 backend 差异。

### 1. Baseline

<table>
<tr>
<td><img src="assets/test/20260922/anima-baseline.png" width="420" /></td>
<td><img src="assets/test/20260922/aio-baseline.png" width="420" /></td>
</tr>
<tr>
<td align="center">Anima Baseline</td>
<td align="center">Noob1.1E Baseline</td>
</tr>
</table>

### 2. FP8 E4M3

<table>
<tr>
<td><img src="assets/test/20260922/anima-fp8-e4m3fn.png" width="420" /></td>
<td><img src="assets/test/20260922/aio-fp8-e4m3fn.png" width="420" /></td>
</tr>
<tr>
<td align="center">Anima · FP8 E4M3 · CUDA</td>
<td align="center">Noob1.1E · FP8 E4M3 · CUDA</td>
</tr>
</table>

### 3. INT8 ConvRot · CUDA

<table>
<tr>
<td><img src="assets/test/20260922/anima-int8-convrot-cuda.png" width="420" /></td>
<td><img src="assets/test/20260922/aio-int8-convrot-cuda.png" width="420" /></td>
</tr>
<tr>
<td align="center">Anima · INT8 ConvRot · CUDA</td>
<td align="center">Noob1.1E · INT8 ConvRot · CUDA</td>
</tr>
</table>

### 4. INT8 ConvRot · Triton

<table>
<tr>
<td><img src="assets/test/20260922/anima-int8-convrot-triton.png" width="420" /></td>
<td><img src="assets/test/20260922/aio-int8-convrot-triton.png" width="420" /></td>
</tr>
<tr>
<td align="center">Anima · INT8 ConvRot · Triton</td>
<td align="center">Noob1.1E · INT8 ConvRot · Triton</td>
</tr>
</table>

完整原始测试数据：[full-test-report.json](assets/test/20260922/full-test-report.json)

## 已知限制

- 本节点只量化模型中的 2D `Linear` 权重，不量化 TE 或 VAE。
- Anima 的 ConvRot W4A4 INT4/INT8 会被主动拒绝，以避免已知严重失真。
- W4A4 的 Triton backend 当前不可用。
- NVFP4 的硬件级加速依赖 Blackwell 或更新架构；旧架构可能退化为软件路径。
- BNB NF4 主要用于显存压缩，不保证带来推理速度提升。
- CUDA/Triton 的实际速度和输出差异依赖模型、shape、PyTorch、ComfyUI 和 comfy-kitchen 版本。
- 测试图像仅用于展示本次固定环境结果，不代表所有模型和硬件的保证值。

## 项目结构

```text
comfyui-fp8-weight-quantize/
├── __init__.py
├── fp8_quantize_node.py
├── assets/
│   └── test/20260922/
│       ├── full-test-report.json
│       └── test images
├── LICENSE
└── README.md
```

当前版本使用 ComfyUI 原生 MixedPrecision 和 Comfy Kitchen，不再保留独立的自定义 Triton backend、Hadamard 旋转文件或第三个后处理量化节点。

## 许可

本项目使用 GPL-3.0 license。部分实现和算法设计参考 ComfyUI、comfy-kitchen、bitsandbytes 以及相关开源项目。
