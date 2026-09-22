"""
ComfyUI Weight Quantization Node

将 UNet/DiT 的 Linear 层权重量化为低精度格式存储，
推理时自动 upcast 计算，大幅降低显存占用。

支持精度：float8_e4m3fn / int8 / nvfp4 / mxfp8 / bnb-nf4
兼容所有使用 nn.Linear 的模型架构。
可通过外部 torch.compile 节点进一步加速。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import contextlib
from folder_paths import get_filename_list
import comfy.model_management

# ── 依赖检测 ──
try:
    import bitsandbytes as bnb
    _BNB_IMPORTED = True
except (ImportError, Exception) as e:
    _BNB_IMPORTED = False
    _BNB_IMPORT_ERROR = f"import bitsandbytes 失败: {e}"
    _BNB_AVAILABLE = False

if _BNB_IMPORTED:
    try:
        from bitsandbytes.functional import dequantize_4bit as _bnb_dequantize_4bit
        from bitsandbytes.functional import quantize_4bit as _bnb_quantize_4bit
        _BNB_AVAILABLE = True
        _BNB_IMPORT_ERROR = None
    except ImportError as _bnb_import_err:
        _BNB_AVAILABLE = False
        _BNB_IMPORT_ERROR = f"ImportError: {_bnb_import_err}\n(Tip: bitsandbytes 版本可能太新，API 已变更)"
    except Exception as _bnb_exc:
        _BNB_AVAILABLE = False
        _BNB_IMPORT_ERROR = f"{type(_bnb_exc).__name__}: {_bnb_exc}"

# ── Quantization helpers ──

if not _BNB_AVAILABLE and _BNB_IMPORT_ERROR:
    print(f"[WeightQuantize] bitsandbytes 不可用: {_BNB_IMPORT_ERROR}")

_NATIVE_MODE_CONFIGS = {
    "float8_e4m3fn": {
        "format": "float8_e4m3fn",
        "kind": "fp8",
    },
    "int8_tensorwise_convrot": {
        "format": "int8_tensorwise",
        "kind": "int8_convrot",
    },
    "int8_tensorwise": {
        "format": "int8_tensorwise",
        "kind": "int8",
    },
    "convrot_w4a4_int4": {
        "format": "convrot_w4a4",
        "kind": "w4a4",
        "linear_dtype": "int4",
    },
    "convrot_w4a4_int8": {
        "format": "convrot_w4a4",
        "kind": "w4a4",
        "linear_dtype": "int8",
    },
    # Keep the old spelling as a cache/workflow compatibility alias; new UI
    # and metadata use Comfy 0.37's canonical asym_w4a8_int8 format.
    "w4a8_int8": {
        "format": "asym_w4a8_int8",
        "kind": "w4a8",
    },
    "asym_w4a8_int8": {
        "format": "asym_w4a8_int8",
        "kind": "w4a8",
    },
}

_KITCHEN_BACKEND_OPTIONS = ["CUDA", "Triton"]
_AIO_COMPONENT = "无需加载（AIO）"
_FP8_SR_REPORTED = set()

def _native_mode_config(mode: str) -> dict:
    try:
        return _NATIVE_MODE_CONFIGS[mode]
    except KeyError as exc:
        raise ValueError(f"Unsupported native quantization mode: {mode}") from exc


def _native_layout(mode: str):
    """Resolve a Comfy quant algorithm and its registered Kitchen layout."""
    from comfy.quant_ops import QUANT_ALGOS, get_layout_class

    config = _native_mode_config(mode)
    quant_format = config["format"]
    if quant_format not in QUANT_ALGOS:
        raise RuntimeError(
            f"ComfyUI does not expose QUANT_ALGOS['{quant_format}']; "
            "this node needs the native comfy_kitchen quantization path."
        )
    layout_name = QUANT_ALGOS[quant_format]["comfy_tensor_layout"]
    layout_cls = get_layout_class(layout_name)
    if layout_cls is None:
        raise RuntimeError(
            f"ComfyUI registered '{quant_format}', but layout '{layout_name}' "
            "is unavailable in the current comfy_kitchen environment."
        )
    return config, quant_format, layout_name, layout_cls


def _backend_name(value: str | None) -> str | None:
    # Retain the old spelling for cached workflows, but expose only CUDA/Triton.
    if value in (None, "CUDA", "Auto (Comfy CUDA)"):
        return "cuda"
    if value == "Triton":
        return "triton"
    raise ValueError(f"Unsupported comfy_kitchen backend: {value!r}")


def _component_choices(category: str) -> list[str]:
    """AIO uses checkpoint-internal components; other models choose a file."""
    return [_AIO_COMPONENT, *sorted(get_filename_list(category))]


def _component_mode(selection: str | None) -> str:
    """Map new selections and legacy True/False workflow values."""
    if selection in (None, "False"):
        return "none"
    if selection in (_AIO_COMPONENT, "True"):
        return "embedded"
    return "external"


def _infer_text_encoder_type(name: str):
    import comfy.sd

    # The bundled Qwen image encoders need QWEN_IMAGE; standard CLIP remains
    # the safe automatic default for the rest of Comfy's text-encoder files.
    if "qwen" in name.lower():
        return comfy.sd.CLIPType.QWEN_IMAGE
    return comfy.sd.CLIPType.STABLE_DIFFUSION


def _load_external_text_encoder(name: str, disable_dynamic: bool = False):
    import comfy.sd
    import folder_paths

    path = folder_paths.get_full_path_or_raise("text_encoders", name)
    return comfy.sd.load_clip(
        ckpt_paths=[path],
        embedding_directory=folder_paths.get_folder_paths("embeddings"),
        clip_type=_infer_text_encoder_type(name),
        disable_dynamic=disable_dynamic,
    )


def _load_external_vae(name: str):
    import comfy.sd
    import comfy.utils
    import folder_paths

    path = folder_paths.get_full_path_or_raise("vae", name)
    state_dict, metadata = comfy.utils.load_torch_file(path, return_metadata=True)
    vae = comfy.sd.VAE(sd=state_dict, metadata=metadata)
    vae.throw_exception_if_invalid()
    # Mirror core VAELoader: without a rebuild factory the VAE patcher cannot
    # produce non-dynamic delegates or multigpu clones.
    vae.patcher.cached_patcher_init = (comfy.sd.load_vae_patcher, (path, metadata, None))
    return vae


def _load_checkpoint_with_components(
    state_dict: dict,
    metadata: dict | None,
    text_encoder: str | None,
    vae_name: str | None,
    disable_dynamic: bool = False,
):
    """Build the model and choose either embedded AIO or named external parts."""
    import comfy.sd

    text_mode = _component_mode(text_encoder)
    vae_mode = _component_mode(vae_name)
    out = comfy.sd.load_state_dict_guess_config(
        state_dict,
        output_vae=(vae_mode == "embedded"),
        output_clip=(text_mode == "embedded"),
        output_clipvision=False,
        embedding_directory=None,
        output_model=True,
        metadata=metadata or {},
        disable_dynamic=disable_dynamic,
    )
    if out is None or out[0] is None:
        raise RuntimeError("Failed to build MODEL from checkpoint")

    clip = out[1] if text_mode == "embedded" else (
        _load_external_text_encoder(text_encoder, disable_dynamic) if text_mode == "external" else None
    )
    vae = out[2] if vae_mode == "embedded" else (
        _load_external_vae(vae_name) if vae_mode == "external" else None
    )
    if text_mode == "embedded" and clip is None:
        raise RuntimeError("已选择“无需加载（AIO）”，但该 checkpoint 不含内置文本编码器；请选择 text_encoder 文件。")
    if vae_mode == "embedded" and vae is None:
        raise RuntimeError("已选择“无需加载（AIO）”，但该 checkpoint 不含内置 VAE；请选择 vae 文件。")
    return (out[0], clip, vae)


def _prepare_kitchen_backend(value: str | None) -> str | None:
    """Enable the requested backend in-process without changing launch args."""
    backend = _backend_name(value)
    if backend is None:
        return None
    import comfy_kitchen as ck

    ck.enable_backend(backend)
    if not ck.registry.is_available(backend):
        info = ck.list_backends().get(backend, {})
        raise RuntimeError(
            f"Requested comfy_kitchen backend '{backend}' is unavailable: "
            f"{info.get('unavailable_reason') or 'disabled or not registered'}"
        )
    return backend


@contextlib.contextmanager
def _kitchen_backend_context(value: str | None):
    backend = _prepare_kitchen_backend(value)
    if backend is None:
        yield
        return
    import comfy_kitchen as ck

    with ck.use_backend(backend):
        yield


def _wrap_model_kitchen_backend(model: nn.Module, value: str | None) -> None:
    """Keep a selected Kitchen backend active for Comfy MixedPrecisionOps."""
    backend = _backend_name(value)
    import comfy_kitchen as ck
    from comfy_kitchen.tensor import QuantizedTensor

    _prepare_kitchen_backend(value)
    for module in model.modules():
        weight = getattr(module, "weight", None)
        if not isinstance(weight, QuantizedTensor):
            continue
        effective_backend = backend
        if (
            backend == "cuda"
            and getattr(weight, "_layout_cls", None) == "AsymW4A8Int8Layout"
            and getattr(getattr(weight, "_params", None), "convrot_groupsize", 256) != 256
        ):
            # The CUDA fused W4A8 kernels currently only accept ConvRot group
            # 256.  Keep the native MixedPrecision tensor/layout path and use
            # Comfy Kitchen eager for the smaller groups.
            effective_backend = "eager"
            ck.enable_backend("eager")
        if getattr(module, "_forge_kitchen_backend", None) == effective_backend:
            continue
        original_forward = module.forward

        def backend_forward(*args, _original=original_forward, _backend=effective_backend, **kwargs):
            with ck.use_backend(_backend):
                return _original(*args, **kwargs)

        module.forward = backend_forward
        module._forge_kitchen_backend = effective_backend


def _stochastic_fp8_auto(value: torch.Tensor, output_type: torch.dtype, seed: int):
    """FP8 stochastic rounding via Comfy core.

    comfy.float.stochastic_rounding already prefers the kitchen CUDA kernel
    (0.37) and carries a complete pure-torch fallback, so an explicit tiered
    lookup here would only duplicate it.
    """
    import comfy.float

    if "comfy core" not in _FP8_SR_REPORTED:
        _FP8_SR_REPORTED.add("comfy core")
        print(f"[WeightQuantize] FP8 stochastic rounding backend=comfy core (kitchen CUDA preferred internally)")
    return comfy.float.stochastic_rounding(value, output_type, seed=seed)


def _native_compute_dtype(weight: torch.Tensor) -> torch.dtype:
    if weight.dtype in (torch.float16, torch.bfloat16):
        return weight.dtype
    return torch.bfloat16


def _convrot_group_size(in_features: int) -> int | None:
    if in_features % 256 == 0:
        return 256
    if in_features % 64 == 0:
        return 64
    return None


def _w4a8_convrot_group_size(in_features: int) -> int | None:
    """Pick the largest Comfy Kitchen-supported W4A8 ConvRot group."""
    for group_size in (256, 64, 16):
        if in_features % group_size == 0:
            return group_size
    return None


def _comfy_quant_marker(config: dict) -> torch.Tensor:
    return torch.tensor(
        list(json.dumps(config, separators=(",", ":")).encode("utf-8")),
        dtype=torch.uint8,
    )


def _looks_like_anima_checkpoint(sd: dict, prefix: str) -> bool:
    """Recognize the Anima DiT layout for quality-sensitive guards."""
    return (
        f"{prefix}blocks.0.self_attn.q_proj.weight" in sd
        and f"{prefix}blocks.0.cross_attn.q_proj.weight" in sd
        and any(k.startswith(f"{prefix}llm_adapter.") for k in sd)
    )


def _maybe_enable_channels_last(model_patcher, tag: str) -> None:
    """Enable channels_last on standard openaimodel UNets (SD1.x/2.x/XL).

    DiTs (Flux/Qwen/Krea/...) carry no conv2d and never trigger this.
    """
    if not (hasattr(model_patcher, "model") and hasattr(model_patcher.model, "diffusion_model")):
        return
    diffusion_model = model_patcher.model.diffusion_model
    try:
        from comfy.ldm.modules.diffusionmodules.openaimodel import UNetModel
        is_unet = isinstance(diffusion_model, UNetModel)
    except Exception:
        is_unet = False
    if not is_unet:
        is_unet = diffusion_model.__class__.__name__ == "UNetModel"
    if is_unet:
        print(f"[{tag}] Detected standard UNet model, implicitly enabling channels_last memory format for diffusion model.")
        diffusion_model.to(memory_format=torch.channels_last)



def _quantize_native_weight(
    weight: torch.Tensor,
    mode: str,
    layout_cls,
    stochastic_seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict, bool]:
    """Quantize one [out_features, in_features] weight with Comfy's layout.

    Returns qdata, the per-layer scale/auxiliary tensors, the exact
    ``comfy_quant`` config, and whether model-shape protection is required
    because the stored qdata is packed.
    """
    config = _native_mode_config(mode)
    # Offline INT8 rotation/scale/rounding needs FP32; storage and inference
    # remain INT8 with the model's original FP16/BF16 output dtype.
    work_dtype = torch.float32 if config["kind"] in ("int8", "int8_convrot") else _native_compute_dtype(weight)
    work = weight.to(device=device, dtype=work_dtype)
    seed = int(stochastic_seed)

    if config["kind"] == "fp8":
        fp8_dtype = torch.float8_e4m3fn
        scale = (work.float().abs().amax() / torch.finfo(fp8_dtype).max).clamp(min=1e-30)
        qdata = _stochastic_fp8_auto(work / scale.to(work.dtype), fp8_dtype, seed)
        marker = {"format": config["format"]}
        return qdata, {"weight_scale": scale.float()}, marker, False

    if config["kind"] in ("int8", "int8_convrot"):
        convrot = config["kind"] == "int8_convrot" and work.shape[1] % 256 == 0
        qdata, params = layout_cls.quantize(
            work,
            is_weight=True,
            per_channel=True,
            convrot=convrot,
            convrot_groupsize=256,
            stochastic_rounding=seed,
        )
        marker = {"format": config["format"]}
        if convrot:
            marker.update({"convrot": True, "convrot_groupsize": 256})
        return qdata, {"weight_scale": params.scale.float()}, marker, False

    if config["kind"] == "w4a8":
        convrot_groupsize = _w4a8_convrot_group_size(work.shape[1])
        if convrot_groupsize is None:
            raise ValueError(
                f"W4A8 requires in_features divisible by 16, got {work.shape[1]}"
            )
        if convrot_groupsize == 256:
            qdata, params = layout_cls.quantize(
                work,
                group_size=16,
                convrot_groupsize=convrot_groupsize,
                symmetric=True,
                scale_dtype=torch.float8_e4m3fn,
                codebook=False,
                stochastic_rounding=seed,
            )
        else:
            # Comfy Kitchen's CUDA rotation extension currently only accepts
            # group 256.  Eager produces the same native W4A8 tensors for
            # group 64/16; inference still uses the selected MixedPrecision
            # backend after the checkpoint is loaded.
            from comfy_kitchen.backends.eager.w4a8_int8 import (
                quantize_w4a8_int8_weight,
            )

            qdata, s_rel, s_channel, correction, codebook = quantize_w4a8_int8_weight(
                work,
                group_size=16,
                convrot_groupsize=convrot_groupsize,
                symmetric=True,
                scale_dtype=torch.float8_e4m3fn,
                codebook=False,
                stochastic_rounding=seed,
            )
            params = layout_cls.Params(
                scale=s_rel,
                s_channel=s_channel,
                correction=correction,
                codebook=codebook,
                orig_dtype=work.dtype,
                orig_shape=tuple(work.shape),
                group_size=16,
                convrot_groupsize=convrot_groupsize,
            )
        marker = {
            "format": config["format"],
            "group_size": 16,
            "convrot_groupsize": convrot_groupsize,
        }
        scale_fields = {"weight_s_rel": params.scale}
        if params.s_channel is None:
            raise RuntimeError("Comfy W4A8 quantization did not return the required per-channel scale.")
        scale_fields["weight_s_channel"] = params.s_channel
        if params.codebook is not None:
            scale_fields["weight_codebook"] = params.codebook
        if params.correction is not None:
            raise RuntimeError(
                "Comfy 0.37's asym_w4a8_int8 loader path does not consume "
                "weight_correction; symmetric W4A8 is required."
            )
        return qdata, scale_fields, marker, True

    group_size = _convrot_group_size(work.shape[1])
    if group_size is None:
        raise ValueError(
            f"ConvRot W4A4 requires in_features divisible by 64, got {work.shape[1]}"
        )
    qdata, params = layout_cls.quantize(
        work,
        convrot_groupsize=group_size,
        quant_group_size=64,
        stochastic_rounding=seed,
        linear_dtype=config["linear_dtype"],
    )
    marker = {
        "format": config["format"],
        "convrot_groupsize": group_size,
    }
    # Comfy's own state_dict writer omits the default int4 field and emits it
    # only for the int8 MMA variant.
    if config["linear_dtype"] != "int4":
        marker["linear_dtype"] = config["linear_dtype"]
    return qdata, {"weight_scale": params.scale.float()}, marker, True


# ── BNB (nf4) forward patch ──

if _BNB_AVAILABLE:

    def _make_bnb_linear_forward(module):
        def forward(x: torch.Tensor) -> torch.Tensor:
            w = _bnb_dequantize_4bit(
                module._forge_quant_weight,
                quant_state=module._forge_weight_quant_state,
            )
            w = w.to(device=x.device, dtype=x.dtype)

            # Apply LoRA delta if patched in-place on the dummy weight
            if hasattr(module, "weight") and module.weight is not None and module.weight.any():
                w = w + module.weight.to(device=x.device, dtype=x.dtype)

            for f in getattr(module, 'weight_function', []):
                w = f(w)

            b = module.bias
            if b is not None:
                b = b.to(device=x.device, dtype=x.dtype)
                for ff in getattr(module, 'bias_function', []):
                    b = ff(b)
            return F.linear(x, w, b)
        return forward


# ── 量化主逻辑 ──

def _make_dummy_weight(weight_data: torch.Tensor) -> torch.nn.Parameter:
    # Quality fix: requires_grad must be passed to Parameter, NOT to zeros_like.
    # Passing it to zeros_like only affects the plain tensor; Parameter.__init__
    # defaults to requires_grad=True and would silently re-enable grad tracking.
    return torch.nn.Parameter(
        torch.zeros_like(weight_data),
        requires_grad=False,
    )


# ── _forge_* 属性说明 ──
# 由本节点动态注入到 nn.Linear 实例上的量化状态：
#   _node_owned            : bool   — 标记此模块由本节点量化，重复执行时可覆盖
#   （BNB 路径）_forge_quant_weight / _forge_weight_quant_state


def _restore_quantized_weight(module: nn.Module) -> "torch.Tensor | None":
    """尝试从各量化格式恢复原始浮点权重。

    返回恢复后的 Tensor，或 None（表示该模块未被量化）。
    副作用：清除已识别的量化状态属性。
    """
    # BNB 4-bit（自定义 _forge 存储路径）
    if _BNB_AVAILABLE and hasattr(module, "_forge_weight_quant_state"):
        qs = getattr(module, "_forge_weight_quant_state", None)
        if qs is not None and hasattr(module, "_forge_quant_weight"):
            w = _bnb_dequantize_4bit(module._forge_quant_weight, quant_state=qs)
            del module._forge_quant_weight
            module._forge_weight_quant_state = None
            return w

    return None


def _restore_original_weight(module: nn.Module) -> torch.Tensor:
    """恢复模块的原始浮点权重，兼容所有量化格式。

    若模块由本节点量化（_node_owned=True），先清除该标记再恢复。
    若模块未被量化，返回当前权重的 detach 副本。
    """
    if getattr(module, "_node_owned", False):
        module._node_owned = False
    w = _restore_quantized_weight(module)
    return w if w is not None else module.weight.detach()


def _quantize_module(module: nn.Module, weight_data: torch.Tensor):
    if not _BNB_AVAILABLE:
        raise RuntimeError(f"[WeightQuantize] bnb-nf4 需要 bitsandbytes，导入失败: {_BNB_IMPORT_ERROR}")
    gpu_device = comfy.model_management.get_torch_device()
    w = weight_data.to(device=gpu_device, dtype=torch.float16)
    w_4bit, quant_state = _bnb_quantize_4bit(w, blocksize=64, quant_type="nf4")
    module._forge_quant_weight = w_4bit
    module._forge_weight_quant_state = quant_state
    module._parameters["weight"] = _make_dummy_weight(weight_data)
    module.forward = _make_bnb_linear_forward(module)
    module._node_owned = True


# Sensitive-layer skip list. Sources (cross-checked):
#   - official quantized checkpoints' per-tensor dtype tables
#     (BFL FLUX.2[-klein] fp8/nvfp4, Comfy-Org Qwen-Image-2.1 int8_convrot,
#      Comfy-Org Krea-2 fp8)
#   - silveroxides/convert_to_quant presets (qwen/flux2)
#   - real-model W4A8 measurements (JoaoZaokk/comfy-quant-bench):
#     1D norm params and input/output projections of any kind stay high
#     precision; modulation kept BF16 per the official school.
# Not listed (measured safe at 8-bit activations): attn output projections
# (to_out.0/proj/to_add_out); RMSNorm/QK-norm weights are 1D and never
# reach the 2D-weight quantizer anyway.
_SENSITIVE_NAME_PATTERNS = (
    "adaln", "modulation", "_norm", "embed", "final_layer", "llm_adapter",
    # per-block modulation (Qwen-Image 1.x ImgMod/TextMod Sequential,
    # Flux.1 double_blocks img_mod/txt_mod .lin)
    "img_mod", "txt_mod",
    # entrance/tail embedders: patch/text in, time/vec/guidance stacks,
    # Qwen output head (norm_out/proj_out)
    "img_in", "txt_in", "time_in", "vector_in", "guidance_in",
    "norm_out", "proj_out",
    # Krea2 (K2): patch embed, time stack, Qwen3-VL adapter stack, LastLayer
    "first", "tmlp", "tproj", "txtmlp", "txtfusion", "last.linear",
    # Lumina 2 / Z-Image (+ zimage_pixel), shared NextDiT impl:
    # shallow cond refiners (noise/context/siglip incl. their adaLN blocks),
    # entrance embedders (x/cap/siglip/timestep; x/cap/siglip *_embedder keys
    # also carry "embed", t_embedder does not), pooled-text projects,
    # and the whole pixel decoder head of the zimage_pixel variant
    "_refiner", "_embedder", "pooled_proj", "cap_proj", "dec_net",
)

_SAFE_BLOCK_START = (0, 1)  # 首两个 block 跳过量化，保持原始精度


def _is_sensitive(name: str) -> bool:
    n = name.lower()
    if any(p in n for p in _SENSITIVE_NAME_PATTERNS):
        return True
    parts = n.split(".")
    for i in range(len(parts) - 1):
        if parts[i] == "blocks" and parts[i + 1].isdigit() and int(parts[i + 1]) in _SAFE_BLOCK_START:
            return True
    return False


def _is_native_sensitive(name: str, anima_last_block: int | None) -> bool:
    """Keep Anima INT8 boundary blocks, early MLPs and non-projection layers."""
    if anima_last_block is None:
        return _is_sensitive(name)
    parts = name.lower().split(".")
    if "llm_adapter" in parts or "blocks" not in parts:
        return True
    i = parts.index("blocks")
    if i + 3 >= len(parts) or not parts[i + 1].isdigit():
        return True
    block = int(parts[i + 1])
    tail = parts[i + 2:]
    if block in (0, anima_last_block):
        return True
    if tail in (["mlp", "layer1"], ["mlp", "layer2"]):
        return block < 4
    return not (len(tail) == 2 and tail[0] in ("self_attn", "cross_attn")
                and tail[1] in ("q_proj", "k_proj", "v_proj", "output_proj"))


def _is_already_quantized(module: nn.Module) -> bool:
    if getattr(module, "_node_owned", False):
        return False
    try:
        from comfy_kitchen.tensor import QuantizedTensor
        if hasattr(module, "weight") and isinstance(module.weight, QuantizedTensor):
            return True
    except Exception:
        pass
    # BNB 路径：_forge_quant_weight 与 _forge_weight_quant_state 均存在
    if hasattr(module, "_forge_quant_weight"):
        return True
    if _BNB_AVAILABLE and hasattr(module, "_forge_weight_quant_state") and module._forge_weight_quant_state is not None:
        return True
    return False


def quantize_model(
    full_model: nn.Module,
    skip_sensitive: bool,
):
    stats = {
        "total_linear": 0,
        "quantized_linear": 0,
        "skipped_already_quantized": 0,
        "skipped_sensitive": 0,
    }

    for name, module in full_model.named_modules():
        if not isinstance(module, nn.Linear):
            continue

        stats["total_linear"] += 1

        if skip_sensitive and _is_sensitive(name):
            stats["skipped_sensitive"] += 1
            continue

        if _is_already_quantized(module):
            stats["skipped_already_quantized"] += 1
            continue

        weight_data = _restore_original_weight(module)
        _quantize_module(module, weight_data)

        stats["quantized_linear"] += 1

        del weight_data

        # 每量化 50 层主动释放一次 GPU 缓存，防止大模型量化时 OOM
        if stats["quantized_linear"] % 50 == 0:
            comfy.model_management.soft_empty_cache()

    comfy.model_management.soft_empty_cache()
    return stats


# ── ComfyUI loaders ──

def _get_stable_seed(layer_name: str) -> int:
    seed = 0
    for char in layer_name:
        seed = (seed * 31 + ord(char)) & 0xFFFFFFFF
    return max(1, seed % (2**31))


def _quantize_checkpoint_state_dict(
    sd: dict,
    prefix: str,
    mode: str,
    skip_sensitive: bool,
    stochastic_rounding: bool,
    device,
):
    """Quantize checkpoint weights and emit Comfy's per-layer metadata."""
    config, _, _, layout_cls = _native_layout(mode)
    anima_int8 = config["kind"] in ("int8", "int8_convrot") and _looks_like_anima_checkpoint(sd, prefix)
    last_block = max(int(k[len(prefix):].split(".")[1]) for k in sd
                     if k.startswith(prefix + "blocks.") and k[len(prefix):].split(".")[1].isdigit()) if anima_int8 else None

    stats = {
        "total_linear": 0,
        "quantized": 0,
        "skipped_sensitive": 0,
        "skipped_existing": 0,
        "skipped_unsupported": 0,
        "plain_int8_fallback": 0,
    }
    quant_layers = {}
    original_shapes = {}

    for key in list(sd.keys()):
        if not key.startswith(prefix) or not key.endswith(".weight"):
            continue
        weight = sd[key]
        if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
            continue

        stats["total_linear"] += 1
        layer_key = key[:-len(".weight")]
        if f"{layer_key}.comfy_quant" in sd:
            stats["skipped_existing"] += 1
            continue

        layer_name = key[len(prefix):-len(".weight")]
        if skip_sensitive and _is_native_sensitive(layer_name, last_block):
            stats["skipped_sensitive"] += 1
            continue

        seed = _get_stable_seed(layer_name) if stochastic_rounding else 0
        try:
            qdata, scale_fields, marker, packed = _quantize_native_weight(
                weight,
                mode,
                layout_cls,
                seed,
                device,
            )
        except ValueError as exc:
            if config["kind"] == "w4a4" and "divisible by 64" in str(exc):
                stats["skipped_unsupported"] += 1
                continue
            raise

        sd[key] = qdata.detach().contiguous().cpu()
        for field_name, field in scale_fields.items():
            sd[f"{layer_key}.{field_name}"] = field.detach().contiguous().cpu()
        sd[f"{layer_key}.comfy_quant"] = _comfy_quant_marker(marker)
        quant_layers[layer_key] = marker
        if packed:
            original_shapes[key] = tuple(weight.shape)
        if config["kind"] == "int8_convrot" and not marker.get("convrot", False):
            stats["plain_int8_fallback"] += 1
        stats["quantized"] += 1

        if stats["quantized"] % 50 == 0:
            comfy.model_management.soft_empty_cache()

    comfy.model_management.soft_empty_cache()
    return stats, quant_layers, original_shapes


# ── FP8 Checkpoint Loader ──

class Fp8CheckpointLoader:
    """
    加载 checkpoint 并注入量化元数据，让 ComfyUI 自动识别并路由到 MixedPrecisionOps。
    此节点的量化发生在 state_dict 阶段，在模型构建前注入 .comfy_quant 元数据
    并直接按 Layout 量化权重，由 ComfyUI 原生管线接管。
    支持精度：float8_e4m3fn, nvfp4, mxfp8, bnb-nf4 并且支持随机舍入（Stochastic Rounding）。
    """

    @classmethod
    def INPUT_TYPES(cls):
        files = get_filename_list("checkpoints") + get_filename_list("diffusion_models")
        return {
            "required": {
                "ckpt_name": (sorted(set(files)), ),
                "dtype": ([
                    "float8_e4m3fn",
                    "nvfp4",
                    "mxfp8",
                    "bnb-nf4",
                ], {"default": "float8_e4m3fn"}),
                "text_encoder": (_component_choices("text_encoders"), {"default": _AIO_COMPONENT}),
                "vae": (_component_choices("vae"), {"default": _AIO_COMPONENT}),
                "skip_sensitive": (["True", "False"], {"default": "True"}),
                "allow_compile": (["False", "True（torch.compile 兼容模式）"], {"default": "False"}),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    RETURN_NAMES = ("model", "clip", "vae")
    FUNCTION = "load_checkpoint"
    CATEGORY = "loaders"
    DESCRIPTION = (
        "[FP8 Checkpoint Loader] 加载 Checkpoint 并写入 Comfy 每层量化元数据。\n\n"
        "支持精度选项：\n"
        "  - float8_e4m3fn — 原生 FP8 E4M3 精度，具备原生硬件加速 (Ampere+ 架构推荐)。\n"
        "  - nvfp4 — ComfyUI 原生 Block-wise 4-bit 浮点格式 (无需 bitsandbytes)。\n"
        "  - mxfp8 — ComfyUI 原生 Block-wise 8-bit 浮点格式 (无需 bitsandbytes)。\n"
        "  - bnb-nf4 — 传统的 bitsandbytes NF4 量化 (载入后自动量化，极限压缩显存，适用于所有 CUDA)。\n\n"
        "特性：\n"
        "  - FP8 随机舍入：交给 Comfy core 自动选择 comfy_kitchen CUDA 或纯 PyTorch fallback。\n"
        "  - text_encoder / vae：直接选择外置文件；“无需加载（AIO）”使用 checkpoint 内置组件。\n"
        "  - 敏感层保护：自动跳过 adaln/modulation/img_mod·txt_mod/norm 类/embed/proj_out/时间·向量·引导嵌入/"
        "Krea2 的 first·tmlp·txtfusion·last 等全家族敏感层（依据官方 fp8/int8 量化文件的逐层 dtype 表），兼顾速度与质量。\n"
        "  - 兼容编译：可通过 allow_compile 开启对外部 torch.compile 的支持。"
    )

    def load_checkpoint(
        self,
        ckpt_name: str,
        dtype: str,
        stochastic_rounding: str | None = None,
        skip_sensitive: str = "True",
        allow_compile: str = "False",
        kitchen_backend: str = "CUDA",
        text_encoder: str = _AIO_COMPONENT,
        vae: str = _AIO_COMPONENT,
        disable_dynamic: bool = False,
    ):
        import comfy.utils
        import comfy.model_detection
        import folder_paths

        is_bnb = (dtype == "bnb-nf4")
        is_native_mode = dtype in _NATIVE_MODE_CONFIGS
        if dtype.startswith("convrot_w4a4") and kitchen_backend == "Triton":
            raise RuntimeError("当前 comfy_kitchen Triton backend 不提供 ConvRot W4A4；请将 kitchen_backend 设为 CUDA。")
        if dtype == "float8_e4m3fn" and kitchen_backend == "Triton":
            print("[Fp8CheckpointLoader] FP8 忽略 Triton 选择，由 Comfy core 自动选择 comfy_kitchen CUDA 或纯 PyTorch fallback。")

        # dynamic import & environment validation
        if is_native_mode:
            _native_layout(dtype)
        elif not is_bnb:
            try:
                from comfy.quant_ops import get_layout_class, QUANT_ALGOS
            except ImportError:
                raise RuntimeError(
                    "[Fp8CheckpointLoader] comfy.quant_ops could not be imported.\n"
                    "This node requires a newer ComfyUI version with Native Mixed Precision support."
                )

            if dtype not in QUANT_ALGOS:
                raise RuntimeError(f"[Fp8CheckpointLoader] Quantization format '{dtype}' is not defined in ComfyUI's QUANT_ALGOS.")

            layout_name = QUANT_ALGOS[dtype]["comfy_tensor_layout"]
            layout_cls = get_layout_class(layout_name)
            if layout_cls is None:
                raise RuntimeError(
                    f"[Fp8CheckpointLoader] The layout '{layout_name}' for precision '{dtype}' is not available in your environment.\n"
                    "Please make sure comfy_kitchen C++ / CUDA extension is installed and properly configured."
                )

        ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
        if ckpt_path is None:
            ckpt_path = folder_paths.get_full_path("diffusion_models", ckpt_name)
        if ckpt_path is None:
            raise FileNotFoundError(f"[Fp8CheckpointLoader] Model not found: {ckpt_name}")

        sd, metadata = comfy.utils.load_torch_file(ckpt_path, return_metadata=True)
        if metadata is None:
            metadata = {}
        prefix = comfy.model_detection.unet_prefix_from_state_dict(sd)
        if dtype.startswith("convrot_w4a4") and _looks_like_anima_checkpoint(sd, prefix):
            raise RuntimeError(
                "当前 Comfy Kitchen ConvRot W4A4 在 Anima DiT 上会产生雪花/严重失真图，"
                "为保证出图正确已禁用该组合；请使用 FP8、W4A8 或普通 INT8。"
            )
        skip = (skip_sensitive == "True")
        # FP8 stochastic rounding stays automatic. Native INT8 uses FP32
        # nearest rounding; retain the old argument for cached workflows.
        stochastic_enabled = dtype not in ("int8_tensorwise", "int8_tensorwise_convrot")

        device = comfy.model_management.get_torch_device()

        if is_bnb:
            print(f"[Fp8CheckpointLoader] Loading model in default format before BNB quantization...")
            original_shapes = {}
        elif is_native_mode:
            print(f"[Fp8CheckpointLoader] Quantization processing device: {device} mode={dtype}")
            stats, quant_layers, original_shapes = _quantize_checkpoint_state_dict(
                sd=sd,
                prefix=prefix,
                mode=dtype,
                skip_sensitive=skip,
                stochastic_rounding=stochastic_enabled,
                device=device,
            )
            # Keep the legacy file-level marker for old Comfy readers, while
            # the actual current route is the per-layer *.comfy_quant key.
            metadata["_quantization_metadata"] = json.dumps({
                "format_version": "1.0",
                "layers": quant_layers,
            })
            print(
                f"[Fp8CheckpointLoader] Precision: {dtype} | Linear: {stats['quantized']}/{stats['total_linear']} | "
                f"跳过(敏感层): {stats['skipped_sensitive']} | "
                f"跳过(已存在原生元数据): {stats['skipped_existing']} | "
                f"跳过(不满足 W4A4 对齐): {stats['skipped_unsupported']}"
            )
        else:
            print(f"[Fp8CheckpointLoader] Quantization processing device: {device}")

            stats = {"total_linear": 0, "quantized": 0, "skipped_sensitive": 0}
            quant_layers = {}
            original_shapes = {}

            for k in list(sd.keys()):
                if not k.startswith(prefix) or not k.endswith(".weight"):
                    continue
                w = sd[k]
                if w.ndim != 2:
                    continue

                stats["total_linear"] += 1

                layer_name = k[len(prefix):-len(".weight")]
                if skip and _is_sensitive(layer_name):
                    stats["skipped_sensitive"] += 1
                    continue

                w_device = w.to(device)
                w_f = w_device.float()
                seed = _get_stable_seed(layer_name) if stochastic_enabled else 0

                # Quantize using Layout class on device (GPU)
                if dtype == "float8_e4m3fn":
                    fp8_dtype = QUANT_ALGOS[dtype]["storage_t"]
                    fmax = torch.finfo(fp8_dtype).max
                    scale = (w_f.abs().amax() / fmax).clamp(min=1e-30)
                    
                    qdata, params = layout_cls.quantize(w_f, scale=scale, stochastic_rounding=seed)
                    sd[k] = qdata.cpu()
                    sd[k[:-len(".weight")] + ".weight_scale"] = scale.cpu().to(dtype=torch.float32)

                elif dtype == "nvfp4":
                    original_shapes[k] = w.shape
                    # F8_E4M3_MAX * F4_E2M1_MAX = 448.0 * 6.0 = 2688.0
                    scale = (w_f.abs().amax() / 2688.0).clamp(min=1e-30)
                    
                    qdata, params = layout_cls.quantize(w_f, scale=scale, stochastic_rounding=seed)
                    sd[k] = qdata.cpu()
                    sd[k[:-len(".weight")] + ".weight_scale_2"] = scale.cpu().to(dtype=torch.float32)
                    sd[k[:-len(".weight")] + ".weight_scale"] = params.block_scale.cpu().to(dtype=torch.float8_e4m3fn)

                elif dtype == "mxfp8":
                    qdata, params = layout_cls.quantize(w_f, stochastic_rounding=seed)
                    sd[k] = qdata.cpu()
                    sd[k[:-len(".weight")] + ".weight_scale"] = params.scale.cpu().view(torch.uint8)

                # Write the current Comfy per-layer marker immediately.  Keep
                # the file-level legacy marker below as a compatibility
                # bridge, but do not rely on convert_old_quants() for the
                # native mixed-ops detection path.
                layer_key = k[:-len(".weight")]
                layer_config = {"format": dtype}
                sd[f"{layer_key}.comfy_quant"] = _comfy_quant_marker(layer_config)
                quant_layers[layer_key] = layer_config
                stats["quantized"] += 1
                
                # Explicitly garbage collect GPU local variables to free memory
                del w_device, w_f, qdata, params

                if stats["quantized"] % 50 == 0:
                    comfy.model_management.soft_empty_cache()

            comfy.model_management.soft_empty_cache()

            metadata["_quantization_metadata"] = json.dumps({
                "format_version": "1.0",
                "layers": quant_layers,
            })

            print(
                f"[Fp8CheckpointLoader] Precision: {dtype} | Linear: {stats['quantized']}/{stats['total_linear']} | "
                f"跳过(敏感层): {stats['skipped_sensitive']}"
            )

        # Temporarily restore packed-weight shapes while Comfy detects the
        # model.  This is required for NVFP4 and ConvRot W4A4.
        _orig_model_config_from_unet = getattr(comfy.model_detection, "model_config_from_unet", None)
        if _orig_model_config_from_unet is not None and not is_bnb and original_shapes:
            def patched_model_config_from_unet(state_dict, *args, **kwargs):
                restored = {}
                for k, orig_shape in original_shapes.items():
                    if k in state_dict:
                        restored[k] = state_dict[k]
                        state_dict[k] = torch.empty(orig_shape, dtype=torch.float16, device="cpu")
                try:
                    return _orig_model_config_from_unet(state_dict, *args, **kwargs)
                finally:
                    for k, v in restored.items():
                        state_dict[k] = v
            comfy.model_detection.model_config_from_unet = patched_model_config_from_unet

        try:
            out = _load_checkpoint_with_components(sd, metadata, text_encoder, vae, disable_dynamic=disable_dynamic)
        finally:
            if _orig_model_config_from_unet is not None and not is_bnb and original_shapes:
                comfy.model_detection.model_config_from_unet = _orig_model_config_from_unet

        if out is None or out[0] is None:
            raise RuntimeError(f"[Fp8CheckpointLoader] Failed to load model from {ckpt_name}")

        model_patcher = out[0]

        if is_native_mode and _native_mode_config(dtype)["kind"] in ("int8", "int8_convrot", "w4a8"):
            _wrap_model_kitchen_backend(model_patcher.model, kitchen_backend)

        # Implicitly detect standard UNet model and enable channels last memory format
        _maybe_enable_channels_last(model_patcher, "Fp8CheckpointLoader")

        # Post-load BNB quantization if requested
        if is_bnb:
            print(f"[Fp8CheckpointLoader] Applying BNB {dtype} quantization to loaded model...")
            stats = quantize_model(full_model=model_patcher.model, skip_sensitive=skip)
            print(
                f"[Fp8CheckpointLoader] BNB Quantization finished | "
                f"Linear: {stats['quantized_linear']}/{stats['total_linear']} | "
                f"跳过(敏感层): {stats['skipped_sensitive']} | "
                f"跳过(已量化): {stats['skipped_already_quantized']}"
            )

        if allow_compile.startswith("True") and hasattr(model_patcher, "model"):
            model_patcher.model._forge_allow_compile = True

        # 3-tuple (factory, args, 0): core clone()/deepclone_multigpu() calls the
        # factory with disable_dynamic and indexes it to reach the MODEL patcher.
        model_patcher.cached_patcher_init = (
            self.load_checkpoint,
            (
                ckpt_name,
                dtype,
                stochastic_rounding,
                skip_sensitive,
                allow_compile,
                kitchen_backend,
                text_encoder,
                vae,
            ),
            0,
        )

        return (model_patcher, out[1], out[2])


class Int8CheckpointLoader(Fp8CheckpointLoader):
    """Direct checkpoint loader for Comfy-native INT8/W4A4/W4A8 modes."""

    _MODES = [
        "int8_tensorwise",
        "int8_tensorwise_convrot",
        "convrot_w4a4_int4",
        "convrot_w4a4_int8",
        "asym_w4a8_int8",
    ]

    @classmethod
    def INPUT_TYPES(cls):
        files = get_filename_list("checkpoints") + get_filename_list("diffusion_models")
        return {
            "required": {
                "ckpt_name": (sorted(set(files)),),
                "quant_mode": (cls._MODES, {"default": "int8_tensorwise_convrot"}),
                "kitchen_backend": (_KITCHEN_BACKEND_OPTIONS, {"default": _KITCHEN_BACKEND_OPTIONS[0]}),
                "text_encoder": (_component_choices("text_encoders"), {"default": _AIO_COMPONENT}),
                "vae": (_component_choices("vae"), {"default": _AIO_COMPONENT}),
                "skip_sensitive": (["True", "False"], {"default": "True"}),
                "allow_compile": (["False", "True（torch.compile 兼容模式）"], {"default": "False"}),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    RETURN_NAMES = ("model", "clip", "vae")
    FUNCTION = "load_checkpoint"
    CATEGORY = "loaders"
    DESCRIPTION = (
        "[INT8 Checkpoint Loader] 直接选择 checkpoint，加载模型、文本编码器和 VAE。\n\n"
        "支持 Comfy 原生 INT8 W8A8、INT8 ConvRot W8A8、ConvRot W4A4 INT4/INT8。\n"
        "text_encoder / vae：直接选择外置文件；“无需加载（AIO）”使用 checkpoint 内置组件。\n"
        "kitchen_backend：CUDA 或 Comfy Kitchen Triton。\n"
        "W4A8 INT8：使用 Comfy 0.37 原生 MixedPrecision asym_w4a8_int8 路径（旧名 w4a8_int8 继续可用）。"
    )

    def load_checkpoint(
        self,
        ckpt_name: str,
        quant_mode: str,
        kitchen_backend: str = "CUDA",
        text_encoder: str = _AIO_COMPONENT,
        vae: str = _AIO_COMPONENT,
        skip_sensitive: str = "True",
        allow_compile: str = "False",
        disable_dynamic: bool = False,
    ):
        result = super().load_checkpoint(
            ckpt_name=ckpt_name,
            dtype=quant_mode,
            stochastic_rounding=None,
            skip_sensitive=skip_sensitive,
            allow_compile=allow_compile,
            kitchen_backend=kitchen_backend,
            text_encoder=text_encoder,
            vae=vae,
            disable_dynamic=disable_dynamic,
        )
        result[0].cached_patcher_init = (
            self.load_checkpoint,
            (
                ckpt_name,
                quant_mode,
                kitchen_backend,
                text_encoder,
                vae,
                skip_sensitive,
                allow_compile,
            ),
            0,
        )
        return result


# ── 注册 ──

NODE_CLASS_MAPPINGS = {
    "Fp8CheckpointLoader": Fp8CheckpointLoader,
    "Int8CheckpointLoader": Int8CheckpointLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Fp8CheckpointLoader": "FP8 Checkpoint Loader (FP8/FP4 + TE/VAE)",
    "Int8CheckpointLoader": "INT8 Checkpoint Loader (W8A8/W4A8/W4A4)",
}
