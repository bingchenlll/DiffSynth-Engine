import os
import shutil
from functools import lru_cache
from pathlib import Path
from typing import Optional

import torch
import torch.library
from torch.utils.cpp_extension import load

_EXTENSION_NAME = "qwen_image_cuda_ext_v8"
_OP_NAMESPACE = "qwen_image_ext"
_OPS_REGISTERED = False
_LIB_HANDLES: list[torch.library.Library] = []


def _sources() -> list[str]:
    base_dir = Path(__file__).resolve().parent / "csrc"
    return [
        str(base_dir / "qwen_image_rotary_binding.cpp"),
        str(base_dir / "qwen_image_rotary_kernel.cu"),
    ]


def _extension_arch_list() -> str:
    # Build for the local visible GPU arch directly to avoid PTX JIT/toolchain
    # mismatches, while also bypassing problematic global TORCH_CUDA_ARCH_LIST.
    try:
        major, minor = torch.cuda.get_device_capability(0)
    except Exception:
        return "9.0"
    return f"{major}.{minor}"


def _preferred_host_compilers() -> tuple[Optional[str], Optional[str]]:
    # Prefer system GCC/G++ over conda-forge GCC to avoid nvcc host-compiler
    # compatibility errors in some conda environments.
    cc = "/usr/bin/gcc" if Path("/usr/bin/gcc").exists() else shutil.which("gcc")
    cxx = "/usr/bin/g++" if Path("/usr/bin/g++").exists() else shutil.which("g++")
    return cc, cxx


@lru_cache(maxsize=1)
def _load_extension():
    if not torch.cuda.is_available():
        return None

    old_arch = os.getenv("TORCH_CUDA_ARCH_LIST")
    os.environ["TORCH_CUDA_ARCH_LIST"] = _extension_arch_list()
    old_cc = os.getenv("CC")
    old_cxx = os.getenv("CXX")
    cc, cxx = _preferred_host_compilers()
    if cc:
        os.environ["CC"] = cc
    if cxx:
        os.environ["CXX"] = cxx

    try:
        return load(
            name=_EXTENSION_NAME,
            sources=_sources(),
            extra_cflags=["-O3", "-std=c++17"],
            extra_cuda_cflags=["-O3", "--use_fast_math"],
            verbose=os.getenv("QWEN_IMAGE_CUDA_EXT_VERBOSE", "0") == "1",
        )
    except Exception as err:
        if os.getenv("QWEN_IMAGE_CUDA_EXT_WARN", "1") == "1":
            print(f"[QwenImage CUDA] rotary extension disabled: {err}")
        return None
    finally:
        if old_arch is None:
            os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
        else:
            os.environ["TORCH_CUDA_ARCH_LIST"] = old_arch
        if old_cc is None:
            os.environ.pop("CC", None)
        else:
            os.environ["CC"] = old_cc
        if old_cxx is None:
            os.environ.pop("CXX", None)
        else:
            os.environ["CXX"] = old_cxx


def rotary_emb_forward(x: torch.Tensor, freqs_cis: torch.Tensor) -> Optional[torch.Tensor]:
    ext = _load_extension()
    if ext is None:
        return None
    return ext.rotary_emb_forward(x, freqs_cis)


def rotary_emb_indexed_forward(
    x: torch.Tensor, freqs_cis: torch.Tensor, token_indices: torch.Tensor
) -> Optional[torch.Tensor]:
    ext = _load_extension()
    if ext is None:
        return None
    return ext.rotary_emb_indexed_forward(x, freqs_cis, token_indices)


def rotary_emb_pair_forward(
    x1: torch.Tensor, x2: torch.Tensor, freqs_cis: torch.Tensor
) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
    ext = _load_extension()
    if ext is None:
        return None
    return ext.rotary_emb_pair_forward(x1, x2, freqs_cis)


def rotary_emb_pair_indexed_forward(
    x1: torch.Tensor, x2: torch.Tensor, freqs_cis: torch.Tensor, token_indices: torch.Tensor
) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
    ext = _load_extension()
    if ext is None:
        return None
    return ext.rotary_emb_pair_indexed_forward(x1, x2, freqs_cis, token_indices)


def rotary_cat_qkv_forward(
    img_q: torch.Tensor,
    img_k: torch.Tensor,
    img_v: torch.Tensor,
    txt_q: torch.Tensor,
    txt_k: torch.Tensor,
    txt_v: torch.Tensor,
    img_freqs: torch.Tensor,
    txt_freqs: torch.Tensor,
) -> Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    ext = _load_extension()
    if ext is None:
        return None
    return ext.rotary_cat_qkv_forward(img_q, img_k, img_v, txt_q, txt_k, txt_v, img_freqs, txt_freqs)


def modulate_forward(x: torch.Tensor, mod_params: torch.Tensor) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
    ext = _load_extension()
    if ext is None:
        return None
    return ext.modulate_forward(x, mod_params)


def modulate_indexed_forward(
    x: torch.Tensor, mod_params: torch.Tensor, index: torch.Tensor
) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
    ext = _load_extension()
    if ext is None:
        return None
    return ext.modulate_indexed_forward(x, mod_params, index)


def gated_residual_forward(base: torch.Tensor, gate: torch.Tensor, update: torch.Tensor) -> Optional[torch.Tensor]:
    ext = _load_extension()
    if ext is None:
        return None
    return ext.gated_residual_forward(base, gate, update)


def _rotary_emb_pytorch(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    x_rotated = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    x_out = torch.view_as_real(x_rotated * freqs_cis.unsqueeze(1)).flatten(3)
    return x_out.type_as(x)


def _rotary_emb_indexed_pytorch(x: torch.Tensor, freqs_cis: torch.Tensor, token_indices: torch.Tensor) -> torch.Tensor:
    return _rotary_emb_pytorch(x, freqs_cis.index_select(0, token_indices))


def _rotary_emb_pair_pytorch(
    x1: torch.Tensor, x2: torch.Tensor, freqs_cis: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    return _rotary_emb_pytorch(x1, freqs_cis), _rotary_emb_pytorch(x2, freqs_cis)


def _rotary_emb_pair_indexed_pytorch(
    x1: torch.Tensor, x2: torch.Tensor, freqs_cis: torch.Tensor, token_indices: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    freqs_selected = freqs_cis.index_select(0, token_indices)
    return _rotary_emb_pytorch(x1, freqs_selected), _rotary_emb_pytorch(x2, freqs_selected)


def _rotary_cat_qkv_pytorch(
    img_q: torch.Tensor,
    img_k: torch.Tensor,
    img_v: torch.Tensor,
    txt_q: torch.Tensor,
    txt_k: torch.Tensor,
    txt_v: torch.Tensor,
    img_freqs: torch.Tensor,
    txt_freqs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    img_q = _rotary_emb_pytorch(img_q, img_freqs)
    img_k = _rotary_emb_pytorch(img_k, img_freqs)
    txt_q = _rotary_emb_pytorch(txt_q, txt_freqs)
    txt_k = _rotary_emb_pytorch(txt_k, txt_freqs)
    joint_q = torch.cat([txt_q, img_q], dim=1)
    joint_k = torch.cat([txt_k, img_k], dim=1)
    joint_v = torch.cat([txt_v, img_v], dim=1)
    return joint_q, joint_k, joint_v


def _modulate_pytorch(x: torch.Tensor, mod_params: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    shift, scale, gate = mod_params.chunk(3, dim=-1)
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1), gate.unsqueeze(1)


def _modulate_indexed_pytorch(
    x: torch.Tensor, mod_params: torch.Tensor, index: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    shift, scale, gate = mod_params.chunk(3, dim=-1)
    actual_batch = shift.size(0) // 2
    shift_0, shift_1 = shift[:actual_batch], shift[actual_batch:]
    scale_0, scale_1 = scale[:actual_batch], scale[actual_batch:]
    gate_0, gate_1 = gate[:actual_batch], gate[actual_batch:]
    shift_result = torch.where(index == 0, shift_0.unsqueeze(1), shift_1.unsqueeze(1))
    scale_result = torch.where(index == 0, scale_0.unsqueeze(1), scale_1.unsqueeze(1))
    gate_result = torch.where(index == 0, gate_0.unsqueeze(1), gate_1.unsqueeze(1))
    return x * (1 + scale_result) + shift_result, gate_result


def _gated_residual_pytorch(base: torch.Tensor, gate: torch.Tensor, update: torch.Tensor) -> torch.Tensor:
    return base + gate * update


def _register_torch_ops():
    global _OPS_REGISTERED, _LIB_HANDLES
    if _OPS_REGISTERED:
        return

    lib_def = torch.library.Library(_OP_NAMESPACE, "FRAGMENT")
    lib_def.define("rotary_emb(Tensor x, Tensor freqs_cis) -> Tensor")
    lib_def.define("rotary_emb_indexed(Tensor x, Tensor freqs_cis, Tensor token_indices) -> Tensor")
    lib_def.define("rotary_emb_pair(Tensor x1, Tensor x2, Tensor freqs_cis) -> (Tensor, Tensor)")
    lib_def.define(
        "rotary_emb_pair_indexed(Tensor x1, Tensor x2, Tensor freqs_cis, Tensor token_indices) -> (Tensor, Tensor)"
    )
    lib_def.define(
        "rotary_cat_qkv("
        "Tensor img_q, Tensor img_k, Tensor img_v, "
        "Tensor txt_q, Tensor txt_k, Tensor txt_v, "
        "Tensor img_freqs, Tensor txt_freqs"
        ") -> (Tensor, Tensor, Tensor)"
    )
    lib_def.define("modulate(Tensor x, Tensor mod_params) -> (Tensor, Tensor)")
    lib_def.define("modulate_indexed(Tensor x, Tensor mod_params, Tensor index) -> (Tensor, Tensor)")
    lib_def.define("gated_residual(Tensor base, Tensor gate, Tensor update) -> Tensor")

    lib_cuda = torch.library.Library(_OP_NAMESPACE, "IMPL", "CUDA")

    def _rotary_cuda_impl(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        out = rotary_emb_forward(x, freqs_cis)
        if out is None:
            return _rotary_emb_pytorch(x, freqs_cis)
        return out

    def _rotary_indexed_cuda_impl(x: torch.Tensor, freqs_cis: torch.Tensor, token_indices: torch.Tensor) -> torch.Tensor:
        out = rotary_emb_indexed_forward(x, freqs_cis, token_indices)
        if out is None:
            return _rotary_emb_indexed_pytorch(x, freqs_cis, token_indices)
        return out

    def _rotary_pair_cuda_impl(
        x1: torch.Tensor, x2: torch.Tensor, freqs_cis: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        out = rotary_emb_pair_forward(x1, x2, freqs_cis)
        if out is None:
            return _rotary_emb_pair_pytorch(x1, x2, freqs_cis)
        return out

    def _rotary_pair_indexed_cuda_impl(
        x1: torch.Tensor, x2: torch.Tensor, freqs_cis: torch.Tensor, token_indices: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        out = rotary_emb_pair_indexed_forward(x1, x2, freqs_cis, token_indices)
        if out is None:
            return _rotary_emb_pair_indexed_pytorch(x1, x2, freqs_cis, token_indices)
        return out

    def _rotary_cat_qkv_cuda_impl(
        img_q: torch.Tensor,
        img_k: torch.Tensor,
        img_v: torch.Tensor,
        txt_q: torch.Tensor,
        txt_k: torch.Tensor,
        txt_v: torch.Tensor,
        img_freqs: torch.Tensor,
        txt_freqs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        out = rotary_cat_qkv_forward(img_q, img_k, img_v, txt_q, txt_k, txt_v, img_freqs, txt_freqs)
        if out is None:
            return _rotary_cat_qkv_pytorch(img_q, img_k, img_v, txt_q, txt_k, txt_v, img_freqs, txt_freqs)
        return out

    def _modulate_cuda_impl(x: torch.Tensor, mod_params: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = modulate_forward(x, mod_params)
        if out is None:
            return _modulate_pytorch(x, mod_params)
        return out

    def _modulate_indexed_cuda_impl(
        x: torch.Tensor, mod_params: torch.Tensor, index: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        out = modulate_indexed_forward(x, mod_params, index)
        if out is None:
            return _modulate_indexed_pytorch(x, mod_params, index)
        return out

    def _gated_residual_cuda_impl(base: torch.Tensor, gate: torch.Tensor, update: torch.Tensor) -> torch.Tensor:
        out = gated_residual_forward(base, gate, update)
        if out is None:
            return _gated_residual_pytorch(base, gate, update)
        return out

    lib_cuda.impl("rotary_emb", _rotary_cuda_impl)
    lib_cuda.impl("rotary_emb_indexed", _rotary_indexed_cuda_impl)
    lib_cuda.impl("rotary_emb_pair", _rotary_pair_cuda_impl)
    lib_cuda.impl("rotary_emb_pair_indexed", _rotary_pair_indexed_cuda_impl)
    lib_cuda.impl("rotary_cat_qkv", _rotary_cat_qkv_cuda_impl)
    lib_cuda.impl("modulate", _modulate_cuda_impl)
    lib_cuda.impl("modulate_indexed", _modulate_indexed_cuda_impl)
    lib_cuda.impl("gated_residual", _gated_residual_cuda_impl)

    lib_comp = torch.library.Library(_OP_NAMESPACE, "IMPL", "CompositeImplicitAutograd")
    lib_comp.impl("rotary_emb", _rotary_emb_pytorch)
    lib_comp.impl("rotary_emb_indexed", _rotary_emb_indexed_pytorch)
    lib_comp.impl("rotary_emb_pair", _rotary_emb_pair_pytorch)
    lib_comp.impl("rotary_emb_pair_indexed", _rotary_emb_pair_indexed_pytorch)
    lib_comp.impl("rotary_cat_qkv", _rotary_cat_qkv_pytorch)
    lib_comp.impl("modulate", _modulate_pytorch)
    lib_comp.impl("modulate_indexed", _modulate_indexed_pytorch)
    lib_comp.impl("gated_residual", _gated_residual_pytorch)

    # Keep Library handles alive for the process lifetime.
    _LIB_HANDLES.extend((lib_def, lib_cuda, lib_comp))
    _OPS_REGISTERED = True


_register_torch_ops()
