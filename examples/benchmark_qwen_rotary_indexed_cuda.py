#!/usr/bin/env python3
import argparse
import time

import torch

import diffsynth_engine.models.qwen_image.qwen_image_cuda_ext


def rotary_indexed_pytorch(x: torch.Tensor, freqs_cis: torch.Tensor, token_indices: torch.Tensor) -> torch.Tensor:
    x_rotated = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs_selected = freqs_cis.index_select(0, token_indices)
    x_out = torch.view_as_real(x_rotated * freqs_selected.unsqueeze(1)).flatten(3)
    return x_out.type_as(x)


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark indexed rotary CUDA op vs PyTorch index_select path.")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq", type=int, default=4096)
    parser.add_argument("--heads", type=int, default=24)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--freq-seq", type=int, default=8192)
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--check-iters", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def dtype_from_str(dtype_name: str) -> torch.dtype:
    if dtype_name == "bf16":
        return torch.bfloat16
    if dtype_name == "fp16":
        return torch.float16
    return torch.float32


def benchmark(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        _ = fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        _ = fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000.0 / iters


def main():
    args = parse_args()
    if args.dim % 2 != 0:
        raise ValueError("--dim must be even.")
    if args.freq_seq < args.seq:
        raise ValueError("--freq-seq must be >= --seq.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")

    torch.manual_seed(args.seed)
    device = "cuda"
    dtype = dtype_from_str(args.dtype)

    x = torch.randn(args.batch, args.seq, args.heads, args.dim, device=device, dtype=dtype).contiguous()
    phase = torch.randn(args.freq_seq, args.dim // 2, device=device, dtype=torch.float32)
    freqs = torch.polar(torch.ones_like(phase), phase).contiguous()
    token_indices = torch.randperm(args.freq_seq, device=device, dtype=torch.int64)[: args.seq].contiguous()

    atol = 3e-3 if dtype in (torch.float16, torch.bfloat16) else 1e-5
    rtol = 3e-3 if dtype in (torch.float16, torch.bfloat16) else 1e-5

    max_abs = 0.0
    max_rel = 0.0
    for _ in range(args.check_iters):
        y_ref = rotary_indexed_pytorch(x, freqs, token_indices)
        y_cuda = torch.ops.qwen_image_ext.rotary_emb_indexed(x, freqs, token_indices)
        diff = (y_ref - y_cuda).abs()
        denom = y_ref.abs().clamp_min(1e-6)
        max_abs = max(max_abs, diff.max().item())
        max_rel = max(max_rel, (diff / denom).max().item())
        if not torch.allclose(y_ref, y_cuda, atol=atol, rtol=rtol):
            raise AssertionError(
                f"Correctness check failed: max_abs={max_abs:.6e}, max_rel={max_rel:.6e}, "
                f"atol={atol}, rtol={rtol}"
            )

    ref_fn = lambda: rotary_indexed_pytorch(x, freqs, token_indices)
    cuda_fn = lambda: torch.ops.qwen_image_ext.rotary_emb_indexed(x, freqs, token_indices)
    ms_ref = benchmark(ref_fn, args.warmup, args.iters)
    ms_cuda = benchmark(cuda_fn, args.warmup, args.iters)
    speedup = ms_ref / ms_cuda

    print("=== Qwen Indexed Rotary Comparison ===")
    print(
        f"shape: B={args.batch}, S={args.seq}, H={args.heads}, D={args.dim}, "
        f"freq_seq={args.freq_seq}, dtype={args.dtype}"
    )
    print(f"correctness: PASS (max_abs={max_abs:.6e}, max_rel={max_rel:.6e})")
    print(f"pytorch(index_select): {ms_ref:.4f} ms/iter")
    print(f"cuda(indexed)       : {ms_cuda:.4f} ms/iter")
    print(f"speedup: {speedup:.3f}x ({(1.0 - ms_cuda / ms_ref) * 100.0:+.2f}% vs pytorch)")


if __name__ == "__main__":
    main()
