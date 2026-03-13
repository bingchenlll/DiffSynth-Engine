#!/usr/bin/env python3
import argparse
import time

import torch

import diffsynth_engine.models.qwen_image.qwen_image_cuda_ext


def gated_residual_pytorch(base: torch.Tensor, gate: torch.Tensor, update: torch.Tensor) -> torch.Tensor:
    return base + gate * update


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark Qwen gated residual CUDA op vs PyTorch.")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--seq", type=int, default=4096)
    parser.add_argument("--dim", type=int, default=3072)
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--gate-seq", type=int, default=1, help="Gate sequence dim (1 or same as seq).")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
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
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run this benchmark.")
    if args.gate_seq not in (1, args.seq):
        raise ValueError("--gate-seq must be 1 or equal to --seq.")

    torch.manual_seed(args.seed)
    dtype = dtype_from_str(args.dtype)
    device = "cuda"

    base = torch.randn(args.batch, args.seq, args.dim, device=device, dtype=dtype).contiguous()
    gate = torch.randn(args.batch, args.gate_seq, args.dim, device=device, dtype=dtype).contiguous()
    update = torch.randn(args.batch, args.seq, args.dim, device=device, dtype=dtype).contiguous()

    atol = 3e-3 if dtype in (torch.float16, torch.bfloat16) else 1e-5
    rtol = 3e-3 if dtype in (torch.float16, torch.bfloat16) else 1e-5

    max_abs = 0.0
    max_rel = 0.0
    for _ in range(args.check_iters):
        y_ref = gated_residual_pytorch(base, gate, update)
        y_cuda = torch.ops.qwen_image_ext.gated_residual(base, gate, update)
        diff = (y_ref - y_cuda).abs()
        denom = y_ref.abs().clamp_min(1e-6)
        max_abs = max(max_abs, diff.max().item())
        max_rel = max(max_rel, (diff / denom).max().item())
        if not torch.allclose(y_ref, y_cuda, atol=atol, rtol=rtol):
            raise AssertionError(
                f"Correctness check failed: max_abs={max_abs:.6e}, max_rel={max_rel:.6e}, "
                f"atol={atol}, rtol={rtol}"
            )

    ref_fn = lambda: gated_residual_pytorch(base, gate, update)
    cuda_fn = lambda: torch.ops.qwen_image_ext.gated_residual(base, gate, update)
    ms_ref = benchmark(ref_fn, args.warmup, args.iters)
    ms_cuda = benchmark(cuda_fn, args.warmup, args.iters)
    speedup = ms_ref / ms_cuda

    print("=== Qwen Gated Residual Comparison ===")
    print(f"shape: B={args.batch}, S={args.seq}, D={args.dim}, gate_seq={args.gate_seq}, dtype={args.dtype}")
    print(f"correctness: PASS (max_abs={max_abs:.6e}, max_rel={max_rel:.6e})")
    print(f"pytorch: {ms_ref:.4f} ms/iter")
    print(f"cuda   : {ms_cuda:.4f} ms/iter")
    print(f"speedup: {speedup:.3f}x ({(1.0 - ms_cuda / ms_ref) * 100.0:+.2f}% vs pytorch)")


if __name__ == "__main__":
    main()
