"""
Check that PyTorch sees the GPU, computes correctly on it, and compare SAC speed CPU vs GPU.
Run from repo root:  python -m experiments.check_gpu
"""
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents import SAC, ReplayBuffer  # noqa: E402


def device_report():
    print(f"torch {torch.__version__} | built with CUDA {torch.version.cuda} | "
          f"cuda available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("-> GPU NOT usable. Check: CUDA build of torch installed? NVIDIA driver working?")
        return False
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"GPU {i}: {p.name} | {p.total_memory / 2**30:.1f} GiB | compute capability {p.major}.{p.minor}")
    return True


def correctness_check():
    """Same matmul on CPU and GPU must agree."""
    torch.manual_seed(0)
    a, b = torch.randn(512, 512), torch.randn(512, 512)
    cpu = a @ b
    gpu = (a.cuda() @ b.cuda()).cpu()
    err = (cpu - gpu).abs().max().item()
    print(f"matmul CPU vs GPU max abs diff: {err:.2e}  ->  {'OK' if err < 1e-2 else 'MISMATCH'}")
    return err < 1e-2


def bench_sac(device, n=300, batch_size=256):
    buf = ReplayBuffer(9, 2, 10_000, seed=0)
    for _ in range(5_000):
        buf.add(np.random.randn(9), np.random.uniform(-1, 1, 2), 0.0, np.random.randn(9), 0)
    agent = SAC(9, 2, device=device, seed=0)
    assert all(p.device.type == device for p in agent.actor.parameters())
    for _ in range(30):                                   # warm-up (CUDA kernels, allocator)
        agent.act(np.random.randn(9))
        agent.update(buf.sample(batch_size))
    if device == "cuda":
        torch.cuda.synchronize()
    t = time.time()
    for _ in range(n):
        agent.act(np.random.randn(9))                     # one env step's action
        agent.update(buf.sample(batch_size))              # one gradient update
    if device == "cuda":
        torch.cuda.synchronize()
    return (time.time() - t) / n * 1000


if __name__ == "__main__":
    ok = device_report()
    if ok:
        ok = correctness_check()
    torch.set_num_threads(4)
    cpu_ms = bench_sac("cpu")
    print(f"SAC act+update  CPU (4 threads): {cpu_ms:6.1f} ms/step")
    if ok:
        gpu_ms = bench_sac("cuda")
        print(f"SAC act+update  GPU            : {gpu_ms:6.1f} ms/step  ->  "
              f"{cpu_ms / gpu_ms:.1f}x {'faster' if gpu_ms < cpu_ms else 'SLOWER'} on GPU")
        print(f"GPU memory allocated by this process: {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB")
