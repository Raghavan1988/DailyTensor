"""
Tensor-Parallel MLP with REAL collective communication (torch.distributed).

Program 01 *simulated* the parallelism in a single process.  This one spawns
`WORLD_SIZE` actual OS processes that talk to each other over the "gloo" CPU
backend, so you can see the genuine collective call -- dist.all_reduce -- that
production frameworks (Megatron-LM, vLLM, DeepSpeed, ...) issue on every layer.

Run it directly:   python 02_tensor_parallel_mlp_torch_dist.py
No GPUs required: gloo performs the all-reduce over localhost sockets.

Each rank:
  1. deterministically rebuilds the FULL weights (same seed on every rank),
  2. slices out only its own shard (columns of A, rows of B),
  3. computes its partial output with no communication,
  4. all-reduces (sums) the partial outputs, so every rank ends up holding the
     identical, complete result -- exactly like the single-device MLP.
"""

import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

D_MODEL, D_FF, BATCH = 8, 32, 4


def build_full_weights():
    # Every rank seeds identically, so they agree on the full weights before
    # slicing -- this stands in for loading the matching shard of a checkpoint.
    g = torch.Generator().manual_seed(0)
    X = torch.randn(BATCH, D_MODEL, generator=g)
    A = torch.randn(D_MODEL, D_FF, generator=g)   # column-parallel weight
    B = torch.randn(D_FF, D_MODEL, generator=g)   # row-parallel weight
    return X, A, B


def worker(rank, world_size):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    X, A, B = build_full_weights()
    shard = D_FF // world_size
    A_local = A[:, rank * shard:(rank + 1) * shard]   # this rank's columns of A
    B_local = B[rank * shard:(rank + 1) * shard, :]   # this rank's rows of B

    # local compute -- still no communication
    Z_local = torch.nn.functional.gelu(X @ A_local)   # [BATCH, D_FF/P]
    Y_partial = Z_local @ B_local                     # [BATCH, D_MODEL] partial

    # THE collective: sum every rank's partial into the full output, in place.
    # After this line every rank holds the same complete Y.
    dist.all_reduce(Y_partial, op=dist.ReduceOp.SUM)

    if rank == 0:
        reference = torch.nn.functional.gelu(X @ A) @ B
        err = (Y_partial - reference).abs().max().item()
        print(f"world_size={world_size}: max abs diff vs single device = {err:.2e}")
        print("Real dist.all_reduce reproduces the single-device MLP.")
    dist.destroy_process_group()


if __name__ == "__main__":
    world_size = 4
    # spawn() launches `world_size` fresh processes, each running worker().
    mp.spawn(worker, args=(world_size,), nprocs=world_size, join=True)
