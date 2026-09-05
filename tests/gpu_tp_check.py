"""Exercise the installed CUDA/NCCL stack on four devices without model weights."""
import os
import tempfile
import time
from datetime import timedelta
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def worker(rank, init):
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl', init_method=init, rank=rank, world_size=4,
                            timeout=timedelta(seconds=60))
    x = torch.full((24576, 512), rank + 1., dtype=torch.bfloat16, device='cuda')
    start = time.monotonic()
    dist.all_reduce(x)
    torch.cuda.synchronize()
    assert x[0,0].item() == 10
    y = torch.nn.functional.silu(x[:, :256]) * x[:, 256:]
    torch.cuda.synchronize()
    assert torch.isfinite(y).all()
    print(f'rank={rank} PASS all_reduce+silu seconds={time.monotonic()-start:.3f}', flush=True)
    dist.destroy_process_group()


if __name__ == '__main__':
    with tempfile.TemporaryDirectory() as td:
        mp.spawn(worker, args=('file://' + os.path.join(td, 'nccl'),), nprocs=4)
