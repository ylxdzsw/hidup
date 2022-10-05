import config
import time
import sys
import datetime
import torch
import torch.fx
from torch.profiler import profile, record_function, ProfilerActivity
from contextlib import nullcontext
import numpy as np

from torch.nn.parallel import DistributedDataParallel as DDP

from utils import *

def comp(global_rank, local_rank):
    import torch.distributed as dist
    dist.init_process_group('nccl', rank=global_rank, timeout=datetime.timedelta(hours=2))

    with torch.no_grad():
        x = torch.randn(128, 2, 20, 768, requires_grad=True).cuda(local_rank)
        w1 = torch.randn(2, 768, 768*4, requires_grad=True).cuda(local_rank)
        w2 = torch.randn(2, 768*4, 768, requires_grad=True).cuda(local_rank)
        with profile(
            activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule = torch.profiler.schedule(wait=2, warmup=10, active=20)
        ) as prof:
            for i in range(32):
                for _ in range(4):
                    t = torch.einsum("edh,becd->bech", w1, x)
                    x = torch.einsum("ehd,bech->becd", w2, t)
                torch.cuda.synchronize()
                prof.step()

    dist.barrier()

    if local_rank == 0:
        # print(prof.key_averages().table(sort_by="cuda_time_total"))
        prof.export_chrome_trace("trace.json")

def comm(global_rank, local_rank):
    import torch.distributed as dist
    dist.init_process_group('nccl', rank=global_rank, timeout=datetime.timedelta(hours=2))

    with torch.no_grad():
        x = torch.randn(8, 2, 20, 768, requires_grad=True).cuda(local_rank)
        w1 = torch.randn(2, 768, 768*4, requires_grad=True).cuda(local_rank)
        w2 = torch.randn(2, 768*4, 768, requires_grad=True).cuda(local_rank)
        with profile(
            activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule = torch.profiler.schedule(wait=2, warmup=10, active=20)
        ) as prof:
            for i in range(32):
                for _ in range(4):
                    t = torch.einsum("edh,becd->bech", w1, x)
                    x = torch.einsum("ehd,bech->becd", w2, t)
                torch.cuda.synchronize()
                prof.step()

    dist.barrier()

    if local_rank == 0:
        # print(prof.key_averages().table(sort_by="cuda_time_total"))
        prof.export_chrome_trace("trace.json")


if __name__ == '__main__':
    ranks = [ int(x) for x in sys.argv[1].split(',') ]

    if torch.cuda.device_count() != len(ranks):
        print("forget to set CUDA_VISIBLE_DEVICES")
        raise SystemExit

    import os
    os.environ['MASTER_ADDR'] = str(config.master_addr)
    os.environ['MASTER_PORT'] = str(config.master_port)
    os.environ['WORLD_SIZE'] = str(config.world_size)

    import torch.multiprocessing as mp
    mp.set_start_method('spawn')

    for local_rank, global_rank in enumerate(ranks):
        mp.Process(target=comp, args=(global_rank, local_rank)).start()

    for p in mp.active_children():
        p.join()
