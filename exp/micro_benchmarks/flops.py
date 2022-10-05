import config
import time
import sys
import datetime
import torch
import torch.fx
from torch.profiler import profile, record_function, ProfilerActivity
from contextlib import nullcontext
import numpy as np

import math

from annotator import annotate
from compiler import compile
from utils import *


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return x.matmul(y)

shape = {
    'x': (1024, 1024),
    'y': (1024, 1024)
}

model = symbolic_trace(Model())
annotate(model, shape)
print_annotated_graph(model.graph)

raise SystemExit

strategy = load(f"strategy_{config.model_name}")

from pprint import pprint
pprint(strategy)

print(model.code)
compile(model, strategy, global_rank=0, local_rank=0, world_size=config.world_size)
print(model.code)



with torch.no_grad():
    x = torch.randn(256, 1024, 1024, requires_grad=True).cuda(local_rank)
    y = torch.randn(2, 64, 64, requires_grad=True).cuda(local_rank)
    with profile(
        activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule = torch.profiler.schedule(wait=1, warmup=3, active=6)
    ) as prof:
        for i in range(10):
            with torch.cuda.stream(stream1):
                for _ in range(50):
                    x = x.matmul(x)
            if i % 2 == 0:
                with torch.cuda.stream(stream2):
                    for _ in range(10):
                        dist.all_reduce(y)
            with torch.cuda.stream(stream1):
                for _ in range(50):
                    x = x.matmul(x)
            stream1.synchronize()
            stream2.synchronize()

            # dist.barrier()
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
        mp.Process(target=run, args=(global_rank, local_rank)).start()

    for p in mp.active_children():
        p.join()
