import config
import sys
import time
import torch
import torch.fx
from torch.profiler import profile, record_function, ProfilerActivity
from contextlib import nullcontext
import numpy as np

from utils import *

import horovod.torch as hvd

hvd.init()
torch.cuda.set_device(hvd.local_rank())

model = config.get_model(seed=39).cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())
train_data = config.get_data()[1]

hvd.broadcast_parameters(model.state_dict(), root_rank=0)

result_times = []
last_iter_time = time.time()
for iter in range(config.run_iter):
    optimizer.zero_grad()
    x, y = next(train_data)
    x = x.chunk(config.world_size, 0)[hvd.rank()].cuda()
    y = y.chunk(config.world_size, 0)[hvd.rank()].cuda()
    with torch.autocast(device_type="cuda") if config.fp16 else nullcontext() :
        loss = model(x, y)
    if hvd.local_rank() == 0:
        print(f"loss {iter}:", loss.detach().cpu().numpy())

    loss.backward()
    optimizer.step()
    # hvd.allreduce(torch.tensor(0), name='barrier')
    if hvd.local_rank() == 0:
        iter_duration = time.time() - last_iter_time
        print("iter time: ", iter_duration)
        result_times.append(iter_duration)
        print("avg±std:", np.mean(result_times[-config.avg_iter:]), np.std(result_times[-config.avg_iter:]))
        last_iter_time += iter_duration

print("peak memory", torch.cuda.max_memory_allocated())

if not config.trace:
    raise SystemExit

x, y = next(train_data)
x = x.chunk(config.world_size, 0)[hvd.rank()].cuda()
y = y.chunk(config.world_size, 0)[hvd.rank()].cuda()
with profile(
    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA],
    # record_shapes = True,
    # profile_memory = True,
    schedule = torch.profiler.schedule(wait=1, warmup=10, active=4)
) as prof:
    for _ in range(15):
        with record_function("forward"):
            loss = model(x, y)
        with record_function("backward"):
            loss.backward()
            # torch.cuda.synchronize()
        with record_function("update"):
            optimizer.step()
        hvd.allreduce(torch.tensor(0), name='barrier')
        prof.step()

if hvd.local_rank() == 0:
    # print(prof.key_averages().table(sort_by="cuda_time_total"))
    prof.export_chrome_trace("trace.json")
