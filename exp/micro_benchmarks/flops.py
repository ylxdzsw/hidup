import config
import torch
import torch.fx
from torch.profiler import profile, record_function, ProfilerActivity

from annotator import annotate
from utils import *

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    # def forward(self, x):
    #     return torch.nn.functional.relu(x)

    def forward(self, x, y):
        return torch.nn.functional.bmm(x, y)


shape = {
    'x': (4, 1024, 1024),
    'y': (4, 1024, 1024)
}

model = symbolic_trace(Model())
annotate(model, shape)
print_annotated_graph(model.graph)

for node in model.graph.nodes:
    if "flops" in node.meta:
        print(node, node.meta["flops"])

with torch.no_grad():
    x = torch.randn(*shape['x']).cuda(0)
    y = torch.randn(*shape['y']).cuda(0)
    with profile(
        activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule = torch.profiler.schedule(wait=2, warmup=10, active=20)
    ) as prof:
        for i in range(32):
            for _ in range(10):
                # model(x)
                model(x, y)
            torch.cuda.synchronize()
            prof.step()

    prof.export_chrome_trace("trace.json")
