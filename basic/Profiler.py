import torch
import torchvision.models as models
from torch.profiler import profile, record_function, ProfilerActivity, schedule

model = models.resnet18()
inputs = torch.randn(5, 3, 224, 224)

profile_schedule = schedule(skip_first=1,
                            wait=5,
                            warmup=1,
                            active=3,
                            repeat=2)


def trace_handler(prof):
    print(prof.key_averages(group_by_stack_n=5).table(sort_by="cpu_time_total", row_limit=10))
    # prof.export_chrome_trace("trace.json")


with profile(activities=[ProfilerActivity.CPU],
             schedule=profile_schedule,
             with_stack=True,
             profile_memory=True,
             record_shapes=True, on_trace_ready=trace_handler) as prof:
    with record_function("model_inference"):
        for i in range(20):
            model(inputs)
            prof.step()
