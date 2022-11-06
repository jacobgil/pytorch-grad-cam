import argparse
import cv2
import numpy as np
import torch
import time

from pytorch_grad_cam import GradCAM, \
    ScoreCAM, \
    GradCAMPlusPlus, \
    AblationCAM, \
    XGradCAM, \
    EigenCAM, \
    EigenGradCAM, \
    LayerCAM, \
    FullGrad

import torchvision # You may need to install separately
from torchvision import models

from torch.profiler import profile, record_function, ProfilerActivity

number_of_inputs = 1000
model =  models.resnet50()
input_tensor = torch.rand((number_of_inputs, 3, 256, 60)) # TODO: Use real data?

# TODOs:
# Test with numpy v1.4.6 (master)
# Test with torch v1.4.7 (wip)
# Test other CAMs besides GradCAM
# Nice output

# Run on CPU with profiler (save the profile to print later)
dev = torch.device('cpu')
use_cuda = False

model.to(dev)
input_tensor.to(dev)

# Some defaults I use in research code
target_layers = [model.layer4]
batch_size = 8
targets = None # [ClassifierOutputTarget(None)]

# Profile the CPU call
with profile(activities=[ProfilerActivity.CPU], profile_memory=True, record_shapes=True) as prof:
    cam_function = GradCAM(model=model, target_layers=target_layers, use_cuda=use_cuda)
    cam_function.batch_size = batch_size
    heatmap = cam_function(input_tensor=input_tensor, targets=targets)

cpu_profile = prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=15)

# Run on CUDA with profiler (save the profile to print later)
dev = torch.device('cuda')
use_cuda = True

model.to(dev)
input_tensor.to(dev)

with profile(activities=[ProfilerActivity.CPU], profile_memory=True, record_shapes=True) as prof:
    cam_function = GradCAM(model=model, target_layers=target_layers, use_cuda=use_cuda)
    cam_function.batch_size = batch_size
    heatmap = cam_function(input_tensor=input_tensor, targets=targets)

cuda_profile = prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=15)

# Run on CPU x1000 (get min, max, and avg times)
cpu_min_time = 10000000000000
cpu_max_time = 0
cpu_sum_of_times = 0

for i in range(number_of_inputs):
    start_time = time.time()

    input_tensor = torch.rand((number_of_inputs, 3, 256, 60)) # TODO: Use real data?

    dev = torch.device('cpu')
    use_cuda = False

    model.to(dev)
    input_tensor.to(dev)

    # Some defaults I use in research code
    target_layers = [model.layer4]
    batch_size = 8
    targets = None # [ClassifierOutputTarget(None)]

    cam_function = GradCAM(model=model, target_layers=target_layers, use_cuda=use_cuda)
    cam_function.batch_size = batch_size
    heatmap = cam_function(input_tensor=input_tensor, targets=targets)

    end_time = time.time()
    time_difference = end_time - start_time

    cpu_sum_of_times += time_difference

    if time_difference > cpu_max_time:
        cpu_max_time = time_difference

    if time_difference < cpu_min_time:
        cpu_min_time = time_difference

cpu_avg_time = cpu_sum_of_times / number_of_inputs
breakpoint()
# Run on CUDA x1000
