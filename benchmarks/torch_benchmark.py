import argparse
import cv2
import numpy as np
import torch
import time
import tqdm

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

print(f'Benchmarking GradCAM using {number_of_inputs} images for ResNet50...')

def run_gradcam(model, number_of_inputs, batch_size=8, use_cuda=False, workflow_test=False):
    min_time = 10000000000000
    max_time = 0
    sum_of_times = 0

    dev = torch.device('cpu')
    if use_cuda:
        dev = torch.device('cuda:0')

    # TODO: Use real data?
    # TODO: Configurable dimensions?

    # Some defaults I use in research code
    input_tensor = torch.rand((number_of_inputs, 3, 256, 60))
    targets = None # [ClassifierOutputTarget(None)]

    model.to(dev)
    target_layers = [model.layer4] # Last CNN layer of ResNet50

    cam_function = GradCAM(model=model, target_layers=target_layers, use_cuda=use_cuda)
    cam_function.batch_size = batch_size

    pbar = tqdm.tqdm(total=number_of_inputs)

    for i in range(0, number_of_inputs, batch_size):
        start_time = time.time()

        # Actual code to benchmark
        input_image = input_tensor[i:i+batch_size].to(dev)
        heatmap = cam_function(input_tensor=input_image, targets=targets)

        if workflow_test:
            for j in range(batch_size):
                # Create a binary map
                threshold_plot = torch.where(torch.tensor(heatmap[j]) > 0.5, 1, 0)
                output_image = input_image * threshold_plot

        end_time = time.time()
        time_difference = end_time - start_time

        sum_of_times += time_difference

        if time_difference > max_time:
            max_time = time_difference

        if time_difference < min_time:
            min_time = time_difference

        pbar.update(batch_size)

    avg_time = sum_of_times / number_of_inputs
    return [min_time, max_time, avg_time]

# TODOs:
# Test with numpy v1.4.6 (master)
# Test with torch v1.4.7 (wip)
# Test other CAMs besides GradCAM
# Nice output

# Run on CPU with profiler (save the profile to print later)
print('Profile list of images on CPU...')
with profile(activities=[ProfilerActivity.CPU], profile_memory=True, record_shapes=True) as prof:
    cpu_profile_min_time, cpu_profile_max_time, cpu_profile_avg_time = run_gradcam(model, number_of_inputs, batch_size=64, use_cuda=False)
cpu_profile = prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=15)

# Run on CUDA with profiler (save the profile to print later)
print('Profile list of images on Cuda...')
with profile(activities=[ProfilerActivity.CPU], profile_memory=True, record_shapes=True) as prof:
    cuda_profile_min_time, cuda_profile_max_time, cuda_profile_avg_time = run_gradcam(model, number_of_inputs, batch_size=64, use_cuda=True)
cuda_profile = prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=15)

# Run on CUDA with extra workflow
print('Profile list of images on Cuda and then run workflow...')
with profile(activities=[ProfilerActivity.CPU], profile_memory=True, record_shapes=True) as prof:
    cuda_profile_min_time, cuda_profile_max_time, cuda_profile_avg_time = run_gradcam(model, number_of_inputs, batch_size=64, use_cuda=True, workflow_test=True)
work_flow_cuda_profile = prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=15)

# Run on CPU x1000 (get min, max, and avg times)
print('Run list of images on CPU...')
cpu_min_time, cpu_max_time, cpu_avg_time = run_gradcam(model, number_of_inputs, batch_size=64, use_cuda=False)

# Run on CUDA x1000
print('Run list of images on Cuda...')
cuda_min_time, cuda_max_time, cuda_avg_time = run_gradcam(model, number_of_inputs, batch_size=64, use_cuda=True)

# Run Workflow
print('Run list of images on Cuda with a workflow...')
workflow_cuda_min_time, workflow_cuda_max_time, workflow_cuda_avg_time = run_gradcam(model, number_of_inputs, batch_size=64, use_cuda=True, workflow_test=True)

print('Complete!')

print('==============================================================================\n\n')
print('CPU Profile:\n')
print(cpu_profile)

print('==============================================================================\n\n')
print('Cuda Profile:\n')
print(cuda_profile)

print('==============================================================================\n\n')
print('Workflow Cuda Profile:\n')
print(work_flow_cuda_profile)

print('==============================================================================\n\n')
print('CPU Timing (No Profiler):\n')
print(f'Min time: {cpu_min_time}\n')
print(f'Max time: {cpu_max_time}\n')
print(f'Avg time: {cpu_avg_time}\n')

print('==============================================================================\n\n')
print('Cuda Timing (No Profiler):\n')
print(f'Min time: {cuda_min_time}\n')
print(f'Max time: {cuda_max_time}\n')
print(f'Avg time: {cuda_avg_time}\n')

print('==============================================================================\n\n')
print('Workflow Cuda Timing (No Profiler):\n')
print(f'Min time: {workflow_cuda_min_time}\n')
print(f'Max time: {workflow_cuda_max_time}\n')
print(f'Avg time: {workflow_cuda_avg_time}\n')

print('==============================================================================\n\n')
print('Done!')
