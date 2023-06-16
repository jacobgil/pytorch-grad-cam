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

from torch import nn
import torch.nn.functional as F

import torchvision # You may need to install separately
from torchvision import models

from torch.profiler import profile, record_function, ProfilerActivity

import benchmark_functions

number_of_inputs = 1
model =  models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

# Just hard-coding a path for now
image_path = '~/image.jpg'
input_tensor = torchvision.io.read_image(image_path)

print(f'Benchmarking GradCAM using {number_of_inputs} image for ResNet50...')

# Run on CPU with profiler (save the profile to print later)
# print('Profile list of images on CPU...')
# with profile(activities=[ProfilerActivity.CPU], profile_memory=True, record_shapes=True) as prof:
#     cpu_profile_min_time, cpu_profile_max_time, cpu_profile_avg_time, _output_image = benchmark_functions.run_gradcam(model, number_of_inputs, batch_size=64, use_cuda=False)
# cpu_profile = prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=15)

# Run on CUDA with profiler (save the profile to print later)
print('Profile list of images on Cuda...')
with profile(activities=[ProfilerActivity.CPU], profile_memory=True, record_shapes=True) as prof:
    cuda_profile_min_time, cuda_profile_max_time, cuda_profile_avg_time, _output_image = benchmark_functions.run_gradcam(model, number_of_inputs, batch_size=64, use_cuda=True)
cuda_profile = prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=15)

# Run on CUDA with extra workflow
print('Profile list of images on Cuda and then run workflow...')
with profile(activities=[ProfilerActivity.CPU], profile_memory=True, record_shapes=True) as prof:
    cuda_profile_min_time, cuda_profile_max_time, cuda_profile_avg_time, _output_image = benchmark_functions.run_gradcam(model, number_of_inputs, batch_size=64, use_cuda=True, workflow_test=True)
work_flow_cuda_profile = prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=15)

# Run on CUDA with extra workflow
print('Profile list of images on Cuda and then run workflow with a simple CNN...')
model = benchmark_functions.SimpleCNN()
model.apply(benchmark_functions.xavier_uniform_init) # Randomise more weights
with profile(activities=[ProfilerActivity.CPU], profile_memory=True, record_shapes=True) as prof:
    cuda_profile_min_time, cuda_profile_max_time, cuda_profile_avg_time, _output_image = benchmark_functions.run_gradcam(model, number_of_inputs, batch_size=64, use_cuda=True, workflow_test=True)
simple_work_flow_cuda_profile = prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=15)

model =  models.resnet50()
# Run on CPU x1000 (get min, max, and avg times)
# print('Run list of images on CPU...')
# cpu_min_time, cpu_max_time, cpu_avg_time, _output_image = benchmark_functions.run_gradcam(model, number_of_inputs, batch_size=64, use_cuda=False)

# Run on CUDA x1000
print('Run list of images on Cuda...')
cuda_min_time, cuda_max_time, cuda_avg_time, _output_image = benchmark_functions.run_gradcam(model, number_of_inputs, batch_size=64, use_cuda=True)

# Run Workflow
print('Run list of images on Cuda with a workflow...')
workflow_cuda_min_time, workflow_cuda_max_time, workflow_cuda_avg_time, _output_image = benchmark_functions.run_gradcam(model, number_of_inputs, batch_size=64, use_cuda=True, workflow_test=True)

print('Run list of images on Cuda with a workflow using simple CNN...')
model = benchmark_functions.SimpleCNN()
model.apply(benchmark_functions.xavier_uniform_init) # Randomise more weights
simple_workflow_cuda_min_time, simple_workflow_cuda_max_time, simple_workflow_cuda_avg_time, output = benchmark_functions.run_gradcam(model, number_of_inputs, batch_size=64, use_cuda=True, workflow_test=True)

print('Complete!')

# print('==============================================================================\n\n')
# print('CPU Profile:\n')
# print(cpu_profile)

print('==============================================================================\n\n')
print('Cuda Profile:\n')
print(cuda_profile)

print('==============================================================================\n\n')
print('Workflow Cuda Profile:\n')
print(work_flow_cuda_profile)

print('==============================================================================\n\n')
print('Simple Workflow Cuda Profile:\n')
print(simple_work_flow_cuda_profile)

# print('==============================================================================\n\n')
# print('CPU Timing (No Profiler):\n')
# print(f'Min time: {cpu_min_time}\n')
# print(f'Max time: {cpu_max_time}\n')
# print(f'Avg time: {cpu_avg_time}\n')

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
print('Simple Workflow Cuda Timing (No Profiler):\n')
print(f'Min time: {simple_workflow_cuda_min_time}\n')
print(f'Max time: {simple_workflow_cuda_max_time}\n')
print(f'Avg time: {simple_workflow_cuda_avg_time}\n')

print('==============================================================================\n\n')
print('Output the resultant heat-map')
threshold_plot, output_image = output

benchmark_functions.save_image(threshold_plot.to("cpu", torch.uint8), '~/threshold.png')
benchmark_functions.save_image(output_image.to("cpu", torch.uint8), '~/output_image.png')

print('==============================================================================\n\n')
print('Done!')
