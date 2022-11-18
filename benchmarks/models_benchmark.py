import argparse
import cv2
import numpy as np
import torch
import time
import tqdm

from pytorch_grad_cam import GradCAM

from torch import nn
import torch.nn.functional as F

import torchvision # You may need to install separately
from torchvision import models

from torch.profiler import profile, record_function, ProfilerActivity

import benchmark_functions

number_of_inputs = 1000

print(f'Benchmarking GradCAM using {number_of_inputs} images for multiple models...')

models_to_benchmark = [
    ["SimpleCNN", benchmark_functions.SimpleCNN()],
    ["resnet18", models.resnet18()],
    ["resnet34", models.resnet34()],
    ["resnet50", models.resnet50()],
    ["alexnet", models.alexnet()],
    ["vgg16", models.vgg16()],
    ["googlenet", models.googlenet()],
    ["mobilenet_v2", models.mobilenet_v2()],
    ["densenet161", models.densenet161()]
]

for model_name, model in tqdm.tqdm(models_to_benchmark):
    print('==============================================================================\n\n')
    print(f'Simple Workflow for model #{model_name}:\n')

    model.apply(benchmark_functions.xavier_uniform_init) # Randomise more weights
    cpu_min_time, cpu_max_time, cpu_avg_time = benchmark_functions.run_gradcam(model, number_of_inputs, batch_size=8, use_cuda=False, workflow_test=True)
    cuda_min_time, cuda_max_time, cuda_avg_time = benchmark_functions.run_gradcam(model, number_of_inputs, batch_size=8, use_cuda=True, workflow_test=True)

    print(f'Cuda Min time: {cuda_min_time}\n')
    print(f'Cuda Max time: {cuda_max_time}\n')
    print(f'Cuda Avg time: {cuda_avg_time}\n\n')
    print(f'CPU Min time: {cpu_min_time}\n')
    print(f'CPU Max time: {cpu_max_time}\n')
    print(f'CPU Avg time: {cpu_avg_time}\n')


print('==============================================================================\n\n')
print('Done!')
