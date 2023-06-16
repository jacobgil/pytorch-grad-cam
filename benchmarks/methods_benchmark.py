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

number_of_inputs = 1000

print(f'Benchmarking GradCAM using {number_of_inputs} images for multiple models...')

methods_to_benchmark = [
    ['GradCAM', GradCAM],
    ['ScoreCAM', ScoreCAM],
    ['GradCAMPlusPlus', GradCAMPlusPlus],
    ['AblationCAM', AblationCAM],
    ['XGradCAM', XGradCAM],
    ['EigenCAM', EigenCAM],
    ['EigenGradCAM', EigenGradCAM],
    ['LayerCAM', LayerCAM],
    ['FullGrad', FullGrad]
]

model = benchmark_functions.SimpleCNN()
# model = models.resnet18()

model.apply(benchmark_functions.xavier_uniform_init) # Randomise more weights

for method_name, method in tqdm.tqdm(methods_to_benchmark):
    print('==============================================================================\n\n')
    print(f'Simple Workflow for method #{method_name}:\n')

    cpu_min_time, cpu_max_time, cpu_avg_time, _output_image = benchmark_functions.run_gradcam(model, number_of_inputs, batch_size=8, use_cuda=False, workflow_test=True, progress_bar=False, method=method)
    cuda_min_time, cuda_max_time, cuda_avg_time, _output_image = benchmark_functions.run_gradcam(model, number_of_inputs, batch_size=8, use_cuda=True, workflow_test=True, progress_bar=False, method=method)

    print(f'Cuda Min time: {cuda_min_time}\n')
    print(f'Cuda Max time: {cuda_max_time}\n')
    print(f'Cuda Avg time: {cuda_avg_time}\n\n')
    print(f'CPU Min time: {cpu_min_time}\n')
    print(f'CPU Max time: {cpu_max_time}\n')
    print(f'CPU Avg time: {cpu_avg_time}\n')

print('==============================================================================\n\n')
print('Done!')
