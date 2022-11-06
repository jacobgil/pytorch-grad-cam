import argparse
import cv2
import numpy as np
import torch

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

model =  models.resnet50()
input_tensor = torch.rand((1, 3, 256, 60)) # TODO: Use real data?

# TODOs:
# Test with numpy v1.4.6 (master)
# Test with torch v1.4.7 (wip)
# Test other CAMs besides GradCAM

# Run on CPU with profiler (save the profile to print later)
dev = torch.device('cpu')
use_cuda = False

model.to(dev)
input_tensor.to(dev)

# Some defaults I use in research code
target_layers = [model.fc]
batch_size = 8
targets = None # [ClassifierOutputTarget(None)]

# Profile the CPU call
with profile(activities=[ProfilerActivity.CPU], profile_memory=True, record_shapes=True) as prof:
    cam_function = GradCAM(model=model, target_layers=target_layers, use_cuda=use_cuda)
    cam_function.batch_size = batch_size
    heatmap = cam_function(input_tensor=input_tensor, targets=targets)

print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=15))
breakpoint() # For now as I write this

# Run on CUDA with profiler (save the profile to print later)

# Run on CPU x100 (get min, max, and avg times)

# Run on CUDA x100
