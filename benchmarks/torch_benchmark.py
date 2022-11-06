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
random_tensor = torch.rand((256, 60, 3)) # TODO: Use real data?

# Test with numpy v1.4.6 (master)
# Test with torch v1.4.7 (wip)

# Run on CPU with profiler (save the profile to print later)
dev = torch.device('cpu')
use_cuda = False
model.to(dev)
target_layers = [model.blocks[-1].norm1]

with profile(activities=[ProfilerActivity.CPU], profile_memory=True, record_shapes=True) as prof:
    GradCAM(model=model, target_layers=target_layers, use_cuda=use_cuda)
print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=15))
breakpoint() # For now as I write this

# Run on CUDA with profiler (save the profile to print later)

# Run on CPU x100 (get min, max, and avg times)

# Run on CUDA x100
