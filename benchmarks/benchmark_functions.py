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

# Simple model to test
class SimpleCNN(nn.Module):
  def __init__(self):
    super(SimpleCNN, self).__init__()

    # Grad-CAM interface
    self.target_layer = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
    self.target_layers = [self.target_layer]
    self.layer4 = self.target_layer

    self.cnn_stack = nn.Sequential(
      nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
      nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
      nn.ReLU(inplace=True),
      self.target_layer,
      nn.ReLU(inplace=True),
      nn.MaxPool2d((2, 2)),
      nn.Flatten(),
      nn.Linear(122880, 10),
      nn.Linear(10, 1)
    )
    self.features = self.cnn_stack

  def forward(self, x):
    logits = self.cnn_stack(x)
    logits = F.normalize(logits, dim = 0)

    return logits

def xavier_uniform_init(layer):
  if type(layer) == nn.Linear or type(layer) == nn.Conv2d:
    gain = nn.init.calculate_gain('relu')

    if layer.bias is not None:
      nn.init.zeros_(layer.bias)

    nn.init.xavier_uniform_(layer.weight, gain=gain)

def last_cnn_layer(model):
  if hasattr(model, 'layer4'):
    return model.layer4

  if hasattr(model, 'conv3'):
    return model.conv3

  for feature in model.features:
    if isinstance(feature, nn.Conv2d):
      return feature

  return None

# Code to run benchmark
def run_gradcam(model, number_of_inputs, batch_size=1, use_cuda=False, workflow_test=False, progress_bar=True):
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
    target_layers = [last_cnn_layer(model)] # Last CNN layer of ResNet50

    cam_function = GradCAM(model=model, target_layers=target_layers, use_cuda=use_cuda)
    cam_function.batch_size = batch_size

    pbar = tqdm.tqdm(total=number_of_inputs)

    for i in range(0, number_of_inputs, batch_size):
        start_time = time.time()

        # Actual code to benchmark
        input_image = input_tensor[i:i+batch_size].to(dev)
        heatmap = cam_function(input_tensor=input_image, targets=targets)

        if workflow_test:
            for j in range(heatmap.shape[0]):
                # Create a binary map
                threshold_plot = torch.where(torch.tensor(heatmap[j]).to(torch.device('cuda:0')) > 0.5, 1, 0).to(dev)
                output_image = input_image * threshold_plot

        end_time = time.time()
        time_difference = end_time - start_time

        sum_of_times += time_difference

        if time_difference > max_time:
            max_time = time_difference

        if time_difference < min_time:
            min_time = time_difference

        if progress_bar:
          pbar.update(batch_size)

    avg_time = sum_of_times / number_of_inputs
    return [min_time, max_time, avg_time]
