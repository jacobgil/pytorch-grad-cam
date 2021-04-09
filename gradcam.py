import argparse
import cv2
import numpy as np
import torch
from torchvision import models

from pytorch_grad_cam import CAM, GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image, \
                                         deprocess_image

import torchvision.transforms


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=False,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument('--image-path', type=str, default='./examples/both.png',
                        help='Input image path')
    parser.add_argument('--method', type=str, default='gradcam',
                        help='Can be gradcam/gradcam++/scorecam')

    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print('Using GPU for acceleration')
    else:
        print('Using CPU for computation')

    return args

if __name__ == '__main__':
    """ python gradcam.py -image-path <path_to_image>
    Example usage of loading an image, and computing:
        1. CAM
        2. Guided Back Propagation
        3. Combining both
    """

    args = get_args()
    model = models.resnet50(pretrained=True)

    # Choose the target layer you want to compute the visualization for.
    # Usually this will be the last convolutional layer in the model.
    # Some common choices can be:
    # Resnet18 and 50: model.layer4[-1]
    # VGG, densenet161: model.features[-1]
    # mnasnet1_0: model.layers[-1]
    # You can print the model to help chose the layer
    target_layer = model.layer4[-1]

    # Standard imagenet transform, replace with anything else.
    # If you don't want any normalization, just use torchvision.transforms.ToTensor().
    normalize_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])])

    cam = CAM(model=model,
              transform=normalize_transform,
              target_layer=target_layer,
              use_cuda=args.use_cuda)

    rgb_img = cv2.imread(args.image_path, 1)[:, :, ::-1]
    rgb_img = np.float32(rgb_img) / 255

    # If None, returns the map for the highest scoring category.
    # Otherwise, targets the requested category.
    target_category = None

    grayscale_cam = cam(image=rgb_img,
                        method=args.method,
                        target_category=target_category)

    cam_image = show_cam_on_image(rgb_img, grayscale_cam)

    gb_model = GuidedBackpropReLUModel(model=model, use_cuda=args.use_cuda)
    input_tensor = normalize_transform(rgb_img).unsqueeze(0)
    gb = gb_model(input_tensor, target_category=target_category)

    cam_mask = cv2.merge([grayscale_cam, grayscale_cam, grayscale_cam])
    cam_gb = deprocess_image(cam_mask * gb)
    gb = deprocess_image(gb)

    cv2.imwrite(f'{args.method}_cam.jpg', cam_image)
    cv2.imwrite(f'{args.method}_gb.jpg', gb)
    cv2.imwrite(f'{args.method}_cam_gb.jpg', cam_gb)