import argparse
import collections

import cv2
import numpy as np
import torch
import torch.nn
from torchvision import models
import matplotlib.pyplot as plt

from pytorch_grad_cam import GradCAM, \
                             ScoreCAM, \
                             GradCAMPlusPlus, \
                             AblationCAM, \
                             XGradCAM, \
                             EigenCAM, \
                             EigenGradCAM

from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image, \
                                         deprocess_image, \
                                         preprocess_image


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=False,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument('--image-path', type=str, default='./examples/both.png',
                        help='Input image path')
    parser.add_argument('--aug_smooth', action='store_true',
                        help='Apply test time augmentation to smooth the CAM')
    parser.add_argument('--eigen_smooth', action='store_true',
                        help='Reduce noise by taking the first principle componenet'
                        'of cam_weights*activations')
    parser.add_argument('--method', type=str, default='gradcam',
                        choices=['gradcam', 'gradcam++', 'scorecam', 'xgradcam',
                                 'ablationcam', 'eigencam', 'eigengradcam'],
                        help='Can be gradcam/gradcam++/scorecam/xgradcam'
                             '/ablationcam/eigencam/eigengradcam')

    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print('Using GPU for acceleration')
    else:
        print('Using CPU for computation')

    return args


class BaseROI:
    def __init__(self, image = None):
        self.image = image
        self.roi = 1
        self.fullroi = None
        self.i = None
        self.j = None

    def setROIij(self):
        print(f'Shape of ROI:{self.roi.shape}')
        self.i = np.where(self.roi == 1)[0]
        self.j = np.where(self.roi == 1)[1]
        print(f'Lengths of i and j index lists: {len(self.i)}, {len(self.j)}')

    def meshgrid(self):
        ylist = np.linspace(0, self.image.shape[0], self.image.shape[0])
        xlist = np.linspace(0, self.image.shape[1], self.image.shape[1])
        return np.meshgrid(xlist, ylist)

class PixelROI(BaseROI):
    def __init__(self, i, j, image):
        self.image = image
        self.roi = torch.zeros((image.shape[-3], image.shape[-2]))
        self.roi[i, j] = 1
        self.i = i
        self.j = j
#
# class ClassROI(BaseROI):
#     def __init__(self, model, image, cls):
#         preds = model.predict(np.expand_dims(image, 0))[0]
#         max_preds = preds.argmax(axis=-1)
#         self.image = image
#         self.roi = np.round(preds[..., cls] * (max_preds == cls)).reshape(image.shape[-3], image.shape[-2])
#         self.fullroi = self.roi
#         self.setROIij()
#
#     def connectedComponents(self, ignore=None):
#         _, all_labels = cv2.connectedComponents(self.fullroi)
#         # all_labels = measure.label(self.fullroi, background=0)
#
#
#         (values, counts) = np.unique(all_labels * (all_labels != 0), return_counts=True)
#         print("connectedComponents values, counts: ", values, counts)
#         return all_labels, values, counts
#
#     def largestComponent(self):
#         all_labels, values, counts = self.connectedComponents()
#         # find the largest component
#         ind = np.argmax(counts[values != 0]) + 1  # +1 because indexing starts from 0 for the background
#         print("argmax: ", ind)
#         # define RoI
#         self.roi = (all_labels == ind).astype(int)
#         self.setRoIij()
#
#     def smallestComponent(self):
#         all_labels, values, counts = self.connectedComponents()
#         ind = np.argmin(counts[values != 0]) + 1
#         print("argmin: ", ind)  #
#         self.roi = (all_labels == ind).astype(int)
#         self.setRoIij()


def get_output_tensor(output, verbose=True):
    if isinstance(output, torch.Tensor):
        return output
    elif isinstance(output, collections.OrderedDict):
        k = next(iter(output.keys()))
        if verbose: print(f'Select "{k}" from dict {output.keys()}')
        return output[k]
    elif isinstance(output, list):
        if verbose: print(f'Select "[0]" from list(n={len(output)})')
        return output[0]
    else:
        raise RuntimeError(f'Unknown type {type(output)}')

class SegModel(torch.nn.Module):
    def __init__(self, model, roi=None):
        super(SegModel, self).__init__()
        self.model = model
        self.roi = roi

    def forward(self, x):
        output = self.model(x)
        output = get_output_tensor(output)
        if self.roi is not None:
            output = output * self.roi.roi
        output = torch.sum(output, dim=(2, 3))
        return output

if __name__ == '__main__':
    """ python cam.py -image-path <path_to_image>
    Example usage of loading an image, and computing:
        1. CAM
        2. Guided Back Propagation
        3. Combining both
    """

    args = get_args()
    methods = \
        {"gradcam": GradCAM,
         "scorecam": ScoreCAM,
         "gradcam++": GradCAMPlusPlus,
         "ablationcam": AblationCAM,
         "xgradcam": XGradCAM,
         "eigencam": EigenCAM,
         "eigengradcam": EigenGradCAM}

    model = models.segmentation.fcn_resnet50(pretrained=True)

    # Choose the target layer you want to compute the visualization for.
    # Usually this will be the last convolutional layer in the model.
    # Some common choices can be:
    # Resnet18 and 50: model.layer4[-1]
    # VGG, densenet161: model.features[-1]
    # mnasnet1_0: model.layers[-1]
    # You can print the model to help chose the layer
    target_layer = model.backbone.layer4[-1]

    rgb_img = cv2.imread(args.image_path, 1)[:, :, ::-1]
    rgb_img = np.float32(rgb_img) / 255
    input_tensor = preprocess_image(rgb_img, mean=[0.485, 0.456, 0.406], 
                                             std=[0.229, 0.224, 0.225])

    segmodel = SegModel(model, roi=PixelROI(0, 130, rgb_img))

    cam = methods[args.method](model=segmodel,
                               target_layer=target_layer,
                               use_cuda=args.use_cuda)


    modeloutput = torch.argmax(get_output_tensor(model(input_tensor)), dim=1).squeeze(0)

    plt.matshow(modeloutput.cpu().numpy())
    plt.show()

    # If None, returns the map for the highest scoring category.
    # Otherwise, targets the requested category.
    target_category = 8

    # AblationCAM and ScoreCAM have batched implementations.
    # You can override the internal batch size for faster computation.
    cam.batch_size = 32

    grayscale_cam = cam(input_tensor=input_tensor,
                        target_category=target_category,
                        aug_smooth=args.aug_smooth,
                        eigen_smooth=args.eigen_smooth)

    # Here grayscale_cam has only one image in the batch
    grayscale_cam = grayscale_cam[0, :]

    cam_image = show_cam_on_image(rgb_img, grayscale_cam)

    gb_model = GuidedBackpropReLUModel(model=segmodel, use_cuda=args.use_cuda)
    gb = gb_model(input_tensor, target_category=target_category)

    cam_mask = cv2.merge([grayscale_cam, grayscale_cam, grayscale_cam])
    cam_gb = deprocess_image(cam_mask * gb)
    gb = deprocess_image(gb)

    plt.figure()
    plt.imshow(cam_image)
    plt.figure()
    plt.imshow(gb)
    plt.figure()
    plt.imshow(cam_gb)
    plt.show()

    # cv2.imwrite(f'{args.method}_cam.jpg', cam_image)
    # cv2.imwrite(f'{args.method}_gb.jpg', gb)
    # cv2.imwrite(f'{args.method}_cam_gb.jpg', cam_gb)
