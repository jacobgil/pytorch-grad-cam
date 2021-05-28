import numpy as np
import torch
import matplotlib.pyplot as plt
from skimage import measure
import collections

def gui_get_point(image, i=None, j=None):
    fig = plt.figure('Input Pick An Point')
    scale = np.mean(image.shape[:2])
    if len(image.shape)==3:
        pImg = plt.imshow(image)
    else:
        pImg = plt.matshow(image)

    pMarker = plt.scatter(j, i, c='r', s=scale, marker='x')
    ret = plt.ginput(1)
    if ret == None or ret == []:
        pass
    else:
        j, i = ret[0]
        j, i= int(j), int(i)
    pMarker.remove()
    pMarker = plt.scatter(j, i, c='r', s=scale, marker='x')
    plt.close(fig)
    return i, j

class BaseROI:
    def __init__(self, image = None):
        self.image = image
        self.roi = torch.Tensor([1])
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

    def apply_roi(self, output):
        return self.roi.to(output.device) * output


class PixelROI(BaseROI):
    def __init__(self, i, j, image):
        self.image = image
        self.roi = torch.zeros((image.shape[-3], image.shape[-2]))
        self.roi[i, j] = 1
        self.i = i
        self.j = j

    def pickPixel(self):
        self.i, self.j = gui_get_point(self.image, self.i, self.j)
        self.roi.zero_()
        self.roi[self.i, self.j] = 1
        print(f'ROI Point: {self.i},{self.j}')

def filter_connected_components(values, counts, exclude):
    selected_indices=[]
    selected_counts=[]
    selected_values=[]
    for i in range(len(values)):
        if values[i] != exclude:
            selected_indices.append([i])
            selected_values.append(values[i])
            selected_counts.append(counts[i])
    return selected_indices, selected_values, selected_counts

class ClassROI(BaseROI):
    def __init__(self, image, pred, cls, background=0):
        self.image = image
        self.pred = pred
        self.cls = cls
        self.roi = (pred == cls).reshape(image.shape[-3], image.shape[-2])
        self.background = background
        print(f'Valid ROI pixels: {torch.sum(self.roi).numpy()} of class {self.cls}')

    def connectedComponents(self):
        all_labels = measure.label(self.pred)
        (values, counts) = np.unique(all_labels, return_counts=True)
        print("connectedComponents values, counts: ", values, counts)
        return all_labels, values, counts

    def largestComponent(self):
        all_labels, values, counts = self.connectedComponents()
        # find the largest component
        selected_indices, selected_values, selected_counts = filter_connected_components(values,
                                                                                         counts,
                                                                                         self.background)
        ind = selected_indices[np.argmax(selected_counts)]
        print("largestComponent argmax: ", ind)
        self.roi = torch.Tensor(all_labels == ind)
        print(f'Valid ROI pixels: {torch.sum(self.roi).numpy()} of class {values[ind]}')

    def smallestComponent(self):
        all_labels, values, counts = self.connectedComponents()
        selected_indices, selected_values, selected_counts = filter_connected_components(values,
                                                                                         counts,
                                                                                         self.background)
        ind = selected_indices[np.argmin(selected_counts)]
        print("smallestComponent argmin: ", ind)
        self.roi = torch.Tensor(all_labels == ind)
        print(f'Valid ROI pixels: {torch.sum(self.roi).numpy()} of class {values[ind]}')

    def pickClass(self):
        i, j = gui_get_point(self.pred)
        self.cls = self.pred[i, j]
        self.roi = (self.pred == self.cls).reshape(self.image.shape[-3], self.image.shape[-2])
        print(f'Valid ROI pixels: {torch.sum(self.roi).numpy()} of class {self.cls}')

    def pickComponentClass(self):
        i, j = gui_get_point(self.pred)
        all_labels, values, counts = self.connectedComponents()
        ind = all_labels[i, j]
        self.cls = all_labels[i, j]
        self.roi = torch.Tensor(all_labels == ind).reshape(self.image.shape[-3], self.image.shape[-2])
        print(f'Valid ROI pixels: {torch.sum(self.roi).numpy()} of class {self.cls}')

# Get tensor from output of network. Some segmentation network returns more than 1 tensor.
def get_output_tensor(output, verbose=False):
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
        output = self.model(x) # might be multiple tensors
        output = get_output_tensor(output) # Ensure only one tensor

        N = output.shape[-3]
        if N == 1: # if the original problem is binary using sigmoid, change to one-hot style.
            output = torch.log_softmax([-output, output], dim=-3)

        if self.roi is not None:
            output = self.roi.apply_roi(output)
        output = torch.sum(output, dim=(-2, -1))
        return output

