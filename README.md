## Grad-CAM implementation in Pytorch ##

### What makes the network think the image label is 'pug, pug-dog' and 'tabby, tabby cat':
![Dog](https://github.com/jacobgil/pytorch-grad-cam/blob/master/examples/dog.jpg?raw=true) ![Cat](https://github.com/jacobgil/pytorch-grad-cam/blob/master/examples/cat.jpg?raw=true)

### Combining Grad-CAM with Guided Backpropagation for the 'pug, pug-dog' class:
![Combined](https://github.com/jacobgil/pytorch-grad-cam/blob/master/examples/cam_gb_dog.jpg?raw=true)

Gradient class activation maps are a visualization technique for deep learning networks.

See the paper: https://arxiv.org/pdf/1610.02391v1.pdf

The paper authors' torch implementation: https://github.com/ramprs/grad-cam

My Keras implementation: https://github.com/jacobgil/keras-grad-cam


----------

Tested with most of the torchvision models.
You need to choose the target layer to compute CAM for.
Some common choices can be:
- Resnet18 and 50: model.layer4[-1]
- densenet161: model.features[-1]
- mnasnet1_0: model.layers[-1]

----------

# Using from code

`pip install pytorch-grad-cam`

```python
from pytorch_grad_cam import GradCam
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.models import resnet50
model = resnet50(pretrained=True)
target_layer = model.layer4[-1]
input_tensor = # Create an input tensor image for you model..
grad_cam = GradCam(model=model, 
                   target_layer=target_layer,
                   use_cuda=args.use_cuda)
grayscale_cam = grad_cam(input_tensor=input_tensor, 
                         target_category=1)
cam = show_cam_on_image(rgb_img, grayscale_cam)
```

----------

# Running the example script:

Usage: `python gradcam.py --image-path <path_to_image>`

To use with CUDA:
`python gradcam.py --image-path <path_to_image> --use-cuda`