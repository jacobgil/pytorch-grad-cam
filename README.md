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

This uses Resnet50, Vgg16, Vgg19 from torchvision. It will be downloaded when used for the first time.
The code can be modified to work with any model.

----------


Usage: `python gradcam.py --image-path <path_to_image> --model {resnet50, vgg16, vgg19}`

To use with CUDA:
`python gradcam.py --image-path <path_to_image> --model {resnet50, vgg16, vgg19} --use-cuda`