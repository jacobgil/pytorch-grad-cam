## Grad-CAM implementation in Pytorch ##

### What makes the network think the image label is 'pug, pug-dog' and 'tabby, tabby cat':
![Dog](https://github.com/jacobgil/pytorch-grad-cam/blob/master/examples/dog.jpg?raw=true) ![Cat](https://github.com/jacobgil/pytorch-grad-cam/blob/master/examples/cat.jpg?raw=true)

Gradient class activation maps are a visualization technique for deep learning networks.

See the paper: https://arxiv.org/pdf/1610.02391v1.pdf

The paper authors torch implementation: https://github.com/ramprs/grad-cam

My Keras implementation: https://github.com/jacobgil/keras-grad-cam


----------


This uses VGG19 from torchvision. It will be downloaded when used for the first time.

The code can be modified to work with any model.
However the VGG models in torchvision have features/classifier methods for the convolutional part of the network, and the fully connected part.
This code assumes that the model passed supports these two methods.


----------


Usage: `python grad-cam.py --image-path <path_to_image>`

To use with CUDA:
`python grad-cam.py --image-path <path_to_image> --use-cuda`
