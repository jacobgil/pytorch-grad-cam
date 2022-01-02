# How does it work with Vision Transformers

*See [usage_examples/vit_example.py](../usage_examples/vit_example.py)*

In ViT the output of the layers are typically BATCH x 197 x 192.
In the dimension with 197, the first element represents the class token, and the rest represent the 14x14 patches in the image.
We can treat the last 196 elements as a 14x14 spatial image, with 192 channels.

To reshape the activations and gradients to 2D spatial images,
we can pass the CAM constructor a reshape_transform function.

This can also be a starting point for other architectures that will come in the future.

```python

GradCAM(model=model, target_layers=target_layers, reshape_transform=reshape_transform)

def reshape_transform(tensor, height=14, width=14):
    result = tensor[:, 1 :  , :].reshape(tensor.size(0),
        height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result
```

### Which target_layer should we chose for Vision Transformers?

Since the final classification is done on the class token computed in the last attention block,
the output will not be affected by the 14x14 channels in the last layer.
The gradient of the output with respect to them, will be 0!

We should chose any layer before the final attention block, for example:
```python
target_layers = [model.blocks[-1].norm1]
```

----------

# How does it work with Swin Transformers

*See [usage_examples/swinT_example.py](../usage_examples/swinT_example.py)*

In Swin transformer base the output of the layers are typically BATCH x 49 x 1024.
We can treat the last 49 elements as a 7x7 spatial image, with 1024 channels.

To reshape the activations and gradients to 2D spatial images,
we can pass the CAM constructor a reshape_transform function.

This can also be a starting point for other architectures that will come in the future.

```python

GradCAM(model=model, target_layers=target_layers, reshape_transform=reshape_transform)

def reshape_transform(tensor, height=7, width=7):
    result = tensor.reshape(tensor.size(0),
        height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result
```

### Which target_layer should we chose for Swin Transformers?

Since the swin transformer is different from ViT, it does not contains `cls_token` as present in ViT,
therefore we will use all the 7x7 images we get from the last block of the last layer.

We should chose any layer before the final attention block, for example:
```python
target_layers = [model.layers[-1].blocks[-1].norm1]
```
