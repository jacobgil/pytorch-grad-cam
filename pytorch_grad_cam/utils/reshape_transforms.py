import torch


def fasterrcnn_reshape_transform(x):
    target_size = x['pool'].size()[-2:]
    activations = []
    for key, value in x.items():
        activations.append(
            torch.nn.functional.interpolate(
                torch.abs(value),
                target_size,
                mode='bilinear'))
    activations = torch.cat(activations, axis=1)
    return activations


def swinT_reshape_transform(tensor, height=7, width=7):
    result = tensor.reshape(tensor.size(0),
                            height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result


def vit_reshape_transform(tensor, height=14, width=14):
    result = tensor[:, 1:, :].reshape(tensor.size(0),
                                      height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result
