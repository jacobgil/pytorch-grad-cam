import torch

class AblationLayer(torch.nn.Module):
    def __init__(self, layer, indices):
        super(AblationLayer, self).__init__()

        self.layer = layer
        # The channels to zero out:
        self.indices = indices

    def __call__(self, x):
        output = self.layer(x)

        for i in range(output.size(0)):

            # Commonly the minimum activation will be 0,
            # And then it makes sense to zero it out.
            # However depending on the architecture,
            # If the values can be negative, we use very negative values
            # to perform the ablation, deviating from the paper.
            if torch.min(output) == 0:
                output[i, self.indices[i], :] = 0
            else:
                ABLATION_VALUE = 1e7
                output[i, self.indices[i], :] = torch.min(
                    output) - ABLATION_VALUE

        return output

class AblationLayerVit(torch.nn.Module):
    def __init__(self, layer, indices):
        super(AblationLayerVit, self).__init__()

        self.layer = layer
        # The channels to zero out:
        self.indices = indices

    def __call__(self, x):
        output = self.layer(x)

        for i in range(output.size(0)):

            # Commonly the minimum activation will be 0,
            # And then it makes sense to zero it out.
            # However depending on the architecture,
            # If the values can be negative, we use very negative values
            # to perform the ablation, deviating from the paper.
            if torch.min(output) == 0:
                output[i, self.indices[i], :] = 0
            else:
                ABLATION_VALUE = 1e7
                output[i, self.indices[i], :] = torch.min(
                    output) - ABLATION_VALUE

        output = output.transpose(2, 1)

        return output
