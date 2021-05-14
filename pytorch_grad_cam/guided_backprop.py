import numpy as np
import torch
from torch.autograd import Function


class GuidedBackpropReLU(Function):
    @staticmethod
    def forward(self, input_img):
        positive_mask = (input_img > 0).type_as(input_img)
        output = torch.addcmul(torch.zeros(input_img.size()).type_as(input_img), input_img, positive_mask)
        self.save_for_backward(input_img, output)
        return output

    @staticmethod
    def backward(self, grad_output):
        input_img, output = self.saved_tensors
        grad_input = None

        positive_mask_1 = (input_img > 0).type_as(grad_output)
        positive_mask_2 = (grad_output > 0).type_as(grad_output)
        grad_input = torch.addcmul(torch.zeros(input_img.size()).type_as(input_img),
                                   torch.addcmul(torch.zeros(input_img.size()).type_as(input_img), grad_output,
                                                 positive_mask_1), positive_mask_2)
        return grad_input


class GuidedBackpropReLUModel:
    def __init__(self, model, use_cuda):
        self.model = model
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = self.model.cuda()

    def forward(self, input_img):
        return self.model(input_img)

    def recursive_replace_relu_with_guidedrelu(self, module_top):
        for idx, module in module_top._modules.items():
            self.recursive_replace_relu_with_guidedrelu(module)
            if module.__class__.__name__ == 'ReLU':
                module_top._modules[idx] = GuidedBackpropReLU.apply

    def recursive_replace_guidedrelu_with_relu(self, module_top):
        try:
            for idx, module in module_top._modules.items():
                self.recursive_replace_guidedrelu_with_relu(module)
                if module == GuidedBackpropReLU.apply:
                    module_top._modules[idx] = torch.nn.ReLU()
        except:
            pass


    def __call__(self, input_img, target_category=None):
        # replace ReLU with GuidedBackpropReLU
        self.recursive_replace_relu_with_guidedrelu(self.model)

        if self.cuda:
            input_img = input_img.cuda()

        input_img = input_img.requires_grad_(True)

        output = self.forward(input_img)

        if target_category is None:
            target_category = np.argmax(output.cpu().data.numpy())

        loss = output[0, target_category]
        loss.backward(retain_graph=True)

        output = input_img.grad.cpu().data.numpy()
        output = output[0, :, :, :]
        output = output.transpose((1, 2, 0))

        # replace GuidedBackpropReLU back with ReLU
        self.recursive_replace_guidedrelu_with_relu(self.model)

        return output
