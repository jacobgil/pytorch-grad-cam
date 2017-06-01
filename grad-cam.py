import torch
from torchvision import models, transforms
import cv2
import sys
import numpy as np
from torch.autograd import Variable

class FeatureExtractor():
    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []
    
    def save_gradient(self, grad):
    	self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        for name, module in self.model._modules.items():
            x = module(x)
            if name in self.target_layers:
            	x.register_hook(self.save_gradient)
                outputs += [x]
        return outputs + [x]

class ModelOutputs():
	def __init__(self, model, target_layers):
		self.model = model
		self.feature_extractor = FeatureExtractor(self.model.features, target_layers)

	def get_gradients(self):
		return self.feature_extractor.gradients

	def __call__(self, x):
		features = self.feature_extractor(x)
		last_layer_output = features[-1]
		last_layer_output = last_layer_output.view(last_layer_output.size(0), -1)
		result = self.model.classifier(last_layer_output)
		return features, result

def preprocess_image(img):
	means=[0.485, 0.456, 0.406]
	stds=[0.229, 0.224, 0.225]

	preprocessed_img = img.copy()[: , :, ::-1]
	for i in range(3):
		preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
		preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
	preprocessed_img = np.transpose(preprocessed_img, (2, 0, 1))
	preprocessed_img = torch.from_numpy(np.ascontiguousarray(preprocessed_img))
	preprocessed_img.unsqueeze_(0)
	input = Variable(preprocessed_img, requires_grad = True)
	return input

def show_cam_on_image(img, mask):
	heatmap = cv2.applyColorMap(np.uint8(255*mask), cv2.COLORMAP_JET)
	heatmap = np.float32(heatmap) / 255
	cam = heatmap + np.float32(img)
	cam = cam / np.max(cam)
	cv2.imwrite("cam.jpg", np.uint8(255 * cam))

class GradCam:
	def __init__(self, model, target_layer_names):
		self.model = model
		self.extractor = ModelOutputs(self.model, target_layer_names)

	def forward(self, input):
		self.model.features.zero_grad()
		return self.model(input) 

	def get_highest_score_index(self, input):
		output = self.forward(input).data.numpy()
		return np.argmax(output)

	def __call__(self, input):
		index = self.get_highest_score_index(input)
		features, output = self.extractor(input)
		one_hot = np.zeros((1, 1000), dtype = np.float32)
		one_hot[0][index] = 1
		one_hot = Variable(torch.from_numpy(one_hot), requires_grad = True)
		one_hot = torch.sum(one_hot * output)

		one_hot.backward()
		grads_val = self.extractor.get_gradients()[-1].data.numpy()

		target = features[-1]
		target = target.data.numpy()[0, :]

		weights = np.mean(grads_val, axis = (2, 3))[0, :]
		cam = np.ones(target.shape[1 : ], dtype = np.float32)

		for i, w in enumerate(weights):
			cam += w * target[i, :, :]

		cam = np.maximum(cam, 0)
		cam = cv2.resize(cam, (224, 224))
		cam = cam - np.min(cam)
		cam = cam / np.max(cam)
		return cam


if __name__ == '__main__':
	grad_cam = GradCam(model = models.vgg19(pretrained=True), \
					target_layer_names = ["36"])

	img = cv2.imread(sys.argv[1], 1)
	img = np.float32(cv2.resize(img, (224, 224))) / 255
	input = preprocess_image(img)

	mask = grad_cam(input)

	show_cam_on_image(img, mask)