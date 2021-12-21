import argparse
import cv2
import numpy as np
import torch
import torchvision
from pytorch_grad_cam import GradCAM, \
    ScoreCAM, \
    GradCAMPlusPlus, \
    AblationCAM, \
    XGradCAM, \
    EigenCAM, \
    EigenGradCAM, \
    ActivationsAndGradients

from pytorch_grad_cam.utils.image import show_cam_on_image, scale_accross_batch_and_channels, scale_cam_image
import sys

coco_names = [
    '__background__',
    'person',
    'bicycle',
    'car',
    'motorcycle',
    'airplane',
    'bus',
    'train',
    'truck',
    'boat',
    'traffic light',
    'fire hydrant',
    'N/A',
    'stop sign',
    'parking meter',
    'bench',
    'bird',
    'cat',
    'dog',
    'horse',
    'sheep',
    'cow',
    'elephant',
    'bear',
    'zebra',
    'giraffe',
    'N/A',
    'backpack',
    'umbrella',
    'N/A',
    'N/A',
    'handbag',
    'tie',
    'suitcase',
    'frisbee',
    'skis',
    'snowboard',
    'sports ball',
    'kite',
    'baseball bat',
    'baseball glove',
    'skateboard',
    'surfboard',
    'tennis racket',
    'bottle',
    'N/A',
    'wine glass',
    'cup',
    'fork',
    'knife',
    'spoon',
    'bowl',
    'banana',
    'apple',
    'sandwich',
    'orange',
    'broccoli',
    'carrot',
    'hot dog',
    'pizza',
    'donut',
    'cake',
    'chair',
    'couch',
    'potted plant',
    'bed',
    'N/A',
    'dining table',
    'N/A',
    'N/A',
    'toilet',
    'N/A',
    'tv',
    'laptop',
    'mouse',
    'remote',
    'keyboard',
    'cell phone',
    'microwave',
    'oven',
    'toaster',
    'sink',
    'refrigerator',
    'N/A',
    'book',
    'clock',
    'vase',
    'scissors',
    'teddy bear',
    'hair drier',
    'toothbrush']

# this will help us create a different color for each class
COLORS = np.random.uniform(0, 255, size=(len(coco_names), 3))


def draw_boxes(boxes, labels, classes, image):
    # read the image with OpenCV
    #image = cv2.cvtColor(np.asarray(image), cv2.COLOR_BGR2RGB)
    for i, box in enumerate(boxes):
        color = COLORS[labels[i]]
        cv2.rectangle(
            image,
            (int(box[0]), int(box[1])),
            (int(box[2]), int(box[3])),
            color, 2
        )
        cv2.putText(image, classes[i], (int(box[0]), int(box[1] - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2,
                    lineType=cv2.LINE_AA)
    return image

class FasterRCNNBoxScoreTarget:
    def __init__(self, labels, bounding_boxes, iou_threshold=0.9):
        self.labels = labels
        self.bounding_boxes = bounding_boxes
        self.iou_threshold = iou_threshold

    def __call__(self, model_outputs):
        output = 0
        for box, label in zip(self.bounding_boxes, self.labels):
            box = torch.Tensor(box[None, :])
            if torch.cuda.is_available():
                box = box.cuda()

            ious = torchvision.ops.box_iou(box, model_outputs["boxes"]).cpu().numpy()
            index = ious.argmax()

            if ious[0, index] > self.iou_threshold and model_outputs["labels"][index] == label:
                score = model_outputs["scores"][index]
                output = output + score
        return output

class AblationLayerFasterRCNN(torch.nn.Module):
    def __init__(self, layer, indices):
        super(AblationLayerFasterRCNN, self).__init__()

        self.layer = layer
        self.indices = indices

    def __call__(self, x):
        print(result['pool'])
        result = self.layer(x)
        layers = {0: '0', 1: '1', 2:'2', 3:'3', 4:'pool'}
        for i in range(result['pool'].size(0)):
            pyramid_layer = int(self.indices[i]/256)
            index_in_pyramid_layer = int(self.indices[i] % 256)
            result[layers[pyramid_layer]] [i, index_in_pyramid_layer, :, :] = 0
        return result


def reshape_transform(x):
    target_size = x['pool'].size()[-2 : ]
    activations = []
    for key, value in x.items():
        activations.append(torch.nn.functional.interpolate(value, target_size, mode='bilinear'))
    activations = torch.cat(activations, axis=1)
    return activations

def predict(input_tensor, model, device, detection_threshold):
    outputs = model(input_tensor)
    pred_classes = [coco_names[i] for i in outputs[0]['labels'].cpu().numpy()]
    pred_labels = outputs[0]['labels'].cpu().numpy()
    # get score for all the predicted objects
    pred_scores = outputs[0]['scores'].detach().cpu().numpy()
    # get all the predicted bounding boxes
    pred_bboxes = outputs[0]['boxes'].detach().cpu().numpy()
    # get boxes above the threshold score

    boxes, classes, labels, indices = [], [], [], []
    for index in range(len(pred_scores)):
        if pred_scores[index] >= detection_threshold:
            boxes.append(pred_bboxes[index].astype(np.int32))
            classes.append(pred_classes[index])
            labels.append(pred_labels[index])
            indices.append(index)
    boxes = np.int32(boxes)
    return boxes, classes, labels, indices

path = sys.argv[1]
image = cv2.imread(path)[:, :, ::-1].copy()
image = np.float32(image) / 255
# define the torchvision image transforms
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
])
# transform the image to tensor
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_tensor = transform(image).to(device)
input_tensor = input_tensor.unsqueeze(0)  # add a batch dimension

# download or load the model from disk
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
    pretrained=True, box_detections_per_img=400)
model.eval().to(device)

boxes, classes, labels, indices = predict(input_tensor, model, device, 0.9)

targets = [FasterRCNNBoxScoreTarget(labels=labels, bounding_boxes=boxes)]
target_layers = [model.backbone.fpn]

cam = AblationCAM(model,
                  target_layers, 
                  use_cuda=True, 
                  reshape_transform=reshape_transform,
                  ablation_layer=AblationLayerFasterRCNN)

cam.batch_size=16
grayscale_cam = cam(input_tensor, targets=targets)

grayscale_cam = grayscale_cam[0, :, :]
cleaned_grayscale_cam = grayscale_cam * 0

cv2.imwrite("grayscale_cam.png", np.uint8(grayscale_cam*255))

# for (x1, y1, x2, y2) in boxes:
#     cleaned_grayscale_cam[y1:y2, x1:x2] += scale_cam_image(grayscale_cam[y1:y2, x1:x2].copy())
# grayscale_cam = scale_cam_image(cleaned_grayscale_cam)

cam_image = show_cam_on_image(image, grayscale_cam, use_rgb=True)
# cam_image is RGB encoded whereas "cv2.imwrite" requires BGR encoding.
cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)

boxes, classes, labels, indices = predict(input_tensor, model, device, 0.9)
image = draw_boxes(boxes, labels, classes, cam_image)
cv2.imwrite(f'ablcation_cam.jpg', image)