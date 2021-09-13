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

from pytorch_grad_cam.utils.image import show_cam_on_image, scale_cam_image
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


def draw_boxes(boxes, classes, labels, image):
    # read the image with OpenCV
    image = cv2.cvtColor(np.asarray(image), cv2.COLOR_BGR2RGB)
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


def predict(image, model, device, detection_threshold):

    # define the torchvision image transforms
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])

    # transform the image to tensor
    image = transform(image).to(device)
    image = image.unsqueeze(0)  # add a batch dimension
    outputs = model(image)  # get the predictions on the image
    # print the results individually
    # print(f"BOXES: {outputs[0]['boxes']}")
    # print(f"LABELS: {outputs[0]['labels']}")
    # print(f"SCORES: {outputs[0]['scores']}")
    # get all the predicited class names
    pred_classes = [coco_names[i] for i in outputs[0]['labels'].cpu().numpy()]
    # get score for all the predicted objects
    pred_scores = outputs[0]['scores'].detach().cpu().numpy()
    # get all the predicted bounding boxes
    pred_bboxes = outputs[0]['boxes'].detach().cpu().numpy()
    print(pred_bboxes)
    # get boxes above the threshold score
    boxes = pred_bboxes[pred_scores >= detection_threshold].astype(np.int32)
    return boxes, pred_classes, outputs[0]['labels']

def fasterrcnn_reshape_transform(x, target_size):
    result = []
    for pyramid_level in x[0]:
        pyramid_level = pyramid_level[0].detach().cpu().numpy()
        img = scale_cam_image(pyramid_level, target_size=target_size)
        result.extend(img)    
    result = np.concatenate(result, axis = 0)
    result = torch.from_numpy(result)
    return result

path = sys.argv[1]
image = cv2.imread(path)[:, :, ::-1].copy()
image = np.float32(image) / 255
print("image shape", image.shape)

# download or load the model from disk
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
    pretrained=True, min_size=1000)

target_layers = [model.backbone.fpn.extra_blocks]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.eval().to(device)
print(model)
activations_and_grads = ActivationsAndGradients(
    model, target_layers, reshape_transform = 
        lambda x: fasterrcnn_reshape_transform(x, (image.shape[:2]))
        )

boxes, classes, labels = predict(image, model, device, 0.9)
image = draw_boxes(boxes, classes, labels, image)
cv2.imshow('Image', image)
cv2.waitKey(-1)

for a, g in zip(activations_and_grads.activations, activations_and_grads.gradients):
    print(a.shape, g.shape)