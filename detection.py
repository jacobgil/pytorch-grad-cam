import sys
import argparse
import cv2
import numpy as np
import torch
import torchvision
from pytorch_grad_cam import AblationCAM, EigenCAM
from pytorch_grad_cam.ablation_layer import AblationLayerFasterRCNN
from pytorch_grad_cam.utils.image import show_cam_on_image, scale_accross_batch_and_channels, scale_cam_image
from pytorch_grad_cam.utils.model_targets import FasterRCNNBoxScoreTarget
from pytorch_grad_cam.utils.reshape_transforms import fasterrcnn_reshape_transform

coco_names = ['__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', \
              'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 
              'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 
              'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella',
              'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
              'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
              'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass', 'cup', 'fork',
              'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
              'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
              'potted plant', 'bed', 'N/A', 'dining table', 'N/A', 'N/A', 'toilet',
              'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
              'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book', 'clock', 'vase',
              'scissors', 'teddy bear', 'hair drier', 'toothbrush']

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
    def __init__(self, labels, bounding_boxes, iou_threshold=0.5):
        self.labels = labels
        self.bounding_boxes = bounding_boxes
        self.iou_threshold = iou_threshold

    def __call__(self, model_outputs):
        output = torch.Tensor([0])
        if torch.cuda.is_available():
            output = output.cuda()

        if len(model_outputs["boxes"]) == 0:
            return output

        for box, label in zip(self.bounding_boxes, self.labels):
            box = torch.Tensor(box[None, :])
            if torch.cuda.is_available():
                box = box.cuda()

            ious = torchvision.ops.box_iou(box, model_outputs["boxes"])
            index = ious.argmax()
            if ious[0, index] > self.iou_threshold and model_outputs["labels"][index] == label:
                score = ious[0, index] + model_outputs["scores"][index]
                output = output + score
        return output

def predict(input_tensor, model, device, detection_threshold):
    outputs = model(input_tensor)
    pred_classes = [coco_names[i] for i in outputs[0]['labels'].cpu().numpy()]
    pred_labels = outputs[0]['labels'].cpu().numpy()
    pred_scores = outputs[0]['scores'].detach().cpu().numpy()
    pred_bboxes = outputs[0]['boxes'].detach().cpu().numpy()

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
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval().to(device)

boxes, classes, labels, indices = predict(input_tensor, model, device, 0.9)

targets = [FasterRCNNBoxScoreTarget(labels=labels, bounding_boxes=boxes)]
target_layers = [model.backbone]

cam = AblationCAM(model,
                  target_layers, 
                  use_cuda=True, 
                  reshape_transform=fasterrcnn_reshape_transform,
                  ablation_layer=AblationLayerFasterRCNN(),
                  ratio_channels_to_ablate=1.0)

# cam = EigenCAM(model,
#               target_layers, 
#               use_cuda=True, 
#               reshape_transform=reshape_transform)


cam.batch_size=8
import time
t0 = time.time()
grayscale_cam = cam(input_tensor, targets=targets, eigen_smooth=False)
print("Took", time.time()-t0)

grayscale_cam = grayscale_cam[0, :, :]
cam_image = grayscale_cam
cleaned_grayscale_cam = grayscale_cam * 0


cv2.imwrite("grayscale_cam.png", np.uint8(grayscale_cam*255))

for (x1, y1, x2, y2) in boxes:
    cleaned_grayscale_cam[y1:y2, x1:x2] = np.maximum(cleaned_grayscale_cam[y1:y2, x1:x2],
                                             scale_cam_image(grayscale_cam[y1:y2, x1:x2].copy()))
cam_image = scale_cam_image(cleaned_grayscale_cam)

cam_image = show_cam_on_image(image, cam_image, use_rgb=True)
# cam_image is RGB encoded whereas "cv2.imwrite" requires BGR encoding.
cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)

boxes, classes, labels, indices = predict(input_tensor, model, device, 0.9)
image = draw_boxes(boxes, labels, classes, cam_image)
cv2.imwrite(f'ablcation_cam.jpg', image)