# Utility functions for image preprocessing and post-processing the results

import torch
import torchvision.transforms as T
import cv2
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

COCO_INSTANCE_CATEGORY_NAMES = [
'__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog',
'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

def load_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def preprocess_image(image):
    
    transform = T.Compose([
        T.ToTensor(),
    ])
    tensor_image = transform(image)
    return tensor_image

def plot_detections(image, predictions):
    
    # Plot the bounding boxes and masks on the image

    boxes = predictions['boxes'].cpu().numpy()
    labels = predictions['labels'].cpu().numpy()
    masks = predictions['masks'].cpu().numpy()
    scores = predictions['scores'].cpu().numpy()

    # Filter out the low-confidence predictions
    for i in range(len(boxes)):
        if scores[i] >= 0.7:
            box = boxes[i]
            label = labels[i]
            mask = masks[i][0]
            if label < len(COCO_INSTANCE_CATEGORY_NAMES):
                class_name = COCO_INSTANCE_CATEGORY_NAMES[label]

                # Draw the bounding box
                image = cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 2)
                image = cv2.putText(image, class_name, (int(box[0]), int(box[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

                # Add mask overlay
                mask = mask > 0.5
                image[mask]  = [255, 0, 0]

    # Display image
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.axis('off')
    plt.show()

def visualize_graph_on_image(G, image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    image_shape = img.shape

    pos = {}
    for node in G.nodes(data = True):
        bbox = node[1]['bbox']

        # Calculate the node position as the center of the bounding box
        x_center = (bbox[0] + bbox[2]) / 2
        y_center = (bbox[1] + bbox[3]) / 2
        # Normalise now
        # pos = nx.spring_layout(G, pos = pos, k = 0.1)
        pos[node[0]] = (x_center, y_center * 1.1)

    nx.draw(G, pos, with_labels = True, node_color = 'skyblue', node_size = 300, font_size = 8, edge_color = 'gray', width = 0.5)

    plt.show()