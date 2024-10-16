# Construct a graph using the detected objects
import networkx as nx
import numpy as np
from utils import COCO_INSTANCE_CATEGORY_NAMES

def construct_graph(predictions, iou_threshold = 0.5):
    # Construct a graph from the detected objects
    # Two nodes (objects) will have an edge if they are spatially close, i.e., their bounding boxes overlap

    # Create an empty graph
    G = nx.Graph()

    boxes = predictions['boxes'].detach().cpu().numpy()
    labels = predictions['labels'].detach().cpu().numpy()

    num_objects = len(boxes)

    # Add nodes
    for i in range(num_objects):
        box = boxes[i]
        label = labels[i]

        if label < len(COCO_INSTANCE_CATEGORY_NAMES):
            class_name = COCO_INSTANCE_CATEGORY_NAMES[label]
            G.add_node(i, label = class_name, bbox = box)

    # Add edges based on spatial proximity (IoU)
    for i in range(num_objects):
        for j in range(i + 1, num_objects):
            box1 = boxes[i]
            box2 = boxes[j]

            iou = compute_iou(box1, box2)
            if iou > iou_threshold:
                G.add_edge(i, j, weight = iou)

    return G

# Helper function to compute the Intersection over Union (IoU) of two bounding boxes
def compute_iou(box1, box2):
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    # Area of intersection rectangle
    inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)

    # Area of both bounding boxes
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # IoU calculation
    iou = inter_area / float(box1_area + box2_area - inter_area)

    return iou