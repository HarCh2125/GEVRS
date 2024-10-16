import torch

# Device configuration for object detection
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Pre-trained model
MODEL_NAME = 'mask_rcnn_resnet50_fpn'