import torch
import torchvision
from config import DEVICE

class ObjectDetector:
    def __init__(self):
        # Load a pre-trained Mask R-CNN model from torchvision
        self.model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
        self.model.eval()
        self.model.to(DEVICE)

    def predict(self, image):
        # # Preprocess the image
        # image = torch.from_numpy(image).permute(2, 0, 1).float().to(DEVICE)
        # image /= 255.0

        # Make a prediction
        with torch.no_grad():
            predictions = self.model([image.to(DEVICE)])

        return predictions[0]