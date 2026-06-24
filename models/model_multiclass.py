import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
import numpy as np


class MulticlassUNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=6,        
            classes=5,            
            activation=None       
        )

    def forward(self, x):
        return self.model(x)


def create_model():
    return MulticlassUNet()


def calculate_metrics(pred, target, num_classes=5):
    with torch.no_grad():
        pred_classes = torch.argmax(pred, dim=1)  # (B, H, W)

        iou_per_class = []
        f1_per_class = []

        for c in range(num_classes):
            pred_c = (pred_classes == c).float()
            target_c = (target == c).float()

            intersection = (pred_c * target_c).sum()
            union = pred_c.sum() + target_c.sum() - intersection

            iou = (intersection + 1e-6) / (union + 1e-6)
            f1 = (2 * intersection + 1e-6) / (pred_c.sum() + target_c.sum() + 1e-6)

            iou_per_class.append(iou.item())
            f1_per_class.append(f1.item())

        mean_iou = np.mean(iou_per_class[1:])
        mean_f1 = np.mean(f1_per_class[1:])

        return mean_iou, mean_f1, iou_per_class, f1_per_class