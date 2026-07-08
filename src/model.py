
"""
Model module for Project Object Detection.

Defines a one-stage (YOLO/SSD-style) detector: a CNN backbone for
feature extraction, plus a convolutional detection head that predicts
class scores and box adjustments at every anchor position simultaneously.
"""

import sys
import torch
import torch.nn as nn
from src.config import Config
from src.exception import CustomException
from src.logger import Logger

_logger_obj = Logger('Model')
logger = _logger_obj.get_logger()


class DetectionBackbone(nn.Module):
    """
    CNN backbone that extracts features from the raw image.

    Uses 4 conv blocks, each halving the spatial size (stride=2),
    so a 416x416 image becomes a 26x26 feature map (416 / 2^4 = 26),
    matching our anchor grid's stride of 16.
    """
    def __init__(self):
        super().__init__()
        
        self.conv_block1 = self._make_block(in_channels=3, out_channels = 32)
        self.conv_block2 = self._make_block(in_channels = 32, out_channels =64)
        self.conv_block3 = self._make_block(in_channels =64, out_channels = 128)
        self.conv_block4 = self._make_block(in_channels = 128, out_channels = 256)
    
    @staticmethod
    def _make_block(in_channels:int, out_channels:int):
        """
        Build one conv block: conv -> batch norm -> ReLU, stride 2 (halves size).

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.

        Returns:
            An nn.Sequential block.
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = 2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace = True)
        )
    
    def forward(self, x:torch.Tensor):
        """
        Args:
            x: Input image batch, shape (batch, 3, 416, 416).

        Returns:
            Feature map, shape (batch, 256, 26, 26).
        """
        x= self.conv_block1(x)
        x= self.conv_block2(x)
        x= self.conv_block3(x)
        x= self.conv_block4(x)
        return x

class DetectionHead(nn.Module):
    """
    Convolutional head that predicts class scores and box adjustments
    at every grid position, for every anchor.
    """
    def __init__(self, in_channels, num_anchors, num_classes):
        """
        Args:
            in_channels: Number of channels coming from the backbone.
            num_anchors: Number of anchors per grid cell (e.g. 9).
            num_classes: Number of object classes (e.g. 4).
        """
        super().__init__()
        self.outputs_per_anchor= 4+1+num_classes
        self.num_anchors =num_anchors
        
        
        self.predictor = nn.Conv2d(
            in_channels,
            num_anchors*self.outputs_per_anchor,
            kernel_size = 3,
            padding = 1
        )
    
    def forward (self, features):
        """
        Args:
            features: Feature map from the backbone, shape (batch, C, 26, 26).

        Returns:
            Raw predictions, shape (batch, num_anchors * outputs_per_anchor, 26, 26).
        """
        return  self.predictor(features)

class ObjectDetectionModel(nn.Module):
    """
    Full one-stage detector: backbone + detection head, composed together.
    """
    def __init__(self, config:Config):
        super().__init__()
        self.num_anchors = len(config.anchor.scales) * len(config.anchor.aspect_ratios)
        self.num_classes = config.data.num_classes

        self.backbone = DetectionBackbone()
        self.head = DetectionHead(
            in_channels=256,
            num_anchors=self.num_anchors,
            num_classes=self.num_classes,
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input image batch, shape (batch, 3, 416, 416).

        Returns:
            Raw predictions, shape (batch, num_anchors * outputs_per_anchor, 26, 26).
        """
        try:
            features = self.backbone(x)
            predictions = self.head(features)
            return predictions

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    config = Config()
    model = ObjectDetectionModel(config)

    dummy_image = torch.randn(1, 3, config.data.image_size, config.data.image_size)
    output = model(dummy_image)

    logger.info(f"Input shape: {dummy_image.shape}")
    logger.info(f"Output shape: {output.shape}")
    logger.info(f"Anchors per cell: {model.num_anchors}")
    logger.info(f"Outputs per anchor: {model.head.outputs_per_anchor}")