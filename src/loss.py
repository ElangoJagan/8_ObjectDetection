"""
Loss module for Project Object Detection.

Combines classification loss (what is the object) and localization
loss (where is the object) into a single trainable loss value.
"""

import sys
import torch
import torch.nn as nn

from src.exception import CustomException
from src.logger import Logger

_logger_obj = Logger('Loss')
logger = _logger_obj.get_logger()

class DetectionLoss(nn.Module):
    """Combines classification loss and localization loss for object detection.""" 
    
    def __init__(self):
        
        super().__init__()
        
        # Classification: predicting which class (or background) -- standard
        # multi-class loss, same type you'd have used in Projects 1-7.
        self.classification_loss_fn = nn.CrossEntropyLoss(reduction="sum")

        # Localization: predicting 4 box adjustment numbers -- treated as a
        # regression problem (predicting continuous numbers, not classes).
        self.localization_loss_fn = nn.SmoothL1Loss(reduction="sum")
    
    def forward(
        self, 
        predicted_boxes:torch.Tensor,
        predicted_classes,
        target_boxes,
        target_classes,
        positive_mask
    ):
        """
        Compute combined detection loss.

        Args:
            predicted_boxes: Model's predicted box adjustments, shape (N, 4).
            predicted_classes: Model's predicted class scores, shape (N, num_classes).
            target_boxes: Ground truth box adjustments, shape (N, 4).
            target_classes: Ground truth class index per anchor, shape (N,).
            positive_mask: Boolean tensor, shape (N,) -- True where this
                anchor is a positive match (IoU above threshold) and should
                count toward localization loss.

        Returns:
            A dict with "classification_loss", "localization_loss", "total_loss".
        """
        try:
            # Classification loss applies to EVERY anchor (positive and background)
            classification_loss=self.classification_loss_fn(
                predicted_classes, target_classes
            )
            # Localization loss only applies to POSITIVE anchors -- there's no
            # real box to compare against for background anchors.
            
            if positive_mask.sum()>0:
                localization_loss = self.localization_loss_fn(
                    predicted_boxes[positive_mask],target_boxes[positive_mask]
                )
            else:
                localization_loss=torch.tensor(0.0)
            
            total_loss = classification_loss+ localization_loss
            
            return{
                "classification_loss": classification_loss,
                "localization_loss": localization_loss,
                "total_loss": total_loss,
            }
        except Exception as e:
            raise CustomException(e,sys)

if __name__ == "__main__":
    num_anchors = 5
    num_classes = 5
    
    predicted_boxes = torch.randn(num_anchors,4)
    predicted_classes = torch.randn(num_anchors, num_classes)
    
    target_boxes = torch.randn(num_anchors, 4)
    target_classes = torch.tensor([0,4,2,4,4])
    
    positive_mask = torch.tensor([True, False, True, False, False])
    
    loss_fn = DetectionLoss()
    result = loss_fn(predicted_boxes, predicted_classes, target_boxes, target_classes, positive_mask)
    
    logger.info(f"Classification loss: {result['classification_loss'].item()}")
    logger.info(f"Localization loss: {result['localization_loss'].item()}")
    logger.info(f"Total loss: {result['total_loss'].item()}")