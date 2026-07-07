"""
Non-max suppression module for Project 8 - Object Detection.

Removes duplicate overlapping boxes, keeping only the highest
confidence box per real object.
"""

import sys

from src.bbox_utils import BoxMath
from src.exception import CustomException
from src.logger import Logger

_logger_obj = Logger("nms")
logger = _logger_obj.get_logger()

class NonMaxSuppression:
    """Removes overlapping duplicate boxes, keeping the highest-confidence one."""
    
    def __init__(self, iou_threshold=0.5):
        self.iou_threshold = iou_threshold
    
    def suppress(self,boxes, scores ):
        """
        Apply non-max suppression to a list of boxes and their scores.

        Args:
            boxes: List of [xmin, ymin, xmax, ymax] boxes.
            scores: Confidence score for each box, same order as boxes.

        Returns:
            The filtered list of boxes that survived suppression.
        """
        try:
            if len(boxes) ==0:
                return []
            
            # Pair each box with its score, then sort highest score first
            indexed_boxes = list(zip(boxes,scores))
            indexed_boxes.sort(key=lambda pair: pair[1], reverse = True)
            
            kept_boxes =[]
            
            while len(indexed_boxes)>0:
                current_box,current_score = indexed_boxes.pop(0)
                kept_boxes.append(current_box)
                
                remaining = []
                for box, score in indexed_boxes:
                    overlap = BoxMath.iou(current_box,box)
                    if overlap < self.iou_threshold:
                        remaining.append((box,score))
                
                indexed_boxes = remaining
            logger.info(f"Kept {len(kept_boxes)} boxes out of {len(boxes)} after NMS")
            return kept_boxes
        
        except Exception as e:
            raise CustomException(e,sys)
    
if __name__ == "__main__":
    # 3 overlapping boxes on roughly the same dog, 1 box on something else
    boxes = [
        [100, 100, 200, 200],   # dog, high confidence
        [105, 105, 205, 205],   # same dog, slightly shifted
        [98, 102, 198, 202],    # same dog again, slightly shifted
        [400, 400, 500, 500],   # completely different object
    ]
    scores = [0.95, 0.90, 0.85, 0.80]

    nms = NonMaxSuppression(iou_threshold=0.5)
    result = nms.suppress(boxes, scores)

    logger.info(f"Boxes kept: {result}")
        