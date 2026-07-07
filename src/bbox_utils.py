"""
Bounding box utility module for Object Detection.

Implements IoU (Intersection over Union) from scratch -- the core
building block used later for anchor matching, NMS, and evaluation.
"""
import sys

from src.exception import CustomException
from src.logger import Logger

_logger_obj = Logger('Bbox_Utils')
logger = _logger_obj.get_logger()

class BoxMath:
    """Pure geometry functions for working with [xmin, ymin, xmax, ymax] boxes."""
    
    @staticmethod
    def area(box):
        """
        Compute the area of a single box.

        Args:
            box: [xmin, ymin, xmax, ymax]

        Returns:
            The box's area (width * height).
        """
        try: 
            xmin, ymin, xmax, ymax= box
            width = xmax-xmin
            height = ymax-ymin
            return height*width
        
        except Exception as e:
            raise CustomException(e,sys)
    
    @staticmethod
    def intersection_area(box_a, box_b):
        try:
            overlap_xmin= max(box_a[0], box_b[0])
            overlap_ymin= max(box_a[1], box_b[1])
            overlap_xmax= min(box_a[2], box_b[2])
            overlap_ymax= min(box_a[3], box_b[3])
            
            overlap_width  =max(0.0, overlap_xmax - overlap_xmin)
            overlap_height = max(0.0,overlap_ymax - overlap_ymin)
            
            return overlap_width*overlap_height
        except Exception as e:
            raise CustomException(e,sys)
        
    
    @staticmethod
    def iou(box_a, box_b):
        """
        Compute Intersection over Union for two boxes.

        Args:
            box_a: [xmin, ymin, xmax, ymax]
            box_b: [xmin, ymin, xmax, ymax]

        Returns:
            A value between 0.0 (no overlap) and 1.0 (perfect overlap).
        """
        try:
            overlap = BoxMath.intersection_area(box_a, box_b)
            area_a = BoxMath.area(box_a)
            area_b = BoxMath.area(box_b)
            
            union = area_a+area_b - overlap
            
            if union ==0:
                return 0
            return overlap/union
        except Exception as e:
            raise CustomException(e,sys)
        
        
if __name__ == "__main__":
    box_a = [100, 100, 300, 300]
    box_b = [200, 200, 400, 400]

    result = BoxMath.iou(box_a, box_b)
    logger.info(f"Box A: {box_a}")
    logger.info(f"Box B: {box_b}")
    logger.info(f"IoU: {result}")
    