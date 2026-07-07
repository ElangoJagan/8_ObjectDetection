"""
Anchor generator module - Object Detection.

Generates a grid of candidate anchor boxes across an image, at
every grid point, in multiple scales and aspect ratios.
"""

import sys
from src.config import Config
from src.exception import CustomException
from src.logger import Logger

_logger_obj = Logger("anchor_generator")
logger = _logger_obj.get_logger()

class AnchorGenerator:
    """Generates anchor boxes across an image using a fixed grid, scales, and ratios."""
    
    def __init__(self, config):
        self.config = config
        self.image_size = config.data.image_size
        self.stride = config.anchor.stride
        self.scales = config.anchor.scales
        self.aspect_ratios = config.anchor.aspect_ratios
        
    def _generate_grid_centers(self):
        """
        Generate every anchor center point across the image, spaced by stride.

        Returns:
            A list of (center_x, center_y) tuples.
        """
        try:
            centers = []
            num_steps = self.image_size // self.stride
            
            for row in range(num_steps):
                for col in range(num_steps):
                    center_x = col*self.stride+self.stride / 2
                    center_y = row*self.stride+self.stride/2
                    centers.append((center_x, center_y))
            return centers
        except Exception as e:
            raise CustomException(e,sys)
    
    def _generate_boxes_at_center(self,center_x, center_y):
        """
        Generate all scale x aspect_ratio boxes for ONE center point.

        Args:
            center_x: X coordinate of the anchor center.
            center_y: Y coordinate of the anchor center.

        Returns:
            A list of [xmin, ymin, xmax, ymax] boxes, one per
            (scale, aspect_ratio) combination.
        """
        try:
            boxes = []
            
            for scale in self.scales:
                for ratio in self.aspect_ratios:
                    box_width = scale*(ratio**0.5)
                    box_height = scale / (ratio**0.5)
                    
                    half_width = box_width / 2
                    half_height = box_height / 2
                    
                    xmin= center_x - half_width
                    ymin = center_y- half_height
                    xmax = center_x+ half_width
                    ymax= center_y + half_height
                    
                    boxes.append([xmin, ymin,xmax, ymax])
            return boxes
        except Exception as e:
            raise CustomException(e,sys)
    
    def generate_all_anchors(self):
        """
        Generate every anchor box across the whole image grid.

        Returns:
            A flat list of [xmin, ymin, xmax, ymax] boxes, covering
            every grid center and every scale/ratio combination.
        """
        try:
            centers = self._generate_grid_centers()
            all_anchors = []
            
            for center_x,center_y in centers:
                boxes_here = self._generate_boxes_at_center(center_x,center_y)
                all_anchors.extend(boxes_here)
                
            logger.info(
                f"Generated {len(all_anchors)} anchors "
                f"from {len(centers)} grid centers "
                f"({len(self.scales)} scales x {len(self.aspect_ratios)} ratios each)"
            )
                
            return all_anchors
        except Exception  as e:
            raise CustomException(e,sys)

if __name__ =='__main__':
    config = Config()
    generator = AnchorGenerator(config)
    all_anchors = generator.generate_all_anchors()

    logger.info(f"First anchor box: {all_anchors[0]}")
    logger.info(f"Last anchor box: {all_anchors[-1]}")