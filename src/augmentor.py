"""
Augmentor module for Object Detection.

Applies simple image augmentations (horizontal flip, brightness
adjustment) to processed images, keeping bounding boxes correctly
in sync with whatever transformation is applied.
"""
import random 
import sys
from typing import Any, Callable

from PIL import Image, ImageEnhance

from src.config import Config
from src.exception import CustomException
from src.logger import  Logger

_logger_obj = Logger('Augumentor')
logger = _logger_obj.get_logger()

class Augmentor:
    """Applies augmentations to an image + its boxes, keeping both in sync."""
    
    def __init__(self,config):
        self.config= config
        self.image_size = config.data.image_size
        
        self.augmentation_registry:dict[str, Callable] = {
            'flip_horizontal':self.flip_horizontal, 
            'adjust_brightness':self.adjust_brightness,
        }
        
    def flip_horizontal(self, image, boxes)-> tuple[Image.Image, list[list[float]]]:
        """
        Mirror the image left-right and update boxes to match.

        Args:
            image: The image to flip.
            boxes: List of [xmin, ymin, xmax, ymax] boxes.

        Returns:
            Tuple of (flipped_image, updated_boxes).
        """
        
        try:
            flipped_image = image.transpose(Image.FLIP_LEFT_RIGHT)
            width = image.width
            
            flipped_boxes=[]
            
            for xmin, ymin, xmax, ymax in boxes:
                new_xmin = width -xmax
                new_xmax = width- xmin
                flipped_boxes.append([new_xmin, ymin, new_xmax, ymax])
                
            return flipped_image, flipped_boxes
        
        except Exception as e:
            raise CustomException(e,sys)
    
    def adjust_brightness(self, image, boxes, factor= 1.3):
        """
        Brighten or darken the image. Boxes are unaffected since only
        pixel values change, not object positions.

        Args:
            image: The image to adjust.
            boxes: List of [xmin, ymin, xmax, ymax] boxes (returned unchanged).
            factor: >1.0 brightens, <1.0 darkens, 1.0 is unchanged.

        Returns:
            Tuple of (adjusted_image, same_boxes).
        """
        try:
            enhancer = ImageEnhance.Brightness(image)
            adjusted_image = enhancer.enhance(factor)
            
            return adjusted_image, boxes
        except Exception as e:
            raise CustomException(e,sys)
    
    def apply_random(self,image: Image.Image, boxes):
        """
        Pick one augmentation at random from the registry and apply it.

        Args:
            image: The image to augment.
            boxes: List of [xmin, ymin, xmax, ymax] boxes.

        Returns:
            A dict with keys: "image", "boxes", "applied" (name of the
            augmentation that was used).
        """
        try:
            chosen_name = random.choice(list(self.augmentation_registry.keys()))
            chosen_function = self.augmentation_registry[chosen_name]
            
            augmented_image, augmented_boxes = chosen_function(image, boxes)
            
            return{
                'image': augmented_image,
                'boxes': augmented_boxes, 
                'applied':chosen_name
            }
        
        except Exception as e:
            raise CustomException(e, sys)
        
        
    def process_all(
        self, processed_records: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """
        Apply one random augmentation to every processed image, saving
        each augmented image as a NEW file (originals are kept untouched).

        Args:
            processed_records: Output from preprocessor's process_all
                (list of dicts with "filename", "processed_path", "boxes", "classes").

        Returns:
            A list of dicts for the augmented images: filename,
            augmented_path, boxes, classes, applied (augmentation name).
        """
        try:
            from pathlib import Path

            processed_dir = Path(self.config.path.processed_path)
            augmented_records: list[dict[str, Any]] = []

            for record in processed_records:
                image = Image.open(record["processed_path"])
                result = self.apply_random(image, record["boxes"])

                augmented_filename = f"aug_{record['filename']}"
                save_path = processed_dir / augmented_filename
                result["image"].save(save_path)

                augmented_records.append({
                    "filename": augmented_filename,
                    "processed_path": str(save_path),
                    "boxes": result["boxes"],
                    "classes": record["classes"],
                    "applied": result["applied"],
                })

            logger.info(f"Created {len(augmented_records)} augmented images")
            return augmented_records

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    import json
    from pathlib import Path

    config = Config()
    augmentor = Augmentor(config)

    with open("data/annotations/processed_index.json") as f:
        processed_records = json.load(f)

    # Quick single-image sanity check (kept from before)
    sample = processed_records[0]
    image = Image.open(sample["processed_path"])
    logger.info(f"Original boxes: {sample['boxes']}")
    result = augmentor.apply_random(image, sample["boxes"])
    logger.info(f"Applied augmentation: {result['applied']}")
    logger.info(f"Augmented boxes: {result['boxes']}")

    # Process everything
    augmented_records = augmentor.process_all(processed_records)

    # Save combined index -- original + augmented together
    combined_records = processed_records + augmented_records
    output_path = Path(config.path.annotations_path) / "augmented_index.json"
    with open(output_path, "w") as f:
        json.dump(combined_records, f, indent=2)

    logger.info(
        f"Saved combined index with {len(combined_records)} entries "
        f"({len(processed_records)} original + {len(augmented_records)} augmented) to {output_path}"
    )
        
