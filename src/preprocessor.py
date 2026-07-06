"""
Resizes images to a fixed size and rescales their matching bounding
boxes by the same proportion, so the boxes stay correctly positioned
after the image itself changes size.
"""

import sys
from pathlib import Path
from typing import Any

from PIL import Image

from src.config import Config
from src.exception import CustomException
from src.logger import Logger

_logger_obj = Logger('Preprocessor')
logger= _logger_obj.get_logger()

class BoxScaler:
    
    
    @staticmethod
    def scaled_boxes(
        boxes : list[list[int]],
        old_width:int,
        old_height:int,
        new_width:int,
        new_height:int,
    ):
        try:
            scale_x = new_width/old_width
            scale_y= new_height/old_height
            
            scaled_boxes:list[list[float]] = []
            for xmin, ymin, xmax, ymax in boxes:
                scaled_boxes.append([
                    xmin*scale_x,
                    ymin*scale_y,
                    xmax*scale_x,
                    ymax*scale_y,
                ])
            return scaled_boxes
        except Exception as e:
            raise CustomException(e,sys)
    
class ImagePreprocessor:
    
    def __init__(self,config):
        self.config = config
        self.target_size = config.data.image_size
        self.box_scaler = BoxScaler()
        
    def process_single(self, image_path:Path, annotation:dict[str, Any]):
        try:
            #resizeing the image happens here
            image = Image.open(image_path).convert('RGB')
            old_width = annotation['width']
            old_height = annotation['height']
            resized_image = image.resize((self.target_size, self.target_size))
            
            scaled_boxes = self.box_scaler.scaled_boxes(
                boxes = annotation['boxes'],
                old_width = old_width,
                old_height = old_height, 
                new_width = self.target_size,
                new_height = self.target_size,
            )
            
            return{
                'filename': annotation['filename'],
                'image': resized_image,
                'boxes':scaled_boxes,
                'classes':annotation['classes'],
            }
        
        except Exception as e:
            raise CustomException(e,sys)
    
    
    def process_all(self, annotations):
        try:
            processed_dir = Path(self.config.path.processed_path)
            processed_dir.mkdir(parents = True, exist_ok = True)
            processed_records =[]
            
            for annotation in annotations:
                image_path = Path(self.config.path.voc_images_path) / annotation["filename"]
                result = self.process_single(image_path, annotation)
                
                save_path = processed_dir/result['filename']
                result['image'].save(save_path)
                
                processed_records.append({
                    'filename' :result['filename'],
                    'processed_path' : str(save_path),
                    'boxes' :result['boxes'],
                    'classes': result['classes'],})
            
            logger.info(f"Processed and saved {len(processed_records)} images to {processed_dir}")
            return processed_records

        except Exception as e:
            raise CustomException(e, sys)
        
        
if __name__ == "__main__":
    import json

    config = Config()
    preprocessor = ImagePreprocessor(config)

    with open("data/annotations/index.json") as f:
        annotations = json.load(f)

    # Quick single-image sanity check (kept from before)
    sample_annotation = annotations[0]
    image_path = Path(config.path.voc_images_path) / sample_annotation["filename"]
    result = preprocessor.process_single(image_path, sample_annotation)
    logger.info(f"Single-image check -- resized boxes: {result['boxes']}")

    # Process everything
    processed_records = preprocessor.process_all(annotations)

    # Save the combined processed index, same pattern as data_loader's index.json
    output_path = Path(config.path.annotations_path) / "processed_index.json"
    with open(output_path, "w") as f:
        json.dump(processed_records, f, indent=2)

    logger.info(f"Saved processed index with {len(processed_records)} entries to {output_path}")