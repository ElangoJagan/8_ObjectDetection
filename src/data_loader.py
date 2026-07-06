"""
Data loader module for Project 8 - Object Detection.

Reads Pascal VOC XML annotation files and matching images, filters
them down to our target classes, and produces a clean in-memory
(and optionally on-disk) list of: image path + boxes + class names.
"""

import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from src.exception import CustomException
from src.logger import Logger
from src.config import Config
from dataclasses import dataclass, field

_logger_obj = Logger('dataLoader')
logger = _logger_obj.get_logger()

class DataLoader:
    
    def __init__(self, config):
        self.config = config
        self.image_dir = Path(config.path.voc_images_path)
        self.annotations_dir = Path(config.path.voc_annotations_path)
        
    
    @staticmethod
    def parse_annotation_file(
        xml_path:Path,
        target_classes:list[str],
        include_difficult:bool,
    )-> dict[str, Any]:
        
        
        """
        Parse a single VOC XML annotation file into a clean dictionary.

        This is a static method because parsing one file depends only on
        its inputs (the file path, target classes, and the difficult-object
        switch) -- it does not need any state stored on a DataLoader instance.

        Args:
            xml_path: Path to a single .xml annotation file.
            target_classes: Class names to keep (others are dropped).
            include_difficult: Whether to keep objects marked difficult="1".

        Returns:
            A dict with keys: "filename", "width", "height", "boxes", "classes".
            "boxes" is a list of [xmin, ymin, xmax, ymax] (ints).
            "classes" is a list of class name strings, same length as "boxes".
        """
        
        try:
            tree =ET.parse(xml_path)
            root = tree.getroot()
            
            filename = root.find('filename').text
            size_node = root.find('size')
            width =int(size_node.find('width').text)
            height = int(size_node.find('height').text)
            
            boxes: list[list[int]] = []
            classes: list[str] = []
            
            for obj in root.findall('object'):
                class_name = obj.find('name').text
                is_difficult = obj.find('difficult').text =='1'
                
                if class_name not in target_classes:
                    continue 
                if is_difficult and not include_difficult:
                    continue
                
                bndbox = obj.find('bndbox')
                xmin = int(bndbox.find('xmin').text)
                ymin = int(bndbox.find('ymin').text)
                xmax = int(bndbox.find('xmax').text)
                ymax = int(bndbox.find('ymax').text)
                
                boxes.append([xmin, ymin, xmax, ymax])
                classes.append(class_name)
            
            return {
                'filename':filename,
                'width':width,
                'height':height,
                'boxes':boxes, 
                'classes':classes,
            }
        except Exception  as e:
            raise CustomException(e,sys)
    
    def load_all_annotations(self)->list[dict[str,Any]]:
        """
        Parse every XML file in the annotations folder, keep only images
        that still have at least one box after filtering.

        Returns:
            A list of per-image dicts (same shape as parse_annotation_file's
            return value), one entry per image that has >= 1 valid box.
        """
        try:
            xml_files = sorted(self.annotations_dir.glob('*.xml'))
            logger.info(f"Found {len(xml_files)} annotation files in {self.annotations_dir}")
            
            all_annotations: list[dict[str,Any]] = []
            skipped_empty = 0
            
            for xml_path in xml_files:
                parsed = self.parse_annotation_file(
                    xml_path = xml_path,
                    target_classes = self.config.data.target_classes,
                    include_difficult = self.config.data.include_difficult,
                )
                
                if len(parsed['boxes']) ==0:
                    skipped_empty +=1
                    continue
                
                all_annotations.append(parsed)
            
            logger.info(
                f"Kept {len(all_annotations)} images with valid boxes, "
                f"skipped {skipped_empty} images with none"
            )
            return all_annotations
        except Exception as e:
            raise CustomException(e,sys)
    
    def save_annotations_index(
        self, annotations: list[dict[str, Any]], output_filename: str = "index.json"
    )-> Path:
        """
        Save the parsed annotations list to a single JSON file, so future
        runs don't need to re-parse every XML file from scratch.

        Args:
            annotations: The list returned by load_all_annotations().
            output_filename: Name of the JSON file to write.

        Returns:
            The path to the saved JSON file.
        """
        try:
            output_dir = Path(self.config.path.annotations_path)
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / output_filename

            with open(output_path, "w") as f:
                json.dump(annotations, f, indent=2)

            logger.info(f"Saved {len(annotations)} annotation entries to {output_path}")
            return output_path

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    config = Config()
    loader = DataLoader(config)
    annotations = loader.load_all_annotations()
    loader.save_annotations_index(annotations)