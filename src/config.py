from dataclasses import dataclass, field


@dataclass
class DataConfig:
    """Settings needed to load and prepare image data."""
    image_size:int = 416
    num_classes:int = 3
    batch_size:int = 16
    target_classes: list[str] = field(
        default_factory=lambda: ["person", "dog", "car", "bicycle"]
    )
    include_difficult: bool = False
    
    
@dataclass
class TrainingConfig:
    
    """Settings that control how the training loop behaves."""
    epochs:int = 50
    learning_rate:float = 0.001
    weight_decay:float = 0.0005
    iou_threshold_positive: float = 0.5
    iou_threshold_negative: float = 0.4
    nms_threshold: float = 0.5
    
@dataclass
class PathConfig:
    
    """Folder paths used across the project."""
    raw_path: str = "data/raw"
    processed_path: str = "data/processed"
    annotations_path: str = "data/annotations"
    models_path: str = "artifacts/models"
    experiments_path: str = "artifacts/experiments"
    logs_path: str = "logs"
    voc_images_path: str = "data/raw/VOCdevkit/VOC2007/JPEGImages"
    voc_annotations_path: str = "data/raw/VOCdevkit/VOC2007/Annotations"


@dataclass
class AnchorConfig:
    """Settings needed to generate anchor boxes across the image grid."""
    scales: list[float]= field(default_factory = lambda: [32.0, 64.0, 128.0])
    aspect_ratios:list[float] = field(default_factory= lambda:[0.5,1.0,2.0])
    stride:int = 16

@dataclass
class Config:
    """Top-level config combining all sub-configs into a single object."""
    data:DataConfig = field(default_factory = DataConfig)
    anchor:AnchorConfig = field(default_factory = AnchorConfig)
    training: TrainingConfig =field(default_factory = TrainingConfig)
    path: PathConfig = field(default_factory = PathConfig)