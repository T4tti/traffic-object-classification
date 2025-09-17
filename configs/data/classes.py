"""
Data configuration for traffic object classification.
"""

# Traffic object class definitions
TRAFFIC_CLASSES = [
    'car', 'truck', 'bus', 'motorcycle', 'bicycle',
    'traffic_light', 'traffic_sign', 'pedestrian', 'road', 'background'
]

# Class to ID mapping
CLASS_TO_ID = {cls: idx for idx, cls in enumerate(TRAFFIC_CLASSES)}
ID_TO_CLASS = {idx: cls for idx, cls in enumerate(TRAFFIC_CLASSES)}

# Dataset statistics (to be updated based on actual data)
DATASET_STATS = {
    'balanced': {
        'total_images': 0,
        'total_annotations': 0,
        'class_distribution': {}
    },
    'imbalanced': {
        'total_images': 0,
        'total_annotations': 0,
        'class_distribution': {}
    }
}

# Image preprocessing parameters
IMAGE_MEAN = [0.485, 0.456, 0.406]  # ImageNet means
IMAGE_STD = [0.229, 0.224, 0.225]   # ImageNet stds

# Annotation format specifications
ANNOTATION_FORMATS = {
    'coco': {
        'required_fields': ['images', 'annotations', 'categories'],
        'bbox_format': 'xywh'  # x, y, width, height
    },
    'yolo': {
        'bbox_format': 'normalized_cxcywh'  # center_x, center_y, width, height (normalized)
    },
    'xml': {
        'bbox_format': 'xyxy'  # x_min, y_min, x_max, y_max
    }
}