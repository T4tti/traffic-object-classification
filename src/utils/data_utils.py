"""
Data utilities for downloading and preparing datasets.
"""

import json
import os
import requests
import zipfile
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import shutil


def download_file(url: str, save_path: str, chunk_size: int = 8192) -> None:
    """
    Download file from URL with progress bar.

    Args:
        url: URL to download from
        save_path: Path to save the file
        chunk_size: Chunk size for downloading
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    
    with open(save_path, 'wb') as f, tqdm(
        desc=os.path.basename(save_path),
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))


def extract_zip(zip_path: str, extract_to: str) -> None:
    """
    Extract ZIP file.

    Args:
        zip_path: Path to ZIP file
        extract_to: Directory to extract to
    """
    os.makedirs(extract_to, exist_ok=True)
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    
    print(f"Extracted {zip_path} to {extract_to}")


def download_dataset(
    dataset_name: str,
    data_dir: str = "data",
    urls: Optional[Dict[str, str]] = None,
) -> str:
    """
    Download and prepare dataset.

    Args:
        dataset_name: Name of the dataset
        data_dir: Root data directory
        urls: Optional dictionary of URLs to download

    Returns:
        Path to extracted dataset

    Raises:
        ValueError: If dataset_name is not supported
    """
    dataset_dir = os.path.join(data_dir, dataset_name)
    
    # Default dataset URLs
    default_urls = {
        "coco": {
            "train_images": "http://images.cocodataset.org/zips/train2017.zip",
            "val_images": "http://images.cocodataset.org/zips/val2017.zip",
            "annotations": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
        },
        "cityscapes": {
            "images": "https://www.cityscapes-dataset.com/file-handling/?packageID=3",
            "annotations": "https://www.cityscapes-dataset.com/file-handling/?packageID=1",
        },
    }
    
    if urls is None:
        if dataset_name not in default_urls:
            raise ValueError(f"No default URLs for dataset: {dataset_name}")
        urls = default_urls[dataset_name]
    
    # Create dataset directory
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Download files
    for file_type, url in urls.items():
        filename = f"{file_type}.zip"
        save_path = os.path.join(dataset_dir, filename)
        
        if not os.path.exists(save_path):
            print(f"Downloading {file_type}...")
            try:
                download_file(url, save_path)
            except Exception as e:
                print(f"Failed to download {file_type}: {e}")
                continue
        
        # Extract if it's a ZIP file
        if save_path.endswith('.zip'):
            extract_dir = os.path.join(dataset_dir, file_type)
            if not os.path.exists(extract_dir):
                print(f"Extracting {file_type}...")
                extract_zip(save_path, extract_dir)
    
    return dataset_dir


def prepare_annotations(
    images_dir: str,
    annotations_file: str,
    output_file: str,
    class_mapping: Optional[Dict[str, int]] = None,
) -> None:
    """
    Prepare annotations in a standardized format.

    Args:
        images_dir: Directory containing images
        annotations_file: Path to annotations file (COCO format)
        output_file: Path to save processed annotations
        class_mapping: Optional class name to ID mapping
    """
    with open(annotations_file, 'r') as f:
        coco_data = json.load(f)
    
    # Create class mapping if not provided
    if class_mapping is None:
        class_mapping = {}
        for category in coco_data['categories']:
            class_mapping[category['name']] = category['id']
    
    # Process images and annotations
    processed_data = {}
    
    # Create image ID to filename mapping
    image_id_to_filename = {}
    for image_info in coco_data['images']:
        image_id_to_filename[image_info['id']] = image_info['file_name']
    
    # Group annotations by image
    annotations_by_image = {}
    for annotation in coco_data['annotations']:
        image_id = annotation['image_id']
        if image_id not in annotations_by_image:
            annotations_by_image[image_id] = []
        annotations_by_image[image_id].append(annotation)
    
    # Process each image
    for image_id, filename in image_id_to_filename.items():
        image_path = os.path.join(images_dir, filename)
        
        if not os.path.exists(image_path):
            continue
        
        objects = []
        if image_id in annotations_by_image:
            for annotation in annotations_by_image[image_id]:
                category_id = annotation['category_id']
                
                # Find category name
                category_name = None
                for category in coco_data['categories']:
                    if category['id'] == category_id:
                        category_name = category['name']
                        break
                
                if category_name and category_name in class_mapping:
                    bbox = annotation['bbox']  # [x, y, w, h]
                    
                    obj = {
                        'class': category_name,
                        'bbox': bbox,
                        'area': annotation.get('area', bbox[2] * bbox[3]),
                        'iscrowd': annotation.get('iscrowd', 0),
                    }
                    objects.append(obj)
        
        processed_data[str(image_id)] = {
            'filename': filename,
            'objects': objects,
        }
    
    # Save processed annotations
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(processed_data, f, indent=2)
    
    print(f"Processed annotations saved to: {output_file}")
    print(f"Total images: {len(processed_data)}")
    print(f"Total objects: {sum(len(data['objects']) for data in processed_data.values())}")


def split_dataset(
    annotations_file: str,
    output_dir: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
    random_seed: int = 42,
) -> Tuple[str, str, str]:
    """
    Split dataset into train/val/test sets.

    Args:
        annotations_file: Path to annotations file
        output_dir: Directory to save split files
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
        test_ratio: Ratio for test set
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of (train_file, val_file, test_file) paths
    """
    import random
    
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1.0")
    
    # Load annotations
    with open(annotations_file, 'r') as f:
        data = json.load(f)
    
    # Get all image IDs
    image_ids = list(data.keys())
    
    # Shuffle with seed
    random.seed(random_seed)
    random.shuffle(image_ids)
    
    # Calculate split indices
    total_images = len(image_ids)
    train_end = int(total_images * train_ratio)
    val_end = train_end + int(total_images * val_ratio)
    
    # Split data
    train_ids = image_ids[:train_end]
    val_ids = image_ids[train_end:val_end]
    test_ids = image_ids[val_end:]
    
    # Create split datasets
    os.makedirs(output_dir, exist_ok=True)
    
    splits = {
        'train': train_ids,
        'val': val_ids,
        'test': test_ids,
    }
    
    split_files = []
    for split_name, ids in splits.items():
        split_data = {img_id: data[img_id] for img_id in ids}
        
        split_file = os.path.join(output_dir, f"{split_name}_annotations.json")
        with open(split_file, 'w') as f:
            json.dump(split_data, f, indent=2)
        
        split_files.append(split_file)
        print(f"{split_name.capitalize()} set: {len(ids)} images")
    
    return tuple(split_files)


def create_class_mapping(annotations_file: str) -> Dict[str, int]:
    """
    Create class name to ID mapping from annotations.

    Args:
        annotations_file: Path to annotations file

    Returns:
        Dictionary mapping class names to IDs
    """
    with open(annotations_file, 'r') as f:
        data = json.load(f)
    
    classes = set()
    for image_data in data.values():
        for obj in image_data.get('objects', []):
            classes.add(obj['class'])
    
    # Sort classes for consistent mapping
    sorted_classes = sorted(list(classes))
    
    # Create mapping (start from 1, 0 is usually background)
    class_mapping = {class_name: idx + 1 for idx, class_name in enumerate(sorted_classes)}
    
    return class_mapping


def verify_dataset(dataset_dir: str, annotations_file: str) -> Dict[str, int]:
    """
    Verify dataset integrity and return statistics.

    Args:
        dataset_dir: Directory containing images
        annotations_file: Path to annotations file

    Returns:
        Dictionary with dataset statistics
    """
    with open(annotations_file, 'r') as f:
        data = json.load(f)
    
    stats = {
        'total_images': 0,
        'total_objects': 0,
        'missing_images': 0,
        'empty_images': 0,
        'class_counts': {},
    }
    
    for image_id, image_data in data.items():
        stats['total_images'] += 1
        
        filename = image_data['filename']
        image_path = os.path.join(dataset_dir, filename)
        
        if not os.path.exists(image_path):
            stats['missing_images'] += 1
            continue
        
        objects = image_data.get('objects', [])
        if not objects:
            stats['empty_images'] += 1
        
        stats['total_objects'] += len(objects)
        
        for obj in objects:
            class_name = obj['class']
            stats['class_counts'][class_name] = stats['class_counts'].get(class_name, 0) + 1
    
    return stats


def copy_images_by_split(
    source_dir: str,
    target_dir: str,
    annotations_file: str,
    split_name: str,
) -> None:
    """
    Copy images to target directory based on split annotations.

    Args:
        source_dir: Source directory containing all images
        target_dir: Target directory for split images
        annotations_file: Annotations file for the split
        split_name: Name of the split (train/val/test)
    """
    with open(annotations_file, 'r') as f:
        data = json.load(f)
    
    split_dir = os.path.join(target_dir, split_name)
    os.makedirs(split_dir, exist_ok=True)
    
    for image_data in tqdm(data.values(), desc=f"Copying {split_name} images"):
        filename = image_data['filename']
        source_path = os.path.join(source_dir, filename)
        target_path = os.path.join(split_dir, filename)
        
        if os.path.exists(source_path):
            shutil.copy2(source_path, target_path)
    
    print(f"Copied {len(data)} images to {split_dir}")