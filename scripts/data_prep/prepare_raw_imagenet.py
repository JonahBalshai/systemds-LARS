#!/usr/bin/env python3
"""
Raw ImageNet Data Preprocessing Pipeline
=========================================

This script processes raw ImageNet JPG images with metadata CSV files and prepares them
for SystemDS AlexNet training. It handles:

1. Reading metadata CSV files with file_path,label format
2. Loading JPG images and keeping them at 256x256 resolution
3. Converting to normalized feature vectors
4. Creating one-hot encoded labels
5. Saving in SystemDS-compatible CSV format

Usage:
    python prepare_raw_imagenet.py --input_dir "C:/Users/romer/Desktop/Code/HuggingFace" --output_dir "imagenet_data/systemds_ready"
    python prepare_raw_imagenet.py --input_dir "C:/Users/romer/Desktop/Code/HuggingFace" --dry_run
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import time
import gc
from PIL import Image
import csv


class RawImageNetProcessor:
    """Raw ImageNet JPG image processor for SystemDS."""
    
    def __init__(self, input_dir: str, output_dir: str = "imagenet_data/systemds_ready"):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Target specifications for SystemDS AlexNet
        self.target_size = 224
        self.channels = 3
        self.features = self.target_size * self.target_size * self.channels  # 150528
        self.num_classes = 1000
        
        print(f"Raw ImageNet Processor initialized")
        print(f"Input directory: {self.input_dir}")
        print(f"Output directory: {self.output_dir}")
        print(f"Target format: {self.target_size}x{self.target_size}x{self.channels} images ({self.features} features), {self.num_classes} classes")
    
    def inspect_raw_data(self) -> Dict:
        """Inspect the raw data structure and return metadata."""
        print("\n=== Raw Data Inspection ===")
        
        # Look for metadata files
        train_metadata_file = self.input_dir / "imagenet_train_metadata.csv"
        test_metadata_file = self.input_dir / "imagenet_test_metadata.csv"
        
        if not train_metadata_file.exists():
            raise FileNotFoundError(f"Training metadata file not found: {train_metadata_file}")
        if not test_metadata_file.exists():
            raise FileNotFoundError(f"Test metadata file not found: {test_metadata_file}")
        
        # Read metadata
        print(f"Reading training metadata from: {train_metadata_file}")
        train_df = pd.read_csv(train_metadata_file)
        print(f"Reading test metadata from: {test_metadata_file}")
        test_df = pd.read_csv(test_metadata_file)
        
        # Inspect structure
        print(f"\nTraining metadata shape: {train_df.shape}")
        print(f"Training columns: {list(train_df.columns)}")
        print(f"Training label range: {train_df['label'].min()} to {train_df['label'].max()}")
        print(f"Training unique labels: {train_df['label'].nunique()}")
        
        print(f"\nTest metadata shape: {test_df.shape}")
        print(f"Test columns: {list(test_df.columns)}")
        print(f"Test label range: {test_df['label'].min()} to {test_df['label'].max()}")
        print(f"Test unique labels: {test_df['label'].nunique()}")
        
        # Check if images actually exist
        print(f"\nChecking image availability...")
        train_available = self._count_available_images(train_df)
        test_available = self._count_available_images(test_df)
        
        # Sample a few images to check dimensions
        sample_dims = self._check_sample_image_dimensions(train_df.head(5))
        
        metadata = {
            'train_total': len(train_df),
            'train_available': train_available,
            'test_total': len(test_df),
            'test_available': test_available,
            'train_labels': sorted(train_df['label'].unique()),
            'test_labels': sorted(test_df['label'].unique()),
            'sample_dimensions': sample_dims
        }
        
        print(f"\n=== Summary ===")
        print(f"Training: {train_available}/{len(train_df)} images available")
        print(f"Test: {test_available}/{len(test_df)} images available")
        print(f"Sample image dimensions: {sample_dims}")
        
        return metadata
    
    def _count_available_images(self, df: pd.DataFrame) -> int:
        """Count how many images actually exist on disk."""
        available = 0
        total = len(df)
        
        print(f"  Checking {total} image files...")
        for i, row in df.iterrows():
            image_path = self.input_dir / row['file_path']
            if image_path.exists():
                available += 1
            
            # Progress update every 1000 images
            if (i + 1) % 1000 == 0:
                print(f"    Checked {i + 1}/{total} images, {available} available")
        
        print(f"  Final: {available}/{total} images available")
        return available
    
    def _check_sample_image_dimensions(self, sample_df: pd.DataFrame) -> List[Tuple]:
        """Check dimensions of a few sample images."""
        dimensions = []
        
        for _, row in sample_df.iterrows():
            image_path = self.input_dir / row['file_path']
            if image_path.exists():
                try:
                    with Image.open(image_path) as img:
                        dimensions.append((img.width, img.height, len(img.getbands())))
                except Exception as e:
                    print(f"    Error reading {image_path}: {e}")
            
            if len(dimensions) >= 3:  # Just check a few
                break
        
        return dimensions
    
    def process_dataset(self, max_samples: Optional[int] = None, dry_run: bool = False) -> Dict:
        """Process the complete dataset."""
        print(f"\n=== Processing Dataset (dry_run={dry_run}) ===")
        
        # Read metadata
        train_df = pd.read_csv(self.input_dir / "imagenet_train_metadata.csv")
        test_df = pd.read_csv(self.input_dir / "imagenet_test_metadata.csv")
        
        # Filter to only available images
        print("Filtering to available images...")
        train_df = self._filter_available_images(train_df)
        test_df = self._filter_available_images(test_df)
        
        # Limit samples if requested
        if max_samples:
            print(f"Limiting to {max_samples} samples per split...")
            train_df = train_df.head(max_samples)
            test_df = test_df.head(max_samples)
        
        print(f"Processing {len(train_df)} training samples...")
        print(f"Processing {len(test_df)} test samples...")
        
        if dry_run:
            print("DRY RUN: Would process the above samples")
            return {'dry_run': True, 'train_samples': len(train_df), 'test_samples': len(test_df)}
        
        # Process training data
        train_results = self._process_split(train_df, "train")
        
        # Process test data (as validation)
        test_results = self._process_split(test_df, "val")
        
        return {
            'train': train_results,
            'validation': test_results
        }
    
    def _filter_available_images(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter dataframe to only include images that exist on disk."""
        available_mask = []
        
        for _, row in df.iterrows():
            image_path = self.input_dir / row['file_path']
            available_mask.append(image_path.exists())
        
        filtered_df = df[available_mask].copy()
        print(f"  Filtered {len(df)} -> {len(filtered_df)} available images")
        return filtered_df
    
    def _process_split(self, df: pd.DataFrame, split_name: str) -> Dict:
        """Process a data split (train or val)."""
        print(f"\nProcessing {split_name} split...")
        
        # Prepare output files
        features_file = self.output_dir / f"imagenet_{split_name}_6GB.csv"
        labels_file = self.output_dir / f"imagenet_{split_name}_labels_6GB.csv"
        
        # Process images in batches to manage memory
        batch_size = 1000
        total_samples = len(df)
        num_batches = (total_samples + batch_size - 1) // batch_size
        
        print(f"Processing {total_samples} samples in {num_batches} batches of {batch_size}")
        
        # Initialize CSV files
        features_written = 0
        labels_written = 0
        
        with open(features_file, 'w', newline='') as f_feat, \
             open(labels_file, 'w', newline='') as f_label:
            
            feat_writer = csv.writer(f_feat)
            label_writer = csv.writer(f_label)
            
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, total_samples)
                batch_df = df.iloc[start_idx:end_idx]
                
                print(f"  Batch {batch_idx + 1}/{num_batches}: Processing samples {start_idx}-{end_idx-1}")
                
                # Process batch
                batch_features, batch_labels = self._process_image_batch(batch_df)
                
                # Write to CSV
                for features_row in batch_features:
                    feat_writer.writerow(features_row)
                    features_written += 1
                
                for labels_row in batch_labels:
                    label_writer.writerow(labels_row)
                    labels_written += 1
                
                # Memory cleanup
                del batch_features, batch_labels
                gc.collect()
                
                print(f"    Wrote {len(batch_df)} samples to CSV")
        
        result = {
            'samples_processed': features_written,
            'features_file': str(features_file),
            'labels_file': str(labels_file),
            'features_shape': (features_written, self.features),
            'labels_shape': (labels_written, self.num_classes)
        }
        
        print(f"  {split_name} processing complete: {features_written} samples")
        return result
    
    def _process_image_batch(self, batch_df: pd.DataFrame) -> Tuple[List, List]:
        """Process a batch of images."""
        batch_features = []
        batch_labels = []
        
        for _, row in batch_df.iterrows():
            try:
                # Load and process image
                image_path = self.input_dir / row['file_path']
                features = self._process_single_image(image_path)
                
                # Process label
                label = int(row['label'])
                # Convert to 0-indexed if needed (ImageNet labels are usually 1-indexed)
                if label > 0:
                    label = label - 1
                
                # Create one-hot encoding
                one_hot = [0.0] * self.num_classes
                if 0 <= label < self.num_classes:
                    one_hot[label] = 1.0
                
                batch_features.append(features)
                batch_labels.append(one_hot)
                
            except Exception as e:
                print(f"    Error processing {row['file_path']}: {e}")
                # Skip this sample
                continue
        
        return batch_features, batch_labels
    
    def _process_single_image(self, image_path: Path) -> List[float]:
        """Process a single image: load, normalize, flatten."""
        # Load image
        with Image.open(image_path) as img:
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Verify image is 224x224, resize if needed
            if img.size != (224, 224):
                img = img.resize((224, 224), Image.LANCZOS)
            
            # Convert to numpy array and normalize to [0,1]
            img_array = np.array(img, dtype=np.float32) / 255.0
            
            # Flatten to feature vector
            features = img_array.flatten().tolist()
            
            return features


def main():
    parser = argparse.ArgumentParser(description='Process raw ImageNet JPG data for SystemDS')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing raw ImageNet data')
    parser.add_argument('--output_dir', type=str, default='imagenet_data/systemds_ready',
                        help='Output directory for processed data')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum number of samples per split (for testing)')
    parser.add_argument('--dry_run', action='store_true',
                        help='Just inspect data without processing')
    
    args = parser.parse_args()
    
    # Initialize processor
    processor = RawImageNetProcessor(args.input_dir, args.output_dir)
    
    # Inspect data first
    try:
        metadata = processor.inspect_raw_data()
    except Exception as e:
        print(f"Error during inspection: {e}")
        return 1
    
    # Process if not dry run
    if not args.dry_run:
        try:
            results = processor.process_dataset(max_samples=args.max_samples, dry_run=False)
            print(f"\n=== Processing Complete ===")
            print(f"Results: {results}")
        except Exception as e:
            print(f"Error during processing: {e}")
            return 1
    else:
        processor.process_dataset(dry_run=True)
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 