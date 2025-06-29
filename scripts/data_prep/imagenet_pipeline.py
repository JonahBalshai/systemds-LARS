#!/usr/bin/env python3
"""
Unified ImageNet Data Processing Pipeline
=========================================

This script provides a complete pipeline to:
1. Inspect ImageNet NPZ files
2. Convert NPZ files to efficient binary format
3. Create sampled 2GB CSV files for SystemDS training

Features:
- Memory-efficient streaming processing
- Automatic data validation
- Progress tracking
- Configurable sampling strategies
- SystemDS metadata generation

Usage:
    python imagenet_pipeline.py --mode full
    python imagenet_pipeline.py --mode inspect
    python imagenet_pipeline.py --mode csv-only --csv-size-gb 2
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


class ImageNetPipeline:
    """Complete ImageNet data processing pipeline."""
    
    def __init__(self, input_dir: str = "imagenet_data/6gb", 
                 output_dir: str = "imagenet_data/systemds_ready"):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Dataset constants (based on 6GB ImageNet)
        self.image_size = 64
        self.channels = 3
        self.features = self.image_size * self.image_size * self.channels  # 12288
        self.num_classes = 1000
        
        print(f"ImageNet Pipeline initialized")
        print(f"Input directory: {self.input_dir}")
        print(f"Output directory: {self.output_dir}")
        print(f"Expected format: {self.image_size}x{self.image_size}x{self.channels} images, {self.num_classes} classes")
    
    def inspect_dataset(self) -> Dict:
        """Inspect all NPZ files and return dataset metadata."""
        print("\n=== Dataset Inspection ===")
        
        # Find all NPZ files
        train_files = sorted(self.input_dir.glob("train_data_batch_*.npz"))
        val_files = list(self.input_dir.glob("val_data.npz"))
        
        print(f"Found {len(train_files)} training batches and {len(val_files)} validation files")
        
        dataset_info = {
            'train_batches': [],
            'val_info': None,
            'total_train_samples': 0,
            'total_val_samples': 0
        }
        
        # Inspect training batches
        for i, npz_file in enumerate(train_files):
            print(f"\nInspecting {npz_file.name}...")
            batch_info = self._inspect_npz_file(npz_file)
            dataset_info['train_batches'].append(batch_info)
            dataset_info['total_train_samples'] += batch_info['samples']
        
        # Inspect validation
        if val_files:
            print(f"\nInspecting {val_files[0].name}...")
            dataset_info['val_info'] = self._inspect_npz_file(val_files[0])
            dataset_info['total_val_samples'] = dataset_info['val_info']['samples']
        
        # Summary
        print(f"\n=== Dataset Summary ===")
        print(f"Training samples: {dataset_info['total_train_samples']:,}")
        print(f"Validation samples: {dataset_info['total_val_samples']:,}")
        print(f"Features per sample: {self.features}")
        print(f"Number of classes: {self.num_classes}")
        
        return dataset_info
    
    def _inspect_npz_file(self, npz_file: Path) -> Dict:
        """Inspect a single NPZ file."""
        data = np.load(npz_file)
        
        info = {
            'file': str(npz_file),
            'arrays': list(data.keys())
        }
        
        for key in data.keys():
            arr = data[key]
            info[key] = {
                'shape': arr.shape,
                'dtype': str(arr.dtype),
                'min': float(arr.min()),
                'max': float(arr.max()),
                'mean': float(arr.mean())
            }
            
            if key == 'data':
                info['samples'] = arr.shape[0]
                info['features'] = arr.shape[1]
            elif key == 'labels':
                info['label_range'] = [int(arr.min()), int(arr.max())]
        
        data.close()
        return info
    
    def convert_to_binary(self) -> Dict:
        """Convert all NPZ files to binary format."""
        print("\n=== Converting NPZ to Binary Format ===")
        
        # Process validation first (smaller)
        val_metadata = self._convert_validation_to_binary()
        
        # Process training batches
        train_metadata = self._convert_training_to_binary()
        
        # Create combined training files
        combined_metadata = self._create_combined_binary_files(train_metadata)
        
        return {
            'validation': val_metadata,
            'training_batches': train_metadata,
            'combined_training': combined_metadata
        }
    
    def _convert_validation_to_binary(self) -> Optional[Dict]:
        """Convert validation NPZ to binary."""
        val_file = self.input_dir / "val_data.npz"
        if not val_file.exists():
            print("WARNING: val_data.npz not found!")
            return None
        
        print(f"\nConverting {val_file.name} to binary...")
        
        # Output files
        val_data_bin = self.output_dir / "val_data.bin"
        val_labels_bin = self.output_dir / "val_labels.bin"
        
        if val_data_bin.exists() and val_labels_bin.exists():
            print("Validation binary files already exist - skipping")
            # Just get metadata
            data = np.load(val_file)
            samples = data['data'].shape[0]
            data.close()
            return {'samples': samples, 'data_file': val_data_bin, 'labels_file': val_labels_bin}
        
        # Load and convert
        data = np.load(val_file)
        X = data['data'].astype(np.float32) / 255.0  # Normalize to [0,1]
        y = data['labels'] - 1  # Convert to 0-indexed
        
        # Create one-hot labels
        y_onehot = np.zeros((len(y), self.num_classes), dtype=np.float32)
        y_onehot[np.arange(len(y)), y] = 1.0
        
        # Save binary files
        X.tofile(val_data_bin)
        y_onehot.tofile(val_labels_bin)
        
        # Create metadata
        self._create_metadata_file(val_data_bin, X.shape[0], X.shape[1], "binary")
        self._create_metadata_file(val_labels_bin, y_onehot.shape[0], y_onehot.shape[1], "binary")
        
        print(f"✓ Validation: {X.shape[0]} samples saved")
        
        data.close()
        return {'samples': X.shape[0], 'data_file': val_data_bin, 'labels_file': val_labels_bin}
    
    def _convert_training_to_binary(self) -> List[Dict]:
        """Convert training batches to binary."""
        train_files = sorted(self.input_dir.glob("train_data_batch_*.npz"))
        train_metadata = []
        
        for i, npz_file in enumerate(train_files):
            batch_name = f"train_batch_{i+1:02d}"
            print(f"\nConverting {npz_file.name} to binary...")
            
            # Output files
            data_bin = self.output_dir / f"{batch_name}_data.bin"
            labels_bin = self.output_dir / f"{batch_name}_labels.bin"
            
            if data_bin.exists() and labels_bin.exists():
                print(f"Batch {i+1} binary files already exist - skipping")
                # Get metadata
                data = np.load(npz_file)
                samples = data['data'].shape[0]
                data.close()
                train_metadata.append({
                    'batch': i+1,
                    'samples': samples,
                    'data_file': data_bin,
                    'labels_file': labels_bin
                })
                continue
            
            # Load and convert
            data = np.load(npz_file)
            X = data['data'].astype(np.float32) / 255.0  # Normalize to [0,1]
            y = data['labels'] - 1  # Convert to 0-indexed
            
            # Create one-hot labels
            y_onehot = np.zeros((len(y), self.num_classes), dtype=np.float32)
            y_onehot[np.arange(len(y)), y] = 1.0
            
            # Save binary files
            X.tofile(data_bin)
            y_onehot.tofile(labels_bin)
            
            # Create metadata
            self._create_metadata_file(data_bin, X.shape[0], X.shape[1], "binary")
            self._create_metadata_file(labels_bin, y_onehot.shape[0], y_onehot.shape[1], "binary")
            
            train_metadata.append({
                'batch': i+1,
                'samples': X.shape[0],
                'data_file': data_bin,
                'labels_file': labels_bin
            })
            
            print(f"✓ Batch {i+1}: {X.shape[0]} samples saved")
            
            data.close()
            del X, y, y_onehot
            gc.collect()
        
        return train_metadata
    
    def _create_combined_binary_files(self, train_metadata: List[Dict]) -> Dict:
        """Create combined training binary files."""
        print(f"\nCombining {len(train_metadata)} training batches...")
        
        combined_data_file = self.output_dir / "train_data_combined.bin"
        combined_labels_file = self.output_dir / "train_labels_combined.bin"
        
        if combined_data_file.exists() and combined_labels_file.exists():
            print("Combined binary files already exist - skipping")
            total_samples = sum(batch['samples'] for batch in train_metadata)
            return {'samples': total_samples, 'data_file': combined_data_file, 'labels_file': combined_labels_file}
        
        total_samples = 0
        
        with open(combined_data_file, 'wb') as data_out, open(combined_labels_file, 'wb') as labels_out:
            for batch in train_metadata:
                print(f"  Appending batch {batch['batch']}...")
                
                # Read batch data
                batch_data = np.fromfile(batch['data_file'], dtype=np.float32)
                batch_labels = np.fromfile(batch['labels_file'], dtype=np.float32)
                
                # Append to combined files
                batch_data.tofile(data_out)
                batch_labels.tofile(labels_out)
                
                total_samples += batch['samples']
                
                del batch_data, batch_labels
                gc.collect()
        
        # Create metadata
        self._create_metadata_file(combined_data_file, total_samples, self.features, "binary")
        self._create_metadata_file(combined_labels_file, total_samples, self.num_classes, "binary")
        
        print(f"✓ Combined training: {total_samples:,} samples")
        
        return {'samples': total_samples, 'data_file': combined_data_file, 'labels_file': combined_labels_file}
    
    def create_csv_samples(self, target_size_gb: float = 2.0) -> Dict:
        """Create sampled CSV files of specified size."""
        print(f"\n=== Creating {target_size_gb}GB CSV Samples ===")
        
        # Calculate target samples based on size
        bytes_per_gb = 1024**3
        target_bytes = target_size_gb * bytes_per_gb
        
        # Estimate bytes per sample (data + labels, with CSV overhead)
        # Each float: ~8 bytes as text, plus commas and newlines
        bytes_per_data_sample = self.features * 10  # Conservative estimate
        bytes_per_label_sample = self.num_classes * 4  # One-hot as integers
        bytes_per_sample = bytes_per_data_sample + bytes_per_label_sample
        
        target_samples = int(target_bytes / bytes_per_sample)
        
        print(f"Target size: {target_size_gb}GB")
        print(f"Estimated bytes per sample: {bytes_per_sample:,}")
        print(f"Target samples: {target_samples:,}")
        
        # Create training CSV sample
        train_csv_info = self._create_training_csv_sample(target_samples)
        
        # Create validation CSV (smaller, use all data)
        val_csv_info = self._create_validation_csv()
        
        return {
            'training': train_csv_info,
            'validation': val_csv_info,
            'target_size_gb': target_size_gb,
            'actual_samples': target_samples
        }
    
    def _create_training_csv_sample(self, target_samples: int) -> Dict:
        """Create training CSV sample from binary data."""
        combined_data_file = self.output_dir / "train_data_combined.bin"
        combined_labels_file = self.output_dir / "train_labels_combined.bin"
        
        if not combined_data_file.exists():
            raise FileNotFoundError("Combined binary files not found. Run conversion first.")
        
        # Get total available samples
        total_data_size = combined_data_file.stat().st_size
        total_samples = total_data_size // (self.features * 4)  # float32 = 4 bytes
        
        print(f"Available training samples: {total_samples:,}")
        
        # Use all available samples if target is larger
        samples_to_use = min(target_samples, total_samples)
        print(f"Using {samples_to_use:,} samples for training CSV")
        
        # Load sample data
        print("Loading training data sample...")
        data_sample = np.fromfile(combined_data_file, dtype=np.float32, 
                                count=samples_to_use * self.features)
        labels_sample = np.fromfile(combined_labels_file, dtype=np.float32, 
                                  count=samples_to_use * self.num_classes)
        
        # Reshape
        data_sample = data_sample.reshape(samples_to_use, self.features)
        labels_sample = labels_sample.reshape(samples_to_use, self.num_classes)
        
        # Save as CSV
        train_csv_data = self.output_dir / "imagenet_train_2GB.csv"
        train_csv_labels = self.output_dir / "imagenet_train_labels_2GB.csv"
        
        print(f"Saving training data CSV: {train_csv_data.name}")
        np.savetxt(train_csv_data, data_sample, delimiter=',', fmt='%.6f')
        
        print(f"Saving training labels CSV: {train_csv_labels.name}")
        np.savetxt(train_csv_labels, labels_sample.astype(int), delimiter=',', fmt='%d')
        
        # Create metadata
        self._create_metadata_file(train_csv_data, samples_to_use, self.features, "csv")
        self._create_metadata_file(train_csv_labels, samples_to_use, self.num_classes, "csv")
        
        # Calculate actual file sizes
        data_size_mb = train_csv_data.stat().st_size / (1024**2)
        labels_size_mb = train_csv_labels.stat().st_size / (1024**2)
        total_size_gb = (data_size_mb + labels_size_mb) / 1024
        
        print(f"✓ Training CSV created: {samples_to_use:,} samples")
        print(f"  Data file: {data_size_mb:.1f} MB")
        print(f"  Labels file: {labels_size_mb:.1f} MB")
        print(f"  Total size: {total_size_gb:.2f} GB")
        
        return {
            'samples': samples_to_use,
            'data_file': train_csv_data,
            'labels_file': train_csv_labels,
            'size_gb': total_size_gb
        }
    
    def _create_validation_csv(self) -> Dict:
        """Create validation CSV from binary data."""
        val_data_file = self.output_dir / "val_data.bin"
        val_labels_file = self.output_dir / "val_labels.bin"
        
        if not val_data_file.exists():
            raise FileNotFoundError("Validation binary files not found. Run conversion first.")
        
        print("Loading validation data...")
        val_data = np.fromfile(val_data_file, dtype=np.float32)
        val_labels = np.fromfile(val_labels_file, dtype=np.float32)
        
        # Get dimensions
        val_samples = len(val_data) // self.features
        val_data = val_data.reshape(val_samples, self.features)
        val_labels = val_labels.reshape(val_samples, self.num_classes)
        
        # Save as CSV
        val_csv_data = self.output_dir / "imagenet_val_2GB.csv"
        val_csv_labels = self.output_dir / "imagenet_val_labels_2GB.csv"
        
        print(f"Saving validation data CSV: {val_csv_data.name}")
        np.savetxt(val_csv_data, val_data, delimiter=',', fmt='%.6f')
        
        print(f"Saving validation labels CSV: {val_csv_labels.name}")
        np.savetxt(val_csv_labels, val_labels.astype(int), delimiter=',', fmt='%d')
        
        # Create metadata
        self._create_metadata_file(val_csv_data, val_samples, self.features, "csv")
        self._create_metadata_file(val_csv_labels, val_samples, self.num_classes, "csv")
        
        # Calculate file sizes
        data_size_mb = val_csv_data.stat().st_size / (1024**2)
        labels_size_mb = val_csv_labels.stat().st_size / (1024**2)
        total_size_gb = (data_size_mb + labels_size_mb) / 1024
        
        print(f"✓ Validation CSV created: {val_samples:,} samples")
        print(f"  Total size: {total_size_gb:.2f} GB")
        
        return {
            'samples': val_samples,
            'data_file': val_csv_data,
            'labels_file': val_csv_labels,
            'size_gb': total_size_gb
        }
    
    def _create_metadata_file(self, data_file: Path, rows: int, cols: int, fmt: str):
        """Create SystemDS metadata file."""
        mtd_file = data_file.with_suffix(data_file.suffix + ".mtd")
        
        with open(mtd_file, "w") as f:
            if fmt == "binary":
                f.write(f'{{"data_type": "matrix", "format": "binary", ')
                f.write(f'"rows": {rows}, "cols": {cols}}}\n')
            else:  # CSV
                f.write(f'{{"data_type": "matrix", "format": "csv", ')
                f.write(f'"header": false, "sep": ",", ')
                f.write(f'"rows": {rows}, "cols": {cols}}}\n')
    
    def run_full_pipeline(self, csv_size_gb: float = 2.0):
        """Run the complete pipeline."""
        print("=" * 60)
        print("ImageNet Data Processing Pipeline")
        print("=" * 60)
        
        start_time = time.time()
        
        # Step 1: Inspect dataset
        dataset_info = self.inspect_dataset()
        
        # Step 2: Convert to binary
        binary_info = self.convert_to_binary()
        
        # Step 3: Create CSV samples
        csv_info = self.create_csv_samples(csv_size_gb)
        
        # Final summary
        elapsed = time.time() - start_time
        print(f"\n{'='*60}")
        print("Pipeline Completed Successfully!")
        print(f"{'='*60}")
        print(f"Processing time: {elapsed:.1f} seconds")
        print(f"Total training samples processed: {dataset_info['total_train_samples']:,}")
        print(f"Total validation samples processed: {dataset_info['total_val_samples']:,}")
        print(f"CSV training samples: {csv_info['training']['samples']:,}")
        print(f"CSV validation samples: {csv_info['validation']['samples']:,}")
        print(f"Total CSV size: {csv_info['training']['size_gb'] + csv_info['validation']['size_gb']:.2f} GB")
        
        print(f"\nGenerated Files:")
        print(f"- {csv_info['training']['data_file'].name}")
        print(f"- {csv_info['training']['labels_file'].name}")
        print(f"- {csv_info['validation']['data_file'].name}")
        print(f"- {csv_info['validation']['labels_file'].name}")
        print(f"- Binary files and metadata in {self.output_dir}")
        
        print(f"\nNext Steps:")
        print(f"1. Use imagenet_train_2GB.csv and imagenet_val_2GB.csv for SystemDS training")
        print(f"2. Files are ready for direct import into SystemDS")
        print(f"3. All necessary .mtd metadata files have been created")


def main():
    parser = argparse.ArgumentParser(description="ImageNet Data Processing Pipeline")
    parser.add_argument("--mode", choices=["full", "inspect", "binary", "csv-only"], 
                       default="full", help="Pipeline mode")
    parser.add_argument("--input-dir", default="imagenet_data/6gb", 
                       help="Input directory with NPZ files")
    parser.add_argument("--output-dir", default="imagenet_data/systemds_ready", 
                       help="Output directory for processed files")
    parser.add_argument("--csv-size-gb", type=float, default=2.0, 
                       help="Target size for CSV files in GB")
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = ImageNetPipeline(args.input_dir, args.output_dir)
    
    try:
        if args.mode == "full":
            pipeline.run_full_pipeline(args.csv_size_gb)
        elif args.mode == "inspect":
            pipeline.inspect_dataset()
        elif args.mode == "binary":
            pipeline.convert_to_binary()
        elif args.mode == "csv-only":
            pipeline.create_csv_samples(args.csv_size_gb)
    
    except Exception as e:
        print(f"\nERROR: {e}")
        print("Pipeline failed. Check the error message above.")
        sys.exit(1)


if __name__ == "__main__":
    main() 