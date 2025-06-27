#!/usr/bin/env python3
"""
Convert combined raw binary files to CSV chunks for SystemDS.
Reads the large binary files and creates smaller CSV chunks.
"""

import numpy as np
import pandas as pd
from pathlib import Path

def binary_to_csv_chunks(data_dir, chunk_size=10000):
    """
    Convert combined binary files to CSV chunks.
    
    Args:
        data_dir: Directory containing binary files
        chunk_size: Number of samples per CSV chunk
    """
    data_dir = Path(data_dir)
    
    print(f"Converting binary files to CSV chunks of {chunk_size} samples...")
    
    # File paths
    train_data_bin = data_dir / "train_data_combined.bin"
    train_labels_bin = data_dir / "train_labels_combined.bin"
    val_data_bin = data_dir / "val_data.bin" 
    val_labels_bin = data_dir / "val_labels.bin"
    
    # Dataset dimensions
    total_train_samples = 1281167  # From your output
    features = 12288  # 64x64x3
    classes = 1000
    val_samples = 50000
    
    print(f"Dataset info:")
    print(f"- Training samples: {total_train_samples:,}")
    print(f"- Features: {features}")
    print(f"- Classes: {classes}")
    print(f"- Validation samples: {val_samples}")
    print()
    
    # Convert validation data (smaller, single chunk)
    print("Converting validation data...")
    val_data = np.fromfile(val_data_bin, dtype=np.float32).reshape(val_samples, features)
    val_labels = np.fromfile(val_labels_bin, dtype=np.float32).reshape(val_samples, classes)
    
    val_data_csv = data_dir / "val_data.csv"
    val_labels_csv = data_dir / "val_labels.csv"
    
    print(f"- Saving {val_data_csv.name}")
    np.savetxt(val_data_csv, val_data, delimiter=',', fmt='%.6f')
    print(f"- Saving {val_labels_csv.name}")
    np.savetxt(val_labels_csv, val_labels.astype(int), delimiter=',', fmt='%d')
    
    # Create metadata for validation
    create_metadata_file(val_data_csv, val_samples, features)
    create_metadata_file(val_labels_csv, val_samples, classes)
    
    # Convert training data in chunks
    print(f"Converting training data in chunks of {chunk_size}...")
    
    num_chunks = (total_train_samples + chunk_size - 1) // chunk_size
    print(f"Will create {num_chunks} training chunks")
    
    # Open binary files for reading
    with open(train_data_bin, 'rb') as data_file, open(train_labels_bin, 'rb') as labels_file:
        
        for chunk_idx in range(num_chunks):
            start_sample = chunk_idx * chunk_size
            end_sample = min(start_sample + chunk_size, total_train_samples)
            actual_chunk_size = end_sample - start_sample
            
            print(f"  Chunk {chunk_idx + 1}/{num_chunks}: samples {start_sample} to {end_sample-1}")
            
            # Read chunk from binary files
            data_bytes = actual_chunk_size * features * 4  # float32 = 4 bytes
            labels_bytes = actual_chunk_size * classes * 4
            
            # Seek to correct position
            data_file.seek(start_sample * features * 4)
            labels_file.seek(start_sample * classes * 4)
            
            # Read chunk data
            chunk_data = np.frombuffer(data_file.read(data_bytes), dtype=np.float32)
            chunk_labels = np.frombuffer(labels_file.read(labels_bytes), dtype=np.float32)
            
            # Reshape
            chunk_data = chunk_data.reshape(actual_chunk_size, features)
            chunk_labels = chunk_labels.reshape(actual_chunk_size, classes)
            
            # Save as CSV
            chunk_data_csv = data_dir / f"train_data_chunk_{chunk_idx:03d}.csv"
            chunk_labels_csv = data_dir / f"train_labels_chunk_{chunk_idx:03d}.csv"
            
            np.savetxt(chunk_data_csv, chunk_data, delimiter=',', fmt='%.6f')
            np.savetxt(chunk_labels_csv, chunk_labels.astype(int), delimiter=',', fmt='%d')
            
            # Create metadata
            create_metadata_file(chunk_data_csv, actual_chunk_size, features)
            create_metadata_file(chunk_labels_csv, actual_chunk_size, classes)
            
            print(f"    Saved {chunk_data_csv.name} and {chunk_labels_csv.name}")
    
    print("\nConversion completed!")
    print(f"Generated {num_chunks} training chunks + validation files")
    print("Files are ready for SystemDS!")
    
    # Create a simple test chunk for quick testing
    print("\nCreating test subset (first 5000 samples)...")
    test_data = np.fromfile(train_data_bin, dtype=np.float32, count=5000 * features).reshape(5000, features)
    test_labels = np.fromfile(train_labels_bin, dtype=np.float32, count=5000 * classes).reshape(5000, classes)
    
    test_data_csv = data_dir / "train_data_test.csv"
    test_labels_csv = data_dir / "train_labels_test.csv"
    
    np.savetxt(test_data_csv, test_data, delimiter=',', fmt='%.6f')
    np.savetxt(test_labels_csv, test_labels.astype(int), delimiter=',', fmt='%d')
    
    create_metadata_file(test_data_csv, 5000, features)
    create_metadata_file(test_labels_csv, 5000, classes)
    
    print(f"Test files: {test_data_csv.name}, {test_labels_csv.name}")

def create_metadata_file(csv_file, rows, cols):
    """Create SystemDS metadata file."""
    mtd_file = str(csv_file) + ".mtd"
    with open(mtd_file, "w") as f:
        f.write(f'{{"data_type": "matrix", "format": "csv", ')
        f.write(f'"header": false, "sep": ",", ')
        f.write(f'"rows": {rows}, "cols": {cols}}}\n')

if __name__ == "__main__":
    binary_to_csv_chunks("imagenet_data/systemds_ready", chunk_size=20000)