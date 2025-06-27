#!/usr/bin/env python3
"""
Convert raw binary files to CSV format for SystemDS.
Creates a subset for testing since full dataset is too large.
"""

import numpy as np
import pandas as pd
from pathlib import Path

def convert_subset_to_csv(data_dir, subset_size=10000):
    """Convert a subset of raw binary files to CSV."""
    
    data_dir = Path(data_dir)
    
    print(f"Converting subset of {subset_size} samples to CSV...")
    
    # Load validation data (smaller, good for testing)
    print("Loading validation data...")
    val_data = np.fromfile(data_dir / "val_data.bin", dtype=np.float32)
    val_labels = np.fromfile(data_dir / "val_labels.bin", dtype=np.float32)
    
    # Reshape validation data
    val_data = val_data.reshape(50000, 12288)
    val_labels = val_labels.reshape(50000, 1000)
    
    print(f"Validation data shape: {val_data.shape}")
    print(f"Validation labels shape: {val_labels.shape}")
    
    # Save validation as CSV
    print("Saving validation CSV files...")
    pd.DataFrame(val_data).to_csv(data_dir / "val_data.csv", header=False, index=False, float_format='%.6f')
    pd.DataFrame(val_labels.astype(int)).to_csv(data_dir / "val_labels.csv", header=False, index=False)
    
    # Load subset of training data
    print(f"Loading first {subset_size} training samples...")
    
    # Calculate bytes to read
    samples_to_read = subset_size
    features = 12288
    classes = 1000
    
    # Read subset of training data
    train_data_bytes = samples_to_read * features * 4  # float32 = 4 bytes
    train_labels_bytes = samples_to_read * classes * 4
    
    train_data = np.fromfile(data_dir / "train_data_combined.bin", dtype=np.float32, count=samples_to_read * features)
    train_labels = np.fromfile(data_dir / "train_labels_combined.bin", dtype=np.float32, count=samples_to_read * classes)
    
    # Reshape training data
    train_data = train_data.reshape(samples_to_read, features)
    train_labels = train_labels.reshape(samples_to_read, classes)
    
    print(f"Training subset shape: {train_data.shape}")
    print(f"Training labels subset shape: {train_labels.shape}")
    
    # Save training subset as CSV
    print("Saving training subset CSV files...")
    pd.DataFrame(train_data).to_csv(data_dir / "train_data_subset.csv", header=False, index=False, float_format='%.6f')
    pd.DataFrame(train_labels.astype(int)).to_csv(data_dir / "train_labels_subset.csv", header=False, index=False)
    
    # Create metadata files
    create_metadata_file(data_dir / "train_data_subset.csv", samples_to_read, features)
    create_metadata_file(data_dir / "train_labels_subset.csv", samples_to_read, classes)
    create_metadata_file(data_dir / "val_data.csv", 50000, features)
    create_metadata_file(data_dir / "val_labels.csv", 50000, classes)
    
    print("Conversion completed!")
    print("\nGenerated files:")
    print("- train_data_subset.csv (for testing)")
    print("- train_labels_subset.csv")
    print("- val_data.csv")
    print("- val_labels.csv")
    print("- *.mtd metadata files")

def create_metadata_file(csv_file, rows, cols):
    """Create SystemDS metadata file."""
    mtd_file = str(csv_file) + ".mtd"
    with open(mtd_file, "w") as f:
        f.write(f'{{"data_type": "matrix", "format": "csv", ')
        f.write(f'"header": false, "sep": ",", ')
        f.write(f'"rows": {rows}, "cols": {cols}}}\n')

if __name__ == "__main__":
    convert_subset_to_csv("imagenet_data/systemds_ready", subset_size=10000)