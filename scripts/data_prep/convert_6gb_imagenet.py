#!/usr/bin/env python3
"""
Convert 6GB ImageNet NPZ dataset to SystemDS-compatible binary format.

Based on inspection results:
- Data: (N, 12288) uint8 [0, 255] - 64x64x3 flattened images
- Labels: (N,) int64 [1, 1000] - 1-indexed class labels
- Mean: (12288,) float64 - per-pixel mean for normalization
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

def convert_batch(npz_file, output_dir, batch_name):
    """
    Convert a single NPZ batch to binary format.
    
    Args:
        npz_file: Path to NPZ file
        output_dir: Output directory for binary files
        batch_name: Name prefix for output files
        
    Returns:
        dict: Metadata about the converted batch
    """
    # Check if binary files already exist
    data_file = output_dir / f"{batch_name}_data.bin"
    labels_file = output_dir / f"{batch_name}_labels.bin"
    
    if data_file.exists() and labels_file.exists():
        print(f"\nSkipping {npz_file} - binary files already exist")
        # Load NPZ just to get metadata
        data = np.load(npz_file)
        N = data['data'].shape[0]
        D = data['data'].shape[1]
        data.close()
        
        return {
            'samples': N,
            'features': D,
            'classes': 1000,
            'data_file': str(data_file),
            'labels_file': str(labels_file),
            'data_size_mb': data_file.stat().st_size / (1024 * 1024),
            'labels_size_mb': labels_file.stat().st_size / (1024 * 1024),
            'mean': None
        }
    
    print(f"\nProcessing {npz_file}...")
    
    # Load NPZ data
    data = np.load(npz_file)
    
    X = data['data']  # Shape: (N, 12288) uint8
    y = data['labels']  # Shape: (N,) int64 [1, 1000]
    mean = data['mean'] if 'mean' in data else None  # Optional per-pixel mean
    
    N = X.shape[0]
    D = X.shape[1]  # 12288 = 64*64*3
    
    print(f"- Samples: {N}")
    print(f"- Features: {D} (64x64x3)")
    print(f"- Label range: [{y.min()}, {y.max()}]")
    
    # Convert data to float32 and normalize to [0, 1] (float32 saves memory)
    X_norm = X.astype(np.float32) / 255.0
    
    # Convert 1-indexed labels to 0-indexed
    y_zero_indexed = y - 1  # Now [0, 999]
    
    # Validate label range
    print(f"- Label range after conversion: [{y_zero_indexed.min()}, {y_zero_indexed.max()}]")
    if y_zero_indexed.min() < 0 or y_zero_indexed.max() >= 1000:
        print(f"ERROR: Labels out of range! Min: {y_zero_indexed.min()}, Max: {y_zero_indexed.max()}")
        return None
    
    # Create one-hot encoded labels as float32
    num_classes = 1000
    y_onehot = np.zeros((N, num_classes), dtype=np.float32)
    y_onehot[np.arange(N), y_zero_indexed.astype(int)] = 1.0
    
    # Validate one-hot encoding
    row_sums = y_onehot.sum(axis=1)
    print(f"- One-hot validation: min_sum={row_sums.min()}, max_sum={row_sums.max()}, mean_sum={row_sums.mean():.3f}")
    if not np.allclose(row_sums, 1.0):
        print("ERROR: One-hot encoding failed!")
        return None
    
    # Save as binary files
    data_file = output_dir / f"{batch_name}_data.bin"
    labels_file = output_dir / f"{batch_name}_labels.bin"
    
    print(f"- Saving data to {data_file.name}")
    X_norm.tofile(data_file)
    
    print(f"- Saving labels to {labels_file.name}")
    y_onehot.tofile(labels_file)
    
    # Calculate memory usage
    data_size_mb = X_norm.nbytes / (1024 * 1024)
    labels_size_mb = y_onehot.nbytes / (1024 * 1024)
    
    metadata = {
        'samples': N,
        'features': D,
        'classes': num_classes,
        'data_file': str(data_file),
        'labels_file': str(labels_file),
        'data_size_mb': data_size_mb,
        'labels_size_mb': labels_size_mb,
        'mean': mean
    }
    
    print(f"- Memory: {data_size_mb:.1f} MB (data) + {labels_size_mb:.1f} MB (labels)")
    
    return metadata

def create_combined_files(output_dir, train_batches, val_metadata):
    """
    Create combined training files and validation files for SystemDS.
    """
    print("\n=== Creating combined binary files ===")
    
    # Check if combined files already exist
    train_data_file = output_dir / "train_data_combined.bin"
    train_labels_file = output_dir / "train_labels_combined.bin"
    subset_data_file = output_dir / "train_data_subset.bin"
    subset_labels_file = output_dir / "train_labels_subset.bin"
    
    if (train_data_file.exists() and train_labels_file.exists() and 
        subset_data_file.exists() and subset_labels_file.exists()):
        print("Combined files already exist - skipping combination step")
        return
    
    # Combine all training batches
    if len(train_batches) > 0:
        print(f"Combining {len(train_batches)} training batches...")
        
        total_samples = sum(metadata['samples'] for metadata in train_batches)
        
        # Save combined binary files by streaming batches
        
        print(f"Streaming combined training data to disk...")
        
        # Open files for writing
        with open(train_data_file, 'wb') as data_out, open(train_labels_file, 'wb') as labels_out:
            for i, metadata in enumerate(train_batches):
                print(f"  Appending {Path(metadata['data_file']).name} ({i+1}/{len(train_batches)})...")
                
                # Read and write one batch at a time
                batch_data = np.fromfile(metadata['data_file'], dtype=np.float32)
                batch_labels = np.fromfile(metadata['labels_file'], dtype=np.float32)
                
                # Append to combined files
                batch_data.tofile(data_out)
                batch_labels.tofile(labels_out)
                
                # Clear memory
                del batch_data, batch_labels
        
        print(f"Combined shape: ({total_samples}, 12288) (data), ({total_samples}, 1000) (labels)")
        
        # Create metadata files for SystemDS
        create_metadata_file(train_data_file, total_samples, 12288, fmt="binary")
        create_metadata_file(train_labels_file, total_samples, 1000, fmt="binary")
        
        print(f"✓ Combined training data: {total_samples} samples")
        
        # Create a smaller subset for testing (from first batch only)
        first_batch = train_batches[0]
        subset_size = min(10000, first_batch['samples'])
        print(f"Creating test subset with {subset_size} samples from first batch...")
        
        # Read first batch and take subset
        batch_data = np.fromfile(first_batch['data_file'], dtype=np.float32).reshape(first_batch['samples'], first_batch['features'])
        batch_labels = np.fromfile(first_batch['labels_file'], dtype=np.float32).reshape(first_batch['samples'], first_batch['classes'])
        
        batch_data[:subset_size].tofile(subset_data_file)
        batch_labels[:subset_size].tofile(subset_labels_file)
        
        create_metadata_file(subset_data_file, subset_size, 12288, fmt="binary")
        create_metadata_file(subset_labels_file, subset_size, 1000, fmt="binary")
        
        print(f"✓ Test subset: {subset_size} samples")
        
        # Clear memory
        del batch_data, batch_labels
    
    # Validation files (already individual)
    if val_metadata:
        print("✓ Validation data ready")

def create_metadata_file(data_file, rows, cols, fmt="binary"):
    """Create SystemDS metadata file."""
    mtd_file = str(data_file) + ".mtd"
    with open(mtd_file, "w") as f:
        if fmt == "binary":
            f.write(f'{{"data_type": "matrix", "format": "binary", ')
            f.write(f'"rows": {rows}, "cols": {cols}}}\n')
        else:
            f.write(f'{{"data_type": "matrix", "format": "csv", ')
            f.write(f'"header": false, "sep": ",", ')
            f.write(f'"rows": {rows}, "cols": {cols}}}\n')

def create_usage_scripts(output_dir):
    """Create example usage scripts for the converted data."""
    
    # DML script for data validation
    validation_script = """# Validate ImageNet binary data format
print("Validating ImageNet binary data...");

# Read combined training data
print("Loading combined training data...");
X_train = read("train_data_combined.bin", format="binary");
Y_train = read("train_labels_combined.bin", format="binary");

# Read validation data
print("Loading validation data...");
X_val = read("val_data.bin", format="binary");
Y_val = read("val_labels.bin", format="binary");

print("Dataset info:");
print("- Training samples: " + nrow(X_train));
print("- Training features: " + ncol(X_train));
print("- Validation samples: " + nrow(X_val));
print("- Classes: " + ncol(Y_train));

# Validate data ranges
print("Data validation:");
print("- X_train range: [" + min(X_train) + ", " + max(X_train) + "]");
print("- Y_train sum per row: " + mean(rowSums(Y_train)));

print("Binary data validation completed!");
"""
    
    with open(output_dir / "validate_binary.dml", "w", encoding="utf-8") as f:
        f.write(validation_script)
    
    # Example training script usage
    usage_script = f"""# Example usage of converted ImageNet binary data

# For ResNet training with subset (quick test):
systemds imagenet_resnet.dml -nvargs \\
  train_data={output_dir}/train_data_subset.bin \\
  train_labels={output_dir}/train_labels_subset.bin \\
  val_data={output_dir}/val_data.bin \\
  val_labels={output_dir}/val_labels.bin \\
  C=3 Hin=64 Win=64 epochs=10 batch_size=256 fmt=binary

# For AlexNet training with full data:
systemds imagenet_alexnet.dml -nvargs \\
  train_data={output_dir}/train_data_combined.bin \\
  train_labels={output_dir}/train_labels_combined.bin \\
  val_data={output_dir}/val_data.bin \\
  val_labels={output_dir}/val_labels.bin \\
  C=3 Hin=64 Win=64 epochs=100 batch_size=1024 fmt=binary

# Validate data first:
systemds validate_binary.dml
"""
    
    with open(output_dir / "usage_examples.sh", "w", encoding="utf-8") as f:
        f.write(usage_script)

def debug_labels(npz_file, num_samples=10):
    """Debug function to check label conversion."""
    print(f"\n=== DEBUG: Checking labels in {npz_file} ===")
    
    data = np.load(npz_file)
    y = data['labels']
    
    print(f"Original labels (first {num_samples}): {y[:num_samples]}")
    
    # Convert to 0-indexed
    y_zero = y - 1
    print(f"Zero-indexed labels: {y_zero[:num_samples]}")
    
    # Create one-hot
    y_onehot = np.zeros((len(y), 1000))
    y_onehot[np.arange(len(y)), y_zero] = 1
    
    # Check first few samples
    for i in range(min(3, len(y))):
        hot_idx = np.where(y_onehot[i] == 1)[0]
        print(f"Sample {i}: original={y[i]}, zero_idx={y_zero[i]}, one_hot_pos={hot_idx}")
    
    print(f"One-hot shape: {y_onehot.shape}")
    print(f"One-hot row sums (first 5): {y_onehot[:5].sum(axis=1)}")
    print(f"One-hot non-zero count: {np.count_nonzero(y_onehot)}")
    
    data.close()

def main():
    """Main conversion function."""
    input_dir = Path("imagenet_data/6gb")
    output_dir = Path("imagenet_data/systemds_ready")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=== 6GB ImageNet NPZ to CSV Converter ===")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Expected format: 64x64x3 images, 1000 classes")
    
    train_batches = []
    val_metadata = None
    
    # Process validation data first
    val_file = input_dir / "val_data.npz"
    if val_file.exists():
        print("\n--- Processing Validation Data ---")
        val_metadata = convert_batch(val_file, output_dir, "val")
    else:
        print("WARNING: val_data.npz not found!")
    
    # Process training batches
    print("\n--- Processing Training Batches ---")
    for i in range(1, 11):  # train_data_batch_1.npz to train_data_batch_10.npz
        batch_file = input_dir / f"train_data_batch_{i}.npz"
        if batch_file.exists():
            metadata = convert_batch(batch_file, output_dir, f"train_batch_{i}")
            train_batches.append(metadata)
        else:
            print(f"WARNING: train_data_batch_{i}.npz not found!")
    
    # Create combined files
    create_combined_files(output_dir, train_batches, val_metadata)
    
    # Create usage scripts
    create_usage_scripts(output_dir)
    
    # Summary
    print("\n=== Conversion Summary ===")
    total_train_samples = sum(batch['samples'] for batch in train_batches)
    val_samples = val_metadata['samples'] if val_metadata else 0
    
    print(f"Training batches processed: {len(train_batches)}")
    print(f"Total training samples: {total_train_samples:,}")
    print(f"Validation samples: {val_samples:,}")
    print(f"Image dimensions: 64x64x3")
    print(f"Classes: 1000")
    print(f"Output directory: {output_dir}")
    
    print("\nGenerated files:")
    print("- train_data_combined.bin / train_labels_combined.bin (full training set)")
    print("- train_data_subset.bin / train_labels_subset.bin (subset for testing)")  
    print("- val_data.bin / val_labels.bin (validation set)")
    print("- validate_binary.dml (script to validate binary files)")
    print("- usage_examples.sh (example commands)")
    
    print("\nNext steps:")
    print("1. Validate data: Run validate_binary.dml to check binary files")
    print("2. Test with subset: Use train_data_subset.bin for quick experiments")
    print("3. Full training: Use train_data_combined.bin for complete training")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "debug":
        # Debug mode - check label conversion
        debug_labels("imagenet_data/6gb/val_data.npz")
        debug_labels("imagenet_data/6gb/train_data_batch_1.npz")
    else:
        main()