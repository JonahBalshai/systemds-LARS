#!/usr/bin/env python3
"""
Inspect NPZ file contents to understand the data format.
"""

import numpy as np
import sys

def inspect_npz(file_path):
    """Inspect contents of an NPZ file."""
    print(f"Inspecting: {file_path}")
    print("=" * 50)
    
    # Load the NPZ file
    data = np.load(file_path)
    
    # List all arrays in the file
    print(f"Arrays in file: {list(data.keys())}")
    print()
    
    # Inspect each array
    for key in data.keys():
        arr = data[key]
        print(f"Array: '{key}'")
        print(f"  Shape: {arr.shape}")
        print(f"  Dtype: {arr.dtype}")
        print(f"  Min: {arr.min()}")
        print(f"  Max: {arr.max()}")
        print(f"  Mean: {arr.mean():.4f}")
        
        # Show sample values
        if len(arr.shape) == 1:
            print(f"  First 10 values: {arr[:10]}")
        elif len(arr.shape) == 2:
            print(f"  First row shape: {arr[0].shape}")
            print(f"  Sample values: {arr[0, :5]}...")
        elif len(arr.shape) == 4:
            print(f"  Image shape: {arr.shape[1:]} (H, W, C)")
            print(f"  First image pixel values: {arr[0, 0, 0, :]}")
        
        print()
    
    # Memory usage
    total_size = sum(data[key].nbytes for key in data.keys())
    print(f"Total memory usage: {total_size / (1024**3):.2f} GB")
    
    data.close()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inspect_npz.py <npz_file>")
        print("Example: python inspect_npz.py imagenet_data/6gb/train_data_batch_1.npz")
    else:
        inspect_npz(sys.argv[1])

# python scripts/data_prep/inspect_npz.py imagenet_data/6gb/train_data_batch_1.npz