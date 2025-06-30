#!/usr/bin/env python3
"""
Debug ImageNet CSV Data
Checks if the processed ImageNet data has any issues that could cause training problems
"""

import numpy as np
import pandas as pd
import os

def check_csv_file(filepath, name):
    """Check a CSV file and return basic statistics"""
    if not os.path.exists(filepath):
        print(f"❌ {name}: File not found: {filepath}")
        return None
    
    print(f"\n=== {name} ===")
    print(f"📁 Path: {filepath}")
    
    # Check file size
    size_mb = os.path.getsize(filepath) / (1024 * 1024)
    print(f"📊 File size: {size_mb:.1f} MB")
    
    # Load and check data
    try:
        data = pd.read_csv(filepath, header=None)
        print(f"📐 Shape: {data.shape}")
        print(f"📊 Data range: [{data.values.min():.6f}, {data.values.max():.6f}]")
        print(f"📊 Data mean: {data.values.mean():.6f}")
        print(f"📊 Data std: {data.values.std():.6f}")
        
        # Check for NaN or infinite values
        nan_count = data.isnull().sum().sum()
        inf_count = np.isinf(data.values).sum()
        print(f"🔍 NaN values: {nan_count}")
        print(f"🔍 Infinite values: {inf_count}")
        
        return data
    except Exception as e:
        print(f"❌ Error loading {name}: {e}")
        return None

def check_labels(data, name):
    """Check if labels are properly one-hot encoded"""
    print(f"\n=== {name} Label Analysis ===")
    
    # Check row sums (should all be 1.0 for one-hot)
    row_sums = data.sum(axis=1)
    print(f"🏷️ Row sums - min: {row_sums.min():.6f}, max: {row_sums.max():.6f}, mean: {row_sums.mean():.6f}")
    
    # Check number of classes
    num_classes = data.shape[1]
    print(f"🏷️ Number of classes: {num_classes}")
    
    # Check class distribution
    class_indices = np.argmax(data.values, axis=1)
    unique_classes = np.unique(class_indices)
    print(f"🏷️ Unique classes range: [{unique_classes.min()}, {unique_classes.max()}]")
    print(f"🏷️ Number of unique classes: {len(unique_classes)}")
    
    # Check class frequency distribution
    class_counts = np.bincount(class_indices)
    non_zero_classes = np.nonzero(class_counts)[0]
    print(f"🏷️ Classes with samples: {len(non_zero_classes)}")
    print(f"🏷️ Sample distribution - min: {class_counts[non_zero_classes].min()}, max: {class_counts[non_zero_classes].max()}")
    
    return class_indices

def main():
    print("🔍 ImageNet Data Debug")
    print("=" * 50)
    
    base_path = "imagenet_data/systemds_ready"
    
    # Check all files
    files_to_check = [
    ("imagenet_train_2GB.csv", "Training Data"),
        ("imagenet_train_labels_2GB.csv", "Training Labels"),
        ("imagenet_val_2GB.csv", "Validation Data"),
        ("imagenet_val_labels_2GB.csv", "Validation Labels")
    ]
    
    data_dict = {}
    
    for filename, display_name in files_to_check:
        filepath = os.path.join(base_path, filename)
        data = check_csv_file(filepath, display_name)
        data_dict[filename] = data
    
    # Check label formatting
    if data_dict["imagenet_train_labels_2GB.csv"] is not None:
        train_classes = check_labels(data_dict["imagenet_train_labels_2GB.csv"], "Training")
    
    if data_dict["imagenet_val_labels_2GB.csv"] is not None:
        val_classes = check_labels(data_dict["imagenet_val_labels_2GB.csv"], "Validation")
    
    # Check data-label alignment
    print(f"\n=== Data-Label Alignment ===")
    train_data = data_dict["imagenet_train_2GB.csv"]
    train_labels = data_dict["imagenet_train_labels_2GB.csv"]
    val_data = data_dict["imagenet_val_2GB.csv"]
    val_labels = data_dict["imagenet_val_labels_2GB.csv"]
    
    if train_data is not None and train_labels is not None:
        print(f"✅ Train data/labels alignment: {train_data.shape[0]} == {train_labels.shape[0]} ✓" if train_data.shape[0] == train_labels.shape[0] else f"❌ Train alignment issue: {train_data.shape[0]} != {train_labels.shape[0]}")
    
    if val_data is not None and val_labels is not None:
        print(f"✅ Val data/labels alignment: {val_data.shape[0]} == {val_labels.shape[0]} ✓" if val_data.shape[0] == val_labels.shape[0] else f"❌ Val alignment issue: {val_data.shape[0]} != {val_labels.shape[0]}")
    
    # Sample some examples
    if train_data is not None and train_labels is not None:
        print(f"\n=== Sample Training Examples ===")
        for i in range(min(3, len(train_data))):
            pixel_sum = train_data.iloc[i].sum()
            class_idx = np.argmax(train_labels.iloc[i].values)
            print(f"Example {i+1}: pixel_sum={pixel_sum:.3f}, class={class_idx}")
    
    # Check expected vs actual dimensions
    print(f"\n=== Expected Dimensions ===")
    print(f"Expected image features: 64x64x3 = 12,288")
    print(f"Expected classes: 1,000")
    
    if train_data is not None:
        print(f"Actual train features: {train_data.shape[1]} {'✅' if train_data.shape[1] == 12288 else '❌'}")
    if train_labels is not None:
        print(f"Actual train classes: {train_labels.shape[1]} {'✅' if train_labels.shape[1] == 1000 else '❌'}")
    
    print(f"\n🔍 Debug Complete!")

if __name__ == "__main__":
    main() 