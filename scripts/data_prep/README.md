# ImageNet Data Processing Pipeline

This directory contains a unified pipeline for processing ImageNet data from NPZ format to SystemDS-compatible CSV files.

## Overview

The pipeline performs these key steps:
1. **Inspect** NPZ files to understand data structure
2. **Convert** NPZ files to efficient binary format
3. **Sample** and create 2GB CSV files for SystemDS training

## Files

- `imagenet_pipeline.py` - Main pipeline class with full functionality
- `run_imagenet_pipeline.py` - Simple runner script (recommended)
- `binary_to_csv_chunks.py` - Legacy chunking script (from original workflow)
- `convert_6gb_imagenet.py` - Legacy conversion script
- `inspect_6gb_imagenet.py` - Legacy inspection script

## Quick Start

### 1. Run the Complete Pipeline (Recommended)

```bash
cd scripts/data_prep
python run_imagenet_pipeline.py
```

This will:
- Process all NPZ files in `imagenet_data/6gb/`
- Create binary files in `imagenet_data/systemds_ready/`
- Generate the final 2GB CSV files:
  - `imagenet_train_2GB.csv`
  - `imagenet_train_labels_2GB.csv`
  - `imagenet_val_2GB.csv`
  - `imagenet_val_labels_2GB.csv`

### 2. Just Inspect Data (Optional)

```bash
python run_imagenet_pipeline.py inspect
```

### 3. Advanced Usage

```bash
# Full control with custom parameters
python imagenet_pipeline.py --mode full --csv-size-gb 3.0

# Only create CSV files (if binary files already exist)
python imagenet_pipeline.py --mode csv-only --csv-size-gb 1.5

# Only convert to binary format
python imagenet_pipeline.py --mode binary

# Just inspect the data
python imagenet_pipeline.py --mode inspect
```

## Expected Data Structure

### Input (NPZ files in `imagenet_data/6gb/`):
```
imagenet_data/6gb/
├── train_data_batch_1.npz
├── train_data_batch_2.npz
├── ...
├── train_data_batch_10.npz
└── val_data.npz
```

### Output (in `imagenet_data/systemds_ready/`):
```
imagenet_data/systemds_ready/
├── imagenet_train_2GB.csv          # Final training data
├── imagenet_train_labels_2GB.csv   # Final training labels
├── imagenet_val_2GB.csv            # Final validation data
├── imagenet_val_labels_2GB.csv     # Final validation labels
├── *.mtd                           # SystemDS metadata files
└── [binary files...]               # Intermediate binary files
```

## NPZ File Format

The pipeline expects NPZ files with:
- `data`: Shape (N, 12288) uint8 [0, 255] - 64x64x3 flattened images
- `labels`: Shape (N,) int64 [1, 1000] - 1-indexed class labels
- `mean`: Shape (12288,) float64 - optional per-pixel mean

## Data Transformations

1. **Normalization**: Images converted from uint8 [0,255] to float32 [0,1]
2. **Label Conversion**: 1-indexed labels → 0-indexed labels → one-hot encoding
3. **Format**: Binary format for efficiency, CSV format for SystemDS compatibility

## Memory and Storage

- **Memory**: Processes data in batches to minimize RAM usage
- **Storage**: Creates both binary (efficient) and CSV (compatible) formats
- **Size**: 2GB CSV files are sampled from the full dataset

## Troubleshooting

### Common Issues

1. **"No NPZ files found"**
   - Check that NPZ files are in the correct directory
   - Verify file names match the expected pattern

2. **"Combined binary files not found"**
   - Run the full pipeline (`--mode full`) first
   - Or run binary conversion (`--mode binary`) before CSV creation

3. **Memory errors**
   - The pipeline processes data in batches
   - If still having issues, try reducing `--csv-size-gb`

4. **Permission errors**
   - Check write permissions in output directory
   - Make sure the output directory can be created

### Validation

The pipeline includes automatic validation:
- Checks data ranges and formats
- Validates one-hot encoding
- Verifies file sizes and metadata

## SystemDS Integration

The generated CSV files are ready for direct use in SystemDS:

```dml
# Load the processed data
X_train = read("imagenet_train_2GB.csv", format="csv");
Y_train = read("imagenet_train_labels_2GB.csv", format="csv");
X_val = read("imagenet_val_2GB.csv", format="csv");
Y_val = read("imagenet_val_labels_2GB.csv", format="csv");

print("Training data: " + nrow(X_train) + " x " + ncol(X_train));
print("Validation data: " + nrow(X_val) + " x " + ncol(X_val));
```

## Performance

- **Processing time**: ~5-15 minutes depending on system
- **Disk space**: Requires ~15-20GB free space during processing
- **Final output**: ~2-4GB total for CSV files

## Migration from Legacy Scripts

If you were using the old scripts:
- `inspect_6gb_imagenet.py` → use `run_imagenet_pipeline.py inspect`
- `convert_6gb_imagenet.py` → use `run_imagenet_pipeline.py`
- `binary_to_csv_chunks.py` → functionality integrated into main pipeline

The new pipeline is more efficient and produces the exact output format you requested. 