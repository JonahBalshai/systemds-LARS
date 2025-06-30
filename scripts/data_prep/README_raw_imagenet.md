# Raw ImageNet Data Preprocessing for SystemDS

This pipeline processes raw ImageNet JPG images with metadata CSV files and prepares them for training with the SystemDS AlexNet implementation.

## Data Structure Expected

Your input directory should contain:

```
C:\Users\romer\Desktop\Code\HuggingFace\
├── imagenet_train_metadata.csv
├── imagenet_test_metadata.csv  
├── train_images/
│   ├── image_0.jpg
│   ├── image_1.jpg
│   └── ...
└── test_images/
    ├── image_0.jpg
    ├── image_1.jpg
    └── ...
```

### Metadata CSV Format

Both `imagenet_train_metadata.csv` and `imagenet_test_metadata.csv` should have:
- Column 1: `file_path` - relative path to the image file
- Column 2: `label` - integer label for the image

Example:
```csv
file_path,label
train_images/image_0.jpg,0
train_images/image_1.jpg,0
train_images/image_2.jpg,1
```

## Requirements

Install required Python packages:
```bash
pip install numpy pandas pillow
```

## Usage

### Option 1: Simple Interactive Runner

Run the interactive script:
```bash
python scripts/data_prep/run_raw_imagenet_preprocessing.py
```

This will present you with options:
1. **Inspect data only** - Check your data structure without processing
2. **Process small sample** - Process 1000 images per split for testing
3. **Process full dataset** - Process all available images
4. **Custom processing** - Specify custom parameters

### Option 2: Direct Script Usage

For more control, run the preprocessing script directly:

```bash
# Inspect data structure (dry run)
python scripts/data_prep/prepare_raw_imagenet.py \
    --input_dir "C:/Users/romer/Desktop/Code/HuggingFace" \
    --dry_run

# Process small sample for testing
python scripts/data_prep/prepare_raw_imagenet.py \
    --input_dir "C:/Users/romer/Desktop/Code/HuggingFace" \
    --output_dir "imagenet_data/systemds_ready" \
    --max_samples 1000

# Process full dataset
python scripts/data_prep/prepare_raw_imagenet.py \
    --input_dir "C:/Users/romer/Desktop/Code/HuggingFace" \
    --output_dir "imagenet_data/systemds_ready"
```

## What the Pipeline Does

1. **Data Inspection**: 
   - Reads metadata CSV files
   - Checks which images actually exist on disk
   - Reports dataset statistics
   - Validates image dimensions

2. **Image Processing**:
   - Loads JPG images (keeping original 256x256 resolution)
   - Converts to RGB if needed
   - Normalizes pixel values to [0,1] range
   - Flattens to 196,608-dimensional feature vectors (256×256×3)

3. **Label Processing**:
   - Converts integer labels to 0-indexed (if needed)
   - Creates one-hot encoded labels for 1000 classes
   - Handles missing/invalid labels gracefully

4. **Output Generation**:
   - Creates CSV files compatible with SystemDS AlexNet training
   - Memory-efficient batch processing
   - Progress tracking and error handling

## Output Files

The pipeline generates these files in `imagenet_data/systemds_ready/`:

- `imagenet_train_6GB.csv` - Training image features (N × 196608)
- `imagenet_train_labels_6GB.csv` - Training labels (N × 1000, one-hot)
- `imagenet_val_6GB.csv` - Validation image features (M × 196608)  
- `imagenet_val_labels_6GB.csv` - Validation labels (M × 1000, one-hot)

## Training with AlexNet

After preprocessing, you can train AlexNet:

```bash
java -Xmx16g -Xms16g -Xmn1600m \
    -cp "target/systemds-3.4.0-SNAPSHOT.jar:target/lib/*" \
    org.apache.sysds.api.DMLScript \
    -f scripts/nn/examples/imagenet_alexnet.dml \
    -exec singlenode -gpu
```

## Memory Considerations

- **Processing**: Images are processed in batches of 1000 to manage memory
- **Testing**: Use `--max_samples 1000` for initial testing
- **Large datasets**: The pipeline handles memory efficiently, but ensure you have enough disk space for output CSV files

## Troubleshooting

### Common Issues

1. **"Training metadata file not found"**
   - Check that `imagenet_train_metadata.csv` exists in your input directory
   - Verify the file path is correct

2. **"No images available"**
   - Ensure image files exist at the paths specified in the metadata CSV
   - Check that the relative paths in CSV match your directory structure

3. **Memory errors**
   - Reduce batch size in the script if needed
   - Use `--max_samples` to process a smaller subset first

4. **Pillow/PIL errors**
   - Install Pillow: `pip install pillow`
   - Some images might be corrupted - the script will skip them with warnings

### Validation

After processing, check the output:

```python
import pandas as pd

# Check shapes
train_features = pd.read_csv("imagenet_data/systemds_ready/imagenet_train_6GB.csv")
train_labels = pd.read_csv("imagenet_data/systemds_ready/imagenet_train_labels_6GB.csv")

print(f"Training features shape: {train_features.shape}")  # Should be (N, 196608)
print(f"Training labels shape: {train_labels.shape}")      # Should be (N, 1000)
print(f"Feature range: [{train_features.min().min():.3f}, {train_features.max().max():.3f}]")  # Should be [0, 1]
print(f"Label sum check: {train_labels.sum(axis=1).mean():.3f}")  # Should be 1.0
```

## Performance Notes

- **Processing time**: ~0.1-0.2 seconds per image (depends on disk I/O)
- **Disk space**: Each image becomes ~785KB in CSV format (196,608 features × 4 bytes)
- **For 40,000 images**: Expect ~31GB output file and 2-4 hours processing time 