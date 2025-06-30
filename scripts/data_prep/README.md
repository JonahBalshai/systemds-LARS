# ImageNet Data Pipeline for SystemDS

This pipeline processes ImageNet data from NPZ format to SystemDS-ready CSV files with proper normalization and formatting.

## Quick Start

**1. Run the complete pipeline:**
```bash
python scripts/data_prep/run_imagenet_pipeline.py
```
This creates 6GB CSV files with 80:20 train/validation split.

**2. Train your models:**
```bash
# ResNet-18
java -Xmx16g -Xms16g -cp "target/systemds-3.4.0-SNAPSHOT.jar:target/lib/*" \
  org.apache.sysds.api.DMLScript -f scripts/nn/examples/imagenet_resnet.dml -exec singlenode -gpu

# AlexNet
java -Xmx16g -Xms16g -cp "target/systemds-3.4.0-SNAPSHOT.jar:target/lib/*" \
  org.apache.sysds.api.DMLScript -f scripts/nn/examples/imagenet_alexnet.dml -exec singlenode -gpu
```

## Pipeline Overview

### Input Data Structure
```
imagenet_data/6gb/
├── train_data_batch_1.npz    # Training batches
├── train_data_batch_2.npz
├── ...
└── val_data.npz              # Validation data
```

### Output Files
```
imagenet_data/systemds_ready/
├── imagenet_train_6GB.csv           # Training data (80% of 6GB target)
├── imagenet_train_labels_6GB.csv    # Training labels (one-hot encoded)
├── imagenet_val_6GB.csv             # Validation data (20% of 6GB target)
├── imagenet_val_labels_6GB.csv      # Validation labels (one-hot encoded)
├── *.mtd files                      # SystemDS metadata
└── Binary files for processing
```

## Advanced Usage

### Custom Dataset Size

**Create different sized datasets:**
```bash
# 2GB dataset
python scripts/data_prep/imagenet_pipeline.py --csv-size-gb 2.0

# 10GB dataset  
python scripts/data_prep/imagenet_pipeline.py --csv-size-gb 10.0

# 8GB dataset with 80:20 split
python scripts/data_prep/imagenet_pipeline.py --csv-size-gb 8.0
```

### Step-by-Step Processing

**1. Inspect data only:**
```bash
python scripts/data_prep/imagenet_pipeline.py --mode inspect
```

**2. Convert to binary (efficient processing):**
```bash
python scripts/data_prep/imagenet_pipeline.py --mode binary
```

**3. Create CSV samples:**
```bash
python scripts/data_prep/imagenet_pipeline.py --mode csv-only --csv-size-gb 6.0
```

## Data Processing Details

### Train/Validation Split
- **80%** for training
- **20%** for validation
- Files are named based on target size (e.g., `6GB`, `2GB`)

### Image Preprocessing
- **Normalization**: uint8 [0,255] → float32 [0,1]
- **Format**: 64×64×3 images flattened to 12,288 features
- **Memory-efficient**: Streaming processing for large datasets

### Label Processing
- **Input**: 1-indexed class labels [1-1000]
- **Output**: 0-indexed one-hot vectors [1000 classes]
- **Format**: Dense integer matrices for SystemDS

### SystemDS Integration
- **Metadata**: Auto-generated .mtd files for all CSV outputs
- **Format**: CSV format optimized for SystemDS reading
- **Memory**: Designed for 8GB+ heap sizes in SystemDS

## Sample Data Statistics

**6GB Dataset Example:**
- Training: ~40,000 samples (4.8GB)
- Validation: ~10,000 samples (1.2GB)
- Features: 12,288 per sample (64×64×3)
- Classes: 1,000 (ImageNet)

## Training Configuration

The pipeline generates files that work with these SystemDS training scripts:

**ResNet-18:**
- Uses `imagenet_train_6GB.csv` and `imagenet_val_6GB.csv`
- Batch size: 256
- 90 epochs with LARS optimizer
- Memory: 16GB+ recommended

**AlexNet:**
- Uses same 6GB CSV files
- Batch size: 256  
- With batch normalization
- Memory: 16GB+ recommended

## Troubleshooting

### Memory Issues
**Problem**: OutOfMemoryError during training
**Solution**: Use larger heap size
```bash
java -Xmx20g -Xms20g -cp "..." ...
```

### Dataset Size Issues
**Problem**: Generated CSV is smaller than expected
**Solution**: Check available samples in your NPZ files
```bash
python scripts/data_prep/imagenet_pipeline.py --mode inspect
```

### File Not Found Errors
**Problem**: Training scripts can't find CSV files
**Solution**: Check file naming matches your target size
```bash
ls imagenet_data/systemds_ready/imagenet_*6GB.csv
```

### Training Accuracy Issues
**Problem**: Very low accuracy (< 1%)
**Solution**: 
1. Verify data integrity with debug script
2. Try AlexNet instead of ResNet for 64×64 images
3. Check learning rate and batch size

## File Size Calculator

**Estimated sizes for different targets:**
- 2GB target → ~16,000 train + 4,000 val samples
- 6GB target → ~40,000 train + 10,000 val samples  
- 10GB target → ~65,000 train + 16,000 val samples

**Note**: Actual sizes depend on CSV formatting overhead and available data in your NPZ files. 