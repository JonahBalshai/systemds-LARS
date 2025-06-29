#!/usr/bin/env python3
"""
Quick runner for ImageNet data processing pipeline.
Configured for the specific directory structure in systemds-LARS.
"""

import sys
from pathlib import Path
from imagenet_pipeline import ImageNetPipeline

def main():
    """Run the ImageNet pipeline with predefined settings."""
    
    # Set up paths for systemds-LARS project
    project_root = Path(__file__).parent.parent.parent  # Go up to systemds-LARS root
    input_dir = project_root / "imagenet_data" / "6gb"
    output_dir = project_root / "imagenet_data" / "systemds_ready"
    
    print("ImageNet Data Processing Pipeline - Quick Runner")
    print("=" * 50)
    print(f"Project root: {project_root}")
    print(f"Input dir: {input_dir}")
    print(f"Output dir: {output_dir}")
    
    # Check if input directory exists
    if not input_dir.exists():
        print(f"\nERROR: Input directory not found: {input_dir}")
        print("Make sure you have NPZ files in imagenet_data/6gb/")
        sys.exit(1)
    
    # Check for NPZ files
    npz_files = list(input_dir.glob("*.npz"))
    if not npz_files:
        print(f"\nERROR: No NPZ files found in {input_dir}")
        print("Expected files: train_data_batch_*.npz and val_data.npz")
        sys.exit(1)
    
    print(f"Found {len(npz_files)} NPZ files")
    
    # Initialize and run pipeline
    pipeline = ImageNetPipeline(str(input_dir), str(output_dir))
    
    try:
        # Run full pipeline to create 2GB CSV files
        pipeline.run_full_pipeline(csv_size_gb=2.0)
        
        print("\n" + "="*50)
        print("SUCCESS! Pipeline completed.")
        print("Your files are ready:")
        print("- imagenet_train_2GB.csv")
        print("- imagenet_train_labels_2GB.csv") 
        print("- imagenet_val_2GB.csv")
        print("- imagenet_val_labels_2GB.csv")
        print(f"\nLocation: {output_dir}")
        
    except Exception as e:
        print(f"\nERROR: {e}")
        print("\nTroubleshooting:")
        print("1. Check that NPZ files are in imagenet_data/6gb/")
        print("2. Ensure you have enough disk space")
        print("3. Check file permissions")
        sys.exit(1)

def inspect_only():
    """Just inspect the dataset without processing."""
    project_root = Path(__file__).parent.parent.parent
    input_dir = project_root / "imagenet_data" / "6gb"
    output_dir = project_root / "imagenet_data" / "systemds_ready"
    
    pipeline = ImageNetPipeline(str(input_dir), str(output_dir))
    pipeline.inspect_dataset()

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "inspect":
        inspect_only()
    else:
        main() 