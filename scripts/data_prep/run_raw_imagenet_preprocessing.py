#!/usr/bin/env python3
"""
Simple runner for raw ImageNet preprocessing
"""

import sys
import subprocess
from pathlib import Path

def main():
    # Default paths
    input_dir = r"C:\Users\romer\Desktop\Code\HuggingFace"
    output_dir = "imagenet_data/systemds_ready"
    
    print("Raw ImageNet Preprocessing Runner")
    print("=" * 50)
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Ask user what they want to do
    print("Choose an option:")
    print("1. Inspect data only (dry run)")
    print("2. Process small sample (1000 images per split)")
    print("3. Process full dataset")
    print("4. Custom processing")
    print()
    
    choice = input("Enter choice (1-4): ").strip()
    
    if choice == "1":
        # Dry run
        cmd = [
            sys.executable, "scripts/data_prep/prepare_raw_imagenet.py",
            "--input_dir", input_dir,
            "--output_dir", output_dir,
            "--dry_run"
        ]
    elif choice == "2":
        # Small sample
        cmd = [
            sys.executable, "scripts/data_prep/prepare_raw_imagenet.py",
            "--input_dir", input_dir,
            "--output_dir", output_dir,
            "--max_samples", "1000"
        ]
    elif choice == "3":
        # Full dataset
        cmd = [
            sys.executable, "scripts/data_prep/prepare_raw_imagenet.py",
            "--input_dir", input_dir,
            "--output_dir", output_dir
        ]
    elif choice == "4":
        # Custom
        custom_input = input(f"Input directory [{input_dir}]: ").strip()
        if custom_input:
            input_dir = custom_input
        
        custom_output = input(f"Output directory [{output_dir}]: ").strip()
        if custom_output:
            output_dir = custom_output
        
        max_samples = input("Max samples per split (leave empty for all): ").strip()
        
        cmd = [
            sys.executable, "scripts/data_prep/prepare_raw_imagenet.py",
            "--input_dir", input_dir,
            "--output_dir", output_dir
        ]
        
        if max_samples:
            cmd.extend(["--max_samples", max_samples])
    else:
        print("Invalid choice!")
        return 1
    
    print(f"\nRunning command: {' '.join(cmd)}")
    print()
    
    # Run the command
    try:
        result = subprocess.run(cmd, check=True)
        print("\nProcessing completed successfully!")
        return 0
    except subprocess.CalledProcessError as e:
        print(f"\nError during processing: {e}")
        return 1
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 