#!/usr/bin/env python3
"""
Quick runner for ImageNet data processing pipeline.
Configured for the specific directory structure in systemds-LARS.
"""

import sys
from pathlib import Path
from imagenet_pipeline import ImageNetPipeline

def main():
    """Run the ImageNet pipeline with simple configuration."""
    print("🚀 ImageNet Pipeline Runner")
    print("=" * 40)
    
    # Auto-detect project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    
    # Default paths
    input_dir = project_root / "imagenet_data" / "6gb"
    output_dir = project_root / "imagenet_data" / "systemds_ready"
    
    print(f"📁 Input directory: {input_dir}")
    print(f"📁 Output directory: {output_dir}")
    
    # Check if input directory exists
    if not input_dir.exists():
        print(f"❌ Input directory not found: {input_dir}")
        print("Please ensure ImageNet NPZ files are in imagenet_data/6gb/")
        return
    
    # Initialize and run pipeline
    try:
        pipeline = ImageNetPipeline(str(input_dir), str(output_dir))
        
        # Run with 6GB target size (gives good training data amount)
        results = pipeline.run_full_pipeline(csv_size_gb=6.0)
        
        if results['success']:
            print("\n🎉 Pipeline completed successfully!")
            print("\n📋 Summary:")
            print(f"  - Training samples: {results['csv_info']['training']['samples']:,}")
            print(f"  - Validation samples: {results['csv_info']['validation']['samples']:,}")
            print(f"  - Training CSV size: {results['csv_info']['training']['size_gb']:.2f} GB")
            print(f"  - Validation CSV size: {results['csv_info']['validation']['size_gb']:.2f} GB")
            
            print("\n🔗 Next Steps:")
            print("1. Use these commands to train your models:")
            print(f"   ResNet: java -Xmx16g -cp 'target/...' ... -f scripts/nn/examples/imagenet_resnet.dml")
            print(f"   AlexNet: java -Xmx16g -cp 'target/...' ... -f scripts/nn/examples/imagenet_alexnet.dml")
        else:
            print(f"\n❌ Pipeline failed: {results['error']}")
    
    except Exception as e:
        print(f"\n💥 Error: {e}")
        import traceback
        traceback.print_exc()

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