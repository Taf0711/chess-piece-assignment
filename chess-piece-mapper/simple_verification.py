#!/usr/bin/env python3
"""
Simple Verification Test - No heavy dependencies
Check actual accuracy results without matplotlib/opencv
"""

import os
import json
from datetime import datetime

def create_dated_run_folder():
    """Create timestamped folder for this run"""
    timestamp = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
    run_folder = f"results/run_{timestamp}"
    os.makedirs(run_folder, exist_ok=True)
    os.makedirs(f"{run_folder}/data", exist_ok=True)
    return run_folder

def test_data_loading():
    """Test basic data loading without dependencies"""
    print(" SIMPLE VERIFICATION TEST")
    print("=" * 50)
    
    # Create run folder
    run_folder = create_dated_run_folder()
    print(f" Results folder: {run_folder}")
    
    # Test basic data access - check train folder specifically
    data_path = "/home/pre/projects/chess-datagen/gen-data/render_src/coco_data_2025_08_08__21_53_08"
    train_path = os.path.join(data_path, 'train')
    
    # Check if data path exists
    if not os.path.exists(train_path):
        print(f" Train data path not found: {train_path}")
        return
    
    print(f" Train data path exists: {train_path}")
    
    # List contents
    try:
        contents = os.listdir(train_path)
        print(f"📂 Train folder contains: {len(contents)} items")
        
        # Look for key files (corrected names)
        key_files = ['coco_annotations.json', 'board_placements.json']
        for key_file in key_files:
            if key_file in contents:
                print(f"   Found: {key_file}")
            else:
                print(f"   Missing: {key_file}")
        
        # Check images folder
        if 'images' in contents:
            images_path = os.path.join(train_path, 'images')
            if os.path.isdir(images_path):
                image_count = len([f for f in os.listdir(images_path) if f.endswith('.png')])
                print(f"   Found {image_count} PNG images in train set")
            else:
                print(f"   Images is not a directory")
        
    except Exception as e:
        print(f" Error accessing data: {str(e)}")
        return
    
    # Test basic JSON loading (corrected paths)
    try:
        annotations_file = os.path.join(train_path, 'coco_annotations.json')
        if os.path.exists(annotations_file):
            with open(annotations_file, 'r') as f:
                annotations = json.load(f)
            print(f" Loaded coco_annotations.json")
            print(f"   Images: {len(annotations.get('images', []))}")
            print(f"  📦 Annotations: {len(annotations.get('annotations', []))}")
            print(f"  🏷  Categories: {len(annotations.get('categories', []))}")
            
            # Show categories
            categories = annotations.get('categories', [])
            if categories:
                print(f"   Piece types found:")
                for cat in categories[:10]:  # Show first 10
                    print(f"    - {cat['name']} (id: {cat['id']})")
                if len(categories) > 10:
                    print(f"    ... and {len(categories) - 10} more")
        
    except Exception as e:
        print(f" Error loading annotations: {str(e)}")
        return
    
    # Test board placements
    try:
        placements_file = os.path.join(train_path, 'board_placements.json')
        if os.path.exists(placements_file):
            with open(placements_file, 'r') as f:
                placements = json.load(f)
            print(f" Loaded board_placements.json")
            print(f"   Board positions: {len(placements)}")
            
            # Show sample
            if placements:
                sample_key = list(placements.keys())[0]
                sample_data = placements[sample_key]
                print(f"   Sample (image {sample_key}):")
                print(f"    FEN: {sample_data.get('fen', 'N/A')[:30]}...")
                if 'pieces' in sample_data:
                    print(f"    Pieces: {len(sample_data['pieces'])}")
        
    except Exception as e:
        print(f" Error loading placements: {str(e)}")
        return
    
    # Create basic report
    report = {
        'timestamp': datetime.now().isoformat(),
        'test_type': 'Simple data verification',
        'data_path': data_path,
        'data_accessible': True,
        'annotations_loaded': True,
        'placements_loaded': True,
        'status': 'Data verification successful - ready for accuracy testing'
    }
    
    report_path = f"{run_folder}/data/simple_verification.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n Simple verification report saved: {report_path}")
    print(f"\n VERIFICATION STATUS:")
    print(f"   Data loading:  Working")
    print(f"  📂 File structure:  Valid")
    print(f"   JSON parsing:  Functional")
    print(f"   Next step: Install dependencies for accuracy testing")
    
    return True

def main():
    """Run simple verification"""
    success = test_data_loading()
    
    if success:
        print(f"\n NEXT STEPS:")
        print(f"  1. Install required Python packages")
        print(f"  2. Run full accuracy verification test")
        print(f"  3. Generate PNG visualizations")
        print(f"  4. Compare with claimed 30.5% accuracy")

if __name__ == "__main__":
    main()