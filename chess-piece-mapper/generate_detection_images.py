#!/usr/bin/env python3
"""
Generate Detection Images for Chess Piece Mapping
Creates CV-style detection visualizations showing piece mapping results
"""

from src.training_pipeline import ChessMappingTrainer
from src.piece_mapper import PieceToSquareMapper, create_test_pieces
from src.image_visualizer import ChessDetectionVisualizer
import numpy as np
import cv2
from pathlib import Path

def generate_test_detection_images():
    """Generate detection images using test data"""
    print("🎯 GENERATING CHESS PIECE DETECTION IMAGES")
    print("=" * 60)
    
    # Initialize components
    visualizer = ChessDetectionVisualizer(output_dir="results/detection_images")
    mapper = PieceToSquareMapper()
    
    # Create test scenario
    test_pieces = create_test_pieces()
    assignments = mapper.solve_assignment(test_pieces)
    
    # Create synthetic chess board image for demonstration
    board_image = create_synthetic_board_image()
    
    print(f"📊 Test Data:")
    print(f"  • {len(test_pieces)} detected pieces")
    print(f"  • {len(assignments)} assignments generated")
    print(f"  • Board image: 256×256 pixels")
    
    # Generate detection visualization  
    detection_viz = visualizer.create_detection_image(
        original_image=board_image,
        warped_image=board_image,
        pieces=test_pieces,
        assignments=assignments,
        image_id=999  # Test image ID
    )
    
    # Generate single detection overlay
    single_detection = visualizer.create_single_detection_overlay(
        image=board_image,
        pieces=test_pieces,
        assignments=assignments,
        image_id=999
    )
    
    print(f"✅ Generated test detection images")
    return detection_viz, single_detection

def generate_real_detection_images():
    """Generate detection images from real dataset"""
    print("\n🔍 PROCESSING REAL DATASET IMAGES")
    print("=" * 60)
    
    # Initialize trainer
    data_path = "/home/pre/projects/chess-datagen/gen-data/render_src/coco_data_2025_08_08__21_53_08"
    trainer = ChessMappingTrainer(data_path, output_dir="results")
    
    # Process sample images
    sample_images = [0, 1, 2, 5, 10, 15, 20, 25, 30]
    successful_results = []
    
    print(f"Processing {len(sample_images)} sample images...")
    
    for image_id in sample_images:
        print(f"  • Processing image {image_id}...")
        result = trainer.process_single_image(image_id, visualize=False)
        
        if result['success']:
            successful_results.append(result)
            
            # Generate detection images even if no assignments (shows detection process)
            trainer.visualizer.create_detection_image(
                original_image=result['original_image'],
                warped_image=result['board_image'],
                pieces=result['pieces'], 
                assignments=result['assignments'],
                ground_truth=result['ground_truth'],
                image_id=image_id
            )
            
            # Only create single overlay if we have assignments
            if result['assignments']:
                trainer.visualizer.create_single_detection_overlay(
                    image=result['board_image'],
                    pieces=result['pieces'],
                    assignments=result['assignments'],
                    image_id=image_id
                )
        else:
            print(f"    ❌ Failed: {result.get('error', 'Unknown error')}")
    
    # Create summary grid
    if successful_results:
        trainer.visualizer.create_summary_grid(successful_results)
        print(f"✅ Generated {len(successful_results)} real detection images")
    
    return successful_results

def create_synthetic_board_image() -> np.ndarray:
    """Create a synthetic chess board for testing"""
    # Create 256x256 checkered board
    board = np.zeros((256, 256, 3), dtype=np.uint8)
    
    # Checkered pattern
    square_size = 32
    for row in range(8):
        for col in range(8):
            if (row + col) % 2 == 1:  # Dark squares
                y1, y2 = row * square_size, (row + 1) * square_size
                x1, x2 = col * square_size, (col + 1) * square_size
                board[y1:y2, x1:x2] = [139, 69, 19]  # Brown
            else:  # Light squares
                y1, y2 = row * square_size, (row + 1) * square_size
                x1, x2 = col * square_size, (col + 1) * square_size
                board[y1:y2, x1:x2] = [245, 245, 220]  # Beige
    
    return board

def create_detection_report():
    """Create a comprehensive detection report"""
    print("\n📋 DETECTION REPORT GENERATION")
    print("=" * 60)
    
    results_dir = Path("results/detection_images")
    
    # Count generated files
    detection_images = list(results_dir.glob("detection_image_*.png"))
    single_detections = list(results_dir.glob("single_detection_*.png")) 
    summary_files = list(results_dir.glob("detection_summary.png"))
    
    print(f"📊 Generated Files:")
    print(f"  • Detection visualizations: {len(detection_images)}")
    print(f"  • Single detection overlays: {len(single_detections)}")
    print(f"  • Summary grids: {len(summary_files)}")
    print(f"  • Total output files: {len(detection_images) + len(single_detections) + len(summary_files)}")
    
    print(f"\n📁 Output Directory:")
    print(f"  {results_dir.absolute()}")
    
    print(f"\n📄 File Types Generated:")
    print(f"  • detection_image_XXXXXX.png - Complete 4-panel visualization")
    print(f"  • single_detection_XXXXXX.png - CV-style overlay detection")  
    print(f"  • detection_summary.png - Grid summary of multiple results")
    
    return {
        'detection_images': len(detection_images),
        'single_detections': len(single_detections), 
        'summary_files': len(summary_files),
        'output_directory': str(results_dir.absolute())
    }

def main():
    """Generate all detection images"""
    print("🚀 CHESS PIECE DETECTION IMAGE GENERATOR")
    print("=" * 70)
    
    # Create output directory
    Path("results/detection_images").mkdir(parents=True, exist_ok=True)
    
    # Generate test detection images  
    generate_test_detection_images()
    
    # Generate real detection images
    generate_real_detection_images()
    
    # Create comprehensive report
    report = create_detection_report()
    
    print(f"\n🎉 DETECTION IMAGE GENERATION COMPLETE!")
    print("=" * 70)
    print(f"✅ Successfully generated detection visualizations")
    print(f"📁 All outputs saved to: {report['output_directory']}")
    print(f"🖼️  Total images: {report['detection_images'] + report['single_detections'] + report['summary_files']}")
    print(f"\nNext steps:")
    print(f"  • Check the results/detection_images/ folder for PNG outputs")
    print(f"  • View detection_summary.png for overview")
    print(f"  • Open individual detection_image_*.png for detailed analysis")

if __name__ == "__main__":
    main()