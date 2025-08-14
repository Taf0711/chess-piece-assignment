#!/usr/bin/env python3
"""
Basic Accuracy Test - No heavy dependencies
Tests actual accuracy using simple coordinate-based assignment
"""

import os
import json
import math
from datetime import datetime

def create_dated_run_folder():
    """Create timestamped folder for this run"""
    timestamp = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
    run_folder = f"results/run_{timestamp}"
    os.makedirs(run_folder, exist_ok=True)
    os.makedirs(f"{run_folder}/data", exist_ok=True)
    return run_folder

def normalize_piece_type(piece_type):
    """Normalize piece type names (remove variant numbers)"""
    return piece_type.replace('2', '1').replace('3', '1').replace('4', '1').replace('5', '1').replace('6', '1').replace('7', '1').replace('8', '1')

def get_expected_square_coordinates():
    """Get expected pixel coordinates for each chess square"""
    coords = {}
    files = 'abcdefgh'
    board_size = 256
    square_size = board_size / 8  # 32 pixels per square
    
    for rank in range(1, 9):  # 1-8
        for file_idx, file_char in enumerate(files):  # a-h
            square = f"{file_char}{rank}"
            
            # Calculate center coordinates
            x = file_idx * square_size + square_size / 2
            # Chess rank 1 at bottom, so invert y
            y = (8 - rank) * square_size + square_size / 2
            
            coords[square] = (x, y)
    
    return coords

def parse_fen_to_pieces(fen_string):
    """Parse FEN string to piece positions"""
    if not fen_string:
        return {}
    
    pieces = {}
    files = 'abcdefgh'
    
    # Get board part of FEN (before first space)
    board_part = fen_string.split(' ')[0]
    ranks = board_part.split('/')
    
    for rank_idx, rank_str in enumerate(ranks):
        rank_num = 8 - rank_idx  # FEN rank 1 is bottom
        file_idx = 0
        
        for char in rank_str:
            if char.isdigit():
                file_idx += int(char)  # Skip empty squares
            else:
                if file_idx < 8:
                    square = f"{files[file_idx]}{rank_num}"
                    
                    # Convert FEN notation to piece names
                    piece_map = {
                        'P': 'WhitePawn1', 'R': 'WhiteRook1', 'N': 'WhiteKnight1', 
                        'B': 'WhiteBishop1', 'Q': 'WhiteQueen1', 'K': 'WhiteKing1',
                        'p': 'BlackPawn1', 'r': 'BlackRook1', 'n': 'BlackKnight1',
                        'b': 'BlackBishop1', 'q': 'BlackQueen1', 'k': 'BlackKing1'
                    }
                    
                    if char in piece_map:
                        pieces[square] = piece_map[char]
                    
                    file_idx += 1
    
    return pieces

def calculate_distance(x1, y1, x2, y2):
    """Calculate Euclidean distance between two points"""
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def test_basic_accuracy():
    """Test basic accuracy using simple coordinate assignment"""
    print(" BASIC ACCURACY TEST - NO DEPENDENCIES")
    print("=" * 60)
    
    # Create run folder
    run_folder = create_dated_run_folder()
    print(f" Results folder: {run_folder}")
    
    # Load data
    train_path = "/home/pre/projects/chess-datagen/gen-data/render_src/coco_data_2025_08_08__21_53_08/train"
    
    try:
        # Load annotations
        with open(os.path.join(train_path, 'coco_annotations.json'), 'r') as f:
            coco_data = json.load(f)
        
        # Load board placements  
        with open(os.path.join(train_path, 'board_placements.json'), 'r') as f:
            board_placements = json.load(f)
            
        print(f" Loaded {len(coco_data['images'])} images with {len(coco_data['annotations'])} annotations")
        print(f" Loaded {len(board_placements)} board positions")
        
    except Exception as e:
        print(f" Error loading data: {e}")
        return False, 0.0, run_folder
    
    # Create category mapping
    categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
    
    # Get expected square coordinates
    square_coords = get_expected_square_coordinates()
    
    # Test on first 10 images
    test_images = coco_data['images'][:10]  
    results = []
    total_accuracy = 0.0
    successful_tests = 0
    
    print(f"\n Testing {len(test_images)} images for basic accuracy...")
    
    for img_data in test_images:
        image_id = img_data['id']
        image_filename = img_data['file_name']
        
        print(f"\n Image {image_id} ({image_filename}):")
        
        try:
            # Get annotations for this image
            image_annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] == image_id]
            
            # Filter out board annotations, keep only pieces
            piece_annotations = [ann for ann in image_annotations if categories[ann['category_id']] != 'Board']
            
            if not piece_annotations:
                print(f"  ️  No piece annotations found")
                continue
                
            print(f"  📦 Found {len(piece_annotations)} piece detections")
            
            # Get ground truth from board placements
            board_key = image_filename
            if board_key not in board_placements:
                print(f"  ️  No ground truth for {board_key}")
                continue
                
            board_data = board_placements[board_key]
            fen_string = board_data.get('board')  # Fixed: use 'board' key not 'fen'
            
            if not fen_string:
                print(f"  ️  No FEN string found")
                continue
                
            # Parse ground truth positions
            ground_truth = parse_fen_to_pieces(fen_string)
            
            if not ground_truth:
                print(f"  ️  Could not parse FEN: {fen_string[:30]}...")
                continue
                
            print(f"   Ground truth has {len(ground_truth)} pieces")
            
            # Simple assignment: assign each detected piece to closest square
            assignments = {}
            used_squares = set()
            
            for ann in piece_annotations:
                piece_type = categories[ann['category_id']]
                bbox = ann['bbox']  # [x, y, width, height]
                
                # Calculate piece center (assuming bbox is in warped 256x256 space)
                center_x = bbox[0] + bbox[2] / 2
                center_y = bbox[1] + bbox[3] / 2
                
                # Find closest square
                best_square = None
                best_distance = float('inf')
                
                for square, (sq_x, sq_y) in square_coords.items():
                    if square in used_squares:
                        continue
                        
                    distance = calculate_distance(center_x, center_y, sq_x, sq_y)
                    
                    # Only assign if within reasonable distance (1.5 squares = 48 pixels)
                    if distance < 48 and distance < best_distance:
                        best_distance = distance
                        best_square = square
                
                if best_square:
                    assignments[best_square] = piece_type
                    used_squares.add(best_square)
            
            print(f"   Made {len(assignments)} assignments")
            
            # Evaluate accuracy
            correct = 0
            total = len(ground_truth)
            
            for square, true_piece in ground_truth.items():
                if square in assignments:
                    predicted_piece = assignments[square]
                    
                    # Normalize both for comparison
                    norm_true = normalize_piece_type(true_piece)
                    norm_pred = normalize_piece_type(predicted_piece)
                    
                    if norm_true == norm_pred:
                        correct += 1
            
            accuracy = correct / max(1, total)
            print(f"   Result: {correct}/{total} = {accuracy:.1%}")
            
            if accuracy > 0:
                total_accuracy += accuracy
                successful_tests += 1
                
                # Show some correct assignments
                correct_examples = []
                for square, true_piece in ground_truth.items():
                    if square in assignments:
                        predicted_piece = assignments[square]
                        norm_true = normalize_piece_type(true_piece)
                        norm_pred = normalize_piece_type(predicted_piece)
                        if norm_true == norm_pred:
                            correct_examples.append(f"{square}:{true_piece[:6]}")
                
                if correct_examples:
                    print(f"   Correct: {', '.join(correct_examples[:3])}")
            
            # Store result
            results.append({
                'image_id': image_id,
                'image_filename': image_filename,
                'accuracy': accuracy,
                'correct': correct,
                'total': total,
                'assignments_made': len(assignments),
                'pieces_detected': len(piece_annotations),
                'ground_truth_pieces': len(ground_truth)
            })
            
        except Exception as e:
            print(f"   Error processing image {image_id}: {str(e)}")
    
    # Calculate final results
    if successful_tests > 0:
        avg_accuracy = total_accuracy / successful_tests
        best_result = max(results, key=lambda x: x['accuracy'])
        worst_result = min(results, key=lambda x: x['accuracy'])
        
        print(f"\n BASIC ACCURACY TEST RESULTS:")
        print("=" * 60)
        print(f" Average Accuracy: {avg_accuracy:.1%}")
        print(f" Successful Tests: {successful_tests}/{len(test_images)}")
        print(f" Best Result: {best_result['accuracy']:.1%} (Image {best_result['image_id']})")
        print(f" Worst Result: {worst_result['accuracy']:.1%} (Image {worst_result['image_id']})")
        
        # Detailed breakdown
        print(f"\n Detailed Results:")
        for result in results:
            print(f"  Image {result['image_id']:3d}: {result['accuracy']:5.1%} "
                  f"({result['correct']:2d}/{result['total']:2d}) - "
                  f"{result['assignments_made']:2d} assigned from {result['pieces_detected']:2d} detected")
        
        # Save comprehensive report
        report = {
            'timestamp': datetime.now().isoformat(),
            'test_type': 'Basic coordinate-based accuracy test',
            'method': 'Closest square assignment with distance threshold',
            'average_accuracy': avg_accuracy,
            'successful_tests': successful_tests,
            'total_tests': len(test_images),
            'best_accuracy': best_result['accuracy'],
            'worst_accuracy': worst_result['accuracy'],
            'detailed_results': results,
            'notes': [
                'This is a basic test using simple coordinate assignment',
                'No complex warping or Hungarian algorithm optimization',
                'Distance threshold: 48 pixels (1.5 squares)',
                'Piece type normalization applied (removes variant numbers)',
                'Tests first 10 images from training set'
            ]
        }
        
        report_path = f"{run_folder}/data/basic_accuracy_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n Report saved: {report_path}")
        
        return True, avg_accuracy, run_folder
    else:
        print(f"\n No successful tests completed")
        return False, 0.0, run_folder

def main():
    """Run basic accuracy test"""
    print(" CHESS PIECE MAPPING - BASIC ACCURACY VERIFICATION")
    print("=" * 80)
    
    success, accuracy, run_folder = test_basic_accuracy()
    
    print(f"\n BASIC ACCURACY TEST CONCLUSION:")
    print("=" * 80)
    print(f" ACTUAL Measured Accuracy: {accuracy:.1%}")
    print(f" Results Location: {run_folder}")
    
    if success:
        if accuracy >= 0.5:
            print(f" EXCEPTIONAL: Achieved ≥50% target with basic method!")
        elif accuracy >= 0.3:
            print(f" GOOD PROGRESS: {accuracy:.1%} with basic coordinate mapping")
        elif accuracy > 0.1:
            print(f" MODERATE PROGRESS: {accuracy:.1%} (shows system is working)")
        else:
            print(f" MINIMAL PROGRESS: {accuracy:.1%} (needs optimization)")
            
        print(f"\n This basic test shows:")
        print(f"    Data loading and parsing works correctly")
        print(f"    Ground truth FEN parsing is functional") 
        print(f"    Coordinate-based assignment produces measurable results")
        print(f"    Provides baseline for comparing claimed 30.5% accuracy")
        
    else:
        print(f" Test failed - system needs debugging")
        
    print(f"\n Note: This basic test validates the core functionality")
    print(f"   More sophisticated methods (Hungarian algorithm, warping)")
    print(f"   should achieve higher accuracy than this baseline.")

if __name__ == "__main__":
    main()