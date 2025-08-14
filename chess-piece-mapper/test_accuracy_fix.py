#!/usr/bin/env python3
"""
Test the Accuracy-Fixed Chess Piece Mapper
"""

from src.optimized_data_loader import OptimizedChessDataLoader
from src.accuracy_fixed_mapper import AccuracyFixedMapper

def test_accuracy_improvements():
    """Test accuracy improvements with multiple images"""
    print(" TESTING ACCURACY IMPROVEMENTS")
    print("=" * 60)
    
    # Initialize components
    data_path = "/home/pre/projects/chess-datagen/gen-data/render_src/coco_data_2025_08_08__21_53_08"
    data_loader = OptimizedChessDataLoader(data_path)
    fixed_mapper = AccuracyFixedMapper()
    
    # Test coordinate system first
    print("📍 Testing coordinate system fixes:")
    test_squares = ['a1', 'h1', 'a8', 'h8', 'e4', 'd5']
    for square in test_squares:
        file, rank = fixed_mapper.algebraic_to_square(square)
        center_x, center_y = fixed_mapper.get_square_center(file, rank)
        back_square = fixed_mapper.square_to_algebraic(file, rank)
        print(f"  {square} -> ({file},{rank}) -> center=({center_x:.0f},{center_y:.0f}) -> {back_square}")
    
    # Test on multiple images
    test_images = [0, 1, 2, 5, 10]
    total_accuracy = 0.0
    successful_tests = 0
    
    print(f"\n Testing on {len(test_images)} images:")
    for image_id in test_images:
        result = data_loader.process_image_optimized(image_id)
        
        if result and result.get('success'):
            pieces = result['pieces']
            ground_truth = result['ground_truth']
            
            if pieces and ground_truth:
                # Get assignments with fixed mapper
                assignments = fixed_mapper.solve_assignment(pieces)
                evaluation = fixed_mapper.evaluate_assignment(assignments, ground_truth)
                
                accuracy = evaluation['accuracy']
                total_accuracy += accuracy
                successful_tests += 1
                
                print(f"  Image {image_id}: {evaluation['correct']}/{evaluation['total']} = {accuracy:.1%}")
                
                # Show detailed results for first image
                if image_id == 0:
                    print(f"    Sample assignments:")
                    for i, assignment in enumerate(assignments[:5]):
                        square = assignment.square
                        predicted = assignment.piece.piece_type
                        actual = ground_truth.get(square, 'NOT_FOUND')
                        norm_pred = fixed_mapper.normalize_piece_type(predicted)
                        norm_actual = fixed_mapper.normalize_piece_type(actual)
                        correct = norm_pred == norm_actual
                        status = "✓" if correct else ""
                        print(f"      {square}: {predicted} -> {actual} {status}")
            else:
                print(f"  Image {image_id}: No pieces or ground truth")
        else:
            print(f"  Image {image_id}: Processing failed")
    
    if successful_tests > 0:
        avg_accuracy = total_accuracy / successful_tests
        print(f"\n RESULTS SUMMARY:")
        print(f"  Average accuracy: {avg_accuracy:.1%}")
        print(f"  Successful tests: {successful_tests}/{len(test_images)}")
        
        if avg_accuracy >= 0.5:
            print(f"   SUCCESS: Achieved ≥50% accuracy target!")
        else:
            print(f"  ️  Still below 50% target, needs more fixes")
        
        return avg_accuracy
    else:
        print(f"   No successful tests")
        return 0.0

def compare_before_after():
    """Compare accuracy before and after fixes"""
    print(f"\n BEFORE vs AFTER COMPARISON:")
    print("=" * 60)
    
    from src.optimized_mapper import OptimizedPieceMapper
    
    data_path = "/home/pre/projects/chess-datagen/gen-data/render_src/coco_data_2025_08_08__21_53_08"
    data_loader = OptimizedChessDataLoader(data_path)
    
    original_mapper = OptimizedPieceMapper()
    fixed_mapper = AccuracyFixedMapper()
    
    image_id = 0
    result = data_loader.process_image_optimized(image_id)
    
    if result and result.get('success'):
        pieces = result['pieces']
        ground_truth = result['ground_truth']
        
        # Original mapper results
        original_assignments = original_mapper.solve_assignment_improved(pieces)
        original_eval = {'accuracy': 0.0, 'correct': 0, 'total': len(ground_truth)}
        if original_assignments:
            # Manual evaluation for original mapper
            correct = 0
            predicted_dict = {a.square: a.piece.piece_type for a in original_assignments}
            for square, true_piece in ground_truth.items():
                if square in predicted_dict and predicted_dict[square] == true_piece:
                    correct += 1
            original_eval = {'accuracy': correct/len(ground_truth), 'correct': correct, 'total': len(ground_truth)}
        
        # Fixed mapper results  
        fixed_assignments = fixed_mapper.solve_assignment(pieces)
        fixed_eval = fixed_mapper.evaluate_assignment(fixed_assignments, ground_truth)
        
        print(f"ORIGINAL MAPPER:")
        print(f"  Assignments: {len(original_assignments)}")
        print(f"  Accuracy: {original_eval['accuracy']:.1%}")
        print(f"  Correct: {original_eval['correct']}/{original_eval['total']}")
        
        print(f"\nFIXED MAPPER:")
        print(f"  Assignments: {len(fixed_assignments)}")
        print(f"  Accuracy: {fixed_eval['accuracy']:.1%}")
        print(f"  Correct: {fixed_eval['correct']}/{fixed_eval['total']}")
        
        improvement = fixed_eval['accuracy'] - original_eval['accuracy']
        print(f"\nIMPROVEMENT: {improvement:+.1%}")

def main():
    """Run accuracy improvement tests"""
    print(" CHESS PIECE MAPPING - ACCURACY IMPROVEMENT TEST")
    print("=" * 70)
    
    # Test the improvements
    accuracy = test_accuracy_improvements()
    
    # Compare before and after
    compare_before_after()
    
    print(f"\n ACCURACY IMPROVEMENT SUMMARY:")
    print("=" * 70)
    print("Key fixes implemented:")
    print("   Fixed Y-coordinate inversion (ranks now correctly oriented)")
    print("   Normalized piece type matching (WhitePawn2 -> WhitePawn1)")
    print("   Improved chess logic bonuses for piece placement")
    print("   Lower cost threshold for better assignment acceptance")
    
    if accuracy >= 0.5:
        print(f"\n TARGET ACHIEVED: {accuracy:.1%} accuracy (≥50% target)")
    else:
        print(f"\n Still working towards 50% target (current: {accuracy:.1%})")

if __name__ == "__main__":
    main()