#!/usr/bin/env python3
"""
Debug Accuracy Issues in Chess Piece Mapping
Analyze why accuracy is 0% and identify problems
"""

from src.optimized_data_loader import OptimizedChessDataLoader
from src.optimized_mapper import OptimizedPieceMapper
import json
import numpy as np

def analyze_accuracy_issues():
    """Analyze what's causing 0% accuracy"""
    print(" DEBUGGING ACCURACY ISSUES")
    print("=" * 50)
    
    # Initialize components
    data_path = "/home/pre/projects/chess-datagen/gen-data/render_src/coco_data_2025_08_08__21_53_08"
    data_loader = OptimizedChessDataLoader(data_path)
    mapper = OptimizedPieceMapper()
    
    # Test image processing
    image_id = 0
    result = data_loader.process_image_optimized(image_id, debug=True)
    
    if not result or not result.get('success'):
        print(" Failed to process image")
        return
    
    pieces = result['pieces']
    ground_truth = result['ground_truth']
    
    print(f"\n GROUND TRUTH ANALYSIS:")
    print(f"Ground truth squares: {len(ground_truth)}")
    for square, piece_type in sorted(ground_truth.items())[:10]:
        print(f"  {square}: {piece_type}")
    
    print(f"\n DETECTED PIECES ANALYSIS:")
    print(f"Detected pieces: {len(pieces)}")
    for i, piece in enumerate(pieces[:10]):
        print(f"  {i}: {piece.piece_type} at ({piece.center_x:.1f}, {piece.center_y:.1f})")
    
    # Get assignments
    assignments = mapper.solve_assignment_improved(pieces)
    
    print(f"\n🔄 ASSIGNMENT ANALYSIS:")
    print(f"Assignments made: {len(assignments)}")
    for i, assignment in enumerate(assignments[:10]):
        predicted_piece = assignment.piece.piece_type
        assigned_square = assignment.square
        ground_truth_piece = ground_truth.get(assigned_square, 'NOT_FOUND')
        match = predicted_piece == ground_truth_piece
        
        print(f"  {i}: {predicted_piece} -> {assigned_square}")
        print(f"     Expected: {ground_truth_piece}, Match: {match}")
    
    # Analyze coordinate system issues
    print(f"\n COORDINATE SYSTEM ANALYSIS:")
    analyze_coordinate_issues(pieces, ground_truth)
    
    # Analyze piece type mismatches
    print(f"\n PIECE TYPE ANALYSIS:")
    analyze_piece_type_issues(pieces, assignments, ground_truth)

def analyze_coordinate_issues(pieces, ground_truth):
    """Analyze if coordinate transformation is causing issues"""
    print("Checking if pieces are in expected locations...")
    
    # Expected positions for starting chess position
    expected_positions = {
        'a1': (16, 240), 'b1': (48, 240), 'c1': (80, 240), 'd1': (112, 240),
        'e1': (144, 240), 'f1': (176, 240), 'g1': (208, 240), 'h1': (240, 240),
        'a2': (16, 208), 'b2': (48, 208), 'c2': (80, 208), 'd2': (112, 208),
        'e2': (144, 208), 'f2': (176, 208), 'g2': (208, 208), 'h2': (240, 208),
        'a7': (16, 48), 'b7': (48, 48), 'c7': (80, 48), 'd7': (112, 48),
        'e7': (144, 48), 'f7': (176, 48), 'g7': (208, 48), 'h7': (240, 48),
        'a8': (16, 16), 'b8': (48, 16), 'c8': (80, 16), 'd8': (112, 16),
        'e8': (144, 16), 'f8': (176, 16), 'g8': (208, 16), 'h8': (240, 16),
    }
    
    # Check if any pieces are near expected positions
    for piece in pieces[:5]:
        print(f"Piece {piece.piece_type} at ({piece.center_x:.1f}, {piece.center_y:.1f})")
        
        # Find closest expected position
        min_dist = float('inf')
        closest_square = None
        for square, (exp_x, exp_y) in expected_positions.items():
            dist = np.sqrt((piece.center_x - exp_x)**2 + (piece.center_y - exp_y)**2)
            if dist < min_dist:
                min_dist = dist
                closest_square = square
        
        print(f"  Closest to {closest_square} (distance: {min_dist:.1f})")

def analyze_piece_type_issues(pieces, assignments, ground_truth):
    """Analyze piece type naming mismatches"""
    print("Analyzing piece type naming...")
    
    # Collect all detected piece types
    detected_types = set(piece.piece_type for piece in pieces)
    ground_truth_types = set(ground_truth.values())
    
    print(f"Detected piece types: {sorted(detected_types)}")
    print(f"Ground truth piece types: {sorted(ground_truth_types)}")
    
    # Check for naming mismatches
    print(f"\nPiece type comparison:")
    for detected in sorted(detected_types)[:10]:
        # Try to find similar ground truth type
        similar = [gt for gt in ground_truth_types if detected.replace('1', '').replace('2', '') in gt or gt.replace('1', '').replace('2', '') in detected]
        print(f"  Detected: '{detected}' -> Similar in GT: {similar}")

def test_board_orientation():
    """Test if board orientation is correct"""
    print(f"\n BOARD ORIENTATION TEST:")
    print("Checking if coordinate system matches chess notation...")
    
    # Test square coordinate conversion
    mapper = OptimizedPieceMapper()
    
    test_squares = ['a1', 'h1', 'a8', 'h8', 'e4']
    for square in test_squares:
        file, rank = mapper._algebraic_to_square(square)
        center_x = file * 32 + 16
        center_y = rank * 32 + 16
        print(f"  {square} -> file={file}, rank={rank} -> center=({center_x}, {center_y})")

def create_fixed_mapper():
    """Create a mapper with better coordinate handling"""
    print(f"\n🛠 TESTING COORDINATE FIXES:")
    
    class FixedPieceMapper(OptimizedPieceMapper):
        def _square_to_algebraic(self, file: int, rank: int) -> str:
            """Convert square coordinates to algebraic notation - FIXED"""
            files = 'abcdefgh'
            # Fix: Chess ranks are 1-8, but our array is 0-7
            # Rank 0 should be rank 1, rank 7 should be rank 8
            return f"{files[file]}{rank + 1}"
        
        def _algebraic_to_square(self, notation: str) -> tuple:
            """Convert algebraic notation to square coordinates - FIXED"""
            file = ord(notation[0]) - ord('a')  # a=0, b=1, ..., h=7
            rank = int(notation[1]) - 1         # 1=0, 2=1, ..., 8=7
            return file, rank
    
    # Test the fix
    fixed_mapper = FixedPieceMapper()
    
    test_squares = ['a1', 'h1', 'a8', 'h8', 'e4']
    for square in test_squares:
        file, rank = fixed_mapper._algebraic_to_square(square)
        back_to_square = fixed_mapper._square_to_algebraic(file, rank)
        print(f"  {square} -> ({file}, {rank}) -> {back_to_square} ✓" if square == back_to_square else f"  {square} -> ({file}, {rank}) -> {back_to_square} ")

def main():
    """Run accuracy debugging"""
    print(" CHESS PIECE MAPPING - ACCURACY DEBUGGING")
    print("=" * 60)
    
    analyze_accuracy_issues()
    test_board_orientation()
    create_fixed_mapper()
    
    print(f"\n IDENTIFIED ISSUES:")
    print("1. Coordinate system may be flipped or offset")
    print("2. Piece type naming might not match exactly")
    print("3. Board orientation could be incorrect")
    print("4. Ground truth format may not align with detection format")

if __name__ == "__main__":
    main()