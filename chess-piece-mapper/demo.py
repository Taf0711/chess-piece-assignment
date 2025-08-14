#!/usr/bin/env python3
"""
Chess Piece to Square Mapping - Complete Demo
Demonstrates the full pipeline from data loading to piece assignment
"""

from src.data_loader import ChessDataLoader
from src.piece_mapper import PieceToSquareMapper, PieceDetection
from src.training_pipeline import ChessMappingTrainer

def demo_single_image():
    """Demo processing of a single image"""
    print("=== CHESS PIECE MAPPING DEMO ===")
    
    # Initialize trainer (contains both data loader and mapper)
    data_path = "/home/pre/projects/chess-datagen/gen-data/render_src/coco_data_2025_08_08__21_53_08"
    trainer = ChessMappingTrainer(data_path)
    
    # Process a sample image
    image_id = 0
    print(f"\nProcessing Image {image_id}...")
    
    result = trainer.process_single_image(image_id, visualize=False)
    
    if result['success']:
        print(f"Successfully processed image {image_id}")
        print(f"Detected pieces: {len(result['pieces'])}")
        print(f"Generated assignments: {len(result['assignments'])}")
        
        if result['ground_truth']:
            print(f" Ground truth positions: {len(result['ground_truth'])}")
            
        if result['evaluation']:
            eval_data = result['evaluation']
            print(f" Accuracy: {eval_data['accuracy']:.3f} ({eval_data['correct']}/{eval_data['total']})")
        
        print(f"\n Final Assignments:")
        for assignment in result['assignments']:
            piece_name = assignment.piece.piece_type.replace('1', '').replace('2', '').replace('3', '')
            print(f"  {piece_name} -> {assignment.square} (cost: {assignment.cost:.3f})")
            
        if result['ground_truth']:
            print(f"\n Ground Truth (from FEN):")
            for square, piece in sorted(result['ground_truth'].items()):
                piece_clean = piece.replace('1', '').replace('2', '').replace('3', '')
                print(f"  {piece_clean} -> {square}")
        
        # Show visual mapping output
        if result['assignments']:
            print(f"\n Visualizing piece-to-square mappings...")
            trainer.mapper.visualize_assignments(result['assignments'], result['board_image'])
                
        return result
    else:
        print(f"Failed to process image {image_id}: {result.get('error', 'Unknown error')}")
        return None

def demo_cost_function():
    """Demonstrate the cost function with test pieces"""
    print("\n=== COST FUNCTION DEMO ===")
    
    mapper = PieceToSquareMapper()
    
    # Create test pieces at known positions
    test_pieces = [
        PieceDetection(16, 16, [0, 0, 32, 32], "WhiteRook1"),      # a1 center
        PieceDetection(48, 16, [32, 0, 32, 32], "WhiteKnight1"),   # b1 center  
        PieceDetection(16, 48, [0, 32, 32, 32], "WhitePawn1"),     # a2 center
    ]
    
    print(f"Testing cost function with {len(test_pieces)} pieces:")
    
    for i, piece in enumerate(test_pieces):
        print(f"\nPiece {i+1}: {piece.piece_type} at ({piece.center_x}, {piece.center_y})")
        
        # Test costs to nearby squares
        test_squares = ['a1', 'b1', 'c1', 'a2', 'b2']
        
        for square in test_squares:
            file, rank = mapper.algebraic_to_square(square)
            
            distance_cost = mapper.compute_distance_cost(piece, file, rank)
            overlap_cost = mapper.compute_overlap_cost(piece, file, rank) 
            piece_type_cost = mapper.compute_piece_type_cost(piece, file, rank)
            total_cost = mapper.compute_total_cost(piece, file, rank)
            
            print(f"  -> {square}: dist={distance_cost:.3f}, overlap={overlap_cost:.3f}, type={piece_type_cost:.3f}, total={total_cost:.3f}")

def demo_assignment_optimization():
    """Demonstrate Hungarian algorithm optimization"""
    print("\n=== ASSIGNMENT OPTIMIZATION DEMO ===")
    
    mapper = PieceToSquareMapper()
    
    # Create challenging test case with multiple pieces
    pieces = [
        PieceDetection(20, 20, [4, 4, 32, 32], "WhiteRook1"),      # Near a1
        PieceDetection(44, 20, [28, 4, 32, 32], "WhiteKnight1"),   # Near b1, close to a1
        PieceDetection(20, 44, [4, 28, 32, 32], "WhitePawn1"),     # Near a2, close to a1
        PieceDetection(176, 176, [160, 160, 32, 32], "BlackQueen1"), # Near f6
        PieceDetection(144, 144, [128, 128, 32, 32], "BlackKing1"),  # Near e5
    ]
    
    print(f"Optimizing assignment for {len(pieces)} pieces...")
    
    # Show cost matrix
    cost_matrix, square_labels = mapper.create_cost_matrix(pieces)
    print(f"Created {cost_matrix.shape[0]}×{cost_matrix.shape[1]} cost matrix")
    
    # Solve assignment
    assignments = mapper.solve_assignment(pieces)
    
    print(f"Hungarian algorithm found {len(assignments)} assignments:")
    for assignment in assignments:
        piece_name = assignment.piece.piece_type.replace('1', '')
        print(f"  {piece_name} -> {assignment.square} (cost: {assignment.cost:.3f})")
    
    # Show detailed mapping visualization
    print(f"\nDETAILED ASSIGNMENT VISUALIZATION:")
    print("=" * 50)
    print(f"{'Piece':<12} {'From Position':<15} {'→':<3} {'To Square':<10} {'Cost':<8}")
    print("-" * 50)
    
    for assignment in assignments:
        piece_short = assignment.piece.piece_type.replace('1', '').replace('White', 'W').replace('Black', 'B')[:8]
        from_pos = f"({assignment.piece.center_x:.0f},{assignment.piece.center_y:.0f})"
        to_square = assignment.square
        cost = assignment.cost
        print(f"{piece_short:<12} {from_pos:<15} {'→':<3} {to_square:<10} {cost:<8.3f}")
    
    # Create and show ASCII board representation
    print(f"\n FINAL BOARD STATE:")
    create_ascii_board_visualization(assignments)

def create_mapping_visualization():
    """Create detailed visualization of the piece mapping process"""
    print("\n=== DETAILED MAPPING VISUALIZATION ===")
    
    # Initialize trainer
    data_path = "/home/pre/projects/chess-datagen/gen-data/render_src/coco_data_2025_08_08__21_53_08"
    trainer = ChessMappingTrainer(data_path)
    
    # Process multiple images to show different scenarios
    sample_images = [0, 1, 2, 5, 10]  # Various images
    
    print(f"Processing {len(sample_images)} sample images for visualization...\n")
    
    for image_id in sample_images:
        print(f" IMAGE {image_id} MAPPING RESULTS:")
        print("=" * 50)
        
        result = trainer.process_single_image(image_id, visualize=False)
        
        if result['success'] and result['assignments']:
            # Show detailed mapping table
            print(f"DETECTED PIECES → BOARD SQUARES")
            print("-" * 50)
            print(f"{'Piece Type':<15} {'Position':<12} {'→':<3} {'Square':<8} {'Cost':<8}")
            print("-" * 50)
            
            for assignment in result['assignments']:
                piece_name = assignment.piece.piece_type.replace('1', '').replace('2', '').replace('3', '')
                pos_str = f"({assignment.piece.center_x:.0f},{assignment.piece.center_y:.0f})"
                print(f"{piece_name:<15} {pos_str:<12} {'→':<3} {assignment.square:<8} {assignment.cost:<8.3f}")
            
            # Show evaluation if available
            if result['evaluation']:
                eval_data = result['evaluation']
                print(f"\nACCURACY METRICS:")
                print(f"  Overall Accuracy: {eval_data['accuracy']:.1%}")
                print(f"  Correct Assignments: {eval_data['correct']}/{eval_data['total']}")
                
                if eval_data.get('missed_pieces'):
                    print(f"  Missed Pieces: {', '.join(eval_data['missed_pieces'])}")
                if eval_data.get('false_positives'):
                    print(f"  False Positives: {', '.join(eval_data['false_positives'])}")
            
            # Create visual board representation
            print(f"\n FINAL BOARD MAPPING:")
            create_ascii_board_visualization(result['assignments'])
            
        else:
            print(f"Failed to process or no assignments found")
        
        print("\n" + "="*60 + "\n")

def create_ascii_board_visualization(assignments):
    """Create ASCII representation of the board with piece assignments"""
    # Initialize 8x8 board
    board = [['.' for _ in range(8)] for _ in range(8)]
    
    # Place pieces on board
    for assignment in assignments:
        file, rank = assignment.file, assignment.rank
        piece_type = assignment.piece.piece_type.replace('1', '').replace('2', '').replace('3', '')
        
        # Create short piece symbol
        if 'White' in piece_type:
            if 'Pawn' in piece_type: symbol = 'P'
            elif 'Rook' in piece_type: symbol = 'R'
            elif 'Knight' in piece_type: symbol = 'N'
            elif 'Bishop' in piece_type: symbol = 'B'
            elif 'Queen' in piece_type: symbol = 'Q'
            elif 'King' in piece_type: symbol = 'K'
            else: symbol = 'W'
        else:
            if 'Pawn' in piece_type: symbol = 'p'
            elif 'Rook' in piece_type: symbol = 'r'
            elif 'Knight' in piece_type: symbol = 'n'
            elif 'Bishop' in piece_type: symbol = 'b'
            elif 'Queen' in piece_type: symbol = 'q'
            elif 'King' in piece_type: symbol = 'k'
            else: symbol = 'b'
            
        board[7-rank][file] = symbol  # Flip rank for proper display
    
    # Print board
    print("    a b c d e f g h")
    print("    ---------------")
    for rank_idx, rank_row in enumerate(board):
        rank_num = 8 - rank_idx
        print(f"{rank_num} | {' '.join(rank_row)} |")
    print("    ---------------")
    print("    a b c d e f g h")

def main():
    """Run complete demo"""
    # Demo 1: Process real image from dataset
    demo_single_image()
    
    # Demo 2: Show cost function details  
    demo_cost_function()
    
    # Demo 3: Show assignment optimization
    demo_assignment_optimization()
    
    # Demo 4: Create detailed mapping visualization
    create_mapping_visualization()
    
    print(f"\n Demo complete! Key achievements:")
    print(f"  Data loading from COCO format")
    print(f"  Board warping to 256×256 standard view")
    print(f"  Piece detection to square mapping")
    print(f"  Hungarian algorithm assignment optimization")
    print(f"  FEN ground truth loading and evaluation")
    print(f"  Cost function with multiple criteria")
    print(f"  Visual output showing final piece mappings")

if __name__ == "__main__":
    main()