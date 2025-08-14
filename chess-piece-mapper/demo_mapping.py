#!/usr/bin/env python3
"""
Simple Demo of Chess Piece Mapping Method
Shows clear visualization of pieces being mapped to board squares
"""

from src.piece_mapper import PieceToSquareMapper, PieceDetection

def create_test_scenario():
    """Create a realistic test scenario showing the mapping method"""
    print("🎯 CHESS PIECE TO SQUARE MAPPING DEMONSTRATION")
    print("=" * 60)
    
    # Initialize mapper
    mapper = PieceToSquareMapper()
    
    # Create test pieces at various board positions
    test_pieces = [
        # Starting position pieces
        PieceDetection(16, 16, [0, 0, 32, 32], "WhiteRook1", 0.95),     # a1
        PieceDetection(48, 16, [32, 0, 32, 32], "WhiteKnight1", 0.89),  # b1  
        PieceDetection(80, 16, [64, 0, 32, 32], "WhiteBishop1", 0.92),  # c1
        PieceDetection(112, 16, [96, 0, 32, 32], "WhiteQueen1", 0.97),  # d1
        PieceDetection(144, 16, [128, 0, 32, 32], "WhiteKing1", 0.99),  # e1
        
        # Pawn line
        PieceDetection(16, 48, [0, 32, 32, 32], "WhitePawn1", 0.88),    # a2
        PieceDetection(48, 48, [32, 32, 32, 32], "WhitePawn2", 0.90),   # b2
        PieceDetection(144, 112, [128, 96, 32, 32], "WhitePawn3", 0.85), # e4 (moved pawn)
        
        # Black pieces
        PieceDetection(16, 208, [0, 192, 32, 32], "BlackPawn1", 0.87),   # a7
        PieceDetection(48, 240, [32, 224, 32, 32], "BlackKnight1", 0.91), # b8
        PieceDetection(240, 240, [224, 224, 32, 32], "BlackRook1", 0.94), # h8
    ]
    
    print(f"\n📊 INPUT: {len(test_pieces)} Detected Chess Pieces")
    print("-" * 60)
    print(f"{'Piece Type':<15} {'Position (x,y)':<15} {'Bbox':<20} {'Conf':<6}")
    print("-" * 60)
    
    for i, piece in enumerate(test_pieces):
        bbox_str = f"[{piece.bbox[0]:.0f},{piece.bbox[1]:.0f},{piece.bbox[2]:.0f},{piece.bbox[3]:.0f}]"
        print(f"{piece.piece_type:<15} ({piece.center_x:.0f},{piece.center_y:.0f}){'':>6} {bbox_str:<20} {piece.confidence:.2f}")
    
    print(f"\n⚡ PROCESSING: Hungarian Algorithm Optimization")
    print("-" * 60)
    
    # Show cost matrix creation
    cost_matrix, square_labels = mapper.create_cost_matrix(test_pieces)
    print(f"✅ Created {cost_matrix.shape[0]}×{cost_matrix.shape[1]} cost matrix")
    
    # Find optimal assignments  
    assignments = mapper.solve_assignment(test_pieces)
    print(f"✅ Found {len(assignments)} optimal assignments")
    
    print(f"\n🎯 OUTPUT: Piece-to-Square Assignments")
    print("-" * 60)
    print(f"{'Piece':<12} {'From (x,y)':<12} {'→':<3} {'Square':<8} {'Cost':<8} {'Quality':<10}")
    print("-" * 60)
    
    for assignment in assignments:
        piece_short = assignment.piece.piece_type.replace('1', '').replace('2', '').replace('3', '')
        piece_short = piece_short.replace('White', 'W').replace('Black', 'B')
        from_pos = f"({assignment.piece.center_x:.0f},{assignment.piece.center_y:.0f})"
        
        # Quality assessment based on cost
        if assignment.cost < 0.5:
            quality = "Excellent"
        elif assignment.cost < 1.0:
            quality = "Good"  
        elif assignment.cost < 2.0:
            quality = "Fair"
        else:
            quality = "Poor"
            
        print(f"{piece_short:<12} {from_pos:<12} {'→':<3} {assignment.square:<8} {assignment.cost:<8.3f} {quality:<10}")
    
    # Show final board state
    print(f"\n🏁 RESULT: Final Board Position")
    print("-" * 60)
    create_ascii_board_with_pieces(assignments)
    
    # Show mapping statistics
    print(f"\n📈 STATISTICS")
    print("-" * 60)
    total_cost = sum(a.cost for a in assignments)
    avg_cost = total_cost / len(assignments) if assignments else 0
    excellent_count = sum(1 for a in assignments if a.cost < 0.5)
    good_count = sum(1 for a in assignments if 0.5 <= a.cost < 1.0)
    
    print(f"  Total Pieces Mapped: {len(assignments)}")
    print(f"  Average Assignment Cost: {avg_cost:.3f}")
    print(f"  Excellent Assignments: {excellent_count}")
    print(f"  Good Assignments: {good_count}")
    print(f"  Success Rate: {len(assignments)}/{len(test_pieces)} = {len(assignments)/len(test_pieces)*100:.1f}%")

def create_ascii_board_with_pieces(assignments):
    """Create ASCII board showing the mapped pieces"""
    # Initialize 8x8 board
    board = [['.' for _ in range(8)] for _ in range(8)]
    
    # Piece symbol mapping
    piece_symbols = {
        'WhiteRook': 'R', 'WhiteKnight': 'N', 'WhiteBishop': 'B', 
        'WhiteQueen': 'Q', 'WhiteKing': 'K', 'WhitePawn': 'P',
        'BlackRook': 'r', 'BlackKnight': 'n', 'BlackBishop': 'b',
        'BlackQueen': 'q', 'BlackKing': 'k', 'BlackPawn': 'p'
    }
    
    # Place pieces on board
    for assignment in assignments:
        file, rank = assignment.file, assignment.rank
        piece_type = assignment.piece.piece_type.replace('1', '').replace('2', '').replace('3', '')
        
        # Get symbol
        symbol = piece_symbols.get(piece_type, '?')
        board[7-rank][file] = symbol  # Flip rank for proper display
    
    # Print board with coordinates
    print("    a b c d e f g h")
    print("  ┌─────────────────┐")
    for rank_idx, rank_row in enumerate(board):
        rank_num = 8 - rank_idx
        print(f"{rank_num} │ {' '.join(rank_row)} │")
    print("  └─────────────────┘")
    print("    a b c d e f g h")

def show_cost_breakdown():
    """Show how the cost function works"""
    print(f"\n🔍 COST FUNCTION BREAKDOWN")
    print("=" * 60)
    
    mapper = PieceToSquareMapper()
    
    # Test piece at a specific position
    test_piece = PieceDetection(20, 20, [4, 4, 32, 32], "WhiteRook1", 0.95)
    
    print(f"Analyzing costs for {test_piece.piece_type} at position ({test_piece.center_x}, {test_piece.center_y})")
    print(f"Testing assignment to nearby squares:\n")
    
    test_squares = ['a1', 'b1', 'a2', 'b2']
    
    print(f"{'Square':<8} {'Distance':<10} {'Overlap':<10} {'PieceType':<12} {'BoardEdge':<12} {'Total':<10}")
    print("-" * 60)
    
    for square in test_squares:
        file, rank = mapper.algebraic_to_square(square)
        
        dist_cost = mapper.compute_distance_cost(test_piece, file, rank)
        overlap_cost = mapper.compute_overlap_cost(test_piece, file, rank)
        piece_cost = mapper.compute_piece_type_cost(test_piece, file, rank)
        edge_cost = mapper.compute_board_edge_cost(test_piece, file, rank)
        total_cost = mapper.compute_total_cost(test_piece, file, rank)
        
        print(f"{square:<8} {dist_cost:<10.3f} {overlap_cost:<10.3f} {piece_cost:<12.3f} {edge_cost:<12.3f} {total_cost:<10.3f}")
    
    print(f"\n💡 Cost Components:")
    print(f"  • Distance: How far piece center is from square center")
    print(f"  • Overlap: IoU between piece bounding box and square")
    print(f"  • PieceType: Bonus/penalty based on chess logic (e.g., rooks on back rank)")
    print(f"  • BoardEdge: Penalty for pieces near image boundaries")
    print(f"  • Lower total cost = better assignment")

def main():
    """Run the mapping demonstration"""
    create_test_scenario()
    show_cost_breakdown()
    
    print(f"\n🎉 DEMONSTRATION COMPLETE!")
    print("=" * 60)
    print("This ML approach successfully:")
    print("  ✅ Detects chess pieces from bounding boxes")
    print("  ✅ Warps board to standardized 256×256 view") 
    print("  ✅ Uses multi-criteria cost function for assignment quality")
    print("  ✅ Applies Hungarian algorithm for optimal mapping")
    print("  ✅ Provides clear visualization of results")
    print("  ✅ Evaluates assignment quality and accuracy")

if __name__ == "__main__":
    main()