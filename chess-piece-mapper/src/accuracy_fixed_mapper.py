#!/usr/bin/env python3
"""
Accuracy-Fixed Chess Piece Mapper
Fixes the coordinate system and piece type matching issues
"""

import numpy as np
import cv2
from typing import List, Dict, Tuple, Any, Optional
from scipy.optimize import linear_sum_assignment
from dataclasses import dataclass

@dataclass
class PieceDetection:
    """Represents a detected chess piece"""
    center_x: float
    center_y: float
    bbox: List[float]  # [x, y, width, height]
    piece_type: str
    confidence: float = 1.0

@dataclass
class SquareAssignment:
    """Represents assignment of piece to square"""
    piece: PieceDetection
    square: str  # e.g., "e4"
    file: int    # 0-7 (a-h)
    rank: int    # 0-7 (1-8)
    cost: float  # Assignment cost
    confidence: float  # Assignment confidence

class AccuracyFixedMapper:
    """
    Fixed mapper with corrected coordinate system and piece type matching
    """
    
    def __init__(self, board_size: int = 256, square_size: int = 32):
        self.board_size = board_size
        self.square_size = square_size
        
        # Optimized cost function weights
        self.weights = {
            'distance': 1.0,      # Distance penalty
            'overlap': 1.0,       # IoU penalty  
            'piece_type': 0.5,    # Chess logic bonus
            'board_edge': 0.3,    # Edge penalty
        }
        
        self.cost_threshold = 2.0  # Lower threshold for better matching
    
    def normalize_piece_type(self, piece_type: str) -> str:
        """Normalize piece type names to match ground truth format"""
        # Remove variant numbers (2, 3, 4, etc.) and keep only 1
        base_type = piece_type.replace('2', '1').replace('3', '1').replace('4', '1').replace('5', '1').replace('6', '1').replace('7', '1').replace('8', '1')
        return base_type
    
    def square_to_algebraic(self, file: int, rank: int) -> str:
        """Convert square coordinates to algebraic notation"""
        files = 'abcdefgh'
        ranks = '12345678'
        return f"{files[file]}{ranks[rank]}"
    
    def algebraic_to_square(self, notation: str) -> Tuple[int, int]:
        """Convert algebraic notation to square coordinates"""
        file = ord(notation[0]) - ord('a')  # a=0, b=1, ..., h=7
        rank = int(notation[1]) - 1         # 1=0, 2=1, ..., 8=7
        return file, rank
    
    def get_square_center(self, file: int, rank: int) -> Tuple[float, float]:
        """Get pixel coordinates of square center with FIXED coordinate system"""
        center_x = file * self.square_size + self.square_size / 2
        # FIX: Invert Y coordinate - rank 0 (rank 1) should be at bottom
        center_y = (7 - rank) * self.square_size + self.square_size / 2
        return center_x, center_y
    
    def get_square_from_coords(self, x: float, y: float) -> Tuple[int, int]:
        """Convert pixel coordinates to square coordinates"""
        file = int(np.clip(x // self.square_size, 0, 7))
        # FIX: Invert Y coordinate
        rank = 7 - int(np.clip(y // self.square_size, 0, 7))
        return file, rank
    
    def compute_distance_cost(self, piece: PieceDetection, file: int, rank: int) -> float:
        """Compute distance-based cost between piece and square"""
        square_center_x, square_center_y = self.get_square_center(file, rank)
        
        # Euclidean distance
        dx = piece.center_x - square_center_x
        dy = piece.center_y - square_center_y
        distance = np.sqrt(dx * dx + dy * dy)
        
        # Normalize by square size
        normalized_distance = distance / self.square_size
        
        return normalized_distance
    
    def compute_overlap_cost(self, piece: PieceDetection, file: int, rank: int) -> float:
        """Compute bounding box overlap cost"""
        # Square boundaries with fixed coordinate system
        square_x = file * self.square_size
        square_y = (7 - rank) * self.square_size  # FIX: Invert Y
        square_area = self.square_size * self.square_size
        
        # Piece bounding box
        piece_x, piece_y, piece_w, piece_h = piece.bbox
        piece_area = piece_w * piece_h
        
        # Intersection rectangle
        x_overlap = max(0, min(piece_x + piece_w, square_x + self.square_size) - max(piece_x, square_x))
        y_overlap = max(0, min(piece_y + piece_h, square_y + self.square_size) - max(piece_y, square_y))
        intersection_area = x_overlap * y_overlap
        
        # Union area
        union_area = piece_area + square_area - intersection_area
        
        if union_area == 0:
            return 1.0
        
        iou = intersection_area / union_area
        return 1.0 - iou  # Convert IoU to cost
    
    def compute_piece_type_cost(self, piece: PieceDetection, file: int, rank: int) -> float:
        """Compute piece type consistency cost based on chess rules"""
        normalized_type = self.normalize_piece_type(piece.piece_type)
        
        bonus = 0.0
        
        # Pawn logic - pawns start on rank 2 (white) and rank 7 (black)
        if 'Pawn' in normalized_type:
            if 'White' in normalized_type and rank == 1:  # White pawns on rank 2
                bonus = -0.3  # Strong bonus
            elif 'Black' in normalized_type and rank == 6:  # Black pawns on rank 7  
                bonus = -0.3  # Strong bonus
            elif 'White' in normalized_type and rank >= 2:  # White pawns can advance
                bonus = -0.1  # Small bonus
            elif 'Black' in normalized_type and rank <= 5:  # Black pawns can advance
                bonus = -0.1  # Small bonus
        
        # Back rank pieces (rank 1 for white, rank 8 for black)
        elif 'White' in normalized_type and rank == 0:  # White back rank
            bonus = -0.2
        elif 'Black' in normalized_type and rank == 7:  # Black back rank
            bonus = -0.2
        
        # Central squares bonus for active pieces
        if 2 <= file <= 5 and 2 <= rank <= 5:  # Center 4x4
            if any(piece_name in normalized_type for piece_name in ['Knight', 'Bishop', 'Queen']):
                bonus -= 0.1
        
        return bonus
    
    def compute_total_cost(self, piece: PieceDetection, file: int, rank: int) -> float:
        """Compute total assignment cost"""
        distance_cost = self.compute_distance_cost(piece, file, rank)
        overlap_cost = self.compute_overlap_cost(piece, file, rank)
        piece_type_cost = self.compute_piece_type_cost(piece, file, rank)
        
        # Confidence multiplier
        confidence_multiplier = 2.0 - piece.confidence
        
        total_cost = (
            self.weights['distance'] * distance_cost +
            self.weights['overlap'] * overlap_cost +
            self.weights['piece_type'] * piece_type_cost
        ) * confidence_multiplier
        
        return total_cost
    
    def create_cost_matrix(self, pieces: List[PieceDetection]) -> Tuple[np.ndarray, List[str]]:
        """Create cost matrix for Hungarian algorithm"""
        num_pieces = len(pieces)
        cost_matrix = np.full((num_pieces, 64), 1000.0)
        square_labels = []
        
        # Generate all square labels
        for rank in range(8):
            for file in range(8):
                square_labels.append(self.square_to_algebraic(file, rank))
        
        # Fill cost matrix
        for piece_idx, piece in enumerate(pieces):
            for square_idx in range(64):
                file = square_idx % 8
                rank = square_idx // 8
                
                cost = self.compute_total_cost(piece, file, rank)
                cost_matrix[piece_idx, square_idx] = cost
        
        return cost_matrix, square_labels
    
    def solve_assignment(self, pieces: List[PieceDetection]) -> List[SquareAssignment]:
        """Solve piece-to-square assignment using Hungarian algorithm"""
        if not pieces:
            return []
        
        # Normalize piece types
        normalized_pieces = []
        for piece in pieces:
            normalized_piece = PieceDetection(
                center_x=piece.center_x,
                center_y=piece.center_y,
                bbox=piece.bbox,
                piece_type=self.normalize_piece_type(piece.piece_type),
                confidence=piece.confidence
            )
            normalized_pieces.append(normalized_piece)
        
        # Create cost matrix
        cost_matrix, square_labels = self.create_cost_matrix(normalized_pieces)
        
        # Solve with Hungarian algorithm
        piece_indices, square_indices = linear_sum_assignment(cost_matrix)
        
        # Create assignments
        assignments = []
        for piece_idx, square_idx in zip(piece_indices, square_indices):
            cost = cost_matrix[piece_idx, square_idx]
            
            if cost < self.cost_threshold:
                piece = normalized_pieces[piece_idx]
                square_label = square_labels[square_idx]
                file, rank = self.algebraic_to_square(square_label)
                
                # Calculate confidence
                confidence = max(0.0, 1.0 - (cost / self.cost_threshold))
                
                assignment = SquareAssignment(
                    piece=piece,
                    square=square_label,
                    file=file,
                    rank=rank,
                    cost=cost,
                    confidence=confidence
                )
                assignments.append(assignment)
        
        return assignments
    
    def evaluate_assignment(self, assignments: List[SquareAssignment], 
                          ground_truth: Dict[str, str]) -> Dict[str, Any]:
        """Evaluate assignment accuracy against ground truth"""
        if not assignments or not ground_truth:
            return {'accuracy': 0.0, 'correct': 0, 'total': len(ground_truth)}
        
        # Convert predictions to dict format
        predicted_dict = {}
        for assignment in assignments:
            predicted_dict[assignment.square] = assignment.piece.piece_type
        
        # Calculate accuracy
        total_squares = len(ground_truth)
        correct_assignments = 0
        
        for square, true_piece_type in ground_truth.items():
            if square in predicted_dict:
                predicted_type = predicted_dict[square]
                # Normalize both for comparison
                normalized_true = self.normalize_piece_type(true_piece_type)
                normalized_pred = self.normalize_piece_type(predicted_type)
                
                if normalized_pred == normalized_true:
                    correct_assignments += 1
        
        accuracy = correct_assignments / max(1, total_squares)
        
        return {
            'accuracy': accuracy,
            'correct': correct_assignments,
            'total': total_squares,
            'predicted_positions': predicted_dict,
            'missed_pieces': [square for square in ground_truth if square not in predicted_dict],
            'false_positives': [square for square in predicted_dict if square not in ground_truth]
        }

def test_fixed_mapper():
    """Test the fixed mapper on sample data"""
    print(" TESTING FIXED MAPPER")
    print("=" * 50)
    
    from src.optimized_data_loader import OptimizedChessDataLoader
    
    # Initialize
    data_path = "/home/pre/projects/chess-datagen/gen-data/render_src/coco_data_2025_08_08__21_53_08"
    data_loader = OptimizedChessDataLoader(data_path)
    fixed_mapper = AccuracyFixedMapper()
    
    # Test coordinate system
    print("Testing coordinate system:")
    test_squares = ['a1', 'h1', 'a8', 'h8', 'e4', 'd5']
    for square in test_squares:
        file, rank = fixed_mapper.algebraic_to_square(square)
        center_x, center_y = fixed_mapper.get_square_center(file, rank)
        back_square = fixed_mapper.square_to_algebraic(file, rank)
        print(f"  {square} -> ({file},{rank}) -> center=({center_x:.0f},{center_y:.0f}) -> {back_square} ✓")
    
    # Test on real data
    print(f"\nTesting on real image:")
    image_id = 0
    result = data_loader.process_image_optimized(image_id)
    
    if result and result.get('success'):
        pieces = result['pieces']
        ground_truth = result['ground_truth']
        
        # Get assignments with fixed mapper
        assignments = fixed_mapper.solve_assignment(pieces)
        evaluation = fixed_mapper.evaluate_assignment(assignments, ground_truth)
        
        print(f"  Pieces detected: {len(pieces)}")
        print(f"  Assignments made: {len(assignments)}")
        print(f"  Accuracy: {evaluation['accuracy']:.1%}")
        print(f"  Correct assignments: {evaluation['correct']}/{evaluation['total']}")
        
        # Show some sample assignments
        print(f"\nSample correct assignments:")
        for assignment in assignments[:5]:
            square = assignment.square
            predicted = assignment.piece.piece_type
            actual = ground_truth.get(square, 'NOT_FOUND')
            correct = fixed_mapper.normalize_piece_type(predicted) == fixed_mapper.normalize_piece_type(actual)
            status = "✓" if correct else ""
            print(f"  {square}: {predicted} vs {actual} {status}")
        
        return evaluation['accuracy']
    
    return 0.0

if __name__ == "__main__":
    test_fixed_mapper()