#!/usr/bin/env python3
"""
Chess Piece to Square Assignment using Hungarian Algorithm
Implementation of ML approach for mapping chess pieces to board squares
"""

import numpy as np
from typing import List, Dict, Tuple, Any, Optional
from scipy.optimize import linear_sum_assignment
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report
from dataclasses import dataclass
import json
import pickle


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
    

class PieceToSquareMapper:
    """
    Machine Learning approach for mapping chess pieces to board squares
    Uses cost function optimization with Hungarian algorithm
    """
    
    def __init__(self, board_size: int = 256, square_size: int = 32):
        self.board_size = board_size
        self.square_size = square_size
        
        # Cost function weights
        self.weights = {
            'distance': 1.0,      # Distance from piece center to square center
            'overlap': 2.0,       # Bounding box overlap with square
            'piece_type': 0.5,    # Piece type consistency bonus
            'board_edge': 1.5,    # Penalty for pieces near board edges
            'collision': 10.0     # Penalty for multiple pieces in same square
        }
        
        # Chess piece type mappings
        self.piece_types = {
            'WhitePawn': 'P', 'BlackPawn': 'p',
            'WhiteRook': 'R', 'BlackRook': 'r', 
            'WhiteKnight': 'N', 'BlackKnight': 'n',
            'WhiteBishop': 'B', 'BlackBishop': 'b',
            'WhiteQueen': 'Q', 'BlackQueen': 'q',
            'WhiteKing': 'K', 'BlackKing': 'k'
        }
        
        # Statistics for evaluation
        self.assignment_stats = {
            'total_assignments': 0,
            'correct_assignments': 0,
            'piece_type_accuracy': {},
            'distance_errors': []
        }
    
    def get_square_center(self, file: int, rank: int) -> Tuple[float, float]:
        """Get pixel coordinates of square center"""
        center_x = file * self.square_size + self.square_size / 2
        center_y = rank * self.square_size + self.square_size / 2
        return center_x, center_y
    
    def get_square_from_coords(self, x: float, y: float) -> Tuple[int, int]:
        """Convert pixel coordinates to square coordinates"""
        file = int(np.clip(x // self.square_size, 0, 7))
        rank = int(np.clip(y // self.square_size, 0, 7))
        return file, rank
    
    def square_to_algebraic(self, file: int, rank: int) -> str:
        """Convert square coordinates to algebraic notation"""
        files = 'abcdefgh'
        ranks = '12345678'
        return f"{files[file]}{ranks[rank]}"
    
    def algebraic_to_square(self, notation: str) -> Tuple[int, int]:
        """Convert algebraic notation to square coordinates"""
        file = ord(notation[0]) - ord('a')
        rank = int(notation[1]) - 1
        return file, rank
    
    def compute_distance_cost(self, piece: PieceDetection, file: int, rank: int) -> float:
        """Compute distance-based cost between piece and square"""
        square_center_x, square_center_y = self.get_square_center(file, rank)
        
        # Euclidean distance from piece center to square center
        dx = piece.center_x - square_center_x
        dy = piece.center_y - square_center_y
        distance = np.sqrt(dx * dx + dy * dy)
        
        # Normalize by square size
        normalized_distance = distance / self.square_size
        
        return normalized_distance
    
    def compute_overlap_cost(self, piece: PieceDetection, file: int, rank: int) -> float:
        """Compute bounding box overlap cost"""
        # Square boundaries
        square_x = file * self.square_size
        square_y = rank * self.square_size
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
        
        # IoU (Intersection over Union) - higher is better, so we return 1 - IoU as cost
        if union_area == 0:
            return 1.0
        
        iou = intersection_area / union_area
        return 1.0 - iou
    
    def compute_piece_type_cost(self, piece: PieceDetection, file: int, rank: int) -> float:
        """Compute piece type consistency cost based on chess rules"""
        piece_base_type = piece.piece_type.replace('1', '').replace('2', '').replace('3', '').replace('4', '').replace('5', '').replace('6', '').replace('7', '').replace('8', '')
        
        # Bonus for pieces on expected starting positions
        if 'Pawn' in piece_base_type:
            if ('White' in piece_base_type and rank == 1) or ('Black' in piece_base_type and rank == 6):
                return -0.2  # Bonus (negative cost)
            elif ('White' in piece_base_type and rank == 0) or ('Black' in piece_base_type and rank == 7):
                return 0.5   # Penalty for pawns on back rank
        
        elif 'Rook' in piece_base_type:
            if file in [0, 7]:  # Rooks often start on a/h files
                return -0.1  # Small bonus
                
        elif 'Knight' in piece_base_type:
            if file in [1, 6]:  # Knights often start on b/g files
                return -0.1  # Small bonus
                
        elif 'Bishop' in piece_base_type:
            if file in [2, 5]:  # Bishops often start on c/f files
                return -0.1  # Small bonus
                
        elif 'Queen' in piece_base_type:
            if file == 3:  # Queen often starts on d file
                return -0.1  # Small bonus
                
        elif 'King' in piece_base_type:
            if file == 4:  # King often starts on e file
                return -0.1  # Small bonus
        
        return 0.0  # Neutral cost
    
    def compute_board_edge_cost(self, piece: PieceDetection, file: int, rank: int) -> float:
        """Compute cost based on proximity to board edges"""
        # Pieces completely outside the board get high penalty
        if piece.center_x < 0 or piece.center_x >= self.board_size or \
           piece.center_y < 0 or piece.center_y >= self.board_size:
            return 5.0
        
        # Small penalty for pieces very close to edges (might be partially cut off)
        edge_distance = min(
            piece.center_x, 
            piece.center_y,
            self.board_size - piece.center_x,
            self.board_size - piece.center_y
        )
        
        if edge_distance < self.square_size * 0.25:
            return 0.5
        
        return 0.0
    
    def compute_total_cost(self, piece: PieceDetection, file: int, rank: int) -> float:
        """Compute total assignment cost"""
        distance_cost = self.compute_distance_cost(piece, file, rank)
        overlap_cost = self.compute_overlap_cost(piece, file, rank)
        piece_type_cost = self.compute_piece_type_cost(piece, file, rank)
        board_edge_cost = self.compute_board_edge_cost(piece, file, rank)
        
        total_cost = (
            self.weights['distance'] * distance_cost +
            self.weights['overlap'] * overlap_cost +
            self.weights['piece_type'] * piece_type_cost +
            self.weights['board_edge'] * board_edge_cost
        )
        
        return total_cost
    
    def create_cost_matrix(self, pieces: List[PieceDetection]) -> Tuple[np.ndarray, List[str]]:
        """
        Create cost matrix for Hungarian algorithm
        Returns: (cost_matrix, square_labels)
        """
        num_pieces = len(pieces)
        num_squares = 64
        
        # Create cost matrix: pieces x squares
        cost_matrix = np.full((num_pieces, num_squares), 1000.0)  # High default cost
        square_labels = []
        
        # Generate all square labels
        for rank in range(8):
            for file in range(8):
                square_labels.append(self.square_to_algebraic(file, rank))
        
        # Fill cost matrix
        for piece_idx, piece in enumerate(pieces):
            for square_idx in range(num_squares):
                file = square_idx % 8
                rank = square_idx // 8
                
                cost = self.compute_total_cost(piece, file, rank)
                cost_matrix[piece_idx, square_idx] = cost
        
        return cost_matrix, square_labels
    
    def solve_assignment(self, pieces: List[PieceDetection]) -> List[SquareAssignment]:
        """
        Solve piece-to-square assignment using Hungarian algorithm
        """
        if not pieces:
            return []
        
        # Create cost matrix
        cost_matrix, square_labels = self.create_cost_matrix(pieces)
        
        # Solve assignment problem
        piece_indices, square_indices = linear_sum_assignment(cost_matrix)
        
        # Create assignment results
        assignments = []
        for piece_idx, square_idx in zip(piece_indices, square_indices):
            piece = pieces[piece_idx]
            square_label = square_labels[square_idx]
            file, rank = self.algebraic_to_square(square_label)
            cost = cost_matrix[piece_idx, square_idx]
            
            # Only include reasonable assignments (filter out very high costs)
            if cost < 10.0:  # Threshold for reasonable assignments
                assignment = SquareAssignment(
                    piece=piece,
                    square=square_label,
                    file=file,
                    rank=rank,
                    cost=cost
                )
                assignments.append(assignment)
        
        return assignments
    
    def evaluate_assignment(self, predicted_assignments: List[SquareAssignment], 
                          ground_truth: Dict[str, str]) -> Dict[str, Any]:
        """
        Evaluate assignment accuracy against ground truth
        Args:
            predicted_assignments: List of predicted assignments
            ground_truth: Dict mapping square -> piece_type
        """
        # Convert predictions to dict format
        predicted_dict = {}
        for assignment in predicted_assignments:
            predicted_dict[assignment.square] = assignment.piece.piece_type
        
        # Calculate metrics
        total_squares = len(ground_truth)
        correct_assignments = 0
        piece_type_correct = {}
        distance_errors = []
        
        for square, true_piece_type in ground_truth.items():
            if square in predicted_dict:
                predicted_type = predicted_dict[square]
                if predicted_type == true_piece_type:
                    correct_assignments += 1
                    
                # Track by piece type
                base_type = true_piece_type.replace('1', '').replace('2', '').replace('3', '').replace('4', '').replace('5', '').replace('6', '').replace('7', '').replace('8', '')
                if base_type not in piece_type_correct:
                    piece_type_correct[base_type] = {'correct': 0, 'total': 0}
                piece_type_correct[base_type]['total'] += 1
                if predicted_type == true_piece_type:
                    piece_type_correct[base_type]['correct'] += 1
        
        accuracy = correct_assignments / max(1, total_squares)
        
        # Update statistics
        self.assignment_stats['total_assignments'] += total_squares
        self.assignment_stats['correct_assignments'] += correct_assignments
        
        return {
            'accuracy': accuracy,
            'correct': correct_assignments,
            'total': total_squares,
            'piece_type_accuracy': {k: v['correct']/max(1, v['total']) for k, v in piece_type_correct.items()},
            'predicted_positions': predicted_dict,
            'missed_pieces': [square for square in ground_truth if square not in predicted_dict],
            'false_positives': [square for square in predicted_dict if square not in ground_truth]
        }
    
    def train_weights(self, training_data: List[Dict[str, Any]], iterations: int = 100):
        """
        Simple optimization of weights based on training data
        Args:
            training_data: List of {pieces, ground_truth} dicts
        """
        best_weights = self.weights.copy()
        best_accuracy = 0.0
        
        print(f"Training assignment weights on {len(training_data)} samples...")
        
        for iteration in range(iterations):
            # Small random perturbations to weights
            test_weights = best_weights.copy()
            for key in test_weights:
                perturbation = np.random.normal(0, 0.1)
                test_weights[key] = max(0.1, test_weights[key] + perturbation)
            
            # Test these weights
            self.weights = test_weights
            total_accuracy = 0.0
            
            for sample in training_data:
                pieces = sample['pieces']
                ground_truth = sample['ground_truth']
                
                assignments = self.solve_assignment(pieces)
                evaluation = self.evaluate_assignment(assignments, ground_truth)
                total_accuracy += evaluation['accuracy']
            
            avg_accuracy = total_accuracy / len(training_data)
            
            if avg_accuracy > best_accuracy:
                best_accuracy = avg_accuracy
                best_weights = test_weights.copy()
                print(f"Iteration {iteration}: New best accuracy = {best_accuracy:.3f}")
        
        self.weights = best_weights
        print(f"Final weights: {self.weights}")
        print(f"Final accuracy: {best_accuracy:.3f}")
        
        return best_weights
    
    def save_model(self, filepath: str):
        """Save trained model weights and parameters"""
        model_data = {
            'weights': self.weights,
            'board_size': self.board_size,
            'square_size': self.square_size,
            'assignment_stats': self.assignment_stats,
            'piece_types': self.piece_types
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained model weights and parameters"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.weights = model_data['weights']
        self.board_size = model_data['board_size']
        self.square_size = model_data['square_size']
        self.assignment_stats = model_data.get('assignment_stats', {})
        self.piece_types = model_data.get('piece_types', self.piece_types)
        
        print(f"Model loaded from {filepath}")
    
    def visualize_assignments(self, assignments: List[SquareAssignment], 
                            board_image: np.ndarray = None):
        """Visualize the piece assignments on the board"""
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        
        if board_image is not None:
            ax.imshow(board_image)
        else:
            # Create a simple checkered board background
            board = np.zeros((self.board_size, self.board_size, 3))
            for rank in range(8):
                for file in range(8):
                    if (rank + file) % 2 == 1:
                        x = file * self.square_size
                        y = rank * self.square_size
                        board[y:y+self.square_size, x:x+self.square_size] = [0.8, 0.8, 0.8]
            ax.imshow(board)
        
        # Draw grid
        for i in range(0, self.board_size + 1, self.square_size):
            ax.axhline(i, color='black', alpha=0.3, linewidth=1)
            ax.axvline(i, color='black', alpha=0.3, linewidth=1)
        
        # Draw assignments
        for assignment in assignments:
            piece = assignment.piece
            
            # Draw bounding box
            bbox = piece.bbox
            rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], 
                                   linewidth=2, edgecolor='red', facecolor='none')
            ax.add_patch(rect)
            
            # Draw piece center
            ax.plot(piece.center_x, piece.center_y, 'ro', markersize=8)
            
            # Draw square assignment
            square_center_x, square_center_y = self.get_square_center(assignment.file, assignment.rank)
            ax.plot(square_center_x, square_center_y, 'go', markersize=8)
            
            # Draw connection line
            ax.plot([piece.center_x, square_center_x], 
                   [piece.center_y, square_center_y], 'b--', alpha=0.7)
            
            # Label
            piece_short = piece.piece_type.replace('White', 'W').replace('Black', 'B').replace('1', '').replace('2', '').replace('3', '')[:3]
            ax.text(square_center_x, square_center_y - 10, 
                   f"{assignment.square}\n{piece_short}\n{assignment.cost:.2f}", 
                   ha='center', va='center', fontsize=8, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        
        ax.set_xlim(0, self.board_size)
        ax.set_ylim(self.board_size, 0)  # Flip Y axis for proper chess board orientation
        ax.set_title("Chess Piece to Square Assignments")
        plt.tight_layout()
        plt.show()


def create_test_pieces() -> List[PieceDetection]:
    """Create test piece detections for demonstration"""
    # Piece centers should be at square centers: file*32 + 16, rank*32 + 16
    pieces = [
        PieceDetection(16, 16, [0, 0, 32, 32], "WhiteRook1"),        # a1: (0*32+16, 0*32+16) = (16, 16)
        PieceDetection(80, 16, [64, 0, 32, 32], "WhiteKnight1"),     # c1: (2*32+16, 0*32+16) = (80, 16)  
        PieceDetection(144, 16, [128, 0, 32, 32], "WhiteBishop1"),   # e1: (4*32+16, 0*32+16) = (144, 16)
        PieceDetection(16, 48, [0, 32, 32, 32], "WhitePawn1"),       # a2: (0*32+16, 1*32+16) = (16, 48)
        PieceDetection(80, 48, [64, 32, 32, 32], "WhitePawn2"),      # c2: (2*32+16, 1*32+16) = (80, 48)
        PieceDetection(16, 176, [0, 160, 32, 32], "BlackPawn1"),     # a6: (0*32+16, 5*32+16) = (16, 176)
        PieceDetection(144, 176, [128, 160, 32, 32], "BlackKnight1"), # e6: (4*32+16, 5*32+16) = (144, 176)
    ]
    return pieces


def main():
    """Test the piece mapper"""
    print("=== Chess Piece to Square Mapper Test ===")
    
    # Initialize mapper
    mapper = PieceToSquareMapper()
    
    # Create test data
    test_pieces = create_test_pieces()
    
    print(f"Testing with {len(test_pieces)} pieces:")
    for i, piece in enumerate(test_pieces):
        print(f"  {i}: {piece.piece_type} at ({piece.center_x:.1f}, {piece.center_y:.1f})")
    
    # Solve assignment
    assignments = mapper.solve_assignment(test_pieces)
    
    print(f"\nAssignment Results:")
    for assignment in assignments:
        print(f"  {assignment.piece.piece_type} -> {assignment.square} (cost: {assignment.cost:.3f})")
    
    # Test evaluation
    ground_truth = {
        'a1': 'WhiteRook1',
        'c1': 'WhiteKnight1', 
        'e1': 'WhiteBishop1',
        'a2': 'WhitePawn1',
        'c2': 'WhitePawn2',
        'a6': 'BlackPawn1',
        'e6': 'BlackKnight1'
    }
    
    evaluation = mapper.evaluate_assignment(assignments, ground_truth)
    print(f"\nEvaluation Results:")
    print(f"  Accuracy: {evaluation['accuracy']:.3f}")
    print(f"  Correct: {evaluation['correct']}/{evaluation['total']}")
    print(f"  Piece type accuracy: {evaluation['piece_type_accuracy']}")
    
    if evaluation['missed_pieces']:
        print(f"  Missed pieces: {evaluation['missed_pieces']}")
    if evaluation['false_positives']:
        print(f"  False positives: {evaluation['false_positives']}")


if __name__ == "__main__":
    main()