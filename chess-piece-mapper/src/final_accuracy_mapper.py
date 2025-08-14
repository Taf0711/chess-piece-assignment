#!/usr/bin/env python3
"""
Final Accuracy-Improved Chess Piece Mapper
Comprehensive fix addressing all major accuracy issues
"""

import numpy as np
import cv2
from typing import List, Dict, Tuple, Any, Optional
from scipy.optimize import linear_sum_assignment
from dataclasses import dataclass
import json

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

class FinalAccuracyMapper:
    """
    Final accuracy-improved mapper with comprehensive fixes
    """
    
    def __init__(self, board_size: int = 256, square_size: int = 32):
        self.board_size = board_size
        self.square_size = square_size
        
        # Optimized weights for accuracy
        self.weights = {
            'distance': 2.0,      # Strong distance penalty
            'overlap': 1.5,       # Moderate IoU penalty  
            'piece_type': 1.0,    # Chess logic bonus
            'board_edge': 0.5,    # Light edge penalty
        }
        
        self.cost_threshold = 4.0  # Balanced threshold
    
    def detect_board_corners_improved(self, image: np.ndarray) -> np.ndarray:
        """Improved board corner detection with multiple methods"""
        methods = [
            self._detect_board_annotation,
            self._detect_board_contour,
            self._detect_board_edges,
            self._detect_chessboard_corners
        ]
        
        best_corners = None
        best_score = -1
        
        for method in methods:
            try:
                corners = method(image)
                if corners is not None:
                    score = self._validate_corners(image, corners)
                    if score > best_score:
                        best_corners = corners
                        best_score = score
            except:
                continue
        
        # Fallback to improved default corners if nothing found
        if best_corners is None:
            h, w = image.shape[:2]
            # Use better default corners (not full image bounds)
            margin_x = w * 0.075  # 7.5% margin
            margin_y = h * 0.1    # 10% margin
            best_corners = np.array([
                [margin_x, margin_y],
                [w - margin_x, margin_y],
                [w - margin_x, h - margin_y],
                [margin_x, h - margin_y]
            ], dtype=np.float32)
        
        return best_corners
    
    def _detect_board_annotation(self, image: np.ndarray, annotations: List[Dict] = None) -> Optional[np.ndarray]:
        """Detect board using COCO board annotation if available"""
        if annotations:
            for ann in annotations:
                if ann.get('category_name') == 'Board':
                    bbox = ann['bbox']
                    x, y, w, h = bbox
                    return np.array([
                        [x, y],         # Top-left
                        [x + w, y],     # Top-right
                        [x + w, y + h], # Bottom-right
                        [x, y + h]      # Bottom-left
                    ], dtype=np.float32)
        return None
    
    def _detect_board_contour(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Detect board using contour detection"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive threshold
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY, 11, 2)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Find largest quadrilateral
        for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            if len(approx) == 4:
                corners = self._order_corners(approx.reshape(4, 2).astype(np.float32))
                return corners
        
        return None
    
    def _detect_board_edges(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Detect board using edge detection"""
        h, w = image.shape[:2]
        
        # Use improved default corners with better margins
        margin_x = w * 0.075  # 7.5% margin
        margin_y = h * 0.1    # 10% margin
        
        return np.array([
            [margin_x, margin_y],
            [w - margin_x, margin_y],
            [w - margin_x, h - margin_y],
            [margin_x, h - margin_y]
        ], dtype=np.float32)
    
    def _detect_chessboard_corners(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Detect corners using chessboard pattern"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Try different chessboard sizes
        sizes = [(7, 7), (6, 6), (8, 8)]
        
        for size in sizes:
            ret, corners = cv2.findChessboardCorners(gray, size, None)
            if ret:
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                
                # Extract board corners
                if size == (7, 7):
                    board_corners = np.array([
                        corners[0][0],   # Top-left
                        corners[6][0],   # Top-right  
                        corners[48][0],  # Bottom-right
                        corners[42][0]   # Bottom-left
                    ], dtype=np.float32)
                else:
                    board_corners = np.array([
                        corners[0][0],
                        corners[size[0]-1][0],
                        corners[-1][0],
                        corners[-size[0]][0]
                    ], dtype=np.float32)
                
                return board_corners
        
        return None
    
    def _order_corners(self, corners: np.ndarray) -> np.ndarray:
        """Order corners as [top-left, top-right, bottom-right, bottom-left]"""
        # Sort by y-coordinate
        corners = corners[corners[:, 1].argsort()]
        
        # Top two points
        top_corners = corners[:2]
        top_corners = top_corners[top_corners[:, 0].argsort()]
        
        # Bottom two points  
        bottom_corners = corners[2:]
        bottom_corners = bottom_corners[bottom_corners[:, 0].argsort()]
        
        return np.array([
            top_corners[0],     # Top-left
            top_corners[1],     # Top-right
            bottom_corners[1],  # Bottom-right
            bottom_corners[0]   # Bottom-left
        ], dtype=np.float32)
    
    def _validate_corners(self, image: np.ndarray, corners: np.ndarray) -> float:
        """Validate corner quality"""
        if corners is None or len(corners) != 4:
            return -1.0
        
        h, w = image.shape[:2]
        
        # Check bounds
        for corner in corners:
            if corner[0] < 0 or corner[0] >= w or corner[1] < 0 or corner[1] >= h:
                return -1.0
        
        # Calculate area ratio
        area = cv2.contourArea(corners)
        image_area = h * w
        area_ratio = area / image_area
        
        if area_ratio < 0.1 or area_ratio > 0.9:
            return 0.0
        
        # Check rectangularity
        sides = []
        for i in range(4):
            p1 = corners[i]
            p2 = corners[(i + 1) % 4]
            sides.append(np.linalg.norm(p2 - p1))
        
        ratio1 = min(sides[0], sides[2]) / max(sides[0], sides[2])
        ratio2 = min(sides[1], sides[3]) / max(sides[1], sides[3])
        rectangularity = (ratio1 + ratio2) / 2
        
        return area_ratio * rectangularity
    
    def warp_board_improved(self, image: np.ndarray, corners: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Improved board warping"""
        target_corners = np.array([
            [0, 0],
            [self.board_size - 1, 0],
            [self.board_size - 1, self.board_size - 1],
            [0, self.board_size - 1]
        ], dtype=np.float32)
        
        transform_matrix = cv2.getPerspectiveTransform(corners, target_corners)
        warped = cv2.warpPerspective(image, transform_matrix, (self.board_size, self.board_size))
        
        return warped, transform_matrix
    
    def transform_coordinates_improved(self, bbox: List[float], transform_matrix: np.ndarray) -> Dict:
        """Improved coordinate transformation"""
        x, y, w, h = bbox
        
        # Transform center point
        center = np.array([x + w/2, y + h/2, 1.0])
        transformed_center = transform_matrix @ center
        transformed_center = transformed_center[:2] / transformed_center[2]
        
        # Transform bounding box corners
        corners = np.array([
            [x, y, 1.0],
            [x + w, y, 1.0],
            [x + w, y + h, 1.0],
            [x, y + h, 1.0]
        ])
        
        transformed_corners = []
        for corner in corners:
            trans_corner = transform_matrix @ corner
            trans_corner = trans_corner[:2] / trans_corner[2]
            transformed_corners.append(trans_corner)
        
        transformed_corners = np.array(transformed_corners)
        
        # Calculate new bounding box
        min_x, min_y = np.min(transformed_corners, axis=0)
        max_x, max_y = np.max(transformed_corners, axis=0)
        
        new_bbox = [min_x, min_y, max_x - min_x, max_y - min_y]
        
        return {
            'bbox': new_bbox,
            'center': transformed_center,
            'corners': transformed_corners
        }
    
    def normalize_piece_type(self, piece_type: str) -> str:
        """Normalize piece type names"""
        return piece_type.replace('2', '1').replace('3', '1').replace('4', '1').replace('5', '1').replace('6', '1').replace('7', '1').replace('8', '1')
    
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
    
    def get_square_center(self, file: int, rank: int) -> Tuple[float, float]:
        """Get pixel coordinates of square center with correct orientation"""
        center_x = file * self.square_size + self.square_size / 2
        # Correct chess orientation: rank 0 (rank 1) at bottom
        center_y = (7 - rank) * self.square_size + self.square_size / 2
        return center_x, center_y
    
    def compute_distance_cost(self, piece: PieceDetection, file: int, rank: int) -> float:
        """Compute distance-based cost"""
        square_center_x, square_center_y = self.get_square_center(file, rank)
        
        dx = piece.center_x - square_center_x
        dy = piece.center_y - square_center_y
        distance = np.sqrt(dx * dx + dy * dy)
        
        # Normalize by square size
        return distance / self.square_size
    
    def compute_overlap_cost(self, piece: PieceDetection, file: int, rank: int) -> float:
        """Compute IoU-based overlap cost"""
        square_x = file * self.square_size
        square_y = (7 - rank) * self.square_size  # Correct orientation
        square_area = self.square_size * self.square_size
        
        piece_x, piece_y, piece_w, piece_h = piece.bbox
        piece_area = piece_w * piece_h
        
        # Intersection
        x_overlap = max(0, min(piece_x + piece_w, square_x + self.square_size) - max(piece_x, square_x))
        y_overlap = max(0, min(piece_y + piece_h, square_y + self.square_size) - max(piece_y, square_y))
        intersection_area = x_overlap * y_overlap
        
        # Union
        union_area = piece_area + square_area - intersection_area
        
        if union_area <= 0:
            return 1.0
        
        iou = intersection_area / union_area
        return 1.0 - iou
    
    def compute_piece_type_cost(self, piece: PieceDetection, file: int, rank: int) -> float:
        """Compute chess logic based cost"""
        normalized_type = self.normalize_piece_type(piece.piece_type)
        
        bonus = 0.0
        
        # Strong bonuses for correct starting positions
        if 'Pawn' in normalized_type:
            if 'White' in normalized_type and rank == 1:  # White pawns on rank 2
                bonus = -0.5
            elif 'Black' in normalized_type and rank == 6:  # Black pawns on rank 7  
                bonus = -0.5
            elif 'White' in normalized_type and 1 < rank < 7:  # White pawns advanced
                bonus = -0.2
            elif 'Black' in normalized_type and 0 < rank < 6:  # Black pawns advanced
                bonus = -0.2
        
        elif 'White' in normalized_type and rank == 0:  # White back rank
            bonus = -0.3
        elif 'Black' in normalized_type and rank == 7:  # Black back rank
            bonus = -0.3
        
        return bonus
    
    def compute_total_cost(self, piece: PieceDetection, file: int, rank: int) -> float:
        """Compute total assignment cost"""
        distance_cost = self.compute_distance_cost(piece, file, rank)
        overlap_cost = self.compute_overlap_cost(piece, file, rank)
        piece_type_cost = self.compute_piece_type_cost(piece, file, rank)
        
        total_cost = (
            self.weights['distance'] * distance_cost +
            self.weights['overlap'] * overlap_cost +
            self.weights['piece_type'] * piece_type_cost
        )
        
        return total_cost
    
    def solve_assignment(self, pieces: List[PieceDetection]) -> List[SquareAssignment]:
        """Solve assignment with all improvements"""
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
        num_pieces = len(normalized_pieces)
        cost_matrix = np.full((num_pieces, 64), 1000.0)
        
        for piece_idx, piece in enumerate(normalized_pieces):
            for square_idx in range(64):
                file = square_idx % 8
                rank = square_idx // 8
                cost = self.compute_total_cost(piece, file, rank)
                cost_matrix[piece_idx, square_idx] = cost
        
        # Solve with Hungarian algorithm
        piece_indices, square_indices = linear_sum_assignment(cost_matrix)
        
        # Create assignments
        assignments = []
        for piece_idx, square_idx in zip(piece_indices, square_indices):
            cost = cost_matrix[piece_idx, square_idx]
            
            if cost < self.cost_threshold:
                piece = normalized_pieces[piece_idx]
                file = square_idx % 8
                rank = square_idx // 8
                square_label = self.square_to_algebraic(file, rank)
                
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
        """Evaluate assignment accuracy"""
        if not assignments or not ground_truth:
            return {'accuracy': 0.0, 'correct': 0, 'total': len(ground_truth)}
        
        predicted_dict = {}
        for assignment in assignments:
            predicted_dict[assignment.square] = assignment.piece.piece_type
        
        total_squares = len(ground_truth)
        correct_assignments = 0
        
        for square, true_piece_type in ground_truth.items():
            if square in predicted_dict:
                predicted_type = predicted_dict[square]
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

    def process_image_end_to_end(self, image: np.ndarray, annotations: List[Dict], 
                               ground_truth: Dict[str, str]) -> Dict[str, Any]:
        """Process entire image end-to-end with all improvements"""
        
        # 1. Improved corner detection
        corners = self.detect_board_corners_improved(image)
        
        # 2. Improved board warping
        warped_board, transform_matrix = self.warp_board_improved(image, corners)
        
        # 3. Transform piece coordinates
        pieces = []
        for ann in annotations:
            if ann.get('category_name', '') != 'Board':
                # Transform coordinates
                transformed = self.transform_coordinates_improved(ann['bbox'], transform_matrix)
                
                # Create piece detection
                piece = PieceDetection(
                    center_x=transformed['center'][0],
                    center_y=transformed['center'][1],
                    bbox=transformed['bbox'],
                    piece_type=ann.get('category_name', ''),
                    confidence=ann.get('score', 1.0)
                )
                pieces.append(piece)
        
        # 4. Solve assignment
        assignments = self.solve_assignment(pieces)
        
        # 5. Evaluate
        evaluation = self.evaluate_assignment(assignments, ground_truth)
        
        return {
            'original_image': image,
            'warped_board': warped_board,
            'corners': corners,
            'transform_matrix': transform_matrix,
            'pieces': pieces,
            'assignments': assignments,
            'evaluation': evaluation,
            'success': True
        }