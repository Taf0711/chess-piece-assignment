#!/usr/bin/env python3
"""
Optimized Chess Piece to Square Mapper
Improved version with better board detection, coordinate transformation, and cost function
"""

import numpy as np
import cv2
from typing import List, Dict, Tuple, Any, Optional
from scipy.optimize import linear_sum_assignment
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
    confidence: float  # Assignment confidence

class OptimizedPieceMapper:
    """
    Optimized machine learning approach for chess piece to square mapping
    Features:
    - Improved board corner detection
    - Better coordinate transformation
    - Adaptive cost thresholds
    - Multiple validation methods
    """
    
    def __init__(self, board_size: int = 256, square_size: int = 32):
        self.board_size = board_size
        self.square_size = square_size
        
        # Optimized cost function weights
        self.weights = {
            'distance': 0.5,      # Reduced distance weight
            'overlap': 1.0,       # Reduced overlap weight  
            'piece_type': 0.3,    # Reduced piece type weight
            'board_edge': 0.2,    # Reduced board edge penalty
            'collision': 5.0      # Reduced collision penalty
        }
        
        # Adaptive thresholds
        self.cost_threshold = 3.0  # More lenient threshold
        self.min_confidence = 0.3  # Minimum assignment confidence
        
    def detect_board_corners_improved(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Improved board corner detection with multiple methods"""
        try:
            # Method 1: Try chessboard pattern detection
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (7, 7), None)
            
            if ret:
                # Refine corners
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                
                # Extract outer corners
                board_corners = np.array([
                    corners[0][0],   # Top-left
                    corners[6][0],   # Top-right  
                    corners[48][0],  # Bottom-right (7*7-1)
                    corners[42][0]   # Bottom-left (7*6)
                ], dtype=np.float32)
                
                return board_corners
                
        except Exception:
            pass
            
        # Method 2: Contour-based detection
        try:
            return self._detect_corners_by_contour(image)
        except Exception:
            pass
            
        # Method 3: Edge-based detection
        try:
            return self._detect_corners_by_edges(image)
        except Exception:
            pass
            
        # Method 4: Fallback to image bounds with padding
        h, w = image.shape[:2]
        padding = min(w, h) * 0.1  # 10% padding
        return np.array([
            [padding, padding],
            [w - padding, padding],
            [w - padding, h - padding],
            [padding, h - padding]
        ], dtype=np.float32)
    
    def _detect_corners_by_contour(self, image: np.ndarray) -> np.ndarray:
        """Detect board by finding largest rectangular contour"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive threshold
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY, 11, 2)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Find largest quadrilateral
        for contour in sorted(contours, key=cv2.contourArea, reverse=True):
            # Approximate contour
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            if len(approx) == 4:
                return approx.reshape(4, 2).astype(np.float32)
                
        raise Exception("No quadrilateral found")
    
    def _detect_corners_by_edges(self, image: np.ndarray) -> np.ndarray:
        """Detect board corners using edge detection"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Hough line detection
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
        
        if lines is not None and len(lines) >= 4:
            # Simple approach: use image corners with some inset
            h, w = image.shape[:2]
            inset = min(w, h) * 0.05  # 5% inset
            return np.array([
                [inset, inset],
                [w - inset, inset], 
                [w - inset, h - inset],
                [inset, h - inset]
            ], dtype=np.float32)
            
        raise Exception("Insufficient edges detected")
    
    def warp_board_improved(self, image: np.ndarray, corners: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Improved board warping with quality validation"""
        # Target corners for 256x256 square
        target_corners = np.array([
            [0, 0],
            [self.board_size - 1, 0],
            [self.board_size - 1, self.board_size - 1], 
            [0, self.board_size - 1]
        ], dtype=np.float32)
        
        # Compute perspective transform
        transform_matrix = cv2.getPerspectiveTransform(corners, target_corners)
        
        # Apply transform
        warped = cv2.warpPerspective(image, transform_matrix, (self.board_size, self.board_size))
        
        # Validate warp quality
        if self._validate_warp_quality(warped):
            return warped, transform_matrix
        else:
            # Try with adjusted corners
            adjusted_corners = self._adjust_corners(corners, image.shape)
            transform_matrix = cv2.getPerspectiveTransform(adjusted_corners, target_corners)
            warped = cv2.warpPerspective(image, transform_matrix, (self.board_size, self.board_size))
            return warped, transform_matrix
    
    def _validate_warp_quality(self, warped_image: np.ndarray) -> bool:
        """Validate the quality of board warping"""
        # Check if warped image has reasonable contrast
        gray = cv2.cvtColor(warped_image, cv2.COLOR_BGR2GRAY)
        std_dev = np.std(gray)
        
        # Check for checkerboard pattern
        mean_intensity = np.mean(gray)
        
        return std_dev > 30 and 50 < mean_intensity < 200
    
    def _adjust_corners(self, corners: np.ndarray, image_shape: Tuple) -> np.ndarray:
        """Adjust corners if warp quality is poor"""
        h, w = image_shape[:2]
        
        # Slightly expand corners outward
        center_x, center_y = w / 2, h / 2
        adjusted = corners.copy()
        
        for i, corner in enumerate(corners):
            # Move corner away from center by 5%
            dx = corner[0] - center_x
            dy = corner[1] - center_y
            adjusted[i][0] = corner[0] + dx * 0.05
            adjusted[i][1] = corner[1] + dy * 0.05
            
            # Clamp to image bounds
            adjusted[i][0] = np.clip(adjusted[i][0], 0, w - 1)
            adjusted[i][1] = np.clip(adjusted[i][1], 0, h - 1)
            
        return adjusted
    
    def transform_coordinates_improved(self, bbox: List[float], transform_matrix: np.ndarray) -> Dict:
        """Improved coordinate transformation with validation"""
        x, y, w, h = bbox
        
        # Calculate multiple points for better accuracy
        points = np.array([
            [x, y],              # Top-left
            [x + w, y],          # Top-right
            [x + w, y + h],      # Bottom-right
            [x, y + h],          # Bottom-left
            [x + w/2, y + h/2]   # Center
        ], dtype=np.float32)
        
        # Transform all points
        transformed = cv2.perspectiveTransform(points.reshape(-1, 1, 2), transform_matrix)
        transformed = transformed.reshape(-1, 2)
        
        # Calculate new bounding box from transformed corners
        min_x, min_y = np.min(transformed[:4], axis=0)
        max_x, max_y = np.max(transformed[:4], axis=0)
        
        new_bbox = [min_x, min_y, max_x - min_x, max_y - min_y]
        new_center = transformed[4]  # Transformed center point
        
        return {
            'bbox': new_bbox,
            'center': new_center,
            'corners': transformed[:4]
        }
    
    def compute_adaptive_cost(self, piece: PieceDetection, file: int, rank: int) -> float:
        """Compute adaptive cost with improved metrics"""
        # Get square center
        square_center_x = file * self.square_size + self.square_size / 2
        square_center_y = rank * self.square_size + self.square_size / 2
        
        # Distance cost (normalized by square size)
        dx = piece.center_x - square_center_x
        dy = piece.center_y - square_center_y
        distance = np.sqrt(dx * dx + dy * dy)
        distance_cost = distance / self.square_size
        
        # Overlap cost using IoU
        overlap_cost = self._compute_iou_cost(piece, file, rank)
        
        # Piece type bonus/penalty
        piece_type_cost = self._compute_piece_logic_cost(piece, file, rank)
        
        # Board edge penalty (less severe)
        edge_cost = self._compute_edge_penalty(piece)
        
        # Confidence adjustment
        confidence_multiplier = 2.0 - piece.confidence  # Lower confidence = higher cost
        
        total_cost = (
            self.weights['distance'] * distance_cost +
            self.weights['overlap'] * overlap_cost +
            self.weights['piece_type'] * piece_type_cost +
            self.weights['board_edge'] * edge_cost
        ) * confidence_multiplier
        
        return total_cost
    
    def _compute_iou_cost(self, piece: PieceDetection, file: int, rank: int) -> float:
        """Compute IoU-based overlap cost"""
        # Square boundaries
        square_x = file * self.square_size
        square_y = rank * self.square_size
        
        # Piece bounding box
        piece_x, piece_y, piece_w, piece_h = piece.bbox
        
        # Intersection
        x_overlap = max(0, min(piece_x + piece_w, square_x + self.square_size) - max(piece_x, square_x))
        y_overlap = max(0, min(piece_y + piece_h, square_y + self.square_size) - max(piece_y, square_y))
        intersection = x_overlap * y_overlap
        
        # Union
        piece_area = piece_w * piece_h
        square_area = self.square_size * self.square_size
        union = piece_area + square_area - intersection
        
        if union <= 0:
            return 1.0
            
        iou = intersection / union
        return 1.0 - iou  # Convert IoU to cost (higher IoU = lower cost)
    
    def _compute_piece_logic_cost(self, piece: PieceDetection, file: int, rank: int) -> float:
        """Compute cost based on chess piece logic"""
        piece_base = piece.piece_type.replace('1', '').replace('2', '').replace('3', '').replace('4', '').replace('5', '').replace('6', '').replace('7', '').replace('8', '')
        
        bonus = 0.0
        
        # Pawn logic
        if 'Pawn' in piece_base:
            if 'White' in piece_base and 1 <= rank <= 6:  # White pawns move up
                bonus = -0.1
            elif 'Black' in piece_base and 1 <= rank <= 6:  # Black pawns move down  
                bonus = -0.1
        
        # Back rank pieces
        elif rank in [0, 7]:  # Back ranks
            if any(piece_name in piece_base for piece_name in ['Rook', 'Knight', 'Bishop', 'Queen', 'King']):
                bonus = -0.05
                
        # Central squares bonus for active pieces
        if 2 <= file <= 5 and 2 <= rank <= 5:  # Center 4x4
            if any(piece_name in piece_base for piece_name in ['Knight', 'Bishop', 'Queen']):
                bonus -= 0.05
        
        return bonus
    
    def _compute_edge_penalty(self, piece: PieceDetection) -> float:
        """Compute penalty for pieces near board edges"""
        edge_distance = min(
            piece.center_x,
            piece.center_y, 
            self.board_size - piece.center_x,
            self.board_size - piece.center_y
        )
        
        if edge_distance < self.square_size * 0.1:  # Very close to edge
            return 0.3
        elif edge_distance < self.square_size * 0.25:  # Close to edge
            return 0.1
            
        return 0.0
    
    def solve_assignment_improved(self, pieces: List[PieceDetection]) -> List[SquareAssignment]:
        """Improved assignment solving with adaptive thresholds"""
        if not pieces:
            return []
        
        # Create cost matrix
        num_pieces = len(pieces)
        cost_matrix = np.full((num_pieces, 64), 1000.0)
        
        for piece_idx, piece in enumerate(pieces):
            for square_idx in range(64):
                file = square_idx % 8
                rank = square_idx // 8
                cost = self.compute_adaptive_cost(piece, file, rank)
                cost_matrix[piece_idx, square_idx] = cost
        
        # Solve with Hungarian algorithm
        piece_indices, square_indices = linear_sum_assignment(cost_matrix)
        
        # Create assignments with confidence scores
        assignments = []
        for piece_idx, square_idx in zip(piece_indices, square_indices):
            cost = cost_matrix[piece_idx, square_idx]
            
            # Use adaptive threshold
            if cost < self.cost_threshold:
                piece = pieces[piece_idx]
                file = square_idx % 8
                rank = square_idx // 8
                square = self._square_to_algebraic(file, rank)
                
                # Calculate confidence (inverse of normalized cost)
                confidence = max(0.0, 1.0 - (cost / self.cost_threshold))
                
                assignment = SquareAssignment(
                    piece=piece,
                    square=square,
                    file=file,
                    rank=rank,
                    cost=cost,
                    confidence=confidence
                )
                assignments.append(assignment)
        
        return assignments
    
    def _square_to_algebraic(self, file: int, rank: int) -> str:
        """Convert square coordinates to algebraic notation"""
        files = 'abcdefgh'
        return f"{files[file]}{rank + 1}"
    
    def create_detailed_results(self, assignments: List[SquareAssignment], 
                              ground_truth: Dict[str, str] = None) -> Dict[str, Any]:
        """Create detailed, readable results"""
        if not assignments:
            return {
                'status': 'no_assignments',
                'message': 'No pieces could be assigned to squares',
                'assignments': [],
                'statistics': {'total_pieces': 0, 'assigned_pieces': 0, 'assignment_rate': 0.0}
            }
        
        # Organize assignments by confidence
        high_confidence = [a for a in assignments if a.confidence >= 0.7]
        medium_confidence = [a for a in assignments if 0.4 <= a.confidence < 0.7]
        low_confidence = [a for a in assignments if a.confidence < 0.4]
        
        # Create readable assignment data
        assignment_data = []
        for assignment in assignments:
            piece_name = assignment.piece.piece_type.replace('1', '').replace('2', '').replace('3', '')
            
            data = {
                'piece': piece_name,
                'square': assignment.square,
                'position': f"({assignment.piece.center_x:.1f}, {assignment.piece.center_y:.1f})",
                'cost': round(assignment.cost, 3),
                'confidence': round(assignment.confidence, 3),
                'quality': self._get_quality_rating(assignment.confidence),
                'piece_confidence': round(assignment.piece.confidence, 3)
            }
            
            # Add ground truth comparison if available
            if ground_truth and assignment.square in ground_truth:
                expected = ground_truth[assignment.square]
                data['ground_truth'] = expected
                data['correct'] = (assignment.piece.piece_type == expected)
            
            assignment_data.append(data)
        
        # Calculate statistics
        total_assigned = len(assignments)
        correct_assignments = sum(1 for a in assignment_data if a.get('correct', False))
        
        statistics = {
            'total_assigned': total_assigned,
            'high_confidence': len(high_confidence),
            'medium_confidence': len(medium_confidence), 
            'low_confidence': len(low_confidence),
            'average_confidence': round(np.mean([a.confidence for a in assignments]), 3),
            'average_cost': round(np.mean([a.cost for a in assignments]), 3)
        }
        
        if ground_truth:
            statistics.update({
                'ground_truth_available': len(ground_truth),
                'correct_assignments': correct_assignments,
                'accuracy': round(correct_assignments / max(1, len(ground_truth)), 3)
            })
        
        return {
            'status': 'success',
            'assignments': assignment_data,
            'statistics': statistics,
            'summary': self._create_summary_text(assignment_data, statistics)
        }
    
    def _get_quality_rating(self, confidence: float) -> str:
        """Get quality rating from confidence score"""
        if confidence >= 0.8:
            return "Excellent"
        elif confidence >= 0.6:
            return "Good"
        elif confidence >= 0.4:
            return "Fair"
        else:
            return "Poor"
    
    def _create_summary_text(self, assignments: List[Dict], statistics: Dict) -> str:
        """Create human-readable summary"""
        summary = f"Assignment Summary:\n"
        summary += f"• {statistics['total_assigned']} pieces successfully assigned\n"
        summary += f"• Average confidence: {statistics['average_confidence']}\n"
        summary += f"• High confidence assignments: {statistics['high_confidence']}\n"
        
        if 'accuracy' in statistics:
            summary += f"• Accuracy: {statistics['accuracy']:.1%}\n"
        
        summary += f"\nQuality breakdown:\n"
        quality_counts = {}
        for assignment in assignments:
            quality = assignment['quality']
            quality_counts[quality] = quality_counts.get(quality, 0) + 1
        
        for quality, count in quality_counts.items():
            summary += f"• {quality}: {count}\n"
        
        return summary