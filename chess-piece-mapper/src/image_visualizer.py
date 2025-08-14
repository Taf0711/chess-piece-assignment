#!/usr/bin/env python3
"""
Image Visualization for Chess Piece Detection and Mapping
Creates CV-style detection images with bounding boxes, assignments, and board grid
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional
from .piece_mapper import PieceDetection, SquareAssignment

class ChessDetectionVisualizer:
    """
    Creates computer vision style detection images showing:
    - Original chess board image
    - Detected piece bounding boxes with labels
    - Square assignments with connecting lines
    - Grid overlay showing 8x8 board squares
    """
    
    def __init__(self, output_dir: str = "results/detection_images"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Colors for different piece types
        self.piece_colors = {
            'WhiteRook': (0, 0, 255),      # Red
            'WhiteKnight': (0, 128, 255),  # Orange  
            'WhiteBishop': (0, 255, 255),  # Yellow
            'WhiteQueen': (128, 0, 255),   # Purple
            'WhiteKing': (255, 0, 128),    # Pink
            'WhitePawn': (0, 255, 0),      # Green
            'BlackRook': (128, 0, 0),      # Dark Red
            'BlackKnight': (128, 64, 0),   # Dark Orange
            'BlackBishop': (128, 128, 0),  # Dark Yellow  
            'BlackQueen': (64, 0, 128),    # Dark Purple
            'BlackKing': (128, 0, 64),     # Dark Pink
            'BlackPawn': (0, 128, 0),      # Dark Green
        }
        
    def get_piece_color(self, piece_type: str) -> Tuple[int, int, int]:
        """Get BGR color for piece type"""
        base_type = piece_type.replace('1', '').replace('2', '').replace('3', '').replace('4', '').replace('5', '').replace('6', '').replace('7', '').replace('8', '')
        return self.piece_colors.get(base_type, (255, 255, 255))  # Default white
    
    def create_detection_image(self, 
                             original_image: np.ndarray,
                             warped_image: np.ndarray, 
                             pieces: List[PieceDetection],
                             assignments: List[SquareAssignment],
                             ground_truth: Dict[str, str] = None,
                             image_id: int = 0,
                             save_individual: bool = True) -> np.ndarray:
        """
        Create comprehensive detection visualization
        Args:
            original_image: Original chess board image (640x480)
            warped_image: Warped board image (256x256)  
            pieces: List of detected pieces
            assignments: List of piece-to-square assignments
            ground_truth: Ground truth positions (optional)
            image_id: Image identifier for filename
            save_individual: Whether to save individual images
        Returns:
            Combined visualization image
        """
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Chess Piece Detection & Mapping - Image {image_id}', fontsize=16, fontweight='bold')
        
        # 1. Original image with detections
        ax1 = axes[0, 0]
        ax1.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        ax1.set_title('1. Original Image', fontweight='bold')
        ax1.axis('off')
        
        # 2. Warped board with piece bounding boxes  
        ax2 = axes[0, 1]
        ax2.imshow(cv2.cvtColor(warped_image, cv2.COLOR_BGR2RGB))
        ax2.set_title('2. Detected Pieces (Bounding Boxes)', fontweight='bold')
        self._draw_piece_detections(ax2, pieces)
        ax2.axis('off')
        
        # 3. Board grid with assignments
        ax3 = axes[1, 0] 
        ax3.imshow(cv2.cvtColor(warped_image, cv2.COLOR_BGR2RGB))
        ax3.set_title('3. Square Assignments', fontweight='bold')
        self._draw_board_grid(ax3)
        self._draw_assignments(ax3, assignments)
        ax3.axis('off')
        
        # 4. Final result with evaluation
        ax4 = axes[1, 1]
        ax4.imshow(cv2.cvtColor(warped_image, cv2.COLOR_BGR2RGB))
        ax4.set_title('4. Final Mapping Result', fontweight='bold')
        self._draw_board_grid(ax4)
        self._draw_final_result(ax4, assignments, ground_truth)
        ax4.axis('off')
        
        # Add statistics text
        self._add_statistics_text(fig, pieces, assignments, ground_truth)
        
        plt.tight_layout()
        
        if save_individual:
            # Save the complete visualization
            output_file = self.output_dir / f"detection_image_{image_id:06d}.png"
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            print(f"Saved detection image: {output_file}")
            
        # Convert to numpy array for return
        fig.canvas.draw()
        try:
            buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        except AttributeError:
            # For newer matplotlib versions
            buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
            buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:,:,:3]  # Remove alpha channel
        else:
            buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        plt.close(fig)
        return buf
        
    def _draw_piece_detections(self, ax, pieces: List[PieceDetection]):
        """Draw bounding boxes around detected pieces"""
        for piece in pieces:
            x, y, w, h = piece.bbox
            color = np.array(self.get_piece_color(piece.piece_type)) / 255.0  # Convert to matplotlib RGB
            
            # Draw bounding box
            rect = patches.Rectangle((x, y), w, h, linewidth=2, 
                                   edgecolor=color, facecolor='none')
            ax.add_patch(rect)
            
            # Draw center point
            ax.plot(piece.center_x, piece.center_y, 'o', color=color, markersize=4)
            
            # Add label
            piece_name = piece.piece_type.replace('1', '').replace('2', '').replace('3', '')
            piece_short = piece_name.replace('White', 'W').replace('Black', 'B')[:6]
            ax.text(x, y-5, f"{piece_short}\n{piece.confidence:.2f}", 
                   fontsize=8, color=color, weight='bold',
                   bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
    
    def _draw_board_grid(self, ax):
        """Draw 8x8 chess board grid"""
        # Draw grid lines
        for i in range(9):
            ax.axhline(i * 32, color='white', alpha=0.6, linewidth=1)
            ax.axvline(i * 32, color='white', alpha=0.6, linewidth=1)
        
        # Add square labels (a1-h8)
        files = 'abcdefgh'
        for file_idx in range(8):
            for rank_idx in range(8):
                x = file_idx * 32 + 16
                y = rank_idx * 32 + 16
                square = f"{files[file_idx]}{rank_idx + 1}"
                ax.text(x, y, square, ha='center', va='center', 
                       fontsize=6, color='white', alpha=0.7, weight='bold')
    
    def _draw_assignments(self, ax, assignments: List[SquareAssignment]):
        """Draw assignment connections between pieces and squares"""
        for assignment in assignments:
            piece = assignment.piece
            
            # Square center
            square_x = assignment.file * 32 + 16
            square_y = assignment.rank * 32 + 16
            
            # Get piece color
            color = np.array(self.get_piece_color(piece.piece_type)) / 255.0
            
            # Draw connection line
            ax.plot([piece.center_x, square_x], [piece.center_y, square_y], 
                   color=color, linewidth=2, alpha=0.8, linestyle='--')
            
            # Draw target square
            ax.plot(square_x, square_y, 'o', color=color, markersize=8, markeredgecolor='white', markeredgewidth=1)
            
            # Label assignment
            piece_short = piece.piece_type.replace('1', '').replace('2', '').replace('3', '')
            piece_short = piece_short.replace('White', 'W').replace('Black', 'B')[:4]
            ax.text(square_x, square_y + 12, 
                   f"{assignment.square}\n{piece_short}\n{assignment.cost:.2f}",
                   ha='center', va='center', fontsize=6, color='white', weight='bold',
                   bbox=dict(boxstyle="round,pad=0.2", facecolor=color, alpha=0.7))
    
    def _draw_final_result(self, ax, assignments: List[SquareAssignment], ground_truth: Dict[str, str]):
        """Draw final mapping result with accuracy indicators"""
        correct_assignments = []
        incorrect_assignments = []
        
        # Classify assignments
        for assignment in assignments:
            if ground_truth and assignment.square in ground_truth:
                expected_piece = ground_truth[assignment.square]
                if assignment.piece.piece_type == expected_piece:
                    correct_assignments.append(assignment)
                else:
                    incorrect_assignments.append(assignment)
            else:
                incorrect_assignments.append(assignment)  # No ground truth = uncertain
        
        # Draw correct assignments (green)
        for assignment in correct_assignments:
            square_x = assignment.file * 32 + 16
            square_y = assignment.rank * 32 + 16
            ax.plot(square_x, square_y, 'o', color='green', markersize=10, alpha=0.8)
            ax.text(square_x, square_y, '✓', ha='center', va='center', 
                   fontsize=8, color='white', weight='bold')
        
        # Draw incorrect assignments (red)
        for assignment in incorrect_assignments:
            square_x = assignment.file * 32 + 16
            square_y = assignment.rank * 32 + 16
            ax.plot(square_x, square_y, 'x', color='red', markersize=8, markeredgewidth=2)
    
    def _add_statistics_text(self, fig, pieces: List[PieceDetection], 
                           assignments: List[SquareAssignment], 
                           ground_truth: Dict[str, str]):
        """Add statistics text to the figure"""
        stats_text = f"Statistics:\n"
        stats_text += f"• Detected Pieces: {len(pieces)}\n"
        stats_text += f"• Successful Assignments: {len(assignments)}\n"
        stats_text += f"• Assignment Rate: {len(assignments)}/{len(pieces)} = {len(assignments)/max(1,len(pieces))*100:.1f}%\n"
        
        if ground_truth:
            correct = sum(1 for a in assignments 
                         if a.square in ground_truth and a.piece.piece_type == ground_truth[a.square])
            stats_text += f"• Accuracy: {correct}/{len(assignments)} = {correct/max(1,len(assignments))*100:.1f}%"
        
        # Add text box
        fig.text(0.02, 0.02, stats_text, fontsize=10, 
                bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.8),
                verticalalignment='bottom')
    
    def create_summary_grid(self, image_results: List[Dict], max_images: int = 12) -> np.ndarray:
        """Create a grid summary of multiple detection results"""
        n_images = min(len(image_results), max_images)
        cols = 4
        rows = (n_images + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(20, 5*rows))
        fig.suptitle('Chess Piece Detection Summary', fontsize=16, fontweight='bold')
        
        if rows == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(n_images):
            row = i // cols
            col = i % cols
            ax = axes[row, col]
            
            result = image_results[i]
            if result['success']:
                # Show warped image with assignments  
                board_image_key = 'warped_board' if 'warped_board' in result else 'board_image'
                ax.imshow(cv2.cvtColor(result[board_image_key], cv2.COLOR_BGR2RGB))
                self._draw_board_grid(ax)
                self._draw_assignments(ax, result['assignments'])
                
                # Add title with stats
                n_pieces = len(result['pieces'])
                n_assignments = len(result['assignments'])
                accuracy = result['evaluation']['accuracy'] if result.get('evaluation') else 0
                ax.set_title(f"Image {result['image_id']}\n{n_assignments}/{n_pieces} assigned, {accuracy:.1%} acc", 
                           fontsize=10)
            else:
                ax.text(0.5, 0.5, f"Image {result.get('image_id', i)}\nFailed", 
                       ha='center', va='center', transform=ax.transAxes, fontsize=12)
            
            ax.axis('off')
        
        # Hide unused subplots
        for i in range(n_images, rows * cols):
            row = i // cols
            col = i % cols
            axes[row, col].axis('off')
        
        plt.tight_layout()
        
        # Save summary
        summary_file = self.output_dir / "detection_summary.png"
        plt.savefig(summary_file, dpi=150, bbox_inches='tight')
        print(f"Saved detection summary: {summary_file}")
        
        # Convert to array
        fig.canvas.draw()
        try:
            buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        except AttributeError:
            # For newer matplotlib versions
            buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
            buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:,:,:3]  # Remove alpha channel
        
        plt.close(fig)
        return buf
    
    def create_single_detection_overlay(self, image: np.ndarray, pieces: List[PieceDetection], 
                                      assignments: List[SquareAssignment], 
                                      image_id: int) -> np.ndarray:
        """Create a single image with detection overlay (similar to YOLO/CV outputs)"""
        # Create a copy to draw on
        output_image = image.copy()
        
        # Draw bounding boxes and labels for each piece
        for piece in pieces:
            x, y, w, h = [int(val) for val in piece.bbox]
            color = self.get_piece_color(piece.piece_type)
            
            # Draw bounding box
            cv2.rectangle(output_image, (x, y), (x + w, y + h), color, 2)
            
            # Draw center point
            center_x, center_y = int(piece.center_x), int(piece.center_y)
            cv2.circle(output_image, (center_x, center_y), 3, color, -1)
            
            # Add label
            piece_name = piece.piece_type.replace('1', '').replace('2', '').replace('3', '')
            piece_short = piece_name.replace('White', 'W').replace('Black', 'B')
            label = f"{piece_short} {piece.confidence:.2f}"
            
            # Label background
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(output_image, (x, y - label_h - 5), (x + label_w, y), color, -1)
            cv2.putText(output_image, label, (x, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw grid overlay
        for i in range(0, 257, 32):
            cv2.line(output_image, (i, 0), (i, 256), (255, 255, 255), 1)
            cv2.line(output_image, (0, i), (256, i), (255, 255, 255), 1)
        
        # Draw assignments
        for assignment in assignments:
            piece = assignment.piece
            color = self.get_piece_color(piece.piece_type)
            
            # Square center
            square_x = assignment.file * 32 + 16
            square_y = assignment.rank * 32 + 16
            
            # Draw connection
            cv2.line(output_image, 
                    (int(piece.center_x), int(piece.center_y)),
                    (square_x, square_y), color, 2)
            
            # Draw target
            cv2.circle(output_image, (square_x, square_y), 5, color, -1)
            cv2.circle(output_image, (square_x, square_y), 5, (255, 255, 255), 1)
            
            # Square label
            cv2.putText(output_image, assignment.square, (square_x - 10, square_y + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Save single detection image
        single_file = self.output_dir / f"single_detection_{image_id:06d}.png"
        cv2.imwrite(str(single_file), output_image)
        print(f"Saved single detection: {single_file}")
        
        return output_image