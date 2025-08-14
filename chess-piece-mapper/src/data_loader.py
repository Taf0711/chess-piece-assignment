#!/usr/bin/env python3
"""
Chess Piece to Board Mapping - Data Loader
Loads COCO annotations and implements data transformation pipeline
"""

import json
import cv2
import numpy as np
import os
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt
from pathlib import Path

class ChessDataLoader:
    """Load and process chess data from COCO format with board warping"""
    
    def __init__(self, data_path: str):
        """
        Initialize data loader
        Args:
            data_path: Path to coco_data_YYYY_MM_DD__HH_MM_SS folder
        """
        self.data_path = Path(data_path)
        self.train_path = self.data_path / "train"
        self.image_path = self.train_path / "images"
        self.annotation_file = self.train_path / "coco_annotations.json"
        
        print(f"Loading data from: {self.data_path}")
        
        # Load COCO annotations
        self.coco_data = self._load_coco_data()
        self.images = {img['id']: img for img in self.coco_data['images']}
        self.categories = {cat['id']: cat for cat in self.coco_data['categories']}
        
        # Chess board configuration
        self.board_size = 256  # Target warped board size
        self.square_size = 32  # Each square is 32x32 pixels
        
        print(f"Loaded {len(self.images)} images with {len(self.coco_data.get('annotations', []))} annotations")
        
    def _load_coco_data(self) -> Dict:
        """Load COCO JSON data"""
        with open(self.annotation_file, 'r') as f:
            return json.load(f)
    
    def analyze_data_structure(self):
        """Analyze the structure of the loaded data"""
        print("=== DATA ANALYSIS ===")
        print(f"Images: {len(self.images)}")
        print(f"Categories: {len(self.categories)}")
        print(f"Annotations: {len(self.coco_data.get('annotations', []))}")
        
        print("\n=== CATEGORIES ===")
        for cat_id, cat in self.categories.items():
            print(f"ID {cat_id}: {cat['name']}")
            
        print("\n=== SAMPLE IMAGE ===")
        if self.images:
            sample_img = list(self.images.values())[0]
            print(f"Sample image: {sample_img}")
            
        print("\n=== SAMPLE ANNOTATIONS ===")
        if 'annotations' in self.coco_data:
            for i, ann in enumerate(self.coco_data['annotations'][:5]):
                print(f"Annotation {i}: {ann}")
        else:
            print("No annotations found in COCO data - checking for bbox format...")
            
    def get_image_annotations(self, image_id: int) -> List[Dict]:
        """Get all annotations for a specific image"""
        if 'annotations' not in self.coco_data:
            return []
        
        return [ann for ann in self.coco_data['annotations'] if ann['image_id'] == image_id]
    
    def load_image(self, image_id: int) -> np.ndarray:
        """Load image by ID"""
        if image_id not in self.images:
            raise ValueError(f"Image ID {image_id} not found")
            
        image_info = self.images[image_id]
        # Handle case where file_name already includes 'images/' prefix
        file_name = image_info['file_name']
        if file_name.startswith('images/'):
            file_name = file_name[7:]  # Remove 'images/' prefix
        image_path = self.image_path / file_name
        
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
            
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    def detect_chessboard_corners(self, image: np.ndarray) -> np.ndarray:
        """
        Detect chessboard corners for perspective transformation
        Returns 4 corner points of the board in clockwise order: top-left, top-right, bottom-right, bottom-left
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Try to detect chessboard pattern (7x7 internal corners for 8x8 board)
        ret, corners = cv2.findChessboardCorners(gray, (7, 7), None)
        
        if ret:
            # Refine corner detection
            corners = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1),
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            )
            
            # Get board corners from internal corners
            # Top-left is first corner, bottom-right is last corner
            board_corners = np.array([
                corners[0][0],      # top-left
                corners[6][0],      # top-right  
                corners[48][0],     # bottom-right (7*7 - 1)
                corners[42][0]      # bottom-left (7*6)
            ], dtype=np.float32)
            
        else:
            # Fallback: detect board using contour detection
            board_corners = self._detect_board_contour(gray)
            
        return board_corners
    
    def _detect_board_contour(self, gray: np.ndarray) -> np.ndarray:
        """Fallback method to detect board using contours"""
        # Apply edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Find the largest rectangular contour (should be the board)
        for contour in sorted(contours, key=cv2.contourArea, reverse=True):
            # Approximate contour to polygon
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # If we found a quadrilateral, assume it's the board
            if len(approx) == 4:
                return approx.reshape(4, 2).astype(np.float32)
        
        # If no contour found, use image corners as fallback
        h, w = gray.shape
        margin = min(w, h) // 10
        return np.array([
            [margin, margin],           # top-left
            [w - margin, margin],       # top-right
            [w - margin, h - margin],   # bottom-right
            [margin, h - margin]        # bottom-left
        ], dtype=np.float32)
    
    def warp_board_to_top_down(self, image: np.ndarray, corners: np.ndarray) -> np.ndarray:
        """
        Warp the board to a standardized 256x256 top-down view
        """
        # Define target corners for 256x256 square
        target_corners = np.array([
            [0, 0],                           # top-left
            [self.board_size - 1, 0],         # top-right
            [self.board_size - 1, self.board_size - 1],  # bottom-right
            [0, self.board_size - 1]          # bottom-left
        ], dtype=np.float32)
        
        # Compute perspective transformation matrix
        transform_matrix = cv2.getPerspectiveTransform(corners, target_corners)
        
        # Apply transformation
        warped = cv2.warpPerspective(image, transform_matrix, (self.board_size, self.board_size))
        
        return warped, transform_matrix
    
    def transform_bounding_boxes(self, annotations: List[Dict], transform_matrix: np.ndarray) -> List[Dict]:
        """
        Transform bounding boxes from original image to warped coordinate system
        """
        transformed_annotations = []
        
        for ann in annotations:
            if 'bbox' not in ann:
                continue
                
            bbox = ann['bbox']  # [x, y, width, height] in COCO format
            
            # Convert to corner points
            x, y, w, h = bbox
            corners = np.array([
                [x, y],           # top-left
                [x + w, y],       # top-right
                [x + w, y + h],   # bottom-right
                [x, y + h]        # bottom-left
            ], dtype=np.float32)
            
            # Transform corners
            corners_homogeneous = np.ones((4, 3), dtype=np.float32)
            corners_homogeneous[:, :2] = corners
            
            transformed_corners = transform_matrix @ corners_homogeneous.T
            
            # Convert back to pixel coordinates
            transformed_corners = transformed_corners[:2] / transformed_corners[2]
            transformed_corners = transformed_corners.T
            
            # Get new bounding box
            x_min, y_min = np.min(transformed_corners, axis=0)
            x_max, y_max = np.max(transformed_corners, axis=0)
            
            # Create transformed annotation
            transformed_ann = ann.copy()
            transformed_ann['bbox'] = [
                float(x_min), 
                float(y_min), 
                float(x_max - x_min), 
                float(y_max - y_min)
            ]
            transformed_ann['center'] = [
                float((x_min + x_max) / 2), 
                float((y_min + y_max) / 2)
            ]
            
            transformed_annotations.append(transformed_ann)
            
        return transformed_annotations
    
    def get_square_from_center(self, center_x: float, center_y: float) -> Tuple[int, int]:
        """
        Convert pixel coordinates to chess square coordinates
        Returns (file, rank) where file=0-7 (a-h), rank=0-7 (1-8)
        """
        file = int(center_x // self.square_size)
        rank = int(center_y // self.square_size)
        
        # Clamp to valid range
        file = max(0, min(7, file))
        rank = max(0, min(7, rank))
        
        return file, rank
    
    def square_to_algebraic(self, file: int, rank: int) -> str:
        """Convert square coordinates to algebraic notation"""
        files = 'abcdefgh'
        ranks = '12345678'
        return f"{files[file]}{ranks[rank]}"
    
    def process_image(self, image_id: int) -> Dict[str, Any]:
        """
        Process a single image through the complete pipeline:
        1. Load image
        2. Detect board corners  
        3. Warp to top-down view
        4. Transform bounding boxes
        5. Map pieces to squares
        """
        # Load image and annotations
        image = self.load_image(image_id)
        annotations = self.get_image_annotations(image_id)
        
        print(f"Processing image {image_id} with {len(annotations)} annotations")
        
        # Detect board corners
        try:
            corners = self.detect_chessboard_corners(image)
        except Exception as e:
            print(f"Corner detection failed for image {image_id}: {e}")
            return None
            
        # Warp board to standard view
        warped_board, transform_matrix = self.warp_board_to_top_down(image, corners)
        
        # Transform annotations if they exist
        if annotations:
            transformed_annotations = self.transform_bounding_boxes(annotations, transform_matrix)
        else:
            # If no bounding box annotations, we'll need to detect pieces in the warped image
            transformed_annotations = []
        
        # Map pieces to squares
        piece_positions = {}
        for ann in transformed_annotations:
            if 'center' in ann:
                center_x, center_y = ann['center']
                file, rank = self.get_square_from_center(center_x, center_y)
                square = self.square_to_algebraic(file, rank)
                
                piece_type = self.categories[ann['category_id']]['name']
                piece_positions[square] = piece_type
        
        return {
            'image_id': image_id,
            'original_image': image,
            'warped_board': warped_board,
            'corners': corners,
            'transform_matrix': transform_matrix,
            'annotations': annotations,
            'transformed_annotations': transformed_annotations,
            'piece_positions': piece_positions
        }
    
    def visualize_processing(self, result: Dict[str, Any]):
        """Visualize the processing pipeline"""
        if result is None:
            print("No result to visualize")
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f"Chess Board Processing - Image {result['image_id']}")
        
        # Original image with detected corners
        ax = axes[0, 0]
        ax.imshow(result['original_image'])
        ax.set_title("Original with Corners")
        
        if 'corners' in result:
            corners = result['corners']
            # Plot corners
            ax.plot(corners[:, 0], corners[:, 1], 'ro', markersize=10)
            # Plot board outline
            corner_order = [0, 1, 2, 3, 0]  # close the rectangle
            ax.plot(corners[corner_order, 0], corners[corner_order, 1], 'r-', linewidth=2)
        
        # Warped board
        ax = axes[0, 1]
        ax.imshow(result['warped_board'])
        ax.set_title("Warped Top-Down (256x256)")
        ax.grid(True, alpha=0.3)
        
        # Add grid lines for squares
        for i in range(0, 256, 32):
            ax.axhline(i, color='red', alpha=0.5, linewidth=0.5)
            ax.axvline(i, color='red', alpha=0.5, linewidth=0.5)
        
        # Bounding boxes on warped board
        ax = axes[1, 0]
        ax.imshow(result['warped_board'])
        ax.set_title("Warped with Transformed Bboxes")
        
        for ann in result['transformed_annotations']:
            if 'bbox' in ann:
                x, y, w, h = ann['bbox']
                rect = plt.Rectangle((x, y), w, h, fill=False, edgecolor='red', linewidth=2)
                ax.add_patch(rect)
                
                # Add piece label
                piece_name = self.categories[ann['category_id']]['name']
                ax.text(x + w/2, y + h/2, piece_name.split('e')[-1][:2], 
                       ha='center', va='center', color='red', fontsize=8, weight='bold')
        
        # Final square assignments
        ax = axes[1, 1]
        ax.imshow(result['warped_board'])
        ax.set_title("Square Assignments")
        
        # Draw grid
        for i in range(0, 256, 32):
            ax.axhline(i, color='white', alpha=0.7, linewidth=1)
            ax.axvline(i, color='white', alpha=0.7, linewidth=1)
        
        # Show piece positions
        for square, piece_type in result['piece_positions'].items():
            file = ord(square[0]) - ord('a')
            rank = int(square[1]) - 1
            
            center_x = file * 32 + 16
            center_y = rank * 32 + 16
            
            ax.text(center_x, center_y, f"{square}\n{piece_type.split('e')[-1][:2]}", 
                   ha='center', va='center', color='yellow', fontsize=8, weight='bold',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.7))
        
        plt.tight_layout()
        plt.show()


def main():
    """Test the data loader"""
    # Use the dataset path
    data_path = "/home/pre/projects/chess-datagen/gen-data/render_src/coco_data_2025_08_08__21_53_08"
    
    # Initialize loader
    loader = ChessDataLoader(data_path)
    
    # Analyze data structure
    loader.analyze_data_structure()
    
    # Process a sample image
    if loader.images:
        sample_id = list(loader.images.keys())[0]
        print(f"\nProcessing sample image {sample_id}...")
        
        result = loader.process_image(sample_id)
        
        if result:
            print(f"Successfully processed image {sample_id}")
            print(f"Detected {len(result['piece_positions'])} pieces:")
            for square, piece in result['piece_positions'].items():
                print(f"  {square}: {piece}")
                
            # Visualize results (commented out for now)
            # loader.visualize_processing(result)
        else:
            print(f"Failed to process image {sample_id}")


if __name__ == "__main__":
    main()