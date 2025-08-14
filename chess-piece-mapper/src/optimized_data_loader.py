#!/usr/bin/env python3
"""
Optimized Chess Data Loader
Improved data loading with better board detection and coordinate transformation
"""

import json
import cv2
import numpy as np
import os
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
from .optimized_mapper import OptimizedPieceMapper, PieceDetection

class OptimizedChessDataLoader:
    """Optimized data loader with improved board detection and processing"""
    
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        self.train_path = self.data_path / "train"
        self.image_path = self.train_path / "images"
        self.annotation_file = self.train_path / "coco_annotations.json"
        
        print(f"Loading data from: {self.data_path}")
        
        # Load COCO annotations
        self.coco_data = self._load_coco_data()
        self.images = {img['id']: img for img in self.coco_data['images']}
        self.categories = {cat['id']: cat for cat in self.coco_data['categories']}
        
        # Initialize optimized mapper for board detection
        self.mapper = OptimizedPieceMapper()
        
        # Board configuration
        self.board_size = 256
        self.square_size = 32
        
        print(f"Loaded {len(self.images)} images with {len(self.coco_data.get('annotations', []))} annotations")
    
    def _load_coco_data(self) -> Dict:
        """Load COCO JSON data"""
        with open(self.annotation_file, 'r') as f:
            return json.load(f)
    
    def load_image(self, image_id: int) -> Optional[np.ndarray]:
        """Load image by ID with error handling"""
        try:
            image_info = self.images[image_id]
            image_filename = image_info['file_name']
            
            # Handle different filename formats
            if not image_filename.startswith('images/'):
                image_path = self.image_path / image_filename
            else:
                # Remove 'images/' prefix and use direct path
                clean_filename = image_filename.replace('images/', '')
                image_path = self.image_path / clean_filename
            
            if not image_path.exists():
                print(f"Warning: Image file not found: {image_path}")
                return None
                
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"Warning: Could not load image: {image_path}")
                return None
                
            return image
            
        except Exception as e:
            print(f"Error loading image {image_id}: {e}")
            return None
    
    def get_image_annotations(self, image_id: int) -> List[Dict]:
        """Get all annotations for an image"""
        annotations = []
        for ann in self.coco_data.get('annotations', []):
            if ann['image_id'] == image_id:
                annotations.append(ann)
        return annotations
    
    def process_image_optimized(self, image_id: int, debug: bool = False) -> Optional[Dict[str, Any]]:
        """Optimized image processing with better error handling"""
        try:
            # Load image and annotations
            image = self.load_image(image_id)
            if image is None:
                return None
                
            annotations = self.get_image_annotations(image_id)
            if not annotations:
                print(f"No annotations found for image {image_id}")
                return None
            
            if debug:
                print(f"Processing image {image_id} with {len(annotations)} annotations")
            
            # Detect board corners using optimized method
            corners = self.mapper.detect_board_corners_improved(image)
            
            # Warp board with quality validation
            warped_board, transform_matrix = self.mapper.warp_board_improved(image, corners)
            
            # Transform annotations to warped coordinate system
            transformed_annotations = []
            pieces = []
            
            for ann in annotations:
                category_name = self.categories[ann['category_id']]['name']
                
                # Skip board detections
                if category_name == 'Board':
                    continue
                
                # Transform coordinates
                bbox = ann['bbox']
                transformed = self.mapper.transform_coordinates_improved(bbox, transform_matrix)
                
                # Validate transformed coordinates
                if self._validate_transformed_coordinates(transformed['center']):
                    # Create annotation with transformed coordinates
                    transformed_ann = ann.copy()
                    transformed_ann['bbox'] = transformed['bbox']
                    transformed_ann['center'] = transformed['center']
                    transformed_annotations.append(transformed_ann)
                    
                    # Create piece detection
                    piece = PieceDetection(
                        center_x=transformed['center'][0],
                        center_y=transformed['center'][1],
                        bbox=transformed['bbox'],
                        piece_type=category_name,
                        confidence=ann.get('score', 1.0)
                    )
                    pieces.append(piece)
                elif debug:
                    print(f"Filtered out piece {category_name} with invalid coordinates")
            
            # Load ground truth from board placements
            ground_truth = self._load_ground_truth_optimized(image_id)
            
            if debug:
                print(f"Processed {len(pieces)} valid pieces, {len(ground_truth)} ground truth positions")
            
            return {
                'success': True,
                'image_id': image_id,
                'original_image': image,
                'warped_board': warped_board,
                'corners': corners,
                'transform_matrix': transform_matrix,
                'annotations': annotations,
                'transformed_annotations': transformed_annotations,
                'pieces': pieces,
                'ground_truth': ground_truth,
                'processing_stats': {
                    'original_annotations': len(annotations),
                    'valid_pieces': len(pieces),
                    'ground_truth_positions': len(ground_truth)
                }
            }
            
        except Exception as e:
            if debug:
                print(f"Error processing image {image_id}: {e}")
            return {
                'success': False,
                'image_id': image_id,
                'error': str(e)
            }
    
    def _validate_transformed_coordinates(self, center: Tuple[float, float]) -> bool:
        """Validate that transformed coordinates are within board bounds"""
        x, y = center
        margin = self.square_size * 0.5  # Allow pieces to be half a square outside
        return (-margin <= x <= self.board_size + margin and 
                -margin <= y <= self.board_size + margin)
    
    def _load_ground_truth_optimized(self, image_id: int) -> Dict[str, str]:
        """Optimized ground truth loading with better error handling"""
        board_file = self.train_path / "board_placements.json"
        
        if not board_file.exists():
            return {}
            
        try:
            with open(board_file, 'r') as f:
                board_data = json.load(f)
            
            # Try multiple filename formats
            possible_names = [
                f"images/{image_id:06d}.png",
                f"{image_id:06d}.png", 
                f"image_{image_id:06d}.png"
            ]
            
            for image_name in possible_names:
                if image_name in board_data:
                    entry = board_data[image_name]
                    if 'board' in entry:
                        fen = entry['board']
                        return self._fen_to_square_mapping_optimized(fen)
            
            return {}
            
        except Exception as e:
            print(f"Warning: Could not load ground truth for image {image_id}: {e}")
            return {}
    
    def _fen_to_square_mapping_optimized(self, fen: str) -> Dict[str, str]:
        """Optimized FEN parsing with better error handling"""
        try:
            mapping = {}
            board_fen = fen.split()[0]
            ranks = board_fen.split('/')
            
            files = 'abcdefgh'
            piece_symbols = {
                'P': 'WhitePawn', 'R': 'WhiteRook', 'N': 'WhiteKnight',
                'B': 'WhiteBishop', 'Q': 'WhiteQueen', 'K': 'WhiteKing',
                'p': 'BlackPawn', 'r': 'BlackRook', 'n': 'BlackKnight', 
                'b': 'BlackBishop', 'q': 'BlackQueen', 'k': 'BlackKing'
            }
            
            for rank_idx, rank in enumerate(ranks):
                file_idx = 0
                for char in rank:
                    if char.isdigit():
                        file_idx += int(char)
                    elif char in piece_symbols:
                        square = f"{files[file_idx]}{8 - rank_idx}"
                        piece_type = piece_symbols[char] + "1"  # Add suffix for consistency
                        mapping[square] = piece_type
                        file_idx += 1
                        
            return mapping
            
        except Exception as e:
            print(f"Error parsing FEN '{fen}': {e}")
            return {}
    
    def batch_process_optimized(self, image_ids: List[int], debug: bool = False) -> List[Dict[str, Any]]:
        """Process multiple images with optimized performance"""
        results = []
        successful = 0
        
        print(f"Processing {len(image_ids)} images...")
        
        for i, image_id in enumerate(image_ids):
            if debug or i % 10 == 0:
                print(f"Processing image {image_id} ({i+1}/{len(image_ids)})")
            
            result = self.process_image_optimized(image_id, debug=debug)
            results.append(result)
            
            if result and result.get('success'):
                successful += 1
        
        print(f"Batch processing complete: {successful}/{len(image_ids)} successful")
        return results
    
    def analyze_dataset_quality(self, sample_size: int = 50) -> Dict[str, Any]:
        """Analyze dataset quality and provide recommendations"""
        print(f"Analyzing dataset quality using {sample_size} samples...")
        
        # Sample random images
        image_ids = list(self.images.keys())[:sample_size]
        results = self.batch_process_optimized(image_ids)
        
        # Analyze results
        successful_results = [r for r in results if r and r.get('success')]
        
        analysis = {
            'total_samples': len(results),
            'successful_processing': len(successful_results),
            'processing_success_rate': len(successful_results) / len(results),
            'average_pieces_per_image': 0,
            'average_ground_truth_per_image': 0,
            'board_detection_issues': 0,
            'coordinate_transformation_issues': 0
        }
        
        if successful_results:
            total_pieces = sum(len(r['pieces']) for r in successful_results)
            total_ground_truth = sum(len(r['ground_truth']) for r in successful_results)
            
            analysis.update({
                'average_pieces_per_image': total_pieces / len(successful_results),
                'average_ground_truth_per_image': total_ground_truth / len(successful_results)
            })
        
        # Provide recommendations
        recommendations = []
        if analysis['processing_success_rate'] < 0.9:
            recommendations.append("Consider improving board corner detection methods")
        if analysis['average_pieces_per_image'] < 20:
            recommendations.append("Low piece detection rate - check annotation quality")
        if analysis['average_ground_truth_per_image'] < analysis['average_pieces_per_image']:
            recommendations.append("Ground truth data may be incomplete")
        
        analysis['recommendations'] = recommendations
        
        return analysis