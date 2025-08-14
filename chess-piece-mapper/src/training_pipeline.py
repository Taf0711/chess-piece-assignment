#!/usr/bin/env python3
"""
Chess Piece Mapping Training Pipeline
Combines data loading with piece-to-square assignment training
"""

import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
import matplotlib.pyplot as plt

from .data_loader import ChessDataLoader
from .piece_mapper import PieceToSquareMapper, PieceDetection, SquareAssignment
from .image_visualizer import ChessDetectionVisualizer


class ChessMappingTrainer:
    """
    Training pipeline for chess piece to square mapping
    """
    
    def __init__(self, data_path: str, output_dir: str = "results"):
        """
        Initialize trainer
        Args:
            data_path: Path to chess dataset 
            output_dir: Directory for saving results
        """
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.data_loader = ChessDataLoader(data_path)
        self.mapper = PieceToSquareMapper()
        self.visualizer = ChessDetectionVisualizer(output_dir=str(self.output_dir / "detection_images"))
        
        # Training statistics
        self.training_stats = {
            'processed_images': 0,
            'failed_images': 0,
            'total_pieces': 0,
            'correctly_assigned': 0,
            'accuracy_history': [],
            'cost_history': []
        }
        
        print(f"Initialized trainer with {len(self.data_loader.images)} images")
    
    def convert_annotations_to_pieces(self, transformed_annotations: List[Dict]) -> List[PieceDetection]:
        """Convert COCO annotations to PieceDetection objects"""
        pieces = []
        
        for ann in transformed_annotations:
            if 'bbox' not in ann or 'center' not in ann:
                continue
                
            # Skip board detections
            category_name = self.data_loader.categories[ann['category_id']]['name']
            if category_name == 'Board':
                continue
            
            bbox = ann['bbox']
            center_x, center_y = ann['center']
            
            piece = PieceDetection(
                center_x=center_x,
                center_y=center_y,
                bbox=bbox,
                piece_type=category_name,
                confidence=ann.get('score', 1.0)
            )
            pieces.append(piece)
            
        return pieces
    
    def load_ground_truth_from_board_placements(self, image_id: int) -> Dict[str, str]:
        """
        Load ground truth positions from board_placements.json if available
        """
        board_file = self.data_loader.train_path / "board_placements.json"
        
        if not board_file.exists():
            return {}
            
        try:
            with open(board_file, 'r') as f:
                board_data = json.load(f)
                
            # Find the entry for this image - format: images/000000.png
            image_name = f"images/{image_id:06d}.png"  # Format: images/000000.png
            
            if image_name in board_data:
                entry = board_data[image_name]
                if 'board' in entry:
                    fen = entry['board']
                    return self.fen_to_square_mapping(fen)
                        
        except Exception as e:
            print(f"Warning: Could not load ground truth for image {image_id}: {e}")
            
        return {}
    
    def fen_to_square_mapping(self, fen: str) -> Dict[str, str]:
        """Convert FEN string to square->piece_type mapping"""
        mapping = {}
        
        # Parse FEN board position (first part before space)
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
                    file_idx += int(char)  # Skip empty squares
                else:
                    if char in piece_symbols:
                        square = f"{files[file_idx]}{8 - rank_idx}"
                        mapping[square] = piece_symbols[char] + "1"  # Add "1" suffix like in COCO data
                    file_idx += 1
                    
        return mapping
    
    def process_single_image(self, image_id: int, visualize: bool = False) -> Dict[str, Any]:
        """Process a single image and return results"""
        try:
            # Process image through data loader
            result = self.data_loader.process_image(image_id)
            if result is None:
                return {'success': False, 'error': 'Image processing failed'}
            
            # Convert annotations to pieces
            pieces = self.convert_annotations_to_pieces(result['transformed_annotations'])
            
            if not pieces:
                return {'success': False, 'error': 'No pieces detected'}
            
            # Solve piece assignment
            assignments = self.mapper.solve_assignment(pieces)
            
            # Load ground truth
            ground_truth = self.load_ground_truth_from_board_placements(image_id)
            if not ground_truth:
                # Fallback: use simple position mapping from data loader result
                ground_truth = result.get('piece_positions', {})
            
            # Evaluate assignment
            evaluation = self.mapper.evaluate_assignment(assignments, ground_truth) if ground_truth else None
            
            # Visualize if requested
            if visualize and assignments:
                self.visualize_image_result(result, assignments, ground_truth, evaluation)
            
            return {
                'success': True,
                'image_id': image_id,
                'pieces': pieces,
                'assignments': assignments,
                'ground_truth': ground_truth,
                'evaluation': evaluation,
                'board_image': result['warped_board'],
                'original_image': result['original_image']
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def train_on_dataset(self, max_images: int = 100, visualize_samples: bool = False):
        """
        Train the mapper on a subset of the dataset
        """
        print(f"Training on up to {max_images} images...")
        
        processed_results = []
        training_data = []
        
        # Collect training data
        image_ids = list(self.data_loader.images.keys())[:max_images]
        
        for image_id in tqdm(image_ids, desc="Processing images"):
            result = self.process_single_image(image_id, visualize=False)
            
            if result['success']:
                processed_results.append(result)
                
                if result['ground_truth'] and result['pieces']:
                    training_data.append({
                        'pieces': result['pieces'],
                        'ground_truth': result['ground_truth']
                    })
                    
                self.training_stats['processed_images'] += 1
                self.training_stats['total_pieces'] += len(result['pieces'])
                
                if result['evaluation']:
                    self.training_stats['correctly_assigned'] += result['evaluation']['correct']
                    self.training_stats['accuracy_history'].append(result['evaluation']['accuracy'])
                    
            else:
                self.training_stats['failed_images'] += 1
                if 'error' in result:
                    print(f"Failed image {image_id}: {result['error']}")
        
        print(f"Collected {len(training_data)} training samples")
        
        # Train weights if we have enough data
        if len(training_data) >= 10:
            print("Training assignment weights...")
            best_weights = self.mapper.train_weights(training_data, iterations=50)
            
            # Save trained model
            model_path = self.output_dir / "trained_piece_mapper.pkl"
            self.mapper.save_model(str(model_path))
            
        # Create detection visualizations 
        if processed_results:
            print(f"Generating detection images for {min(10, len(processed_results))} samples...")
            image_results = []
            
            for i, result in enumerate(processed_results[:10]):
                if result['success'] and result.get('original_image') is not None:
                    # Create detection visualization
                    detection_image = self.visualizer.create_detection_image(
                        original_image=result['original_image'],
                        warped_image=result['board_image'], 
                        pieces=result['pieces'],
                        assignments=result['assignments'],
                        ground_truth=result['ground_truth'],
                        image_id=result['image_id']
                    )
                    
                    # Create single overlay detection
                    if result['assignments']:  # Only if we have assignments
                        self.visualizer.create_single_detection_overlay(
                            image=result['board_image'],
                            pieces=result['pieces'],
                            assignments=result['assignments'],
                            image_id=result['image_id']
                        )
                    
                    image_results.append(result)
                    
            # Create summary grid
            if image_results:
                self.visualizer.create_summary_grid(image_results)
                print(f"Generated {len(image_results)} detection visualization images")
        
        # Show sample results (text)
        if visualize_samples and processed_results:
            print(f"Sample Results Summary:")
            for i, result in enumerate(processed_results[:3]):
                print(f"\nSample {i+1}: Image {result['image_id']}")
                self.visualize_image_result_simple(result)
                
        return processed_results
    
    def evaluate_on_test_set(self, test_images: int = 20) -> Dict[str, Any]:
        """
        Evaluate trained model on test images
        """
        print(f"Evaluating on {test_images} test images...")
        
        # Use different images for testing
        image_ids = list(self.data_loader.images.keys())[-test_images:]
        
        test_results = []
        total_accuracy = 0.0
        total_pieces = 0
        correct_pieces = 0
        
        for image_id in tqdm(image_ids, desc="Testing"):
            result = self.process_single_image(image_id)
            
            if result['success'] and result['evaluation']:
                test_results.append(result)
                total_accuracy += result['evaluation']['accuracy']
                total_pieces += result['evaluation']['total']
                correct_pieces += result['evaluation']['correct']
        
        if test_results:
            avg_accuracy = total_accuracy / len(test_results)
            overall_accuracy = correct_pieces / max(1, total_pieces)
            
            print(f"\nTest Results:")
            print(f"  Average per-image accuracy: {avg_accuracy:.3f}")
            print(f"  Overall piece accuracy: {overall_accuracy:.3f}")
            print(f"  Correctly assigned pieces: {correct_pieces}/{total_pieces}")
            
            return {
                'test_results': test_results,
                'avg_accuracy': avg_accuracy,
                'overall_accuracy': overall_accuracy,
                'correct_pieces': correct_pieces,
                'total_pieces': total_pieces
            }
        else:
            print("No successful test results")
            return {}
    
    def visualize_image_result_simple(self, result: Dict[str, Any]):
        """Simple visualization of image processing result"""
        if not result['success']:
            print(f"  Failed: {result.get('error', 'Unknown error')}")
            return
            
        print(f"  Processed {len(result['pieces'])} pieces")
        print(f"  Generated {len(result['assignments'])} assignments")
        
        if result['evaluation']:
            eval_data = result['evaluation']
            print(f"  Accuracy: {eval_data['accuracy']:.3f} ({eval_data['correct']}/{eval_data['total']})")
            
            if eval_data.get('missed_pieces'):
                print(f"  Missed pieces: {eval_data['missed_pieces']}")
            if eval_data.get('false_positives'):
                print(f"  False positives: {eval_data['false_positives']}")
        
        # Show some assignments
        print(f"  Sample assignments:")
        for i, assignment in enumerate(result['assignments'][:5]):
            piece_name = assignment.piece.piece_type.replace('1', '').replace('2', '').replace('3', '')[:6]
            print(f"    {piece_name} -> {assignment.square} (cost: {assignment.cost:.3f})")
    
    def visualize_image_result(self, data_result: Dict, assignments: List[SquareAssignment], 
                             ground_truth: Dict, evaluation: Dict = None):
        """Detailed visualization with matplotlib"""
        try:
            fig, axes = plt.subplots(1, 2, figsize=(15, 7))
            fig.suptitle(f"Chess Piece Assignment - Image {data_result['image_id']}")
            
            # Left: Warped board with detections
            ax = axes[0]
            ax.imshow(data_result['warped_board'])
            ax.set_title("Detected Pieces")
            
            # Draw bounding boxes and centers
            for ann in data_result['transformed_annotations']:
                if ann.get('category_id') == 12:  # Skip board
                    continue
                    
                bbox = ann['bbox']
                x, y, w, h = bbox
                
                # Draw bounding box
                rect = plt.Rectangle((x, y), w, h, fill=False, edgecolor='red', linewidth=2)
                ax.add_patch(rect)
                
                # Draw center
                center_x, center_y = ann.get('center', (x + w/2, y + h/2))
                ax.plot(center_x, center_y, 'ro', markersize=6)
                
                # Label
                piece_name = self.data_loader.categories[ann['category_id']]['name']
                piece_short = piece_name.replace('White', 'W').replace('Black', 'B')[:6]
                ax.text(center_x, center_y - 15, piece_short, ha='center', va='center', 
                       fontsize=8, color='red', weight='bold')
            
            # Right: Assignment results
            ax = axes[1] 
            ax.imshow(data_result['warped_board'])
            ax.set_title("Square Assignments")
            
            # Draw grid
            for i in range(0, 256, 32):
                ax.axhline(i, color='white', alpha=0.5, linewidth=1)
                ax.axvline(i, color='white', alpha=0.5, linewidth=1)
                
            # Draw assignments
            for assignment in assignments:
                square_center_x = assignment.file * 32 + 16
                square_center_y = assignment.rank * 32 + 16
                
                # Piece center
                piece_center_x = assignment.piece.center_x
                piece_center_y = assignment.piece.center_y
                
                # Draw connection
                ax.plot([piece_center_x, square_center_x], 
                       [piece_center_y, square_center_y], 'g--', alpha=0.7, linewidth=2)
                
                # Draw assigned square
                ax.plot(square_center_x, square_center_y, 'go', markersize=8)
                
                # Label assignment
                piece_short = assignment.piece.piece_type.replace('White', 'W').replace('Black', 'B')[:4]
                ax.text(square_center_x, square_center_y + 15, 
                       f"{assignment.square}\n{piece_short}", 
                       ha='center', va='center', fontsize=8, color='white', weight='bold',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor='green', alpha=0.7))
            
            # Add evaluation info
            if evaluation:
                info_text = f"Accuracy: {evaluation['accuracy']:.3f}\n"
                info_text += f"Correct: {evaluation['correct']}/{evaluation['total']}\n"
                if evaluation.get('missed_pieces'):
                    info_text += f"Missed: {len(evaluation['missed_pieces'])}\n"
                if evaluation.get('false_positives'):
                    info_text += f"False+: {len(evaluation['false_positives'])}"
                
                fig.text(0.02, 0.98, info_text, ha='left', va='top', fontsize=10,
                        bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.8))
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"Visualization error: {e}")
    
    def save_results(self, results: List[Dict[str, Any]], filename: str = "training_results.json"):
        """Save training results to file"""
        output_data = {
            'training_stats': self.training_stats,
            'mapper_weights': self.mapper.weights,
            'results_summary': {
                'total_images_processed': len([r for r in results if r['success']]),
                'total_pieces': sum(len(r.get('pieces', [])) for r in results if r['success']),
                'average_accuracy': np.mean([r['evaluation']['accuracy'] for r in results 
                                           if r['success'] and r.get('evaluation')])
            }
        }
        
        # Save results
        output_file = self.output_dir / filename
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2, default=str)
        
        print(f"Results saved to {output_file}")


def main():
    """Run the training pipeline"""
    print("=== Chess Piece Mapping Training Pipeline ===")
    
    # Initialize trainer
    data_path = "/home/pre/projects/chess-datagen/gen-data/render_src/coco_data_2025_08_08__21_53_08"
    trainer = ChessMappingTrainer(data_path)
    
    # Train on subset of data
    results = trainer.train_on_dataset(max_images=50, visualize_samples=True)
    
    # Evaluate on test set
    test_results = trainer.evaluate_on_test_set(test_images=10)
    
    # Save results
    trainer.save_results(results)
    
    print("\n=== Training Complete ===")
    print(f"Training Statistics:")
    for key, value in trainer.training_stats.items():
        if isinstance(value, list):
            continue  # Skip lists in summary
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()