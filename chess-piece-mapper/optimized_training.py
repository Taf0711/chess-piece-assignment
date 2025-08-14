#!/usr/bin/env python3
"""
Optimized Chess Piece Mapping Training
Improved training pipeline with better data processing and results
"""

from src.optimized_data_loader import OptimizedChessDataLoader
from src.optimized_mapper import OptimizedPieceMapper
from src.image_visualizer import ChessDetectionVisualizer
import json
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
from tqdm import tqdm

class OptimizedTrainingPipeline:
    """Optimized training pipeline with improved performance and results"""
    
    def __init__(self, data_path: str, output_dir: str = "results"):
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize optimized components
        self.data_loader = OptimizedChessDataLoader(data_path)
        self.mapper = OptimizedPieceMapper()
        self.visualizer = ChessDetectionVisualizer(output_dir=str(self.output_dir / "optimized_detection_images"))
        
        print(f"Initialized optimized training pipeline with {len(self.data_loader.images)} images")
    
    def run_dataset_analysis(self) -> Dict[str, Any]:
        """Analyze dataset quality before training"""
        print("\n=== DATASET QUALITY ANALYSIS ===")
        analysis = self.data_loader.analyze_dataset_quality(sample_size=30)
        
        print(f"Analysis Results:")
        print(f"  • Processing success rate: {analysis['processing_success_rate']:.1%}")
        print(f"  • Average pieces per image: {analysis['average_pieces_per_image']:.1f}")
        print(f"  • Average ground truth per image: {analysis['average_ground_truth_per_image']:.1f}")
        
        if analysis['recommendations']:
            print(f"  Recommendations:")
            for rec in analysis['recommendations']:
                print(f"    - {rec}")
        
        return analysis
    
    def process_single_image_optimized(self, image_id: int, debug: bool = False) -> Dict[str, Any]:
        """Process single image with optimized pipeline"""
        # Load and process image
        result = self.data_loader.process_image_optimized(image_id, debug=debug)
        
        if not result or not result.get('success'):
            return result
        
        # Extract pieces
        pieces = result['pieces']
        ground_truth = result['ground_truth']
        
        if not pieces:
            return {
                'success': False,
                'error': 'No valid pieces found after processing'
            }
        
        # Solve assignments using optimized mapper
        assignments = self.mapper.solve_assignment_improved(pieces)
        
        # Create detailed results
        detailed_results = self.mapper.create_detailed_results(assignments, ground_truth)
        
        # Combine all results
        final_result = result.copy()
        final_result.update({
            'assignments': assignments,
            'detailed_results': detailed_results,
            'assignment_success': len(assignments) > 0
        })
        
        if debug:
            print(f"Image {image_id}: {len(assignments)} assignments generated")
            if detailed_results['status'] == 'success':
                print(f"  {detailed_results['summary']}")
        
        return final_result
    
    def train_optimized_model(self, max_images: int = 100, debug: bool = False) -> Dict[str, Any]:
        """Train the optimized model"""
        print(f"\n=== OPTIMIZED TRAINING PIPELINE ===")
        print(f"Training on up to {max_images} images...")
        
        # Select training images
        image_ids = list(self.data_loader.images.keys())[:max_images]
        
        # Process all images
        training_results = []
        successful_assignments = 0
        total_pieces = 0
        total_assignments = 0
        
        print("\nProcessing training images...")
        for image_id in tqdm(image_ids, desc="Training"):
            result = self.process_single_image_optimized(image_id, debug=debug)
            
            if result and result.get('success'):
                training_results.append(result)
                total_pieces += len(result.get('pieces', []))
                total_assignments += len(result.get('assignments', []))
                
                if result.get('assignment_success'):
                    successful_assignments += 1
        
        # Calculate training statistics  
        training_stats = {
            'processed_images': len(training_results),
            'successful_assignments': successful_assignments,
            'assignment_success_rate': successful_assignments / max(1, len(training_results)),
            'total_pieces_detected': total_pieces,
            'total_assignments_made': total_assignments,
            'average_pieces_per_image': total_pieces / max(1, len(training_results)),
            'average_assignments_per_image': total_assignments / max(1, len(training_results))
        }
        
        print(f"\nTraining Statistics:")
        print(f"  • Processed images: {training_stats['processed_images']}")
        print(f"  • Assignment success rate: {training_stats['assignment_success_rate']:.1%}")
        print(f"  • Total pieces detected: {training_stats['total_pieces_detected']}")
        print(f"  • Total assignments made: {training_stats['total_assignments_made']}")
        print(f"  • Average assignments per image: {training_stats['average_assignments_per_image']:.1f}")
        
        # Generate visualizations for best results
        self.generate_optimized_visualizations(training_results[:15])
        
        # Save training results
        self.save_optimized_results(training_results, training_stats)
        
        return {
            'training_results': training_results,
            'training_stats': training_stats,
            'model_weights': self.mapper.weights
        }
    
    def generate_optimized_visualizations(self, results: List[Dict[str, Any]]):
        """Generate visualizations for optimized results"""
        print(f"\nGenerating visualizations for {len(results)} results...")
        
        visualization_results = []
        
        for result in results:
            if (result.get('success') and 
                result.get('assignment_success') and 
                len(result.get('assignments', [])) > 0):
                
                # Create detection visualization
                self.visualizer.create_detection_image(
                    original_image=result['original_image'],
                    warped_image=result['warped_board'],
                    pieces=result['pieces'],
                    assignments=result['assignments'],
                    ground_truth=result['ground_truth'],
                    image_id=result['image_id']
                )
                
                # Create single detection overlay
                self.visualizer.create_single_detection_overlay(
                    image=result['warped_board'],
                    pieces=result['pieces'],
                    assignments=result['assignments'], 
                    image_id=result['image_id']
                )
                
                visualization_results.append(result)
        
        # Create summary grid
        if visualization_results:
            self.visualizer.create_summary_grid(visualization_results)
            print(f"Generated visualizations for {len(visualization_results)} images")
    
    def save_optimized_results(self, training_results: List[Dict], training_stats: Dict):
        """Save optimized training results in readable format"""
        
        # Prepare summary data
        summary_data = {
            'training_summary': training_stats,
            'model_configuration': {
                'cost_weights': self.mapper.weights,
                'cost_threshold': self.mapper.cost_threshold,
                'min_confidence': self.mapper.min_confidence
            },
            'detailed_results': []
        }
        
        # Add detailed results for images with assignments
        for result in training_results:
            if result.get('assignment_success'):
                detailed_result = {
                    'image_id': result['image_id'],
                    'pieces_detected': len(result['pieces']),
                    'assignments_made': len(result['assignments']),
                    'detailed_analysis': result['detailed_results']
                }
                summary_data['detailed_results'].append(detailed_result)
        
        # Save comprehensive results
        results_file = self.output_dir / "optimized_training_results.json"
        with open(results_file, 'w') as f:
            json.dump(summary_data, f, indent=2, default=str)
        
        # Save readable summary
        summary_file = self.output_dir / "training_summary.txt"
        with open(summary_file, 'w') as f:
            f.write("OPTIMIZED CHESS PIECE MAPPING - TRAINING SUMMARY\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("TRAINING STATISTICS:\n")
            f.write("-" * 30 + "\n")
            for key, value in training_stats.items():
                if isinstance(value, float):
                    f.write(f"{key.replace('_', ' ').title()}: {value:.3f}\n")
                else:
                    f.write(f"{key.replace('_', ' ').title()}: {value}\n")
            
            f.write(f"\nMODEL CONFIGURATION:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Cost Threshold: {self.mapper.cost_threshold}\n")
            f.write(f"Min Confidence: {self.mapper.min_confidence}\n")
            f.write(f"Cost Weights: {self.mapper.weights}\n")
            
            f.write(f"\nDETAILED RESULTS:\n")
            f.write("-" * 30 + "\n")
            for result in summary_data['detailed_results'][:10]:  # Top 10 results
                f.write(f"Image {result['image_id']}:\n")
                f.write(f"  Pieces: {result['pieces_detected']}, Assignments: {result['assignments_made']}\n")
                if 'summary' in result['detailed_analysis']:
                    lines = result['detailed_analysis']['summary'].strip().split('\n')
                    for line in lines:
                        f.write(f"  {line}\n")
                f.write("\n")
        
        print(f"Results saved to:")
        print(f"  • {results_file}")
        print(f"  • {summary_file}")
    
    def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run comprehensive test of the optimized system"""
        print("\n=== COMPREHENSIVE SYSTEM TEST ===")
        
        # Step 1: Dataset analysis
        analysis = self.run_dataset_analysis()
        
        # Step 2: Train optimized model
        training_results = self.train_optimized_model(max_images=50, debug=False)
        
        # Step 3: Evaluation on test set
        test_results = self.evaluate_test_set()
        
        # Step 4: Generate comprehensive report
        report = self.generate_comprehensive_report(analysis, training_results, test_results)
        
        return report
    
    def evaluate_test_set(self, test_size: int = 20) -> Dict[str, Any]:
        """Evaluate on separate test set"""
        print(f"\nEvaluating on {test_size} test images...")
        
        # Use different images for testing (from the end of the dataset)
        all_image_ids = list(self.data_loader.images.keys())
        test_image_ids = all_image_ids[-test_size:] if len(all_image_ids) >= test_size else all_image_ids[-5:]
        
        test_results = []
        total_accuracy = 0.0
        valid_evaluations = 0
        
        for image_id in tqdm(test_image_ids, desc="Testing"):
            result = self.process_single_image_optimized(image_id)
            
            if result and result.get('success') and result.get('assignment_success'):
                test_results.append(result)
                
                # Extract accuracy from detailed results
                detailed = result.get('detailed_results', {})
                if detailed.get('statistics', {}).get('accuracy'):
                    total_accuracy += detailed['statistics']['accuracy']
                    valid_evaluations += 1
        
        evaluation_stats = {
            'test_images_processed': len(test_results),
            'test_success_rate': len(test_results) / len(test_image_ids),
            'average_accuracy': total_accuracy / max(1, valid_evaluations) if valid_evaluations > 0 else 0.0,
            'valid_evaluations': valid_evaluations
        }
        
        print(f"Test Results:")
        print(f"  • Test success rate: {evaluation_stats['test_success_rate']:.1%}")
        print(f"  • Average accuracy: {evaluation_stats['average_accuracy']:.1%}")
        
        return {
            'test_results': test_results,
            'evaluation_stats': evaluation_stats
        }
    
    def generate_comprehensive_report(self, analysis: Dict, training_results: Dict, test_results: Dict) -> Dict:
        """Generate comprehensive performance report"""
        report = {
            'dataset_analysis': analysis,
            'training_performance': training_results['training_stats'],
            'test_performance': test_results['evaluation_stats'],
            'model_configuration': {
                'cost_weights': self.mapper.weights,
                'cost_threshold': self.mapper.cost_threshold
            },
            'recommendations': []
        }
        
        # Generate recommendations based on results
        training_success_rate = training_results['training_stats']['assignment_success_rate']
        test_accuracy = test_results['evaluation_stats']['average_accuracy']
        
        if training_success_rate < 0.7:
            report['recommendations'].append("Consider adjusting cost threshold or improving board detection")
        if test_accuracy < 0.5:
            report['recommendations'].append("Model may need additional tuning or better features")
        if analysis['processing_success_rate'] < 0.8:
            report['recommendations'].append("Dataset processing issues detected - check image quality")
        
        # Save report
        report_file = self.output_dir / "comprehensive_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\nComprehensive report saved to: {report_file}")
        
        return report

def main():
    """Run the optimized training pipeline"""
    print("🚀 OPTIMIZED CHESS PIECE MAPPING TRAINING PIPELINE")
    print("=" * 70)
    
    # Initialize pipeline
    data_path = "/home/pre/projects/chess-datagen/gen-data/render_src/coco_data_2025_08_08__21_53_08"
    pipeline = OptimizedTrainingPipeline(data_path)
    
    # Run comprehensive test
    results = pipeline.run_comprehensive_test()
    
    print("\n🎉 OPTIMIZED TRAINING COMPLETE!")
    print("=" * 70)
    print("Key Results:")
    print(f"  • Training success rate: {results['training_performance']['assignment_success_rate']:.1%}")
    print(f"  • Test accuracy: {results['test_performance']['average_accuracy']:.1%}")
    print(f"  • Dataset processing rate: {results['dataset_analysis']['processing_success_rate']:.1%}")
    
    if results['recommendations']:
        print("\nRecommendations:")
        for rec in results['recommendations']:
            print(f"  • {rec}")
    
    print(f"\nAll results saved to: results/")

if __name__ == "__main__":
    main()