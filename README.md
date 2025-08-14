# Chess Piece Assignment

A machine learning system for mapping detected chess pieces to board squares using the Hungarian algorithm and computer vision techniques.

## Overview

This project solves the problem of automatically determining which chess pieces belong to which squares on a chess board from computer vision detections. Given bounding box detections of chess pieces (from YOLO, R-CNN, or similar object detectors), the system uses optimization algorithms to assign each detected piece to its most likely board square.

The core challenge is that piece detections may be noisy, incomplete, or imprecise, while the assignment must respect the constraint that each square can contain at most one piece. This is formulated as a classic assignment problem and solved using the Hungarian algorithm.

## Algorithm Overview

The system works in several stages:

1. **Board Standardization**: Transform the input chess board image to a standardized 256x256 pixel top-down view where each square is exactly 32x32 pixels. This removes perspective distortion and provides a consistent coordinate system.

2. **Cost Matrix Computation**: For each detected piece and each of the 64 board squares, compute an assignment cost that considers multiple factors including geometric distance, bounding box overlap, piece type constraints, and board edge penalties.

3. **Optimal Assignment**: Use the Hungarian algorithm (also known as the Kuhn-Munkres algorithm) to find the assignment of pieces to squares that minimizes the total cost.

4. **Post-processing**: Filter assignments based on cost thresholds to remove poor matches, and convert square coordinates to standard chess notation (a1, b2, etc.).

## Detailed Project Structure

```
chess-piece-assignment/
├── chess-piece-mapper/              # Core ML mapping system
│   ├── src/                         # Source code modules
│   │   ├── data_loader.py          # COCO data loading and board warping
│   │   ├── piece_mapper.py         # Hungarian algorithm implementation
│   │   ├── training_pipeline.py    # Training and parameter optimization
│   │   ├── optimized_mapper.py     # Performance-optimized version
│   │   ├── direct_coordinate_mapper.py # Alternative coordinate-based approach
│   │   └── image_visualizer.py     # Visualization and debugging tools
│   ├── results/                     # Training outputs and test results
│   │   ├── trained_piece_mapper.pkl # Saved model weights
│   │   ├── training_results.json   # Training metrics and parameters
│   │   └── detection_images/       # Visualization outputs
│   ├── config/                      # Configuration files
│   ├── models/                      # Model definitions and weights
│   └── requirements.txt            # Python dependencies
├── combined_model/                  # End-to-end pipeline integration
│   ├── predict_board.py            # Complete inference pipeline
│   └── samples/                    # Example input images
├── evaluation/                      # Performance evaluation tools
│   ├── board_segmentation/         # Board detection evaluation
│   └── board_mappings/            # Assignment accuracy evaluation
├── gen-data/                       # Data generation pipeline
│   ├── render_src/                 # Blender rendering scripts
│   │   ├── reorder.py             # Basic rendering pipeline
│   │   ├── reorder_gpu_aggressive.py # GPU-optimized rendering
│   │   └── *.blend                # Blender scene files
│   └── reformat_util/             # Data format conversion tools
└── data_map/                       # Board mapping utilities
    ├── use_board.py               # Board coordinate transformations
    └── use_board_GPT.py          # Enhanced mapping functions
```

## Core Components Documentation

### Data Loading System

The `ChessDataLoader` class handles the complex task of loading chess images and preparing them for analysis:

**Board Detection**: Uses OpenCV's chessboard pattern detection or contour-based methods to identify the four corners of the chess board in the image. This step is critical as it defines the perspective transformation.

**Perspective Correction**: Applies a perspective transformation using `cv2.getPerspectiveTransform()` to warp the detected board region to a standard 256x256 pixel square. This ensures consistent geometry regardless of camera angle.

**Coordinate Transformation**: All piece bounding boxes are transformed to match the warped coordinate system, ensuring that piece positions are correctly mapped to the standardized board.

**Data Validation**: Verifies that the transformed coordinates are reasonable and filters out obviously incorrect detections.

### Piece Assignment Algorithm

The `PieceToSquareMapper` class implements the core assignment logic:

**Cost Function Components**:
- Distance Cost: Euclidean distance from the center of the piece's bounding box to the center of each square
- Overlap Cost: Intersection-over-Union (IoU) between the piece bounding box and the target square
- Piece Type Cost: Chess-specific bonuses and penalties (e.g., pawns are more likely on ranks 2 and 7)
- Board Edge Cost: Penalties for pieces detected near the image boundaries, which are often false positives

**Hungarian Algorithm**: Uses scipy's `linear_sum_assignment` function to solve the assignment problem optimally. This guarantees the minimum total cost assignment.

**Filtering**: Applies cost thresholds to reject assignments with unreasonably high costs, which typically correspond to detection errors.

### Training and Optimization

The training system optimizes the weights of the different cost components:

**Parameter Search**: Uses grid search or gradient-based methods to find optimal weights for distance, overlap, piece type, and edge costs.

**Ground Truth Evaluation**: Compares assignments against FEN (Forsyth-Edwards Notation) strings that encode the true board positions.

**Cross Validation**: Tests on held-out data to ensure the optimized parameters generalize well.

**Performance Metrics**: Tracks per-piece accuracy, overall position accuracy, and assignment confidence scores.

## Running it

```bash
cd chess-piece-mapper
python demo.py
```

For training:
```bash
python src/training_pipeline.py
```

## Cost function

For each piece-square pair, the total cost considers:
- Distance from piece center to square center
- Overlap between piece bounding box and target square
- Chess-specific bonuses (like pawns being more likely on certain ranks)
- Penalties for pieces near image edges

The Hungarian algorithm then finds the assignment that minimizes total cost across all pieces.

## Data format

Expects COCO format annotations with chess piece bounding boxes and FEN strings for ground truth positions. The system handles variable numbers of detected pieces (doesn't assume exactly 32).

## Performance

Processes around 10-15 images per second. Accuracy depends on image quality and how well the board detection works. Supports 32 different piece categories.

## Note

This codebase was derived from Ishraq's chess-gen data generation repository. The piece assignment system was built on top of their existing chess rendering and data processing tools.
