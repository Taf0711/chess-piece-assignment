# Chess Piece Assignment

A machine learning system for mapping detected chess pieces to board squares using the Hungarian algorithm and computer vision.

## What it does

This project takes chess piece detections (bounding boxes from object detectors like YOLO) and figures out which board square each piece belongs to. It uses optimization algorithms to make the best possible assignments.

The main approach:
- Warp chess board images to a standard 256x256 top-down view  
- Calculate costs for assigning each detected piece to each of the 64 squares
- Use the Hungarian algorithm to find the optimal assignment
- Evaluate against ground truth chess positions in FEN format

## Project structure

- `chess-piece-mapper/` - Main system with the assignment algorithms and training code
- `combined_model/` - Integration of different components 
- `evaluation/` - Tools for measuring performance
- `gen-data/` - Scripts for generating training data with Blender
- `data_map/` - Utilities for board coordinate mapping

## Key components

**Data loading** (`chess-piece-mapper/src/data_loader.py`)
- Loads images and COCO format annotations
- Detects board corners and applies perspective correction
- Transforms bounding boxes to the warped coordinate system

**Piece assignment** (`chess-piece-mapper/src/piece_mapper.py`) 
- Computes assignment costs based on distance, overlap, piece type, and board position
- Solves the assignment problem using scipy's Hungarian algorithm implementation
- Filters results by cost thresholds

**Training pipeline** (`chess-piece-mapper/src/training_pipeline.py`)
- Optimizes cost function weights using ground truth data
- Evaluates accuracy and generates performance reports

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

This codebase was derived from my friend's chess data generation repository. The piece assignment system was built on top of their existing chess rendering and data processing tools.