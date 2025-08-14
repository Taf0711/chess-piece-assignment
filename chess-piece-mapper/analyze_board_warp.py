#!/usr/bin/env python3
"""
Analyze Board Warping Issues
Deep analysis of coordinate transformation problems
"""

import cv2
import numpy as np
from src.optimized_data_loader import OptimizedChessDataLoader
import matplotlib.pyplot as plt

def analyze_original_vs_warped():
    """Analyze the board warping transformation"""
    print(" ANALYZING BOARD WARPING TRANSFORMATION")
    print("=" * 60)
    
    data_path = "/home/pre/projects/chess-datagen/gen-data/render_src/coco_data_2025_08_08__21_53_08"
    data_loader = OptimizedChessDataLoader(data_path)
    
    # Load and process an image
    image_id = 0
    result = data_loader.process_image_optimized(image_id, debug=True)
    
    if not result or not result.get('success'):
        print(" Failed to process image")
        return
    
    original_image = result['original_image']
    warped_board = result['warped_board']
    transform_matrix = result['transform_matrix']
    corners = result['corners']
    pieces = result['pieces']
    ground_truth = result['ground_truth']
    
    print(f"Original image shape: {original_image.shape}")
    print(f"Warped board shape: {warped_board.shape}")
    print(f"Detected corners: {corners}")
    print(f"Number of pieces: {len(pieces)}")
    print(f"Ground truth positions: {len(ground_truth)}")
    
    # Analyze piece positions in both coordinate systems
    print(f"\n📍 PIECE POSITION ANALYSIS:")
    annotations = result['annotations'][:10]  # First 10 annotations
    
    for i, ann in enumerate(annotations):
        if ann['category_id'] == 12:  # Skip board annotations
            continue
            
        category_name = data_loader.categories[ann['category_id']]['name']
        original_bbox = ann['bbox']
        original_center = [original_bbox[0] + original_bbox[2]/2, original_bbox[1] + original_bbox[3]/2]
        
        # Find corresponding transformed piece
        transformed_piece = None
        for piece in pieces:
            if piece.piece_type == category_name:
                transformed_piece = piece
                break
        
        if transformed_piece:
            print(f"  {category_name}:")
            print(f"    Original center: ({original_center[0]:.1f}, {original_center[1]:.1f})")
            print(f"    Warped center: ({transformed_piece.center_x:.1f}, {transformed_piece.center_y:.1f})")
            
            # Expected position based on chess starting position
            expected_pos = get_expected_starting_position(category_name)
            if expected_pos:
                print(f"    Expected position: {expected_pos}")
    
    # Test the transformation matrix manually
    print(f"\n TESTING TRANSFORMATION MATRIX:")
    test_points_original = [
        corners[0], corners[1], corners[2], corners[3],  # Board corners
        [original_image.shape[1]/2, original_image.shape[0]/2]  # Center
    ]
    
    for i, point in enumerate(test_points_original):
        # Transform point
        point_homogeneous = np.array([point[0], point[1], 1.0])
        transformed_point = transform_matrix @ point_homogeneous
        transformed_point = transformed_point[:2] / transformed_point[2]
        
        label = f"Corner {i}" if i < 4 else "Center"
        print(f"  {label}: ({point[0]:.1f}, {point[1]:.1f}) -> ({transformed_point[0]:.1f}, {transformed_point[1]:.1f})")

def get_expected_starting_position(piece_name):
    """Get expected starting position for a piece"""
    starting_positions = {
        'WhiteRook1': 'a1', 'WhiteKnight1': 'b1', 'WhiteBishop1': 'c1', 'WhiteQueen1': 'd1',
        'WhiteKing1': 'e1', 'WhiteBishop2': 'f1', 'WhiteKnight2': 'g1', 'WhiteRook2': 'h1',
        'WhitePawn1': 'a2', 'WhitePawn2': 'b2', 'WhitePawn3': 'c2', 'WhitePawn4': 'd2',
        'WhitePawn5': 'e2', 'WhitePawn6': 'f2', 'WhitePawn7': 'g2', 'WhitePawn8': 'h2',
        'BlackPawn1': 'a7', 'BlackPawn2': 'b7', 'BlackPawn3': 'c7', 'BlackPawn4': 'd7',
        'BlackPawn5': 'e7', 'BlackPawn6': 'f7', 'BlackPawn7': 'g7', 'BlackPawn8': 'h7',
        'BlackRook1': 'a8', 'BlackKnight1': 'b8', 'BlackBishop1': 'c8', 'BlackQueen1': 'd8',
        'BlackKing1': 'e8', 'BlackBishop2': 'f8', 'BlackKnight2': 'g8', 'BlackRook2': 'h8',
    }
    return starting_positions.get(piece_name)

def create_improved_board_detector():
    """Create an improved board corner detector"""
    print(f"\n🛠 CREATING IMPROVED BOARD DETECTOR")
    print("=" * 60)
    
    data_path = "/home/pre/projects/chess-datagen/gen-data/render_src/coco_data_2025_08_08__21_53_08"
    data_loader = OptimizedChessDataLoader(data_path)
    
    # Load image
    image_id = 0
    image = data_loader.load_image(image_id)
    
    if image is None:
        print(" Failed to load image")
        return
    
    print(f"Analyzing image shape: {image.shape}")
    
    # Try different corner detection methods
    methods = [
        "chessboard_pattern",
        "contour_detection", 
        "edge_based",
        "manual_annotation"
    ]
    
    best_corners = None
    best_score = -1
    
    for method in methods:
        print(f"\n Testing method: {method}")
        
        try:
            if method == "chessboard_pattern":
                corners = detect_chessboard_corners(image)
            elif method == "contour_detection":
                corners = detect_board_contour(image)
            elif method == "edge_based":
                corners = detect_board_edges(image)
            elif method == "manual_annotation":
                corners = use_board_annotation(data_loader, image_id)
            
            if corners is not None:
                score = validate_corners(image, corners)
                print(f"  Found corners: {corners}")
                print(f"  Validation score: {score:.2f}")
                
                if score > best_score:
                    best_corners = corners
                    best_score = score
        except Exception as e:
            print(f"  Failed: {e}")
    
    if best_corners is not None:
        print(f"\n Best corners found with score {best_score:.2f}:")
        print(f"   {best_corners}")
        return best_corners
    else:
        print(f"\n No valid corners found")
        return None

def detect_chessboard_corners(image):
    """Detect corners using chessboard pattern"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Try different chessboard sizes
    sizes = [(7, 7), (6, 6), (8, 8), (9, 9)]
    
    for size in sizes:
        ret, corners = cv2.findChessboardCorners(gray, size, None)
        if ret:
            # Refine corners
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            
            # Extract board corners from inner corners
            if size == (7, 7):
                board_corners = np.array([
                    corners[0][0],   # Top-left
                    corners[6][0],   # Top-right  
                    corners[48][0],  # Bottom-right
                    corners[42][0]   # Bottom-left
                ], dtype=np.float32)
            else:
                # For other sizes, use corner corners
                board_corners = np.array([
                    corners[0][0],                    # Top-left
                    corners[size[0]-1][0],           # Top-right
                    corners[-1][0],                   # Bottom-right
                    corners[size[0]*(size[1]-1)][0]  # Bottom-left
                ], dtype=np.float32)
            
            return board_corners
    
    return None

def detect_board_contour(image):
    """Detect board using contour detection"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply multiple threshold methods
    thresh_methods = [
        cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2),
        cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
    ]
    
    for thresh in thresh_methods:
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Sort by area
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        for contour in contours[:5]:  # Check top 5 largest contours
            # Approximate contour to polygon
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            if len(approx) == 4:
                # Order corners properly
                corners = order_corners(approx.reshape(4, 2).astype(np.float32))
                return corners
    
    return None

def detect_board_edges(image):
    """Detect board using edge detection and Hough lines"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    # Hough line detection
    lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
    
    if lines is not None and len(lines) >= 4:
        # Find intersection points of lines to form board corners
        # Simplified approach: use image bounds with inset
        h, w = image.shape[:2]
        inset = min(w, h) * 0.1  # 10% inset
        
        corners = np.array([
            [inset, inset],
            [w - inset, inset],
            [w - inset, h - inset],
            [inset, h - inset]
        ], dtype=np.float32)
        
        return corners
    
    return None

def use_board_annotation(data_loader, image_id):
    """Use board annotation from COCO data if available"""
    annotations = data_loader.get_image_annotations(image_id)
    
    for ann in annotations:
        category_name = data_loader.categories[ann['category_id']]['name']
        if category_name == 'Board':
            bbox = ann['bbox']
            # Convert bbox to corners
            x, y, w, h = bbox
            corners = np.array([
                [x, y],         # Top-left
                [x + w, y],     # Top-right
                [x + w, y + h], # Bottom-right
                [x, y + h]      # Bottom-left
            ], dtype=np.float32)
            return corners
    
    return None

def order_corners(corners):
    """Order corners as [top-left, top-right, bottom-right, bottom-left]"""
    # Sort by y-coordinate
    corners = corners[corners[:, 1].argsort()]
    
    # Top two points
    top_corners = corners[:2]
    top_corners = top_corners[top_corners[:, 0].argsort()]  # Sort by x
    
    # Bottom two points
    bottom_corners = corners[2:]
    bottom_corners = bottom_corners[bottom_corners[:, 0].argsort()]  # Sort by x
    
    return np.array([
        top_corners[0],     # Top-left
        top_corners[1],     # Top-right
        bottom_corners[1],  # Bottom-right
        bottom_corners[0]   # Bottom-left
    ], dtype=np.float32)

def validate_corners(image, corners):
    """Validate corner quality"""
    if corners is None or len(corners) != 4:
        return -1.0
    
    # Check if corners form a reasonable rectangle
    h, w = image.shape[:2]
    
    # All corners should be within image bounds
    for corner in corners:
        if corner[0] < 0 or corner[0] >= w or corner[1] < 0 or corner[1] >= h:
            return -1.0
    
    # Calculate area
    area = cv2.contourArea(corners)
    image_area = h * w
    
    # Area should be a reasonable fraction of image
    area_ratio = area / image_area
    if area_ratio < 0.1 or area_ratio > 0.9:
        return 0.0
    
    # Check if it's roughly rectangular
    # Calculate side lengths
    sides = []
    for i in range(4):
        p1 = corners[i]
        p2 = corners[(i + 1) % 4]
        side_length = np.linalg.norm(p2 - p1)
        sides.append(side_length)
    
    # Opposite sides should be similar
    ratio1 = min(sides[0], sides[2]) / max(sides[0], sides[2])
    ratio2 = min(sides[1], sides[3]) / max(sides[1], sides[3])
    
    rectangularity = (ratio1 + ratio2) / 2
    
    return area_ratio * rectangularity

def main():
    """Run board warping analysis"""
    print(" BOARD WARPING ANALYSIS")
    print("=" * 70)
    
    # Analyze current warping
    analyze_original_vs_warped()
    
    # Create improved detector
    improved_corners = create_improved_board_detector()
    
    if improved_corners is not None:
        print(f"\n Improved corner detection successful")
        print(f"Next step: Implement these corners in the mapper")
    else:
        print(f"\n Need to investigate corner detection further")

if __name__ == "__main__":
    main()