import blenderproc as bproc  # On version 2.8.0

import csv
import random
import time
import re
import bpy  # type: ignore
import random
import re
import os
import numpy as np
import re
import json
import os
import math
import datetime
import threading
import subprocess

# Init BlenderProc and Optimize Your Settings
# Using magical numbers found online
# https://blenderartists.org/t/options-to-speed-up-a-render/1515328
bproc.init()


def _force_cycles_gpu(device_type="CUDA"):
    print("=== FORCING GPU CONFIGURATION ===")
    
    # Make sure Cycles is the active engine
    bpy.context.scene.render.engine = "CYCLES"
    print(f"Set render engine to: {bpy.context.scene.render.engine}")

    # 1) Tell Cycles to use the GPU for rendering
    bpy.context.scene.cycles.device = "GPU"
    print(f"Set cycles device to: {bpy.context.scene.cycles.device}")

    # 2) Enable the CUDA backend at the preferences level
    prefs = bpy.context.preferences.addons["cycles"].preferences
    # In 4.x you should refresh before changing devices
    try:
        prefs.refresh_devices()
        print("Refreshed devices successfully")
    except Exception as e:
        print(f"Device refresh failed: {e}")

    # Prefer CUDA (you can change to "OPTIX" if you want OptiX path tracing)
    prefs.compute_device_type = device_type  # "CUDA" or "OPTIX"
    print(f"Set compute device type to: {prefs.compute_device_type}")

    # 3) Turn on all CUDA devices (and leave CPU off) - MORE AGGRESSIVE
    print("=== DEVICE CONFIGURATION ===")
    def _enable_all_cuda(devs):
        for d in devs:
            name = getattr(d, "name", "")
            dev_type = getattr(d, "type", "")
            print(f"Found device: {name} ({dev_type})")
            
            # EXPLICITLY disable CPU and enable only CUDA/OPTIX
            if dev_type == "CPU":
                d.use = False
                print(f"  -> DISABLED CPU: {name}")
            elif dev_type in ("CUDA", "OPTIX"):
                d.use = True
                print(f"  -> ENABLED GPU: {name}")
            else:
                d.use = False
                print(f"  -> DISABLED UNKNOWN: {name}")
    
    try:
        # Flat list style
        _enable_all_cuda(prefs.devices)
    except Exception as e:
        print(f"Flat device config failed: {e}")
        # Nested style ([(type, [devices...]), ...])
        try:
            for typ, devs in getattr(prefs, "get_devices_for_type", lambda *_: [])(device_type):
                print(f"Processing device group: {typ}")
                _enable_all_cuda(devs)
        except Exception as e2:
            print(f"Nested device config also failed: {e2}")

    # 4) Set environment variables for extra enforcement
    os.environ["CYCLES_RENDER_DEVICE"] = device_type
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Force only GPU 0
    print(f"Set CYCLES_RENDER_DEVICE={device_type}")
    print(f"Set CUDA_VISIBLE_DEVICES=0")

    # 5) Force preview compute device to GPU as well
    try:
        bpy.context.scene.cycles.preview_compute_device = "GPU"
        print("Set preview compute device to GPU")
    except AttributeError:
        print("Preview compute device setting not available")
    
    # Log what we ended up with
    enabled = []
    disabled = []
    try:
        for d in prefs.devices:
            name = getattr(d, "name", "")
            dev_type = getattr(d, "type", "")
            is_used = getattr(d, "use", False)
            if is_used:
                enabled.append(f"{dev_type}:{name}")
            else:
                disabled.append(f"{dev_type}:{name}")
    except Exception:
        pass
    
    print(f"=== FINAL GPU CONFIG ===")
    print(f"Render engine: {bpy.context.scene.render.engine}")
    print(f"Cycles device: {bpy.context.scene.cycles.device}")
    print(f"Compute device type: {prefs.compute_device_type}")
    try:
        print(f"Preview compute device: {bpy.context.scene.cycles.preview_compute_device}")
    except AttributeError:
        print("Preview compute device: N/A")
    print(f"ENABLED devices: {enabled}")
    print(f"DISABLED devices: {disabled}")
    print("=== END GPU CONFIG ===")

_force_cycles_gpu(device_type="CUDA")  # Back to CUDA, remove BlenderProc interference

def verify_gpu_usage():
    """Verify that GPU is properly configured for rendering"""
    print("=== VERIFYING GPU USAGE ===")
    
    # Check render engine
    engine = bpy.context.scene.render.engine
    print(f"Render Engine: {engine}")
    if engine != "CYCLES":
        print("WARNING: Not using Cycles engine!")
        return False
    
    # Check cycles device setting
    device = bpy.context.scene.cycles.device
    print(f"Cycles Device: {device}")
    if device != "GPU":
        print("WARNING: Cycles not set to use GPU!")
        return False
    
    # Check enabled devices
    prefs = bpy.context.preferences.addons["cycles"].preferences
    gpu_enabled = False
    cpu_enabled = False
    
    for d in prefs.devices:
        name = getattr(d, "name", "")
        dev_type = getattr(d, "type", "")
        is_used = getattr(d, "use", False)
        print(f"Device: {name} ({dev_type}) - Enabled: {is_used}")
        
        if dev_type == "CUDA" and is_used:
            gpu_enabled = True
        elif dev_type == "CPU" and is_used:
            cpu_enabled = True
    
    if not gpu_enabled:
        print("ERROR: No CUDA GPU devices are enabled!")
        return False
    
    if cpu_enabled:
        print("WARNING: CPU is still enabled for rendering!")
        return False
    
    print("✓ GPU configuration appears correct")
    return True

verify_gpu_usage()
bproc.renderer.set_cpu_threads(0)

# COMPLETELY REMOVE BLENDERPROC DEVICE SETTING - IT'S OVERRIDING OUR CONFIG!
print("=== SKIPPING BlenderProc device setting - using manual GPU config only ===")
# IMMEDIATELY DISABLE ALL CPU CORES IN BLENDER
bpy.context.scene.render.threads_mode = 'FIXED'
bpy.context.scene.render.threads = 0  # Force 0 CPU threads

bproc.renderer.set_noise_threshold(0.1)

# RE-ENFORCE GPU USAGE AFTER BLENDERPROC SETUP
print("=== RE-ENFORCING GPU AFTER BLENDERPROC ===")
_force_cycles_gpu(device_type="CUDA")  # Back to CUDA, remove BlenderProc interference

# FINAL AGGRESSIVE CHECK - ENSURE NO CPU USAGE
prefs = bpy.context.preferences.addons["cycles"].preferences
for d in prefs.devices:
    if getattr(d, "type", "") == "CPU":
        d.use = False
        print(f"FINAL CHECK: Disabled CPU device {getattr(d, 'name', '')}")
    elif getattr(d, "type", "") == "CUDA":
        d.use = True
        print(f"FINAL CHECK: Enabled CUDA device {getattr(d, 'name', '')}")

verify_gpu_usage()

# GPU MONITORING THREAD FOR REAL-TIME FEEDBACK
gpu_monitor_running = True
gpu_utilization_history = []

def monitor_gpu():
    global gpu_monitor_running, gpu_utilization_history
    while gpu_monitor_running:
        try:
            result = subprocess.run([
                'nvidia-smi', '--query-gpu=utilization.gpu,temperature.gpu,power.draw', 
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0:
                gpu_data = result.stdout.strip().split(', ')
                gpu_util = int(gpu_data[0])
                gpu_temp = int(gpu_data[1]) 
                gpu_power = float(gpu_data[2])
                
                gpu_utilization_history.append(gpu_util)
                if len(gpu_utilization_history) > 20:  # Keep last 20 readings
                    gpu_utilization_history.pop(0)
                
                avg_util = sum(gpu_utilization_history) / len(gpu_utilization_history)
                print(f"GPU: {gpu_util}% util, {gpu_temp}°C, {gpu_power}W (avg: {avg_util:.1f}%)")
                
        except Exception as e:
            print(f"GPU monitoring error: {e}")
        
        time.sleep(2)  # Monitor every 2 seconds

# Start GPU monitoring thread
gpu_monitor_thread = threading.Thread(target=monitor_gpu, daemon=True)
gpu_monitor_thread.start()
print(" GPU monitoring started - targeting 70-80% utilization")

# OPTIMIZED SAMPLES FOR 70-80% GPU UTILIZATION
bproc.renderer.set_max_amount_of_samples(256)  # Balanced samples for sustained GPU load

# DISABLE DENOISING TO INCREASE GPU WORKLOAD  
bproc.renderer.set_denoiser(None)  # No denoising = more raw GPU compute

# INCREASE BOUNCES FOR MORE GPU-INTENSIVE RAY TRACING
bproc.renderer.set_light_bounces(12, 12, 12, 12, 24, 16, 4)  # Much higher bounces

# Load your scene
loaded = bproc.loader.load_blend("ChessBoard2.blend")
output_path = datetime.datetime.now().strftime("coco_data_%Y_%m_%d__%H_%M_%S")
os.makedirs(output_path, exist_ok=True)

# ----- PIECE PLACEMENT -----
# This section defines the positions of the chess pieces on the board.
positionsDict = {
    'A1': (-0.87574, -0.87857),
    'A2': (-0.87574, -0.628312),
    'A3': (-0.87574, -0.378054),
    'A4': (-0.87574, -0.127796),
    'A5': (-0.87574, 0.122462),
    'A6': (-0.87574, 0.372720),
    'A7': (-0.87574, 0.622978),
    'A8': (-0.87574, 0.873236),
    'B1': (-0.62914, -0.87857),
    'B2': (-0.62914, -0.628312),
    'B3': (-0.62914, -0.378054),
    'B4': (-0.62914, -0.127796),
    'B5': (-0.62914, 0.122462),
    'B6': (-0.62914, 0.372720),
    'B7': (-0.62914, 0.622978),
    'B8': (-0.62914, 0.873236),
    'C1': (-0.38254, -0.87857),
    'C2': (-0.38254, -0.628312),
    'C3': (-0.38254, -0.378054),
    'C4': (-0.38254, -0.127796),
    'C5': (-0.38254, 0.122462),
    'C6': (-0.38254, 0.372720),
    'C7': (-0.38254, 0.622978),
    'C8': (-0.38254, 0.873236),
    'D1': (-0.13594, -0.87857),
    'D2': (-0.13594, -0.628312),
    'D3': (-0.13594, -0.378054),
    'D4': (-0.13594, -0.127796),
    'D5': (-0.13594, 0.122462),
    'D6': (-0.13594, 0.372720),
    'D7': (-0.13594, 0.622978),
    'D8': (-0.13594, 0.873236),
    'E1': (0.11066, -0.87857),
    'E2': (0.11066, -0.628312),
    'E3': (0.11066, -0.378054),
    'E4': (0.11066, -0.127796),
    'E5': (0.11066, 0.122462),
    'E6': (0.11066, 0.372720),
    'E7': (0.11066, 0.622978),
    'E8': (0.11066, 0.873236),
    'F1': (0.35726, -0.87857),
    'F2': (0.35726, -0.628312),
    'F3': (0.35726, -0.378054),
    'F4': (0.35726, -0.127796),
    'F5': (0.35726, 0.122462),
    'F6': (0.35726, 0.372720),
    'F7': (0.35726, 0.622978),
    'F8': (0.35726, 0.873236),
    'G1': (0.60386, -0.87857),
    'G2': (0.60386, -0.628312),
    'G3': (0.60386, -0.378054),
    'G4': (0.60386, -0.127796),
    'G5': (0.60386, 0.122462),
    'G6': (0.60386, 0.372720),
    'G7': (0.60386, 0.622978),
    'G8': (0.60386, 0.873236),
    'H1': (0.85046, -0.87857),
    'H2': (0.85046, -0.628312),
    'H3': (0.85046, -0.378054),
    'H4': (0.85046, -0.127796),
    'H5': (0.85046, 0.122462),
    'H6': (0.85046, 0.372720),
    'H7': (0.85046, 0.622978),
    'H8': (0.85046, 0.873236),
    'None': (1, 1)
}

pieceToSquareDict = {
    'BlackRook1':   'A8',
    'BlackRook2':   'H8',
    'BlackRook3':   'None',
    'BlackRook4':   'None',
    'BlackRook5':   'None',
    'BlackRook6':   'None',
    'BlackRook7':   'None',
    'BlackRook8':   'None',
    'BlackRook9':   'None',
    'BlackRook10':  'None',

    'BlackKnight1': 'B8',
    'BlackKnight2': 'G8',
    'BlackKnight3': 'None',
    'BlackKnight4': 'None',
    'BlackKnight5': 'None',
    'BlackKnight6': 'None',
    'BlackKnight7': 'None',
    'BlackKnight8': 'None',
    'BlackKnight9': 'None',
    'BlackKnight10':'None',

    'BlackBishop1': 'C8',
    'BlackBishop2': 'E8',
    'BlackBishop3': 'None',
    'BlackBishop4': 'None',
    'BlackBishop5': 'None',
    'BlackBishop6': 'None',
    'BlackBishop7': 'None',
    'BlackBishop8': 'None',
    'BlackBishop9': 'None',
    'BlackBishop10':'None',

    'BlackQueen1':  'D8',
    'BlackQueen2':  'None',
    'BlackQueen3':  'None',
    'BlackQueen4':  'None',
    'BlackQueen5':  'None',
    'BlackQueen6':  'None',
    'BlackQueen7':  'None',
    'BlackQueen8':  'None',
    'BlackQueen9':  'None',

    'BlackKing1':   'E8',

    'BlackPawn1':   'A7',
    'BlackPawn2':   'B7',
    'BlackPawn3':   'C7',
    'BlackPawn4':   'D7',
    'BlackPawn5':   'E7',
    'BlackPawn6':   'F7',
    'BlackPawn7':   'G7',
    'BlackPawn8':   'H7',

    'WhiteRook1':   'A1',
    'WhiteRook2':   'H1',
    'WhiteRook3':   'None',
    'WhiteRook4':   'None',
    'WhiteRook5':   'None',
    'WhiteRook6':   'None',
    'WhiteRook7':   'None',
    'WhiteRook8':   'None',
    'WhiteRook9':   'None',
    'WhiteRook10':  'None',

    'WhiteKnight1': 'B1',
    'WhiteKnight2': 'G1',
    'WhiteKnight3': 'None',
    'WhiteKnight4': 'None',
    'WhiteKnight5': 'None',
    'WhiteKnight6': 'None',
    'WhiteKnight7': 'None',
    'WhiteKnight8': 'None',
    'WhiteKnight9': 'None',
    'WhiteKnight10':'None',

    'WhiteBishop1': 'C1',
    'WhiteBishop2': 'F1',
    'WhiteBishop3': 'None',
    'WhiteBishop4': 'None',
    'WhiteBishop5': 'None',
    'WhiteBishop6': 'None',
    'WhiteBishop7': 'None',
    'WhiteBishop8': 'None',
    'WhiteBishop9': 'None',
    'WhiteBishop10':'None',

    'WhiteQueen1':  'D1',
    'WhiteQueen2':  'None',
    'WhiteQueen3':  'None',
    'WhiteQueen4':  'None',
    'WhiteQueen5':  'None',
    'WhiteQueen6':  'None',
    'WhiteQueen7':  'None',
    'WhiteQueen8':  'None',
    'WhiteQueen9':  'None',

    'WhiteKing1':   'E1',

    'WhitePawn1':   'A2',
    'WhitePawn2':   'B2',
    'WhitePawn3':   'C2',
    'WhitePawn4':   'D2',
    'WhitePawn5':   'E2',
    'WhitePawn6':   'F2',
    'WhitePawn7':   'G2',
    'WhitePawn8':   'H2',
}

def parse_fen(fen):
    files = 'abcdefgh'
    ranks = fen.split()[0].split('/')
    piecePositions = {
        'BlackPawn': [], 'WhitePawn': [],
        'BlackRook': [], 'WhiteRook': [],
        'BlackKnight': [], 'WhiteKnight': [],
        'BlackBishop': [], 'WhiteBishop': [],
        'BlackQueen': [], 'WhiteQueen': [],
        'BlackKing': [],  'WhiteKing': [],
    }
    symbolToType = {
        'p': 'Pawn', 'r': 'Rook', 'n': 'Knight',
        'b': 'Bishop','q': 'Queen','k': 'King'
    }
    for rank_idx, rank in enumerate(ranks):
        file_idx = 0
        for ch in rank:
            if ch.isdigit():
                file_idx += int(ch)
            else:
                color = 'White' if ch.isupper() else 'Black'
                ptype = symbolToType[ch.lower()]
                square = files[file_idx] + str(8 - rank_idx)
                piecePositions[f'{color}{ptype}'].append(square)
                file_idx += 1
    return piecePositions

# update my dict based on new positions
def update_dict_from_positions(d, positions):
    reset()
    for k in d:
        d[k] = 'None'
    bpy.data.objects["BlackKing1"].hide_viewport = False
    bpy.data.objects["BlackKing1"].hide_render = False
    bpy.data.objects["WhiteKing1"].hide_viewport = False
    bpy.data.objects["WhiteKing1"].hide_render = False
    for ptype, squares in positions.items():
        for i, sq in enumerate(squares, start=1):
            key = f'{ptype}{i}'
            d[key] = sq
            bpy.data.objects[key].hide_viewport = False
            bpy.data.objects[key].hide_render = False
    return d

class Piece:
    def __init__(self, name, kind, color, initial_position, home_tile_color):
        self.name_ = name
        self.kind_ = kind
        self.color_ = color
        self.initial_position_ = initial_position
        self.home_tile_color_ = home_tile_color

def allowed_squares_for(piece, free_squares):

    choices = set(free_squares)

    # pawns cant go on rank 1 or 8
    if piece.kind_ == 'Pawn':
        choices = {sq for sq in choices if sq[1] not in ('1', '8')}

    # bishops only on same color they started on
    if piece.kind_ == 'Bishop':
        # piece.home_tile_color_ is a bool: False = Black True = White
        choices = {sq for sq in choices if tile_color[sq] == piece.home_tile_color_}

    return choices

def randomizePositions(chance=0.5):
    free_squares = set(positionsDict.keys())
    assignment = {}

    pieces = []

    # if piece were chosen, add it to pieces otherwise make it disappear
    for p in pieceList:
        # each piece has a 50% chance of selection, but kings are always selected
        if random.random() < chance or 'King' in p.name_:
            pieces.append(p)
        else:
            bpy.data.objects[p.name_].hide_viewport = True
            bpy.data.objects[p.name_].hide_render = True

    # shuffle the pieces
    random.shuffle(pieces)

    for p in pieces:
        legal = allowed_squares_for(p, free_squares)
        if not legal:
            print(f"WARNING: No legal square left for {p.name_}!")
            bpy.data.objects[p.name_].hide_viewport = True
            bpy.data.objects[p.name_].hide_render = True
            continue
            # there are no bugs, only features
            # raise RuntimeError(f"No legal square left for {p.name_}!")

        square = random.choice(list(legal))
        free_squares -= {square}
        assignment[p.name_] = square

    return assignment

def reset():
    # reset the pieces
    for name, square in pieceToSquareDict.items():
        bpy.data.objects[name].hide_viewport = True
        bpy.data.objects[name].hide_render = True

FILES = 'ABCDEFGH'
RANKS = '12345678'

# mapping squares to tile color
tile_color = {}
for f in FILES:
    for r in RANKS:
        # Chessboard coloring: bottom left (A1) is black (False)
        tile_color[f + r] = ((FILES.index(f) + RANKS.index(r)) % 2 == 1)

pieceList = []

# make the piece class and populate pieceList
for piece, square in pieceToSquareDict.items():
    if square == "None":  # fixed string comparison
        continue
    name = piece
    color = 'Black' if 'Black' in name else 'White'
    kind = re.sub(r'\d+$', '', piece[len(color):])
    initial_position = square
    print(square)
    home_tile_color = 'Black' if ((FILES.index(square[0]) + RANKS.index(square[1])) % 2 == 0) else 'White'
    pieceList.append(Piece(name, kind, color, initial_position, home_tile_color))

reset()

# Labeling for COCO annotations
CATEGORY_NAME_TO_ID = {
    "WhitePawn": 13,
    "WhiteRook": 1,
    "WhiteKnight": 2,
    "WhiteBishop": 3,
    "WhiteQueen": 4,
    "WhiteKing": 5,

    "BlackPawn": 6,
    "BlackRook": 7,
    "BlackKnight": 8,
    "BlackBishop": 9,
    "BlackQueen": 10,
    "BlackKing": 11,

    "Board": 12
}

def get_base_name(name):
    # Remove trailing digits: 'WhitePawn1' → 'WhitePawn'
    return re.sub(r'\d+$', '', name)

for obj in loaded:
    full_name = obj.get_name()
    base_name = get_base_name(full_name)
    print(f"Processing object: {full_name} (base name: {base_name})")
    if base_name not in CATEGORY_NAME_TO_ID:
        print(f"WARNING: Unknown category name '{base_name}' for object '{full_name}'")
        continue
    obj.set_cp("category_id", CATEGORY_NAME_TO_ID[base_name])

# ----- CAMERA SETUP -----
# This section sets up the camera to orbit around the chessboard.
# The camera will be positioned at a distance and look towards the center of the board.

# The below code was taken from the BlenderProc documentation
# Create a point light next to it
light = bproc.types.Light()
light.set_location([0.0, 0.0, 2.0])  # Light above the chessboard
light.set_energy(1000.0)
bproc.camera.set_resolution(640, 480)  # Set the resolution of the rendered images
bproc.renderer.set_output_format(enable_transparency=True)
print("K MATRIX", bproc.camera.get_intrinsics_as_K_matrix())

# OPTIMIZED FOR SUSTAINED GPU UTILIZATION
# Balanced workload for 70-80% GPU usage with efficient data generation
N = 5     # More cameras per setup for better GPU utilization
num_random_setup = 200   # More setups for sustained workload
print(f"Generating {num_random_setup} random setups with {N} camera positions each...")
print(f"Total images to generate: {N * num_random_setup}")
print("Optimized for 70-80% GPU utilization...")
# Golden angle in radians
golden_angle = np.pi * (3 - np.sqrt(5))

camera_info = {}

# Add Camera poses
for i in range(N):
    rho = random.uniform(4.5, 6.5)

    z = 1 - (i) / (N - 1)            # z from 1 to -1
    radius = np.sqrt(1 - z * z)      # radius at that z
    theta = golden_angle * i         # azimuthal angle

    x = np.cos(theta) * radius
    y = np.sin(theta) * radius

    pos = rho * np.array([x, y, z])  # scale to radius rho

    # Camera looks at origin
    forward_vec = -pos / np.linalg.norm(pos)
    rotation = bproc.camera.rotation_from_forward_vec(forward_vec)

    cam_pose = bproc.math.build_transformation_mat(pos.tolist(), rotation)
    print(rotation, cam_pose)
    bproc.camera.add_camera_pose(cam_pose)
    camera_info[i] = {"dist": rho, "rot": rotation.tolist(), "pos": pos.tolist()}

# Render for each randomized position
trn_val_tst_split = [6, 2, 2]
gcd = math.gcd(math.gcd(trn_val_tst_split[0], trn_val_tst_split[1]), trn_val_tst_split[2])
trn_val_tst_split = [x // gcd for x in trn_val_tst_split]
split_map = {0: 'train', 1: 'val', 2: 'test'}

fen_visited = set()

# Set up csv reading
csv_file = open('positions.csv', newline='')
reader = csv.reader(csv_file)
next(reader, None)
fen_rows = iter(reader)
avg_time = 0
for z in range(num_random_setup):
    progress_pct = (z+1) / num_random_setup * 100
    print(f"==== Render step {z+1}/{num_random_setup} ({progress_pct:.1f}%) ====")
    current_time = time.time()
    
    # Estimate completion time
    if z > 0:
        time_per_setup = avg_time
        remaining_setups = num_random_setup - (z + 1)
        eta_seconds = remaining_setups * time_per_setup
        eta_hours = eta_seconds / 3600
        print(f"ETA: {eta_hours:.1f} hours ({eta_seconds/60:.0f} minutes) remaining")

    light.set_location(
        [
            random.uniform(-2.0, 2.0),
            random.uniform(-2.0, 2.0),
            random.uniform(0, 2.5)
        ]
    )  # Light above the chessboard

    light.set_energy(random.uniform(250, 750))
    # Randomly shuffle the pieceList to create a new random setup
    # Determine which split this iteration belongs to (0=train, 1=val, 2=test)

    # Magic number stuff to figure out which split this is
    # This is a s̶i̶m̶p̶l̶e̶ bad way to split the data into train, validation, and test sets
    # Please don't fire me in the future or flame me for this i plead innocence
    split_idx = z % sum(trn_val_tst_split)
    if split_idx < trn_val_tst_split[0]:
        split_idx = 0
    elif split_idx < trn_val_tst_split[0] + trn_val_tst_split[1]:
        split_idx = 1
    else:
        split_idx = 2

    dir_pre = split_map[split_idx]

    try:
        row = next(fen_rows)
        fen = row[0]
        while fen in fen_visited:
            print(f"FEN {fen} already visited, skipping...")
            row = next(fen_rows)
            fen = row[0]

    except StopIteration:
        print("No more FEN rows—stopping early.")
        break

    fen_visited.add(fen)

    pos = parse_fen(fen)
    update_dict_from_positions(pieceToSquareDict, pos)
    placement = pieceToSquareDict.copy()

    for name, square in placement.items():
        if square == "None":  # fixed string comparison
            continue
        x, y = positionsDict[square.upper()]
        bpy.data.objects[name].location.x = x
        bpy.data.objects[name].location.y = y

    # OPTIMIZED RENDERING WITH GPU UTILIZATION MONITORING
    avg_util = sum(gpu_utilization_history) / len(gpu_utilization_history) if gpu_utilization_history else 0
    print(f"=== STARTING RENDER {z+1}/{num_random_setup} (GPU avg: {avg_util:.1f}%) ===")
    
    render_start = time.time()
    print(" Rendering segmentation maps...")
    seg_data = bproc.renderer.render_segmap(map_by=["instance", "class", "name"])
    
    print(" Rendering main colors...")
    data = bproc.renderer.render()
    render_time = time.time() - render_start
    
    print(f"⚡ Render {z+1} complete in {render_time:.2f}s")
    
    # BATCH WRITE DATA TO REDUCE I/O BOTTLENECKS  
    write_start = time.time()
    image_paths = bproc.writer.write_coco_annotations(
        f'{output_path}/{dir_pre}',
        instance_segmaps=seg_data["instance_segmaps"],  # type: ignore
        instance_attribute_maps=seg_data["instance_attribute_maps"],  # type: ignore
        colors=data["colors"],  # type: ignore
        color_file_format="PNG",
        append_to_existing_output=True,  # <-- important!
    )
    
    # THE OUTPUT OF WRITE COCO WAS MODIFIED TO RETURN THE IMAGE PATHS; ENSURE THIS IS DONE IN FUTURE CODE REVISIONS

    input_json_path = f'{output_path}/{dir_pre}/board_placements.json'

    # Load existing data or create new list
    if os.path.exists(input_json_path):
        with open(input_json_path, "r") as f:
            try:
                data_list = json.load(f)
            except json.JSONDecodeError:
                data_list = {}
    else:
        data_list = {}

    # Append new entries
    for i, path in enumerate(image_paths):
        if path in data_list.keys():
            print(f"WARNING: File path '{path}' is already in the board_placements.json -> overwriting entry.")
        data_list[path] = {}
        data_list[path]["board"] = fen
        data_list[path]["cam"] = dict(camera_info[i])
        print(path, camera_info[i])
        print(path, data_list[path])

    # Save back to file
    with open(input_json_path, "w") as f:
        json.dump(data_list, f)
    write_time = time.time() - write_start
    print(f"💾 Writing completed in {write_time:.2f}s")
    
    # DYNAMIC GPU UTILIZATION OPTIMIZATION
    current_avg_util = sum(gpu_utilization_history) / len(gpu_utilization_history) if gpu_utilization_history else 0
    current_samples = bpy.context.scene.cycles.samples
    
    if current_avg_util < 60:  # Too low utilization
        new_samples = min(current_samples + 32, 512)
        bproc.renderer.set_max_amount_of_samples(new_samples)
        print(f" GPU util low ({current_avg_util:.1f}%) - increasing samples to {new_samples}")
    elif current_avg_util > 90:  # Too high utilization  
        new_samples = max(current_samples - 32, 128)
        bproc.renderer.set_max_amount_of_samples(new_samples)
        print(f" GPU util high ({current_avg_util:.1f}%) - reducing samples to {new_samples}")
    else:
        print(f" GPU utilization optimal: {current_avg_util:.1f}%")
    
    avg_time = (avg_time * z + time.time() - current_time) / (z + 1)
    step_time = time.time() - current_time
    print(f"⏱  Setup {z+1} completed in {step_time:.2f}s (render: {render_time:.2f}s, write: {write_time:.2f}s)")
    
    # Enhanced progress summary every 5 steps
    if (z + 1) % 5 == 0:
        total_images = (z + 1) * N
        total_elapsed = sum([avg_time * (i + 1) for i in range(z + 1)])
        images_per_hour = total_images / (total_elapsed / 3600) if total_elapsed > 0 else 0
        
        print(f" === PROGRESS CHECKPOINT {z+1}/{num_random_setup} ===")
        print(f" Generated: {total_images} images")
        print(f"⏱  Average time per setup: {avg_time:.2f}s")
        print(f" Current GPU utilization: {current_avg_util:.1f}%") 
        print(f"⚡ Images per hour: {images_per_hour:.0f}")
        print(f"🕐 Total elapsed: {total_elapsed/60:.1f} minutes")
        print("=== END CHECKPOINT ===")

total_images = num_random_setup * N
total_time_hours = (avg_time * num_random_setup) / 3600

# STOP GPU MONITORING
gpu_monitor_running = False
print("🛑 Stopping GPU monitoring...")

# FINAL GPU UTILIZATION SUMMARY
if gpu_utilization_history:
    final_avg_util = sum(gpu_utilization_history) / len(gpu_utilization_history)
    max_util = max(gpu_utilization_history)
    min_util = min(gpu_utilization_history)
    print(f" Final GPU Statistics:")
    print(f"   Average Utilization: {final_avg_util:.1f}%")
    print(f"   Peak Utilization: {max_util}%")
    print(f"   Minimum Utilization: {min_util}%")

print(f" ==== OPTIMIZED DATASET GENERATION COMPLETED ====")
print(f" Total Images Generated: {total_images:,}")
print(f"🔢 Total Setups: {num_random_setup}")  
print(f"📷 Images per Setup: {N}")
print(f"⏱  Total Time: {avg_time*num_random_setup:.2f} seconds ({total_time_hours:.1f} hours)")
print(f" Average Time per Setup: {avg_time:.2f} seconds")
print(f"️  Average Time per Image: {avg_time/N:.3f} seconds")
print(f"⚡ Images per Hour: {total_images/total_time_hours:.0f}")
print(f" Target GPU utilization: 70-80% (achieved: {final_avg_util:.1f}%)")
print(" ==== DATASET READY FOR TRAINING ====")
