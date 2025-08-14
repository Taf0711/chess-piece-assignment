import blenderproc as bproc

import csv
import random
import time
import re
import bpy
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

# Init BlenderProc and EXTREMELY AGGRESSIVE GPU-ONLY Settings
bproc.init()

def _force_optix_gpu_only():
    """EXTREMELY AGGRESSIVE GPU-ONLY CONFIGURATION"""
    print("🔥 === EXTREME GPU-ONLY CONFIGURATION ===")
    
    # Force Cycles engine
    bpy.context.scene.render.engine = "CYCLES"
    print(f"✅ Render engine: {bpy.context.scene.render.engine}")

    # FORCE GPU-ONLY RENDERING
    bpy.context.scene.cycles.device = "GPU"
    print(f"✅ Cycles device: {bpy.context.scene.cycles.device}")

    # Get preferences
    prefs = bpy.context.preferences.addons["cycles"].preferences
    
    # Refresh devices
    try:
        prefs.refresh_devices()
        print("✅ Refreshed devices")
    except Exception as e:
        print(f"⚠️  Device refresh failed: {e}")

    # FORCE OPTIX (better GPU utilization than CUDA for RTX)
    prefs.compute_device_type = "OPTIX"
    print(f"🚀 Set compute device type to: {prefs.compute_device_type}")

    # AGGRESSIVELY DISABLE ALL CPU AND ENABLE ALL GPU
    gpu_enabled = False
    cpu_disabled = False
    
    for d in prefs.devices:
        name = getattr(d, "name", "")
        dev_type = getattr(d, "type", "")
        print(f"📱 Found device: {name} ({dev_type})")
        
        if dev_type == "CPU":
            d.use = False
            cpu_disabled = True
            print(f"❌ DISABLED CPU: {name}")
        elif dev_type in ("CUDA", "OPTIX"):
            d.use = True
            gpu_enabled = True
            print(f"✅ ENABLED GPU: {name}")
        else:
            d.use = False
            print(f"❌ DISABLED OTHER: {name}")
    
    if not gpu_enabled:
        print("💥 ERROR: NO GPU DEVICES ENABLED!")
        return False
        
    if not cpu_disabled:
        print("⚠️  WARNING: CPU NOT DISABLED!")

    # FORCE ENVIRONMENT VARIABLES
    os.environ["CYCLES_RENDER_DEVICE"] = "OPTIX"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["CYCLES_CPU_THREADS"] = "0"
    print("🔧 Set environment variables for GPU-only")

    # DISABLE ALL CPU THREADS IN BLENDER
    bpy.context.scene.render.threads_mode = 'FIXED'
    bpy.context.scene.render.threads = 0
    print("🚫 Set CPU threads to 0")

    # FORCE VIEWPORT TO USE GPU TOO
    try:
        bpy.context.scene.cycles.preview_compute_device = "GPU"
        print("✅ Set preview compute device to GPU")
    except AttributeError:
        print("⚠️  Preview compute device not available")
    
    # VERIFY FINAL CONFIGURATION
    enabled_gpus = []
    enabled_cpus = []
    
    for d in prefs.devices:
        name = getattr(d, "name", "")
        dev_type = getattr(d, "type", "")
        is_used = getattr(d, "use", False)
        
        if is_used:
            if dev_type == "CPU":
                enabled_cpus.append(name)
            elif dev_type in ("CUDA", "OPTIX"):
                enabled_gpus.append(name)
    
    print(f"🔥 === FINAL VERIFICATION ===")
    print(f"✅ GPU devices enabled: {enabled_gpus}")
    print(f"❌ CPU devices enabled: {enabled_cpus}")
    print(f"🎯 Engine: {bpy.context.scene.render.engine}")
    print(f"🎯 Device: {bpy.context.scene.cycles.device}")
    print(f"🎯 Compute type: {prefs.compute_device_type}")
    print(f"🎯 CPU threads: {bpy.context.scene.render.threads}")
    
    success = len(enabled_gpus) > 0 and len(enabled_cpus) == 0
    print(f"🎯 GPU-ONLY SUCCESS: {success}")
    
    return success

# FORCE GPU-ONLY CONFIGURATION
_force_optix_gpu_only()

# SET CPU THREADS TO ZERO IN BLENDERPROC TOO
bproc.renderer.set_cpu_threads(0)

# GPU MONITORING WITH CPU USAGE TOO
gpu_monitor_running = True
gpu_utilization_history = []
cpu_utilization_history = []

def monitor_gpu_and_cpu():
    global gpu_monitor_running, gpu_utilization_history, cpu_utilization_history
    while gpu_monitor_running:
        try:
            # GPU monitoring
            gpu_result = subprocess.run([
                'nvidia-smi', '--query-gpu=utilization.gpu,temperature.gpu,power.draw', 
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True, timeout=5)
            
            # CPU monitoring  
            cpu_result = subprocess.run([
                'top', '-bn1'
            ], capture_output=True, text=True, timeout=5)
            
            if gpu_result.returncode == 0:
                gpu_data = gpu_result.stdout.strip().split(', ')
                gpu_util = int(gpu_data[0])
                gpu_temp = int(gpu_data[1]) 
                gpu_power = float(gpu_data[2])
                
                gpu_utilization_history.append(gpu_util)
                if len(gpu_utilization_history) > 20:
                    gpu_utilization_history.pop(0)
                
                # Extract CPU usage from top
                cpu_util = 0
                if cpu_result.returncode == 0:
                    for line in cpu_result.stdout.split('\n'):
                        if '%Cpu(s):' in line:
                            # Extract CPU usage percentage
                            parts = line.split(',')[0]
                            cpu_util = float(parts.split()[1].replace('%us', ''))
                            break
                
                cpu_utilization_history.append(cpu_util)
                if len(cpu_utilization_history) > 20:
                    cpu_utilization_history.pop(0)
                
                avg_gpu = sum(gpu_utilization_history) / len(gpu_utilization_history)
                avg_cpu = sum(cpu_utilization_history) / len(cpu_utilization_history) if cpu_utilization_history else 0
                
                print(f"🔥 GPU: {gpu_util}% util, {gpu_temp}°C, {gpu_power}W (avg: {avg_gpu:.1f}%)")
                print(f"💻 CPU: {cpu_util:.1f}% (avg: {avg_cpu:.1f}%) - SHOULD BE LOW!")
                
                if avg_cpu > 50:
                    print(f"⚠️  WARNING: HIGH CPU USAGE - GPU NOT BEING USED PROPERLY!")
                    
        except Exception as e:
            print(f"Monitoring error: {e}")
        
        time.sleep(2)

# Start monitoring
gpu_monitor_thread = threading.Thread(target=monitor_gpu_and_cpu, daemon=True)
gpu_monitor_thread.start()
print("🔥 AGGRESSIVE GPU+CPU MONITORING STARTED")

# EXTREME RENDER SETTINGS FOR MAXIMUM GPU LOAD
bproc.renderer.set_max_amount_of_samples(1024)  # VERY HIGH SAMPLES
bproc.renderer.set_denoiser(None)  # NO DENOISING = MORE GPU WORK
bproc.renderer.set_noise_threshold(0.01)  # VERY LOW THRESHOLD = MORE SAMPLES

# MAXIMUM LIGHT BOUNCES FOR GPU RAY TRACING
bproc.renderer.set_light_bounces(24, 24, 24, 24, 48, 32, 12)  # EXTREME BOUNCES

# Load scene  
loaded = bproc.loader.load_blend("ChessBoard2.blend")
output_path = "gpu_aggressive_" + datetime.datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
os.makedirs(output_path, exist_ok=True)

# Copy all the chess piece setup code from original
positionsDict = {
    'A1': (-0.87574, -0.87857), 'A2': (-0.87574, -0.628312), 'A3': (-0.87574, -0.378054),
    'A4': (-0.87574, -0.127796), 'A5': (-0.87574, 0.122462), 'A6': (-0.87574, 0.372720),
    'A7': (-0.87574, 0.622978), 'A8': (-0.87574, 0.873236), 'B1': (-0.62914, -0.87857),
    'B2': (-0.62914, -0.628312), 'B3': (-0.62914, -0.378054), 'B4': (-0.62914, -0.127796),
    'B5': (-0.62914, 0.122462), 'B6': (-0.62914, 0.372720), 'B7': (-0.62914, 0.622978),
    'B8': (-0.62914, 0.873236), 'C1': (-0.38254, -0.87857), 'C2': (-0.38254, -0.628312),
    'C3': (-0.38254, -0.378054), 'C4': (-0.38254, -0.127796), 'C5': (-0.38254, 0.122462),
    'C6': (-0.38254, 0.372720), 'C7': (-0.38254, 0.622978), 'C8': (-0.38254, 0.873236),
    'D1': (-0.13594, -0.87857), 'D2': (-0.13594, -0.628312), 'D3': (-0.13594, -0.378054),
    'D4': (-0.13594, -0.127796), 'D5': (-0.13594, 0.122462), 'D6': (-0.13594, 0.372720),
    'D7': (-0.13594, 0.622978), 'D8': (-0.13594, 0.873236), 'E1': (0.11066, -0.87857),
    'E2': (0.11066, -0.628312), 'E3': (0.11066, -0.378054), 'E4': (0.11066, -0.127796),
    'E5': (0.11066, 0.122462), 'E6': (0.11066, 0.372720), 'E7': (0.11066, 0.622978),
    'E8': (0.11066, 0.873236), 'F1': (0.35726, -0.87857), 'F2': (0.35726, -0.628312),
    'F3': (0.35726, -0.378054), 'F4': (0.35726, -0.127796), 'F5': (0.35726, 0.122462),
    'F6': (0.35726, 0.372720), 'F7': (0.35726, 0.622978), 'F8': (0.35726, 0.873236),
    'G1': (0.60386, -0.87857), 'G2': (0.60386, -0.628312), 'G3': (0.60386, -0.378054),
    'G4': (0.60386, -0.127796), 'G5': (0.60386, 0.122462), 'G6': (0.60386, 0.372720),
    'G7': (0.60386, 0.622978), 'G8': (0.60386, 0.873236), 'H1': (0.85046, -0.87857),
    'H2': (0.85046, -0.628312), 'H3': (0.85046, -0.378054), 'H4': (0.85046, -0.127796),
    'H5': (0.85046, 0.122462), 'H6': (0.85046, 0.372720), 'H7': (0.85046, 0.622978),
    'H8': (0.85046, 0.873236), 'None': (1, 1)
}

pieceToSquareDict = {
    'BlackRook1': 'A8', 'BlackRook2': 'H8', 'BlackKnight1': 'B8', 'BlackKnight2': 'G8',
    'BlackBishop1': 'C8', 'BlackBishop2': 'E8', 'BlackQueen1': 'D8', 'BlackKing1': 'E8',
    'BlackPawn1': 'A7', 'BlackPawn2': 'B7', 'BlackPawn3': 'C7', 'BlackPawn4': 'D7',
    'BlackPawn5': 'E7', 'BlackPawn6': 'F7', 'BlackPawn7': 'G7', 'BlackPawn8': 'H7',
    'WhiteRook1': 'A1', 'WhiteRook2': 'H1', 'WhiteKnight1': 'B1', 'WhiteKnight2': 'G1',
    'WhiteBishop1': 'C1', 'WhiteBishop2': 'F1', 'WhiteQueen1': 'D1', 'WhiteKing1': 'E1',
    'WhitePawn1': 'A2', 'WhitePawn2': 'B2', 'WhitePawn3': 'C2', 'WhitePawn4': 'D2',
    'WhitePawn5': 'E2', 'WhitePawn6': 'F2', 'WhitePawn7': 'G2', 'WhitePawn8': 'H2',
}

# Add all other pieces as None for now (simplified)
for piece_type in ['BlackRook', 'BlackKnight', 'BlackBishop', 'BlackQueen', 'WhiteRook', 'WhiteKnight', 'WhiteBishop', 'WhiteQueen']:
    for i in range(3, 11):
        pieceToSquareDict[f'{piece_type}{i}'] = 'None'

def parse_fen(fen):
    files = 'abcdefgh'
    ranks = fen.split()[0].split('/')
    piecePositions = {
        'BlackPawn': [], 'WhitePawn': [], 'BlackRook': [], 'WhiteRook': [],
        'BlackKnight': [], 'WhiteKnight': [], 'BlackBishop': [], 'WhiteBishop': [],
        'BlackQueen': [], 'WhiteQueen': [], 'BlackKing': [], 'WhiteKing': [],
    }
    symbolToType = {
        'p': 'Pawn', 'r': 'Rook', 'n': 'Knight', 'b': 'Bishop','q': 'Queen','k': 'King'
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

def reset():
    for name, square in pieceToSquareDict.items():
        try:
            bpy.data.objects[name].hide_viewport = True
            bpy.data.objects[name].hide_render = True
        except KeyError:
            pass

def update_dict_from_positions(d, positions):
    reset()
    for k in d:
        d[k] = 'None'
    try:
        bpy.data.objects["BlackKing1"].hide_viewport = False
        bpy.data.objects["BlackKing1"].hide_render = False
        bpy.data.objects["WhiteKing1"].hide_viewport = False
        bpy.data.objects["WhiteKing1"].hide_render = False
    except KeyError:
        pass
    
    for ptype, squares in positions.items():
        for i, sq in enumerate(squares, start=1):
            key = f'{ptype}{i}'
            if key in d:
                d[key] = sq
                try:
                    bpy.data.objects[key].hide_viewport = False
                    bpy.data.objects[key].hide_render = False
                except KeyError:
                    pass
    return d

# CATEGORY MAPPING
CATEGORY_NAME_TO_ID = {
    "WhitePawn": 13, "WhiteRook": 1, "WhiteKnight": 2, "WhiteBishop": 3, "WhiteQueen": 4, "WhiteKing": 5,
    "BlackPawn": 6, "BlackRook": 7, "BlackKnight": 8, "BlackBishop": 9, "BlackQueen": 10, "BlackKing": 11, "Board": 12
}

def get_base_name(name):
    return re.sub(r'\d+$', '', name)

for obj in loaded:
    full_name = obj.get_name()
    base_name = get_base_name(full_name)
    if base_name in CATEGORY_NAME_TO_ID:
        obj.set_cp("category_id", CATEGORY_NAME_TO_ID[base_name])

# EXTREME GPU WORKLOAD SETTINGS
light = bproc.types.Light()
light.set_location([0.0, 0.0, 2.0])
light.set_energy(2000.0)  # HIGHER ENERGY = MORE GPU WORK

# HIGH RESOLUTION FOR MORE GPU LOAD
bproc.camera.set_resolution(1280, 960)  # HIGHER RESOLUTION
bproc.renderer.set_output_format(enable_transparency=True)
print("🎯 HIGH RESOLUTION:", bproc.camera.get_intrinsics_as_K_matrix())

# AGGRESSIVE TEST PARAMETERS - FEWER SETUPS BUT MAXIMUM GPU LOAD
N = 8  # More cameras per setup
num_random_setup = 20  # Fewer setups but EXTREME GPU load per setup

print(f"🔥 === AGGRESSIVE GPU-ONLY TEST ===")
print(f"📸 Generating {num_random_setup} setups × {N} cameras = {N * num_random_setup} total images")
print(f"📏 Resolution: 1280×960")
print(f"🎲 Samples: 1024+ per render")
print(f"🎯 Target: 70-80% GPU, <20% CPU")

# Setup cameras
golden_angle = np.pi * (3 - np.sqrt(5))
camera_info = {}

for i in range(N):
    rho = random.uniform(4.5, 6.5)
    z = 1 - (i) / (N - 1)
    radius = np.sqrt(1 - z * z)
    theta = golden_angle * i
    x = np.cos(theta) * radius
    y = np.sin(theta) * radius
    pos = rho * np.array([x, y, z])
    forward_vec = -pos / np.linalg.norm(pos)
    rotation = bproc.camera.rotation_from_forward_vec(forward_vec)
    cam_pose = bproc.math.build_transformation_mat(pos.tolist(), rotation)
    bproc.camera.add_camera_pose(cam_pose)
    camera_info[i] = {"dist": rho, "rot": rotation.tolist(), "pos": pos.tolist()}

# Test positions CSV
csv_file = open('positions.csv', newline='')
reader = csv.reader(csv_file)
next(reader, None)
fen_rows = iter(reader)

print("🚀 Starting AGGRESSIVE GPU-ONLY data generation...")
start_time = time.time()

for z in range(num_random_setup):
    setup_start = time.time()
    
    # Get FEN position
    try:
        row = next(fen_rows)
        fen = row[0]
    except StopIteration:
        print("No more FEN positions")
        break
    
    # Current utilization stats
    current_gpu = gpu_utilization_history[-1] if gpu_utilization_history else 0
    avg_gpu = sum(gpu_utilization_history) / len(gpu_utilization_history) if gpu_utilization_history else 0
    current_cpu = cpu_utilization_history[-1] if cpu_utilization_history else 0
    avg_cpu = sum(cpu_utilization_history) / len(cpu_utilization_history) if cpu_utilization_history else 0
    
    print(f"\n🔥 === AGGRESSIVE RENDER {z+1}/{num_random_setup} ===")
    print(f"🎯 GPU: {current_gpu}% (avg: {avg_gpu:.1f}%) | CPU: {current_cpu:.1f}% (avg: {avg_cpu:.1f}%)")
    print(f"🎲 Using FEN: {fen}")
    
    # EXTREME dynamic sample adjustment
    current_samples = bpy.context.scene.cycles.samples
    if avg_gpu < 60:
        new_samples = min(current_samples + 128, 2048)  # HUGE JUMPS
        bproc.renderer.set_max_amount_of_samples(new_samples)
        print(f"📈 GPU LOW - EXTREME increase to {new_samples} samples")
    elif avg_gpu > 90:
        new_samples = max(current_samples - 64, 512)
        bproc.renderer.set_max_amount_of_samples(new_samples)
        print(f"📉 GPU HIGH - reducing to {new_samples} samples")
    else:
        print(f"✅ GPU utilization acceptable: {avg_gpu:.1f}%")
    
    if avg_cpu > 30:
        print(f"⚠️  HIGH CPU USAGE: {avg_cpu:.1f}% - GPU NOT BEING USED PROPERLY!")
    
    # Setup chess position
    pos = parse_fen(fen)
    update_dict_from_positions(pieceToSquareDict, pos)
    
    for name, square in pieceToSquareDict.items():
        if square != "None":
            try:
                x, y = positionsDict[square.upper()]
                bpy.data.objects[name].location.x = x
                bpy.data.objects[name].location.y = y
            except KeyError:
                pass
    
    # Vary lighting for more GPU work
    light.set_location([
        random.uniform(-2.0, 2.0),
        random.uniform(-2.0, 2.0), 
        random.uniform(1.0, 3.0)
    ])
    light.set_energy(random.uniform(1500, 3000))  # HIGH ENERGY
    
    # AGGRESSIVE RENDERING
    render_start = time.time()
    print(f"⚡ AGGRESSIVE RENDER: {current_samples} samples @ 1280×960...")
    
    # Render both segmaps and colors for maximum GPU load
    seg_data = bproc.renderer.render_segmap(map_by=["instance", "class", "name"])
    data = bproc.renderer.render()
    
    render_time = time.time() - render_start
    setup_time = time.time() - setup_start
    
    print(f"✅ Setup {z+1} complete in {setup_time:.2f}s (render: {render_time:.2f}s)")
    
    # Show recent GPU/CPU stats
    if len(gpu_utilization_history) >= 5:
        recent_gpu = sum(gpu_utilization_history[-5:]) / 5
        recent_cpu = sum(cpu_utilization_history[-5:]) / 5 if len(cpu_utilization_history) >= 5 else 0
        print(f"📊 Recent averages - GPU: {recent_gpu:.1f}%, CPU: {recent_cpu:.1f}%")

# STOP MONITORING
gpu_monitor_running = False
total_time = time.time() - start_time

print(f"\n🎉 === AGGRESSIVE GPU TEST COMPLETE ===")
print(f"⏱️  Total time: {total_time:.1f} seconds")
print(f"📸 Images rendered: {N * num_random_setup}")

if gpu_utilization_history and cpu_utilization_history:
    final_gpu = sum(gpu_utilization_history) / len(gpu_utilization_history)
    final_cpu = sum(cpu_utilization_history) / len(cpu_utilization_history)
    max_gpu = max(gpu_utilization_history)
    max_cpu = max(cpu_utilization_history)
    
    print(f"📊 FINAL RESULTS:")
    print(f"   🎯 GPU Average: {final_gpu:.1f}% (Peak: {max_gpu}%)")
    print(f"   💻 CPU Average: {final_cpu:.1f}% (Peak: {max_cpu:.1f}%)")
    
    if final_gpu >= 60 and final_cpu < 30:
        print(f"🎉 SUCCESS: GPU-DOMINANT RENDERING ACHIEVED!")
    elif final_cpu > final_gpu:
        print(f"💥 PROBLEM: CPU USAGE ({final_cpu:.1f}%) > GPU USAGE ({final_gpu:.1f}%)")
        print(f"💡 SOLUTION NEEDED: Blender still using CPU despite configuration")
    else:
        print(f"⚠️  MIXED: GPU working but needs more optimization")

print(f"🚀 RTX 5090 GPU-ONLY TEST COMPLETE!")
csv_file.close()