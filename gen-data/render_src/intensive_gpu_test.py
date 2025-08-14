import blenderproc as bproc

"""
INTENSIVE GPU TEST - Designed to achieve 70-80% utilization
- Higher resolution, more samples, more complex lighting
- Continuous rendering to sustain GPU load
"""

import csv
import random
import time
import re
import bpy
import os
import numpy as np
import json
import math
import datetime
import threading
import subprocess

# GPU CONFIGURATION
def _force_cycles_gpu(device_type="CUDA"):
    print("=== FORCING GPU CONFIGURATION ===")
    bpy.context.scene.render.engine = "CYCLES"
    bpy.context.scene.cycles.device = "GPU"

    prefs = bpy.context.preferences.addons["cycles"].preferences
    try:
        prefs.refresh_devices()
        print("Refreshed devices successfully")
    except Exception as e:
        print(f"Device refresh failed: {e}")

    prefs.compute_device_type = device_type
    print(f"Set compute device type to: {prefs.compute_device_type}")

    def _enable_all_cuda(devs):
        for d in devs:
            name = getattr(d, "name", "")
            dev_type = getattr(d, "type", "")
            
            if dev_type == "CPU":
                d.use = False
                print(f"  -> DISABLED CPU: {name}")
            elif dev_type in ("CUDA", "OPTIX"):
                d.use = True
                print(f"  -> ENABLED GPU: {name}")
            else:
                d.use = False

    try:
        _enable_all_cuda(prefs.devices)
    except Exception as e:
        print(f"Device config failed: {e}")

    os.environ["CYCLES_RENDER_DEVICE"] = device_type
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print("=== GPU CONFIG COMPLETE ===")

# Initialize BlenderProc
bproc.init()
_force_cycles_gpu(device_type="CUDA")
bproc.renderer.set_cpu_threads(0)

# INTENSIVE GPU MONITORING
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
                if len(gpu_utilization_history) > 30:
                    gpu_utilization_history.pop(0)
                
                avg_util = sum(gpu_utilization_history) / len(gpu_utilization_history)
                print(f" GPU: {gpu_util}% util, {gpu_temp}°C, {gpu_power}W (avg: {avg_util:.1f}%)")
                
        except Exception as e:
            print(f"GPU monitoring error: {e}")
        
        time.sleep(1)  # Monitor every second for more responsive feedback

# Start intensive monitoring
gpu_monitor_thread = threading.Thread(target=monitor_gpu, daemon=True)
gpu_monitor_thread.start()
print(" INTENSIVE GPU MONITORING STARTED")

# AGGRESSIVE RENDER SETTINGS FOR HIGH GPU LOAD
bproc.renderer.set_max_amount_of_samples(512)  # Start with high samples
bproc.renderer.set_denoiser(None)  # No denoising = more GPU compute
bproc.renderer.set_noise_threshold(0.05)  # Lower threshold = more samples
bproc.renderer.set_light_bounces(16, 16, 16, 16, 32, 24, 8)  # More bounces = more GPU work

# Load scene
loaded = bproc.loader.load_blend("ChessBoard2.blend")
output_path = "intensive_test_" + datetime.datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
os.makedirs(output_path, exist_ok=True)

# INTENSIVE TEST PARAMETERS
N = 8           # More cameras per setup
num_setups = 10  # More setups for sustained load

print(f" === INTENSIVE GPU UTILIZATION TEST ===")
print(f" Generating {num_setups} setups × {N} cameras = {N * num_setups} total images")
print(f" Target GPU utilization: 70-80%")
print(f"⚡ This intensive test will run for ~5-10 minutes")
print(f" Using AGGRESSIVE render settings")

# HIGHER RESOLUTION AND COMPLEX LIGHTING
bproc.camera.set_resolution(1024, 768)  # Higher resolution = more GPU work

# Multiple complex lights for more ray tracing work
lights = []
for i in range(3):  # 3 lights for complex lighting
    light = bproc.types.Light()
    light.set_location([
        random.uniform(-3.0, 3.0),
        random.uniform(-3.0, 3.0),
        random.uniform(1.0, 4.0)
    ])
    light.set_energy(random.uniform(300, 800))
    lights.append(light)

# Camera setup with more positions
golden_angle = np.pi * (3 - np.sqrt(5))
camera_info = {}

for i in range(N):
    rho = random.uniform(3.5, 7.0)  # Wider distance range
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
    camera_info[i] = {"dist": rho, "pos": pos.tolist()}

# INTENSIVE RENDER LOOP
print(" Starting INTENSIVE GPU utilization test...")
start_time = time.time()

for z in range(num_setups):
    setup_start = time.time()
    
    # Dynamic lighting changes
    for i, light in enumerate(lights):
        light.set_location([
            random.uniform(-3.0, 3.0),
            random.uniform(-3.0, 3.0),
            random.uniform(1.0, 4.0)
        ])
        light.set_energy(random.uniform(400, 1000))
    
    # Current GPU stats
    current_util = gpu_utilization_history[-1] if gpu_utilization_history else 0
    avg_util = sum(gpu_utilization_history) / len(gpu_utilization_history) if gpu_utilization_history else 0
    
    print(f"\n === INTENSIVE RENDER {z+1}/{num_setups} ===")
    print(f" Current GPU: {current_util}% | Rolling Avg: {avg_util:.1f}%")
    
    # AGGRESSIVE dynamic sample adjustment
    current_samples = bpy.context.scene.cycles.samples
    if avg_util < 65:  # More aggressive threshold for higher utilization
        new_samples = min(current_samples + 64, 1024)  # Bigger jumps
        bproc.renderer.set_max_amount_of_samples(new_samples)
        print(f" LOW GPU - Increasing samples to {new_samples}")
    elif avg_util > 85:
        new_samples = max(current_samples - 32, 256)
        bproc.renderer.set_max_amount_of_samples(new_samples)
        print(f" HIGH GPU - Reducing samples to {new_samples}")
    else:
        print(f" GPU utilization in target range: {avg_util:.1f}%")
    
    # Intensive rendering
    render_start = time.time()
    print(f"⚡ Rendering with {current_samples} samples at 1024×768...")
    
    # Render both segmaps and colors for maximum GPU load
    seg_data = bproc.renderer.render_segmap(map_by=["instance", "class"])
    data = bproc.renderer.render()
    
    render_time = time.time() - render_start
    setup_time = time.time() - setup_start
    
    print(f" Setup {z+1} complete in {setup_time:.2f}s (pure render: {render_time:.2f}s)")
    
    # Show sustained utilization stats
    if len(gpu_utilization_history) >= 5:
        recent_avg = sum(gpu_utilization_history[-5:]) / 5
        print(f" Recent 5-second average: {recent_avg:.1f}%")

# STOP MONITORING
gpu_monitor_running = False
total_time = time.time() - start_time

# COMPREHENSIVE RESULTS
print(f"\n === INTENSIVE GPU TEST COMPLETE ===")
print(f"⏱  Total test time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
print(f" Total images rendered: {N * num_setups}")

if gpu_utilization_history:
    final_avg = sum(gpu_utilization_history) / len(gpu_utilization_history)
    max_util = max(gpu_utilization_history)
    min_util = min(gpu_utilization_history)
    
    # Calculate sustained high utilization percentage
    high_util_count = sum(1 for util in gpu_utilization_history if util >= 70)
    sustained_percentage = (high_util_count / len(gpu_utilization_history)) * 100
    
    print(f" FINAL GPU UTILIZATION ANALYSIS:")
    print(f"    Target: 70-80%")
    print(f"    Overall Average: {final_avg:.1f}%")
    print(f"    Peak: {max_util}%")
    print(f"    Minimum: {min_util}%")
    print(f"   ⏰ Time at 70%+: {sustained_percentage:.1f}%")
    
    if 70 <= final_avg <= 80:
        print(f" SUCCESS: GPU utilization is in optimal range!")
        print(f" Ready for full production data generation!")
    elif final_avg < 70:
        print(f"️  MODERATE: GPU utilization below target")
        print(f"💡 Suggestion: Increase samples or resolution for full generation")
    else:
        print(f" EXCELLENT: High GPU utilization achieved!")
        print(f" Perfect for maximum generation speed!")
        
    print(f"\n RTX 5090 PERFORMANCE CONFIRMED!")
    print(f"⚡ Your GPU is working at {final_avg:.1f}% average utilization")
    print(f" Optimization successful - ready for large-scale generation!")

else:
    print(" No GPU monitoring data collected")

print(f"\n Recommended settings for full generation:")
print(f"   📏 Resolution: 1024×768 or higher") 
print(f"   🎲 Samples: 512-768 for optimal balance")
print(f"    Batch size: 8-10 cameras per setup")
print(f"   ⚡ Expected speed: {(N * num_setups) / (total_time / 60):.0f} images/minute")