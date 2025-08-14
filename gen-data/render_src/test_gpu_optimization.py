import blenderproc as bproc

"""
TEST VERSION of optimized reorder.py
- Only 5 setups with 3 cameras each (15 images total)  
- Designed to verify 70-80% GPU utilization
- Run this first to test before full generation
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

# Same GPU forcing function
def _force_cycles_gpu(device_type="CUDA"):
    print("=== FORCING GPU CONFIGURATION ===")
    bpy.context.scene.render.engine = "CYCLES"
    print(f"Set render engine to: {bpy.context.scene.render.engine}")
    bpy.context.scene.cycles.device = "GPU"
    print(f"Set cycles device to: {bpy.context.scene.cycles.device}")

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
            print(f"Found device: {name} ({dev_type})")
            
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
        _enable_all_cuda(prefs.devices)
    except Exception as e:
        print(f"Device config failed: {e}")

    os.environ["CYCLES_RENDER_DEVICE"] = device_type
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print("=== END GPU CONFIG ===")

# Initialize BlenderProc and GPU
bproc.init()
_force_cycles_gpu(device_type="CUDA")
bproc.renderer.set_cpu_threads(0)

# GPU MONITORING
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
                if len(gpu_utilization_history) > 20:
                    gpu_utilization_history.pop(0)
                
                avg_util = sum(gpu_utilization_history) / len(gpu_utilization_history)
                print(f"🔥 GPU: {gpu_util}% util, {gpu_temp}°C, {gpu_power}W (avg: {avg_util:.1f}%)")
                
        except Exception as e:
            print(f"GPU monitoring error: {e}")
        
        time.sleep(3)

# Start monitoring
gpu_monitor_thread = threading.Thread(target=monitor_gpu, daemon=True)
gpu_monitor_thread.start()
print("🔥 GPU TEST MONITORING STARTED")

# TEST CONFIGURATION - MINIMAL WORKLOAD
bproc.renderer.set_max_amount_of_samples(256)  # Start with 256 samples
bproc.renderer.set_denoiser(None)
bproc.renderer.set_noise_threshold(0.1)
bproc.renderer.set_light_bounces(12, 12, 12, 12, 24, 16, 4)

# Load scene
loaded = bproc.loader.load_blend("ChessBoard2.blend")
output_path = "gpu_test_" + datetime.datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
os.makedirs(output_path, exist_ok=True)

# TEST PARAMETERS
N = 3           # 3 cameras per setup
num_random_setup = 5  # Only 5 setups for testing

print(f"🧪 === GPU OPTIMIZATION TEST ===")
print(f"📸 Generating {num_random_setup} setups × {N} cameras = {N * num_random_setup} total images")
print(f"🎯 Target GPU utilization: 70-80%")
print(f"⚡ This test will run for ~2-3 minutes")

# Minimal setup - just test GPU utilization
light = bproc.types.Light()
light.set_location([0.0, 0.0, 2.0])
light.set_energy(500.0)
bproc.camera.set_resolution(640, 480)

# Simple camera setup
golden_angle = np.pi * (3 - np.sqrt(5))
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

# TEST RENDER LOOP
print("🚀 Starting GPU utilization test...")
start_time = time.time()

for z in range(num_random_setup):
    setup_start = time.time()
    
    # Show current GPU utilization
    current_util = gpu_utilization_history[-1] if gpu_utilization_history else 0
    avg_util = sum(gpu_utilization_history) / len(gpu_utilization_history) if gpu_utilization_history else 0
    
    print(f"\n⚡ === TEST RENDER {z+1}/{num_random_setup} ===")
    print(f"🔥 Current GPU: {current_util}% | Average: {avg_util:.1f}%")
    
    # Dynamic sample adjustment like in main script
    current_samples = bpy.context.scene.cycles.samples
    if avg_util < 60:
        new_samples = min(current_samples + 32, 512)
        bproc.renderer.set_max_amount_of_samples(new_samples)
        print(f"📈 Increasing samples to {new_samples} (low GPU util)")
    elif avg_util > 85:
        new_samples = max(current_samples - 32, 128)
        bproc.renderer.set_max_amount_of_samples(new_samples)
        print(f"📉 Reducing samples to {new_samples} (high GPU util)")
    
    # Render
    render_start = time.time()
    data = bproc.renderer.render()
    render_time = time.time() - render_start
    
    setup_time = time.time() - setup_start
    print(f"✅ Setup {z+1} complete in {setup_time:.2f}s (render: {render_time:.2f}s)")

# STOP MONITORING
gpu_monitor_running = False
total_time = time.time() - start_time

# FINAL RESULTS
if gpu_utilization_history:
    final_avg = sum(gpu_utilization_history) / len(gpu_utilization_history)
    max_util = max(gpu_utilization_history)
    min_util = min(gpu_utilization_history)
    
    print(f"\n🎉 === GPU OPTIMIZATION TEST COMPLETE ===")
    print(f"⏱️  Total test time: {total_time:.1f} seconds")
    print(f"📊 GPU Utilization Results:")
    print(f"   🎯 Target: 70-80%")
    print(f"   📈 Average: {final_avg:.1f}%")
    print(f"   🔥 Peak: {max_util}%")
    print(f"   📉 Minimum: {min_util}%")
    
    if 70 <= final_avg <= 80:
        print(f"✅ SUCCESS: GPU utilization is in target range!")
    elif final_avg < 70:
        print(f"⚠️  LOW: GPU utilization below target (increase samples/complexity)")
    else:
        print(f"🔥 HIGH: GPU utilization above target (decrease samples)")
        
    print(f"🚀 Optimization test complete - ready for full generation!")

else:
    print("❌ No GPU data collected during test")