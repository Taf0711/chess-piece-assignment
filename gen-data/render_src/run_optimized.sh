#!/bin/bash

echo "🚀 RTX 5090 OPTIMIZED CHESS DATA GENERATION"
echo "============================================"
echo ""
echo "Available options:"
echo "1. Test GPU optimization (5 setups × 3 cameras = 15 images)"
echo "2. Full generation (200 setups × 5 cameras = 1000 images)"
echo ""
echo "The test will verify 70-80% GPU utilization before running full generation."
echo ""

read -p "Choose option (1 or 2): " choice

case $choice in
    1)
        echo ""
        echo "🧪 Running GPU optimization test..."
        echo "This will generate 15 images to test GPU utilization."
        echo ""
        python3 test_gpu_optimization.py
        ;;
    2)
        echo ""
        echo "⚠️  WARNING: Full generation will create 1000 images!"
        echo "This may take several hours with high GPU utilization."
        echo ""
        read -p "Continue? (y/N): " confirm
        if [[ $confirm == [yY] || $confirm == [yY][eE][sS] ]]; then
            echo "🚀 Starting full optimized generation..."
            python3 reorder.py
        else
            echo "❌ Cancelled."
        fi
        ;;
    *)
        echo "❌ Invalid choice. Use 1 or 2."
        ;;
esac