#!/bin/bash
# QLoRA Fine-Tuning Launcher
# Run this script to start the fine-tuning process

set -e

PROJECT_DIR="/home/yapbenzet/Desktop/TurkishMedLLM Project/TurkishMedLLM"
cd "$PROJECT_DIR"

echo "=========================================="
echo "QLoRA Fine-Tuning Launcher"
echo "=========================================="
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Virtual environment not found!"
    echo "Please run: python3 -m venv venv"
    exit 1
fi

# Activate virtual environment
source venv/bin/activate
echo "Virtual environment activated"
echo ""

# Check if Ollama is running (optional - for inference testing later)
echo "📋 Checking system status..."
echo ""

# Check GPU
python3 << 'EOF'
import torch
print(f"GPU Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   Device: {torch.cuda.get_device_name(0)}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
EOF

echo ""
echo "=========================================="
echo "Starting QLoRA Fine-Tuning"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  • Base Model: Qwen/Qwen2-7B"
echo "  • Training Method: QLoRA (4-bit quantized)"
echo "  • Training Data: 210,791 SFT samples"
echo "  • Epochs: 10"
echo "  • Batch Size: 8 (with 4x gradient accumulation)"
echo "  • Learning Rate: 2e-4"
echo "  • LoRA Rank: 16"
echo "  • Estimated Time: 12-18 hours"
echo ""

# Start training
echo "Starting training process..."
echo ""
python3 1_finetune_qlora.py

echo ""
echo "=========================================="
echo "Fine-Tuning Complete!"
echo "=========================================="
