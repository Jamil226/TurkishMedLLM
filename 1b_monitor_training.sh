#!/bin/bash
# TRAINING MONITOR SCRIPT
# Real-time monitoring of QLoRA fine-tuning progress

TRAINING_LOG="qlora_training.log"
CHECKPOINT_DIR="./qlora_checkpoints"

echo "QLoRA Training Monitor"
echo "======================================"
echo "Started: $(date)"
echo ""

# Monitor function
monitor_training() {
    clear
    echo "QLoRA TRAINING STATUS"
    echo "======================================"
    echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""
    
    # Check if training is running
    if pgrep -f "1_finetune_qlora.py" > /dev/null; then
        echo "Training Process: RUNNING"
    else
        echo "Training Process: STOPPED"
    fi
    
    echo ""
    echo "📈 GPU Status:"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,utilization.memory --format=csv,noheader | while read line; do
        echo "   $line"
    done
    
    echo ""
    echo "📁 Checkpoint Status:"
    if [ -d "$CHECKPOINT_DIR" ]; then
        checkpoint_count=$(find "$CHECKPOINT_DIR" -maxdepth 1 -type d -name "checkpoint-*" | wc -l)
        epoch_count=$(find "$CHECKPOINT_DIR" -maxdepth 1 -type d -name "epoch_*" | wc -l)
        echo "   Checkpoints: $checkpoint_count"
        echo "   Epochs completed: $epoch_count"
        
        if [ $epoch_count -gt 0 ]; then
            latest_epoch=$(ls -td "$CHECKPOINT_DIR"/epoch_* 2>/dev/null | head -1)
            if [ -n "$latest_epoch" ]; then
                size=$(du -sh "$latest_epoch" 2>/dev/null | cut -f1)
                echo "   Latest epoch: $(basename $latest_epoch) ($size)"
            fi
        fi
    else
        echo "No checkpoints yet (still initializing)"
    fi
    
    echo ""
    echo "Recent Log Entries:"
    if [ -f "$TRAINING_LOG" ]; then
        tail -5 "$TRAINING_LOG" | sed 's/^/   /'
    fi
    
    echo ""
    echo "======================================"
    echo "Press Ctrl+C to exit"
}

# Continuous monitoring
while true; do
    monitor_training
    sleep 30
done
