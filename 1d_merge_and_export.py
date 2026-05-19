#!/usr/bin/env python3
"""
TurkishMedLLM LoRA Weight Merging and Export Script
This script merges the trained QLoRA adapter checkpoints with the base Qwen3-8B model.
Run this script on a machine with sufficient VRAM/RAM (e.g., Linux GPU server).
"""

import os
import torch
import argparse
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def merge_lora_weights(base_model_name: str, adapter_dir: str, output_dir: str):
    print("====================================================================")
    print("STARTING LORA WEIGHT MERGING PIPELINE")
    print("====================================================================")
    print(f"Base Model: {base_model_name}")
    print(f"Adapter Dir: {adapter_dir}")
    print(f"Output Dir: {output_dir}")
    print("--------------------------------------------------------------------")

    # Ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 1. Load Tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    tokenizer.save_pretrained(output_dir)
    print("Tokenizer loaded and saved.")

    # 2. Load Base Model
    print("Loading base model (this requires ~16GB RAM/VRAM)...")
    device_map = "auto" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() or torch.backends.mps.is_available() else torch.float32
    
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map=device_map,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
    )
    print("Base model loaded successfully.")

    # 3. Load PEFT/LoRA Model
    print("Loading PEFT/LoRA adapter...")
    model = PeftModel.from_pretrained(
        base_model,
        adapter_dir,
        torch_dtype=torch_dtype,
    )
    print("PEFT adapter loaded.")

    # 4. Merge weights
    print("Merging weights (this might take a minute)...")
    merged_model = model.merge_and_unload()
    print("Weights merged successfully.")

    # 5. Save Merged Model
    print(f"Saving merged model to {output_dir}...")
    merged_model.save_pretrained(output_dir, max_shard_size="2GB")
    print("Merged model saved successfully!")
    print("====================================================================")
    print("Merged model is ready for GGUF conversion or deployment!")
    print("====================================================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge LoRA weights with base model.")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen3-8B", help="Base model name on HF Hub")
    parser.add_argument("--adapter_dir", type=str, default="./qlora_checkpoints", help="Path to adapter checkpoints")
    parser.add_argument("--output_dir", type=str, default="./qwen3_8b_merged", help="Path to save merged model")
    
    args = parser.parse_args()
    merge_lora_weights(args.base_model, args.adapter_dir, args.output_dir)
