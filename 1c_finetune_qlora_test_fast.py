#!/usr/bin/env python3
"""
QLoRA Fast Test Fine-Tuning for Turkish Medical LLM (qwen3:8b)
Ultra-fast version: 5% data, 1 epoch, ~3-5 minutes
Perfect for testing the evaluation pipeline without waiting hours
"""

import os
import json
import csv
import torch
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
from dataclasses import dataclass, field
import sys

import numpy as np
from datasets import load_dataset, Dataset
from rouge_score import rouge_scorer
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Add src to path
BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

from src.config import SFT_INSTRUCTION_FILE, BASE_DIR as CONFIG_BASE_DIR
from src.utils.logger import setup_logger

logger = setup_logger("QLoRAFastTest")

# ============================================================================
# FAST TEST CONFIGURATION (5% data, 1 epoch, ~3-5 minutes)
# ============================================================================

@dataclass
class QLoRAFastTestConfig:
    """QLoRA FAST TEST Configuration - 5% Data"""
    
    # Model Configuration
    base_model_name: str = "Qwen/Qwen3-8B"
    
    # Quantization (4-bit)
    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_quant_type: str = "nf4"
    
    # LoRA Configuration
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "v_proj", "k_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    
    # FAST TEST SETTINGS: 5% data, minimal steps
    output_dir: str = "./qlora_checkpoints_test"
    num_epochs: int = 1  # Just 1 epoch
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 1  # Fast
    learning_rate: float = 2e-4
    warmup_steps: int = 10  # Minimal warmup
    max_steps: int = 50  # ONLY 50 STEPS - Ultra fast validation
    logging_steps: int = 10  # Log frequently for progress
    save_steps: int = 50  # Save after all steps
    eval_steps: int = 50  # Eval after training
    save_total_limit: int = 1  # Only keep final checkpoint
    
    # Optimization
    optim: str = "paged_adamw_32bit"
    weight_decay: float = 0.01
    max_grad_norm: float = 0.3
    
    # Data
    max_seq_length: int = 2048
    packing: bool = False
    data_sample_percentage: float = 0.01  # ONLY 1% OF DATA - Ultra fast!
    
    # Training
    device_map: str = "auto"
    mixed_precision: str = "fp16"
    
    # Paths
    training_data_file: str = str(SFT_INSTRUCTION_FILE)
    checkpoint_dir: str = "./qlora_checkpoints_test"
    rouge_eval_dir: str = "./data-evaluation/qlora-rouge-test"


class QLoRAFastTest:
    """Ultra-fast QLoRA fine-tuning for testing (5% data)"""
    
    def __init__(self, config: QLoRAFastTestConfig = None):
        self.config = config or QLoRAFastTestConfig()
        self.model = None
        self.tokenizer = None
        self.trainer = None
        
        # Create output directories
        Path(self.config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.rouge_eval_dir).mkdir(parents=True, exist_ok=True)
    
    def load_model_and_tokenizer(self):
        """Load Qwen3:8B with 4-bit quantization"""
        logger.info(f"⏳ Loading model: {self.config.base_model_name}")
        
        # 4-bit quantization config
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=self.config.load_in_4bit,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=self.config.bnb_4bit_use_double_quant,
            bnb_4bit_quant_type=self.config.bnb_4bit_quant_type,
        )
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model_name,
            quantization_config=quantization_config,
            device_map=self.config.device_map,
            trust_remote_code=True,
            use_cache=False,  # Important for training
        )
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.base_model_name,
            trust_remote_code=True,
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Prepare model for kbit training
        self.model = prepare_model_for_kbit_training(self.model)
        
        logger.info("✅ Model and tokenizer loaded")
    
    def apply_lora(self):
        """Apply LoRA adapters"""
        logger.info("⏳ Applying LoRA adapters...")
        
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            target_modules=self.config.lora_target_modules,
            task_type="CAUSAL_LM",
        )
        
        self.model = get_peft_model(self.model, lora_config)
        
        # Print trainable parameters
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_pct = (trainable_params / total_params) * 100
        
        logger.info(f"✅ LoRA applied")
        logger.info(f"   Trainable parameters: {trainable_params:,}")
        logger.info(f"   Total parameters: {total_params:,}")
        logger.info(f"   Trainable %: {trainable_pct:.2f}%")
    
    def load_and_prepare_data(self):
        """Load and prepare training data (5% sample only)"""
        logger.info(f"⏳ Loading data from: {self.config.training_data_file}")
        
        # Load JSONL file
        data_list = []
        with open(self.config.training_data_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        data_list.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        
        logger.info(f"   Total samples in file: {len(data_list):,}")
        
        # SAMPLE 5% for fast testing
        sample_size = max(1, int(len(data_list) * self.config.data_sample_percentage))
        import random
        random.seed(42)  # For reproducibility
        sampled_data = random.sample(data_list, sample_size)
        
        logger.info(f"   🚀 Using 5% sample: {len(sampled_data):,} samples for FAST TEST")
        
        # Create train/val split (80/20)
        train_size = int(len(sampled_data) * 0.8)
        train_data = sampled_data[:train_size]
        val_data = sampled_data[train_size:]
        
        logger.info(f"   Train samples: {len(train_data):,}")
        logger.info(f"   Validation samples: {len(val_data):,}")
        
        # Convert to HF Dataset - proper format
        def dict_to_dataset(data_list):
            texts = []
            for item in data_list:
                system = item.get('system', '')
                instruction = item.get('instruction', '')
                output = item.get('output', '')
                
                text = f"{system}\n\nInstruction: {instruction}\n\nResponse: {output}"
                texts.append(text)
            
            return Dataset.from_dict({'text': texts})
        
        train_dataset = dict_to_dataset(train_data)
        val_dataset = dict_to_dataset(val_data)
        
        return train_dataset, val_dataset
    
    def tokenize_dataset(self, dataset):
        """Tokenize dataset"""
        def tokenize_function(examples):
            return self.tokenizer(
                examples['text'],
                padding='max_length',
                truncation=True,
                max_length=self.config.max_seq_length,
            )
        
        return dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=['text']
        )
    
    def setup_trainer(self, train_dataset, val_dataset):
        """Setup HuggingFace Trainer"""
        logger.info("⏳ Setting up Trainer...")
        
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_epochs if self.config.max_steps < 0 else 1,
            max_steps=self.config.max_steps if self.config.max_steps > 0 else -1,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            warmup_steps=self.config.warmup_steps,
            learning_rate=self.config.learning_rate,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            eval_strategy="steps" if self.config.eval_steps > 0 else "no",
            eval_steps=self.config.eval_steps if self.config.eval_steps > 0 else None,
            save_total_limit=self.config.save_total_limit,
            optim=self.config.optim,
            weight_decay=self.config.weight_decay,
            max_grad_norm=self.config.max_grad_norm,
            fp16=True,
            logging_dir='./logs',
            logging_first_step=True,
        )
        
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=DataCollatorForLanguageModeling(
                self.tokenizer, mlm=False
            ),
        )
        
        logger.info("✅ Trainer ready")
    
    def train(self):
        """Execute training"""
        logger.info("=" * 80)
        logger.info("🚀 STARTING FAST TEST TRAINING (5% data, 1 epoch)")
        logger.info("=" * 80)
        
        start_time = datetime.now()
        
        # Train
        logger.info("\n⏳ Training...")
        train_result = self.trainer.train()
        
        # Evaluate
        logger.info("\n⏳ Evaluating...")
        eval_result = self.trainer.evaluate()
        
        elapsed = datetime.now() - start_time
        logger.info(f"\n✅ Training completed in {elapsed}")
        logger.info(f"   Train loss: {train_result.training_loss:.4f}")
        logger.info(f"   Eval loss: {eval_result.get('eval_loss', 'N/A')}")
        
        return train_result, eval_result
    
    def save_checkpoint(self):
        """Save model checkpoint"""
        logger.info(f"\n⏳ Saving checkpoint to: {self.config.checkpoint_dir}")
        
        self.model.save_pretrained(self.config.checkpoint_dir)
        self.tokenizer.save_pretrained(self.config.checkpoint_dir)
        
        logger.info(f"✅ Checkpoint saved")
    
    def save_training_report(self, train_result, eval_result):
        """Save training report"""
        report = {
            'test_type': 'FAST TEST (5% data)',
            'timestamp': datetime.now().isoformat(),
            'model': self.config.base_model_name,
            'data_percentage': f"{self.config.data_sample_percentage * 100}%",
            'num_epochs': self.config.num_epochs,
            'batch_size': self.config.per_device_train_batch_size,
            'gradient_accumulation': self.config.gradient_accumulation_steps,
            'learning_rate': self.config.learning_rate,
            'training_loss': float(train_result.training_loss) if hasattr(train_result, 'training_loss') else None,
            'eval_loss': float(eval_result.get('eval_loss', 0)),
            'checkpoint_dir': self.config.checkpoint_dir,
            'note': 'This is a FAST TEST checkpoint for validation. Use full training for production.'
        }
        
        report_file = Path(self.config.checkpoint_dir) / "test_training_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"\n📊 Training Report:")
        logger.info(json.dumps(report, indent=2, ensure_ascii=False))
        logger.info(f"\n✅ Report saved: {report_file}")
    
    def run_complete_pipeline(self):
        """Run complete fast-test pipeline"""
        try:
            # 1. Load model
            self.load_model_and_tokenizer()
            
            # 2. Apply LoRA
            self.apply_lora()
            
            # 3. Load and prepare data (5% sample)
            train_dataset, val_dataset = self.load_and_prepare_data()
            
            # 4. Tokenize
            logger.info("⏳ Tokenizing datasets...")
            train_dataset = self.tokenize_dataset(train_dataset)
            val_dataset = self.tokenize_dataset(val_dataset)
            logger.info(f"✅ Tokenization complete")
            
            # 5. Setup trainer
            self.setup_trainer(train_dataset, val_dataset)
            
            # 6. Train
            train_result, eval_result = self.train()
            
            # 7. Save checkpoint
            self.save_checkpoint()
            
            # 8. Save report
            self.save_training_report(train_result, eval_result)
            
            logger.info("\n" + "=" * 80)
            logger.info("✅ FAST TEST TRAINING COMPLETE!")
            logger.info("=" * 80)
            logger.info(f"\n📌 Checkpoint saved: {self.config.checkpoint_dir}")
            logger.info("   Ready for Configurations 3-5 evaluation!")
            
        except Exception as e:
            logger.error(f"❌ Error during training: {e}")
            raise


def main():
    """Main execution"""
    print("\n" + "=" * 80)
    print("🚀 QWEN3:8B FAST TEST FINE-TUNING")
    print("   Configuration: 5% data, 1 epoch, ~3-5 minutes")
    print("=" * 80)
    
    # Create config
    config = QLoRAFastTestConfig()
    
    # Run training
    ft = QLoRAFastTest(config)
    ft.run_complete_pipeline()


if __name__ == "__main__":
    main()
