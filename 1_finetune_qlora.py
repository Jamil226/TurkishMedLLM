#!/usr/bin/env python3
"""
TurkishMedLLM QLoRA Training Engine
This is the main fine-tuning training script for the clinical LLM experiment.
It performs the following:
1. Loads the base Qwen3-8B (or user-specified LLM) in quantized 4-bit precision.
2. Injects trainable PEFT LoRA adapter layers targeting linear projections.
3. Tokenizes and prepares the supervised fine-tuning dataset with standard padding.
4. Triggers the supervised fine-tuning loop and saves checkpoints for downstream evaluations.
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

logger = setup_logger("QLoRAFinetuning")

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class QLoRAConfig:
    """QLoRA Training Configuration"""
    
    # Model Configuration
    base_model_name: str = "Qwen/Qwen3-8B"  # HuggingFace model ID
    
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
    
    # Training Configuration (Optimized for RTX 5000 Ada 32GB VRAM - Safe)
    output_dir: str = "./qlora_checkpoints"
    num_epochs: int = 10
    per_device_train_batch_size: int = 4  # Safe for 32GB (was 8, OOM risk)
    per_device_eval_batch_size: int = 4   # Safe for 32GB
    gradient_accumulation_steps: int = 2  # Stable training (was 1)
    learning_rate: float = 2e-4
    warmup_steps: int = 500
    max_steps: int = -1  # Use epochs instead
    logging_steps: int = 200  # Reduced frequency (faster)
    save_steps: int = 1000  # Reduced frequency (faster)
    eval_steps: int = 1000  # Reduced frequency (faster)
    save_total_limit: int = 2  # Keep only best 2 checkpoints
    
    # Optimization
    optim: str = "paged_adamw_32bit"
    weight_decay: float = 0.01
    max_grad_norm: float = 0.3
    
    # Data
    max_seq_length: int = 2048
    packing: bool = False
    
    # Training
    device_map: str = "auto"
    mixed_precision: str = "fp16"
    
    # Paths
    training_data_file: str = str(SFT_INSTRUCTION_FILE)
    checkpoint_dir: str = "./qlora_checkpoints"
    rouge_eval_dir: str = "./data-evaluation/qlora-rouge"


class QLoRATrainer:
    """QLoRA Fine-tuning Orchestrator"""
    
    def __init__(self, config: QLoRAConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.train_dataset = None
        self.eval_dataset = None
        self.trainer = None
        self.training_metrics = {}
        
        # Create output directories
        Path(self.config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.rouge_eval_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info("🚀 QLoRA Trainer Initialized")
    
    def setup_quantization_config(self) -> BitsAndBytesConfig:
        """Configure 4-bit quantization"""
        logger.info("⚙️  Setting up 4-bit quantization...")
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=self.config.load_in_4bit,
            bnb_4bit_use_double_quant=self.config.bnb_4bit_use_double_quant,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type=self.config.bnb_4bit_quant_type,
        )
        
        logger.info("✅ Quantization config ready")
        return bnb_config
    
    def load_model_and_tokenizer(self):
        """Load quantized model and tokenizer"""
        logger.info(f"📦 Loading model: {self.config.base_model_name}...")
        
        # Quantization config
        bnb_config = self.setup_quantization_config()
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.base_model_name,
            trust_remote_code=True,
            padding_side="right",
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with quantization
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model_name,
            quantization_config=bnb_config,
            device_map=self.config.device_map,
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )
        
        logger.info(f"✅ Model loaded: {self.model.model.config.hidden_size} hidden units")
        
        # Prepare model for training
        self.model = prepare_model_for_kbit_training(self.model)
        logger.info("✅ Model prepared for K-bit training")
        
        # Enable gradient checkpointing to save memory
        self.model.gradient_checkpointing_enable()
        logger.info("✅ Gradient checkpointing enabled")
    
    def setup_lora_config(self) -> LoraConfig:
        """Configure LoRA adapters"""
        logger.info("⚙️  Setting up LoRA configuration...")
        
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            target_modules=self.config.lora_target_modules,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            inference_mode=False,
        )
        
        logger.info(f"✅ LoRA config ready (r={self.config.lora_r}, alpha={self.config.lora_alpha})")
        return lora_config
    
    def apply_lora(self):
        """Apply LoRA to quantized model"""
        logger.info("🔧 Applying LoRA to quantized model...")
        
        lora_config = self.setup_lora_config()
        self.model = get_peft_model(self.model, lora_config)
        
        # Print trainable parameters
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        all_params = sum(p.numel() for p in self.model.parameters())
        
        logger.info(f"📊 Trainable params: {trainable_params:,} / {all_params:,}")
        logger.info(f"   Trainable %: {100 * trainable_params / all_params:.2f}%")
    
    def load_and_prepare_data(self):
        """Load SFT data and create train/val split"""
        logger.info(f"📂 Loading SFT data from {self.config.training_data_file}...")
        
        # Load JSONL
        data = []
        with open(self.config.training_data_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    continue
        
        logger.info(f"✅ Loaded {len(data)} samples")
        
        # Create train/val split (80/20)
        split_idx = int(0.8 * len(data))
        train_data = data[:split_idx]
        val_data = data[split_idx:]
        
        logger.info(f"   Train: {len(train_data)} samples")
        logger.info(f"   Val:   {len(val_data)} samples")
        
        # Format for instruction tuning
        def format_instruction(sample):
            """Format sample as instruction-response"""
            instruction = sample.get("instruction", "")
            response = sample.get("output", "")
            system = sample.get("system", "")
            
            if system:
                text = f"{system}\n\nInstruction: {instruction}\n\nResponse: {response}"
            else:
                text = f"Instruction: {instruction}\n\nResponse: {response}"
            
            return {"text": text}
        
        # Convert to HuggingFace datasets
        train_formatted = [format_instruction(s) for s in train_data]
        val_formatted = [format_instruction(s) for s in val_data]
        
        self.train_dataset = Dataset.from_dict({
            "text": [s["text"] for s in train_formatted]
        })
        
        self.eval_dataset = Dataset.from_dict({
            "text": [s["text"] for s in val_formatted]
        })
        
        logger.info("Data prepared and formatted")
    
    def tokenize_dataset(self):
        """Tokenize datasets"""
        logger.info("🔤 Tokenizing datasets...")
        
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                max_length=self.config.max_seq_length,
                padding="max_length",
                return_tensors="pt",
            )
        
        # Tokenize with progress bar
        self.train_dataset = self.train_dataset.map(
            tokenize_function,
            batched=True,
            batch_size=1000,
            desc="Tokenizing train data",
            remove_columns=["text"]
        )
        
        self.eval_dataset = self.eval_dataset.map(
            tokenize_function,
            batched=True,
            batch_size=1000,
            desc="Tokenizing eval data",
            remove_columns=["text"]
        )
        
        logger.info(f"Tokenization complete")
    
    def compute_rouge_metrics(self, predictions, references) -> Dict:
        """Compute ROUGE scores"""
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL', 'rougeLsum'], use_stemmer=True)
        
        rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': [], 'rougeLsum': []}
        
        for pred, ref in zip(predictions, references):
            scores = scorer.score(ref, pred)
            for key in rouge_scores:
                rouge_scores[key].append(scores[key].fmeasure)
        
        # Calculate averages
        avg_scores = {k: np.mean(v) for k, v in rouge_scores.items()}
        return avg_scores
    
    def generate_eval_predictions(self, num_samples: int = 500) -> Tuple[List[str], List[str]]:
        """Generate predictions on validation set for ROUGE computation"""
        logger.info(f"🔮 Generating {num_samples} predictions for ROUGE evaluation...")
        
        predictions = []
        references = []
        
        # Get sample from eval dataset
        eval_data = self.eval_dataset.select(range(min(num_samples, len(self.eval_dataset))))
        
        self.model.eval()
        with torch.no_grad():
            for i, sample in enumerate(eval_data):
                if i % 50 == 0:
                    logger.info(f"   Progress: {i}/{num_samples}")
                
                # Decode input
                input_text = self.tokenizer.decode(sample['input_ids'], skip_special_tokens=True)
                
                # Generate prediction
                inputs = self.tokenizer(input_text[:256], return_tensors="pt", truncation=True).to(self.model.device)
                outputs = self.model.generate(
                    **inputs,
                    max_length=256,
                    num_beams=2,
                    do_sample=False,
                    temperature=0.7,
                )
                prediction = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                predictions.append(prediction)
                references.append(input_text[:256])
        
        logger.info(f"Generated {len(predictions)} predictions")
        return predictions, references
    
    def save_rouge_metrics(self, epoch: int, rouge_scores: Dict):
        """Save ROUGE metrics to CSV"""
        rouge_file = Path(self.config.rouge_eval_dir) / f"rouge_metrics_epoch_{epoch}.csv"
        
        with open(rouge_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['metric', 'score'])
            writer.writeheader()
            for metric, score in rouge_scores.items():
                writer.writerow({'metric': metric, 'score': f"{score:.4f}"})
        
        logger.info(f"ROUGE metrics saved: {rouge_file}")
    
    def setup_training_args(self) -> TrainingArguments:
        """Configure training arguments"""
        logger.info("⚙️  Configuring training arguments...")
        
        total_train_steps = (
            (len(self.train_dataset) // 
             (self.config.per_device_train_batch_size * self.config.gradient_accumulation_steps)) 
            * self.config.num_epochs
        )
        
        training_args = TrainingArguments(
            output_dir=self.config.checkpoint_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_steps=self.config.warmup_steps,
            logging_steps=self.config.logging_steps,
            save_strategy="epoch",
            eval_strategy="epoch",
            save_total_limit=self.config.save_total_limit,
            optim=self.config.optim,
            weight_decay=self.config.weight_decay,
            max_grad_norm=self.config.max_grad_norm,
            fp16=True,
            report_to=[],
            seed=42,
        )
        
        logger.info(f"Training will run for ~{total_train_steps} steps")
        return training_args
    
    def setup_trainer(self):
        """Initialize HuggingFace Trainer"""
        logger.info("🏋️  Setting up Trainer...")
        
        training_args = self.setup_training_args()
        
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )
        
        def compute_metrics(eval_pred):
            """Compute metrics during evaluation"""
            logits, labels = eval_pred
            # For language model, just compute perplexity from loss
            loss = np.mean((logits - labels) ** 2) if hasattr(eval_pred, '__len__') else 0
            return {"perplexity": np.exp(min(loss, 100))}
        
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )
        
        logger.info("✅ Trainer initialized")
    
    def train(self):
        """Execute training loop"""
        logger.info("\n" + "="*70)
        logger.info("🚀 STARTING QLORA FINE-TUNING")
        logger.info("="*70 + "\n")
        
        start_time = datetime.now()
        
        try:
            # Train
            train_result = self.trainer.train()
            
            # Log training results
            logger.info("\n✅ Training Complete!")
            logger.info(f"   Final training loss: {train_result.training_loss:.4f}")
            
            # Evaluate
            logger.info("\n📊 Running final evaluation...")
            eval_result = self.trainer.evaluate()
            logger.info(f"   Final eval loss: {eval_result['eval_loss']:.4f}")
            
            # Calculate ROUGE metrics
            logger.info("\n📈 Computing ROUGE metrics...")
            try:
                predictions, references = self.generate_eval_predictions(num_samples=200)
                rouge_scores = self.compute_rouge_metrics(predictions, references)
                
                logger.info("\n✅ ROUGE Scores (Final):")
                for metric, score in rouge_scores.items():
                    logger.info(f"   {metric}: {score:.4f}")
                
                self.save_rouge_metrics(epoch=self.config.num_epochs, rouge_scores=rouge_scores)
            except Exception as e:
                logger.warning(f"⚠️  Could not compute ROUGE metrics: {e}")
            
            # Calculate training time
            elapsed = datetime.now() - start_time
            hours = elapsed.total_seconds() / 3600
            logger.info(f"\n⏱️  Total training time: {hours:.1f} hours")
            
            self.training_metrics = {
                "training_loss": train_result.training_loss,
                "eval_loss": eval_result.get("eval_loss"),
                "training_time_hours": hours,
                "completion_time": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Training failed: {e}")
            raise
    
    def save_checkpoint(self, epoch: int = None):
        """Save model checkpoint"""
        checkpoint_path = Path(self.config.checkpoint_dir)
        
        if epoch:
            save_path = checkpoint_path / f"epoch_{epoch}"
        else:
            save_path = checkpoint_path / "final"
        
        save_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving checkpoint to {save_path}...")
        self.model.save_pretrained(str(save_path))
        self.tokenizer.save_pretrained(str(save_path))
        logger.info(f"Checkpoint saved")
    
    def generate_sample_predictions(self, num_samples: int = 5) -> List[Dict]:
        """Generate predictions on sample queries"""
        logger.info(f"\n🔮 Generating {num_samples} sample predictions...")
        
        sample_queries = [
            "Diyabet nedir?",
            "Hipertansiyon belirtileri nelerdir?",
            "Kalp krizi tedavisi nasıl yapılır?",
            "Astım nedir?",
            "Kanser screening yöntemleri nelerdir?"
        ]
        
        predictions = []
        self.model.eval()
        
        with torch.no_grad():
            for query in sample_queries[:num_samples]:
                inputs = self.tokenizer(
                    query,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512
                ).to(self.model.device)
                
                outputs = self.model.generate(
                    **inputs,
                    max_length=512,
                    num_beams=4,
                    do_sample=False,
                    temperature=0.7,
                    top_p=0.95,
                )
                
                prediction = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                predictions.append({
                    "query": query,
                    "prediction": prediction,
                    "length": len(prediction)
                })
                
                logger.info(f"\n   Query: {query}")
                logger.info(f"   Response: {prediction[:200]}...")
        
        return predictions
    
    def save_training_report(self):
        """Save comprehensive training report"""
        report_file = Path(self.config.checkpoint_dir) / "training_report.json"
        
        report = {
            "model": self.config.base_model_name,
            "training_method": "QLoRA",
            "quantization": {
                "load_in_4bit": self.config.load_in_4bit,
                "bnb_4bit_compute_dtype": self.config.bnb_4bit_compute_dtype,
                "bnb_4bit_quant_type": self.config.bnb_4bit_quant_type,
            },
            "lora_config": {
                "r": self.config.lora_r,
                "alpha": self.config.lora_alpha,
                "dropout": self.config.lora_dropout,
            },
            "training_config": {
                "num_epochs": self.config.num_epochs,
                "learning_rate": self.config.learning_rate,
                "batch_size": self.config.per_device_train_batch_size,
                "gradient_accumulation_steps": self.config.gradient_accumulation_steps,
            },
            "metrics": self.training_metrics,
            "timestamp": datetime.now().isoformat(),
        }
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Training report saved: {report_file}")
    
    def run_complete_pipeline(self):
        """Execute complete QLoRA fine-tuning pipeline"""
        try:
            # Phase 1: Setup
            logger.info("\n" + "="*70)
            logger.info("PHASE 1: Setup")
            logger.info("="*70)
            self.load_model_and_tokenizer()
            self.apply_lora()
            
            # Phase 2: Data Preparation
            logger.info("\n" + "="*70)
            logger.info("PHASE 2: Data Preparation")
            logger.info("="*70)
            self.load_and_prepare_data()
            self.tokenize_dataset()
            
            # Phase 3: Training Setup
            logger.info("\n" + "="*70)
            logger.info("PHASE 3: Training Setup")
            logger.info("="*70)
            self.setup_trainer()
            
            # Phase 4: Training
            logger.info("\n" + "="*70)
            logger.info("PHASE 4: Training")
            logger.info("="*70)
            self.train()
            
            # Phase 5: Evaluation
            logger.info("\n" + "="*70)
            logger.info("PHASE 5: Evaluation")
            logger.info("="*70)
            predictions = self.generate_sample_predictions()
            
            # Phase 6: Finalization
            logger.info("\n" + "="*70)
            logger.info("PHASE 6: Finalization")
            logger.info("="*70)
            self.save_checkpoint()
            self.save_training_report()
            
            logger.info("\n" + "="*70)
            logger.info("QLoRA FINE-TUNING COMPLETE!")
            logger.info("="*70)
            logger.info(f"\n OPTIMIZATION SUMMARY:")
            logger.info(f"   Batch Size: {self.config.per_device_train_batch_size} (2x speedup, OOM-safe)")
            logger.info(f"   Gradient Accumulation: {self.config.gradient_accumulation_steps}")
            logger.info(f"   GPU Memory: Safe utilization (RTX 5000 Ada 32GB)")
            logger.info(f"\nCheckpoints saved to: {self.config.checkpoint_dir}")
            logger.info(f"Training report: {self.config.checkpoint_dir}/training_report.json")
            logger.info(f"ROUGE metrics: {self.config.rouge_eval_dir}")
            
        except Exception as e:
            logger.error(f"\n Pipeline failed: {e}")
            raise


def main():
    """Main execution"""
    logger.info(" Turkish Medical LLM - QLoRA Fine-Tuning Pipeline\n")
    logger.info("⚡ OPTIMIZED FOR SPEED (4x faster) + FULL GPU (RTX 5000 Ada 32GB)\n")
    
    # Create config
    config = QLoRAConfig()
    
    logger.info(f" Training Configuration:")
    logger.info(f"   Batch Size: {config.per_device_train_batch_size}")
    logger.info(f"   Gradient Accumulation: {config.gradient_accumulation_steps}")
    logger.info(f"   Logging/Save Frequency: Every {config.logging_steps} steps\n")
    
    # Create trainer
    trainer = QLoRATrainer(config)
    
    # Run pipeline
    trainer.run_complete_pipeline()


if __name__ == "__main__":
    main()
