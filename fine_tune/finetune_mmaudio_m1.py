#!/usr/bin/env python3
"""
MMAudio Video-to-Audio Fine-tuning Script for MacBook M1
=========================================================
Fine-tune MMAudio models on custom video-audio datasets.
Optimized for Apple Silicon (M1/M2/M3) with MPS support.

Usage:
    # MacBook M1 training
    python finetune_mmaudio_m1.py --exp_id my_experiment --model small_16k
    python finetune_mmaudio_m1.py --exp_id experiment_02 --model small_16k
    # CPU-only training (if MPS has issues)
    python finetune_mmaudio_m1.py --exp_id my_experiment --model small_16k --device cpu
"""

import os
import sys
import logging
import json
import math
import gc
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List
import time
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

import numpy as np
import pandas as pd
import tensordict as td
from tqdm import tqdm
from einops import rearrange

# Import MMAudio modules
from mmaudio.model.flow_matching import FlowMatching
from mmaudio.model.networks import get_my_mmaudio, PreprocessedConditions
from mmaudio.model.utils.features_utils import FeaturesUtils
from mmaudio.model.utils.distributions import DiagonalGaussianDistribution

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Training configuration for MacBook M1"""
    # Model
    model: str = 'small_16k'  # small_16k, small_44k, medium_44k, large_44k
    pretrained_path: Optional[str] = None  # Path to pretrained model
    
    # Data
    data_dir: str = './output/memmap/vgg-my_videos'  # Directory with memory-mapped features
    tsv_file: str = './output/memmap/vgg-my_videos.tsv'  # TSV file with metadata
    
    # Training - Optimized for M1
    exp_id: str = 'finetune_exp'
    batch_size: int = 4  # Reduced for M1 memory constraints
    num_epochs: int = 25  # Reduced for faster training
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    
    # Optimization
    warmup_steps: int = 500
    use_amp: bool = False  # AMP not well supported on MPS yet
    gradient_accumulation_steps: int = 2  # Accumulate gradients to simulate larger batch
    
    # Flow matching
    num_flow_steps: int = 25
    min_sigma: float = 0.0
    cfg_strength: float = 4.5  # Classifier-free guidance strength
    
    # Logging
    log_interval: int = 10
    save_interval: int = 500
    eval_interval: int = 250
    
    # Output
    output_dir: str = './output/finetune'
    
    # Device - Auto-detect best available
    device: str = 'auto'
    num_workers: int = 2  # Reduced for M1
    
    # Memory optimization
    pin_memory: bool = False  # Disable for M1
    persistent_workers: bool = False
    prefetch_factor: int = 2
    
    # Checkpointing
    save_optimizer_state: bool = False  # Save memory by not saving optimizer state
    gradient_checkpointing: bool = False  # Enable if memory is tight


def get_device(device_preference: str = 'auto') -> torch.device:
    """Get the best available device for MacBook M1"""
    if device_preference == 'auto':
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            device = torch.device("mps")
            logger.info("✅ Using MPS (Metal Performance Shaders) on Apple Silicon")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info("✅ Using CUDA GPU")
        else:
            device = torch.device("cpu")
            logger.info("⚠️  Using CPU (will be slower)")
    else:
        device = torch.device(device_preference)
        logger.info(f"Using specified device: {device}")
    
    return device


def clear_memory():
    """Clear memory cache for different devices"""
    gc.collect()
    if torch.backends.mps.is_available():
        try:
            # MPS memory management
            torch.mps.empty_cache()
            torch.mps.synchronize()
        except:
            pass
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()


class VideoAudioDataset(Dataset):
    """Dataset for loading pre-extracted features"""
    
    def __init__(self, data_dir: Path, tsv_file: Path, split: str = 'train'):
        self.data_dir = data_dir
        self.split = split
        
        # Load metadata
        self.metadata = pd.read_csv(tsv_file, sep='\t')
        logger.info(f"Loaded {len(self.metadata)} samples from {tsv_file}")
        
        # Load memory-mapped tensors
        memmap_dir = data_dir / f'vgg-{split.replace("_", "-")}'
        if not memmap_dir.exists():
            # Try alternative naming
            memmap_dir = data_dir / f'vgg-my_videos'
        
        logger.info(f"Loading memory-mapped tensors from {memmap_dir}")
        self.tensor_dict = td.TensorDict.load_memmap(memmap_dir)
        
        # Verify data integrity
        self._verify_data()
        
    def _verify_data(self):
        """Verify that all required features are present"""
        required_keys = ['mean', 'std', 'clip_features', 'sync_features', 'text_features']
        for key in required_keys:
            if key not in self.tensor_dict.keys():
                raise ValueError(f"Missing required feature: {key}")
        
        # Check dimensions
        n_samples = len(self.tensor_dict['mean'])
        logger.info(f"Dataset contains {n_samples} samples")
        logger.info(f"Feature shapes:")
        for key in required_keys:
            logger.info(f"  {key}: {self.tensor_dict[key].shape}")
    
    def __len__(self):
        return len(self.tensor_dict['mean'])
    
    def __getitem__(self, idx):
        return {
            'mean': self.tensor_dict['mean'][idx].clone(),  # Clone to avoid memory issues
            'std': self.tensor_dict['std'][idx].clone(),
            'clip_features': self.tensor_dict['clip_features'][idx].clone(),
            'sync_features': self.tensor_dict['sync_features'][idx].clone(),
            'text_features': self.tensor_dict['text_features'][idx].clone(),
            'caption': self.metadata.iloc[idx]['label'] if idx < len(self.metadata) else ""
        }


class MMAudioFineTunerM1:
    """Fine-tuning class optimized for MacBook M1"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        
        # Device setup
        self.device = get_device(config.device)
        
        # Create output directory
        self.output_dir = Path(config.output_dir) / config.exp_id
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.output_dir / 'checkpoints'
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Log system info
        self._log_system_info()
        
        # Initialize model
        self._init_model()
        
        # Initialize flow matching
        self.flow_matching = FlowMatching(
            min_sigma=config.min_sigma,
            num_steps=config.num_flow_steps
        )
        
        # Setup data
        self._setup_data()
        
        # Setup optimization
        self._setup_optimization()
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_loss = float('inf')
        
        # Memory tracking
        self.memory_check_interval = 50
        
    def _log_system_info(self):
        """Log system information"""
        logger.info("=" * 50)
        logger.info("System Information:")
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"Device: {self.device}")
        
        if self.device.type == 'mps':
            logger.info("MPS (Metal Performance Shaders) is active")
            logger.info("Note: Training may be slower than CUDA but faster than CPU")
        
        # Check available memory (approximate)
        try:
            import subprocess
            result = subprocess.run(['sysctl', 'hw.memsize'], capture_output=True, text=True)
            if result.returncode == 0:
                mem_bytes = int(result.stdout.split(':')[1].strip())
                mem_gb = mem_bytes / (1024**3)
                logger.info(f"System RAM: {mem_gb:.1f} GB")
        except:
            pass
        
        logger.info("=" * 50)
    
    def _init_model(self):
        """Initialize the MMAudio model"""
        logger.info(f"Initializing model: {self.config.model}")
        
        # Load model
        self.model = get_my_mmaudio(self.config.model)
        
        # Load pretrained weights if provided
        if self.config.pretrained_path:
            pretrained_path = Path(self.config.pretrained_path)
        else:
            # Use default pretrained model
            pretrained_path = Path(f'./weights/mmaudio_{self.config.model}.pth')
        
        if pretrained_path.exists():
            logger.info(f"Loading pretrained weights from {pretrained_path}")
            
            # Load with map_location for M1 compatibility
            checkpoint = torch.load(
                pretrained_path, 
                map_location='cpu'  # Load to CPU first
            )
            
            if 'model' in checkpoint:
                self.model.load_state_dict(checkpoint['model'])
            else:
                self.model.load_weights(checkpoint)
            
            logger.info("✅ Pretrained weights loaded successfully")
        else:
            logger.warning(f"⚠️ No pretrained weights found at {pretrained_path}")
            logger.warning("Training from scratch (not recommended)")
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Enable gradient checkpointing if requested
        if self.config.gradient_checkpointing:
            logger.info("Enabling gradient checkpointing to save memory")
            # Note: MMAudio models may need custom implementation
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Model parameters: {total_params/1e6:.2f}M total, {trainable_params/1e6:.2f}M trainable")
        
        # Set model to training mode
        self.model.train()
    
    def _setup_data(self):
        """Setup data loaders optimized for M1"""
        logger.info("Setting up data loaders...")
        
        # Create dataset
        self.train_dataset = VideoAudioDataset(
            Path(self.config.data_dir),
            Path(self.config.tsv_file),
            split='my_videos'
        )
        
        # Create data loader with M1 optimizations
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            persistent_workers=self.config.persistent_workers,
            prefetch_factor=self.config.prefetch_factor,
            drop_last=True
        )
        
        logger.info(f"Training data: {len(self.train_dataset)} samples")
        logger.info(f"Batches per epoch: {len(self.train_loader)}")
        logger.info(f"Effective batch size: {self.config.batch_size * self.config.gradient_accumulation_steps}")
    
    def _setup_optimization(self):
        """Setup optimizer and scheduler"""
        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Learning rate scheduler
        total_steps = len(self.train_loader) * self.config.num_epochs // self.config.gradient_accumulation_steps
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps,
            eta_min=1e-6
        )
        
        logger.info(f"Optimizer: AdamW (lr={self.config.learning_rate})")
        logger.info(f"Scheduler: CosineAnnealingLR (total_steps={total_steps})")
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step optimized for M1"""
        # Move batch to device
        mean = batch['mean'].to(self.device, non_blocking=True)
        std = batch['std'].to(self.device, non_blocking=True)
        clip_features = batch['clip_features'].to(self.device, non_blocking=True)
        sync_features = batch['sync_features'].to(self.device, non_blocking=True)
        text_features = batch['text_features'].to(self.device, non_blocking=True)
        
        batch_size = mean.shape[0]
        
        # Sample from latent distribution
        z1 = mean + std * torch.randn_like(mean)
        
        # Normalize latents
        z1 = self.model.normalize(z1)
        
        # Sample time steps
        t = torch.rand(batch_size, device=self.device)
        
        # Get flow matching targets
        z0 = torch.randn_like(z1)
        zt = self.flow_matching.get_conditional_flow(z0, z1, t)
        
        # Forward pass
        predicted_flow = self.model(zt, clip_features, sync_features, text_features, t)
        
        # Compute loss
        loss = self.flow_matching.loss(predicted_flow, z0, z1).mean()
        
        # Scale loss for gradient accumulation
        loss = loss / self.config.gradient_accumulation_steps
        
        # Backward pass
        loss.backward()
        
        return {
            'loss': loss.item() * self.config.gradient_accumulation_steps,
            'batch_size': batch_size
        }
    
    def evaluate(self, num_batches: int = 10) -> Dict[str, float]:
        """Quick evaluation"""
        self.model.eval()
        total_loss = 0
        num_samples = 0
        
        with torch.no_grad():
            for i, batch in enumerate(self.train_loader):
                if i >= num_batches:
                    break
                
                # Move batch to device
                mean = batch['mean'].to(self.device)
                std = batch['std'].to(self.device)
                clip_features = batch['clip_features'].to(self.device)
                sync_features = batch['sync_features'].to(self.device)
                text_features = batch['text_features'].to(self.device)
                
                batch_size = mean.shape[0]
                
                # Sample from latent distribution
                z1 = mean + std * torch.randn_like(mean)
                z1 = self.model.normalize(z1)
                
                # Sample time steps
                t = torch.rand(batch_size, device=self.device)
                
                # Get flow matching targets
                z0 = torch.randn_like(z1)
                zt = self.flow_matching.get_conditional_flow(z0, z1, t)
                
                # Forward pass
                predicted_flow = self.model(zt, clip_features, sync_features, text_features, t)
                
                # Compute loss
                loss = self.flow_matching.loss(predicted_flow, z0, z1).mean()
                
                total_loss += loss.item() * batch_size
                num_samples += batch_size
        
        self.model.train()
        
        return {'eval_loss': total_loss / num_samples}
    
    def save_checkpoint(self, tag: str = 'latest'):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model': self.model.state_dict(),
            'best_loss': self.best_loss,
            'config': self.config.__dict__
        }
        
        # Optionally save optimizer state
        if self.config.save_optimizer_state:
            checkpoint['optimizer'] = self.optimizer.state_dict()
            checkpoint['scheduler'] = self.scheduler.state_dict()
        
        checkpoint_path = self.checkpoint_dir / f'checkpoint_{tag}.pth'
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"💾 Saved checkpoint: {checkpoint_path}")
        
        # Save model only (for inference)
        if tag == 'best':
            model_path = self.checkpoint_dir / f'model_best.pth'
            torch.save(self.model.state_dict(), model_path)
            logger.info(f"💾 Saved best model: {model_path}")
    
    def load_checkpoint(self, checkpoint_path: Path):
        """Load model checkpoint"""
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        self.model.load_state_dict(checkpoint['model'])
        
        if 'optimizer' in checkpoint and self.config.save_optimizer_state:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.scheduler.load_state_dict(checkpoint['scheduler'])
        
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_loss = checkpoint.get('best_loss', float('inf'))
        
        logger.info(f"✅ Resumed from epoch {self.epoch}, step {self.global_step}")
    
    def train(self):
        """Main training loop optimized for M1"""
        logger.info("🚀 Starting training on Apple Silicon...")
        logger.info(f"Training for {self.config.num_epochs} epochs")
        logger.info(f"Gradient accumulation steps: {self.config.gradient_accumulation_steps}")
        
        self.model.train()
        
        for epoch in range(self.epoch, self.config.num_epochs):
            self.epoch = epoch
            epoch_loss = 0
            num_batches = 0
            
            # Training loop
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config.num_epochs}")
            
            for batch_idx, batch in enumerate(pbar):
                # Training step
                metrics = self.train_step(batch)
                
                epoch_loss += metrics['loss']
                num_batches += 1
                
                # Gradient accumulation
                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.gradient_clip
                    )
                    
                    # Optimizer step
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    
                    self.global_step += 1
                    
                    # Update progress bar
                    pbar.set_postfix({
                        'loss': f"{metrics['loss']:.4f}",
                        'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}",
                        'grad': f"{grad_norm:.2f}"
                    })
                    
                    # Memory management
                    if self.global_step % self.memory_check_interval == 0:
                        clear_memory()
                    
                    # Logging
                    if self.global_step % self.config.log_interval == 0:
                        avg_loss = epoch_loss / num_batches
                        logger.info(
                            f"Step {self.global_step} | "
                            f"Epoch {epoch+1}/{self.config.num_epochs} | "
                            f"Loss: {avg_loss:.4f} | "
                            f"LR: {self.optimizer.param_groups[0]['lr']:.2e}"
                        )
                    
                    # Evaluation
                    if self.global_step % self.config.eval_interval == 0:
                        eval_metrics = self.evaluate()
                        logger.info(f"Evaluation at step {self.global_step}: {eval_metrics}")
                        
                        # Save best model
                        if eval_metrics['eval_loss'] < self.best_loss:
                            self.best_loss = eval_metrics['eval_loss']
                            self.save_checkpoint('best')
                            logger.info(f"🏆 New best model! Loss: {self.best_loss:.4f}")
                    
                    # Save checkpoint
                    if self.global_step % self.config.save_interval == 0:
                        self.save_checkpoint('latest')
            
            # End of epoch
            avg_epoch_loss = epoch_loss / num_batches
            logger.info(f"Epoch {epoch+1} completed. Average loss: {avg_epoch_loss:.4f}")
            
            # Save epoch checkpoint
            self.save_checkpoint(f'epoch_{epoch+1}')
            
            # Clear memory after each epoch
            clear_memory()
        
        # Final checkpoint
        self.save_checkpoint('final')
        logger.info("🎉 Training completed!")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="MMAudio Fine-tuning for MacBook M1")
    
    # Model
    parser.add_argument('--model', type=str, default='small_16k',
                       choices=['small_16k', 'small_44k', 'medium_44k', 'large_44k'],
                       help='Model architecture (large_44k_v2 not recommended for M1)')
    parser.add_argument('--pretrained_path', type=str, default=None,
                       help='Path to pretrained model')
    
    # Data
    parser.add_argument('--data_dir', type=str, default='./output/memmap',
                       help='Directory with memory-mapped features')
    parser.add_argument('--tsv_file', type=str, default='./output/memmap/vgg-my_videos.tsv',
                       help='TSV file with metadata')
    
    # Training
    parser.add_argument('--exp_id', type=str, default='finetune_m1',
                       help='Experiment ID')
    parser.add_argument('--batch_size', type=int, default=2,
                       help='Batch size (2-4 recommended for M1)')
    parser.add_argument('--num_epochs', type=int, default=50,
                       help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                       help='Weight decay')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2,
                       help='Gradient accumulation steps')
    
    # Flow matching
    parser.add_argument('--num_flow_steps', type=int, default=25,
                       help='Number of flow steps')
    parser.add_argument('--cfg_strength', type=float, default=4.5,
                       help='CFG strength')
    
    # Device
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'mps', 'cpu'],
                       help='Device to use (auto will detect best option)')
    
    # Other
    parser.add_argument('--output_dir', type=str, default='./output/finetune',
                       help='Output directory')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from checkpoint')
    parser.add_argument('--num_workers', type=int, default=2,
                       help='Number of data workers (2-4 for M1)')
    parser.add_argument('--save_optimizer_state', action='store_true',
                       help='Save optimizer state (uses more memory)')
    
    args = parser.parse_args()
    
    # Create config
    config = TrainingConfig(
        model=args.model,
        pretrained_path=args.pretrained_path,
        data_dir=args.data_dir,
        tsv_file=args.tsv_file,
        exp_id=args.exp_id,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_flow_steps=args.num_flow_steps,
        cfg_strength=args.cfg_strength,
        device=args.device,
        output_dir=args.output_dir,
        num_workers=args.num_workers,
        save_optimizer_state=args.save_optimizer_state
    )
    
    # Log configuration
    logger.info("Configuration:")
    for key, value in config.__dict__.items():
        logger.info(f"  {key}: {value}")
    
    # Create trainer
    trainer = MMAudioFineTunerM1(config)
    
    # Resume if checkpoint provided
    if args.resume:
        trainer.load_checkpoint(Path(args.resume))
    
    # Start training
    try:
        trainer.train()
    except KeyboardInterrupt:
        logger.info("🛑 Training interrupted by user")
        trainer.save_checkpoint('interrupted')
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        raise
    finally:
        # Final cleanup
        clear_memory()


if __name__ == '__main__':
    main()