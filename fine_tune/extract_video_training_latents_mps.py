#!/usr/bin/env python3
"""
extract_video_training_latents.py - MacBook Version
=========================================================================
This script extracts audio-video-text features for MMAudio fine-tuning.
Optimized for MacBook with Apple Silicon (MPS support).
Simplified version without psutil dependency.



Usage:
    export HF_ENDPOINT=https://hf-mirror.com
    caffeinate python3 training/extract_video_training_latents_mps.py --latent_dir ./output/latents --output_dir ./output/memmap
"""

import logging
import os
import sys
import gc
from argparse import ArgumentParser
from datetime import timedelta
from pathlib import Path

import pandas as pd
import tensordict as td
import torch
import torch.distributed as distributed
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from mmaudio.data.data_setup import error_avoidance_collate
from mmaudio.data.extraction.vgg_sound import VGGSound
from mmaudio.model.utils.features_utils import FeaturesUtils
from mmaudio.utils.dist_utils import local_rank, world_size

# Model configuration for 16kHz
SAMPLING_RATE = 16000
DURATION_SEC = 8.0
NUM_SAMPLES = 128000
vae_path = './ext_weights/v1-16.pth'
bigvgan_path = './ext_weights/best_netG.pt'
mode = '16k'

# For 44.1kHz model
# SAMPLING_RATE = 44100
# DURATION_SEC = 8.0
# NUM_SAMPLES = 353280
# vae_path = './ext_weights/v1-44.pth'
# bigvgan_path = None
# mode = '44k'

synchformer_ckpt = './ext_weights/synchformer_state_dict.pth'

# MacBook optimized settings
BATCH_SIZE = 4          # Reduced for MacBook memory constraints
NUM_WORKERS = 4         # Reduced for MacBook CPU cores
MEMORY_CHECK_INTERVAL = 50  # Check memory every N batches

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

# ==========================================
# DATA CONFIGURATION - MODIFY THIS SECTION
# ==========================================
# Update these paths to match your setup
data_cfg = {
    'my_videos': {
        'root': './training_videos',
        'subset_name': './training/my_video_train.tsv',
        'normalize_audio': True,
    },
    # You can add more datasets here if needed
    # 'additional_videos': {
    #     'root': './additional_videos',
    #     'subset_name': './training/additional_train.tsv',
    #     'normalize_audio': True,
    # },
}

def check_system_resources():
    """Check and log system resources (simplified version)"""
    log.info("=== System Resources Check ===")
    
    # CPU info (basic)
    try:
        import multiprocessing
        cpu_count = multiprocessing.cpu_count()
        log.info(f"CPU cores: {cpu_count}")
    except:
        log.info("CPU cores: Unable to determine")
    
    # GPU/MPS
    if torch.backends.mps.is_available():
        log.info("GPU: MPS (Metal Performance Shaders) available")
    elif torch.cuda.is_available():
        try:
            gpu_name = torch.cuda.get_device_name(0)
            memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            log.info(f"GPU: {gpu_name} ({memory_gb:.1f} GB)")
        except:
            log.info("GPU: CUDA available")
    else:
        log.info("GPU: None available, using CPU")
    
    log.info("================================")

def get_device():
    """Determine the best available device"""
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        log.info("✅ Using MPS (Metal Performance Shaders)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        log.info(f"✅ Using CUDA")
    else:
        device = torch.device("cpu")
        log.info("⚠️  Using CPU (will be very slow)")
    
    return device

def clear_memory_cache():
    """Clear memory cache for different devices"""
    if torch.backends.mps.is_available():
        try:
            torch.mps.empty_cache()
        except:
            pass
    elif torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except:
            pass
    gc.collect()

def check_memory_usage():
    """Simplified memory check without psutil"""
    try:
        # Simple memory cleanup
        clear_memory_cache()
        log.debug("Memory cache cleared")
    except:
        pass

def distributed_setup():
    """Setup distributed training (with fallback for single device)"""
    try:
        if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
            # Running with torchrun
            distributed.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo", 
                                         timeout=timedelta(hours=1))
            rank = distributed.get_rank()
            world_size = distributed.get_world_size()
            log.info(f'Distributed training: rank={rank}, world_size={world_size}')
            return rank, world_size
        else:
            # Single device mode
            log.info('Single device mode (no distributed training)')
            return 0, 1
    except Exception as e:
        log.warning(f"Distributed setup failed: {e}. Using single device mode.")
        return 0, 1

def verify_data_config():
    """Verify that data configuration is valid"""
    log.info("=== Verifying Data Configuration ===")
    
    for split_name, config in data_cfg.items():
        log.info(f"Checking dataset: {split_name}")
        
        # Check video directory
        video_dir = Path(config['root'])
        if not video_dir.exists():
            log.error(f"Video directory not found: {video_dir}")
            log.error("Please create the directory and add your video files")
            return False
        
        # Count video files
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
        video_files = []
        for ext in video_extensions:
            video_files.extend(list(video_dir.glob(f'*{ext}')))
            video_files.extend(list(video_dir.glob(f'*{ext.upper()}')))
        
        if not video_files:
            log.error(f"No video files found in {video_dir}")
            log.error(f"Supported formats: {video_extensions}")
            return False
        
        log.info(f"  Found {len(video_files)} video files")
        
        # Check TSV file
        tsv_path = Path(config['subset_name'])
        if not tsv_path.exists():
            log.error(f"TSV file not found: {tsv_path}")
            log.error("Creating TSV file automatically...")
            
            # Auto-create TSV file
            create_tsv_file(video_files, tsv_path)
        else:
            # Verify TSV content
            try:
                df = pd.read_csv(tsv_path, sep='\t')
                log.info(f"  TSV file: {len(df)} entries")
                
                # Check required columns
                if 'id' not in df.columns or 'label' not in df.columns:
                    log.error("TSV file must have 'id' and 'label' columns")
                    return False
                    
            except Exception as e:
                log.error(f"Error reading TSV file: {e}")
                return False
    
    log.info("✅ Data configuration verified")
    return True

def create_tsv_file(video_files, tsv_path):
    """Auto-create TSV file from video files"""
    log.info(f"Creating TSV file: {tsv_path}")
    
    tsv_data = []
    for video_path in video_files:
        video_id = video_path.stem  # filename without extension
        tsv_data.append({
            'id': video_id,
            'label': 'audio from video'  # Generic label for fine-tuning
        })
    
    df = pd.DataFrame(tsv_data)
    
    # Ensure directory exists
    tsv_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save TSV file
    df.to_csv(tsv_path, sep='\t', index=False)
    log.info(f"✅ Created TSV file with {len(df)} entries")

def setup_dataset(split: str, rank: int, world_size: int):
    """Setup dataset and dataloader"""
    log.info(f"Setting up dataset for split: {split}")
    
    dataset = VGGSound(
        data_cfg[split]['root'],
        tsv_path=data_cfg[split]['subset_name'],
        sample_rate=SAMPLING_RATE,
        duration_sec=DURATION_SEC,
        audio_samples=NUM_SAMPLES,
        normalize_audio=data_cfg[split]['normalize_audio'],
    )
    
    # Setup sampler based on distributed training
    if world_size > 1:
        sampler = DistributedSampler(dataset, rank=rank, shuffle=False)
        log.info(f"Using DistributedSampler: rank={rank}, world_size={world_size}")
    else:
        sampler = None
        log.info("Using regular sampling")
    
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        sampler=sampler,
        shuffle=False if sampler else False,
        drop_last=False,
        collate_fn=error_avoidance_collate,
        pin_memory=False,  # Disable for MacBook to save memory
    )

    log.info(f"Dataset setup complete: {len(dataset)} samples, {len(loader)} batches")
    return dataset, loader

@torch.inference_mode()
def extract():
    """Main extraction function"""
    log.info("🚀 Starting MMAudio Feature Extraction")
    log.info("=" * 50)
    
    # System checks
    check_system_resources()
    
    # Verify data configuration
    if not verify_data_config():
        log.error("❌ Data configuration verification failed")
        sys.exit(1)
    
    # Setup distributed training
    rank, world_size_val = distributed_setup()
    
    # Parse arguments
    parser = ArgumentParser(description="Extract video training features for MMAudio")
    parser.add_argument('--latent_dir', type=Path, 
                        default='./fine_tuning_data/output/video-latents',
                        help='Directory to save intermediate latent files')
    parser.add_argument('--output_dir', type=Path, 
                        default='./fine_tuning_data/output/memmap',
                        help='Directory to save final memory-mapped tensors')
    args = parser.parse_args()

    latent_dir = Path(args.latent_dir)
    output_dir = Path(args.output_dir)
    
    log.info(f"📁 Latent directory: {latent_dir}")
    log.info(f"📁 Output directory: {output_dir}")

    # Device setup
    device = get_device()
    
    # Load feature extractor
    log.info("🔧 Loading feature extractor models...")
    try:
        feature_extractor = FeaturesUtils(
            tod_vae_ckpt=vae_path,
            enable_conditions=True,
            bigvgan_vocoder_ckpt=bigvgan_path,
            synchformer_ckpt=synchformer_ckpt,
            mode=mode
        ).eval().to(device)
        log.info("✅ Feature extractor loaded successfully")
    except Exception as e:
        log.error(f"❌ Failed to load feature extractor: {e}")
        sys.exit(1)

    # Process each split
    for split in data_cfg.keys():
        log.info(f"\n🎬 Processing split: {split}")
        log.info("=" * 30)
        
        # Create output directory for this split
        this_latent_dir = latent_dir / split
        this_latent_dir.mkdir(parents=True, exist_ok=True)

        # Setup dataset
        try:
            dataset, loader = setup_dataset(split, rank, world_size_val)
        except Exception as e:
            log.error(f"❌ Failed to setup dataset for {split}: {e}")
            continue

        # Extract features
        log.info(f"🔄 Starting feature extraction for {len(dataset)} samples...")
        
        batch_count = 0
        for curr_iter, data in enumerate(tqdm(loader, desc=f"Extracting {split}")):
            try:
                # Prepare output dictionary
                output = {
                    'id': data['id'],
                    'caption': data['caption'],
                }

                # Extract audio features
                audio = data['audio'].to(device)
                dist = feature_extractor.encode_audio(audio)
                output['mean'] = dist.mean.detach().cpu().transpose(1, 2)
                output['std'] = dist.std.detach().cpu().transpose(1, 2)

                # Extract video CLIP features
                clip_video = data['clip_video'].to(device)
                clip_features = feature_extractor.encode_video_with_clip(clip_video)
                output['clip_features'] = clip_features.detach().cpu()

                # Extract synchronization features
                sync_video = data['sync_video'].to(device)
                sync_features = feature_extractor.encode_video_with_sync(sync_video)
                output['sync_features'] = sync_features.detach().cpu()

                # Extract text features
                caption = data['caption']
                text_features = feature_extractor.encode_text(caption)
                output['text_features'] = text_features.detach().cpu()

                # Save batch results
                output_file = this_latent_dir / f'r{rank}_{curr_iter}.pth'
                torch.save(output, output_file)
                
                batch_count += 1
                
                # Periodic memory check
                if batch_count % MEMORY_CHECK_INTERVAL == 0:
                    check_memory_usage()
                    log.info(f"Processed {batch_count}/{len(loader)} batches")

            except Exception as e:
                log.error(f"❌ Error processing batch {curr_iter}: {e}")
                continue

        log.info(f"✅ Feature extraction completed for {split}")

        # Synchronize processes if using distributed training
        if world_size_val > 1:
            distributed.barrier()
            log.info("⏳ Waiting for all processes to complete...")

        # Combine results (only rank 0)
        if rank == 0:
            log.info(f"🔄 Combining results for {split}...")
            combine_results(this_latent_dir, output_dir, split)

    log.info("\n🎉 All feature extraction completed!")

def combine_results(latent_dir: Path, output_dir: Path, split: str):
    """Combine individual batch results into final tensors"""
    log.info(f"📦 Combining results for {split}")
    
    used_id = set()
    list_of_ids_and_labels = []
    output_data = {
        'mean': [],
        'std': [],
        'clip_features': [],
        'sync_features': [],
        'text_features': [],
    }

    # Process all batch files
    batch_files = sorted([f for f in os.listdir(latent_dir) if f.endswith('.pth')])
    log.info(f"Processing {len(batch_files)} batch files...")

    for filename in tqdm(batch_files, desc="Combining batches"):
        try:
            data = torch.load(latent_dir / filename, weights_only=True)
            batch_size = len(data['id'])

            for bi in range(batch_size):
                this_id = data['id'][bi]
                this_caption = data['caption'][bi]
                
                if this_id in used_id:
                    log.warning(f'Duplicate id: {this_id}')
                    continue

                list_of_ids_and_labels.append({'id': this_id, 'label': this_caption})
                used_id.add(this_id)
                
                # Collect features
                output_data['mean'].append(data['mean'][bi])
                output_data['std'].append(data['std'][bi])
                output_data['clip_features'].append(data['clip_features'][bi])
                output_data['sync_features'].append(data['sync_features'][bi])
                output_data['text_features'].append(data['text_features'][bi])

        except Exception as e:
            log.error(f"Error processing {filename}: {e}")
            continue

    if not list_of_ids_and_labels:
        log.error("No valid data found to combine")
        return

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save metadata TSV file
    output_df = pd.DataFrame(list_of_ids_and_labels)
    tsv_path = output_dir / f'vgg-{split}.tsv'
    output_df.to_csv(tsv_path, sep='\t', index=False)
    log.info(f"📄 Saved metadata: {tsv_path} ({len(output_df)} samples)")

    # Stack tensors and create memory-mapped files
    log.info("💾 Creating memory-mapped tensors...")
    try:
        output_data = {k: torch.stack(v) for k, v in output_data.items()}
        tensor_dict = td.TensorDict(output_data)
        memmap_path = output_dir / f'vgg-{split}'
        tensor_dict.memmap_(memmap_path)
        log.info(f"✅ Memory-mapped tensors saved: {memmap_path}")
        
        # Log tensor shapes for verification
        for key, tensor in output_data.items():
            log.info(f"  {key}: {tensor.shape}")
            
    except Exception as e:
        log.error(f"❌ Error creating memory-mapped tensors: {e}")
        return

    # Clean up intermediate files (optional)
    cleanup_latent_files = True  # Set to False if you want to keep intermediate files
    if cleanup_latent_files:
        log.info("🧹 Cleaning up intermediate files...")
        try:
            for filename in batch_files:
                (latent_dir / filename).unlink()
            log.info("✅ Cleanup completed")
        except Exception as e:
            log.warning(f"⚠️  Cleanup warning: {e}")

    log.info(f"🎉 Successfully processed {split}: {len(output_df)} samples")

if __name__ == '__main__':
    try:
        extract()
    except KeyboardInterrupt:
        log.info("🛑 Extraction interrupted by user")
        sys.exit(1)
    except Exception as e:
        log.error(f"❌ Fatal error: {e}")
        sys.exit(1)
    finally:
        # Cleanup
        if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
            try:
                distributed.destroy_process_group()
                log.info("🔚 Distributed training cleanup completed")
            except:
                pass
        
        # Final memory cleanup
        clear_memory_cache()
        log.info("✅ Feature extraction finished")