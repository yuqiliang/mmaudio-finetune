#!/usr/bin/env python3
import logging
import os
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
import tensordict as td
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from mmaudio.data.data_setup import error_avoidance_collate
from mmaudio.data.extraction.vgg_sound import VGGSound
from mmaudio.model.utils.features_utils import FeaturesUtils

# for the 16kHz model
SAMPLING_RATE = 16000
DURATION_SEC = 8.0
NUM_SAMPLES = 128000
vae_path = './ext_weights/v1-16.pth'
bigvgan_path = './ext_weights/best_netG.pt'
mode = '16k'

# for the 44.1kHz model
"""
NOTE: 352800 (8*44100) is not divisible by (STFT hop size * VAE downsampling ratio) which is 1024.
353280 is the next integer divisible by 1024.
"""
# SAMPLING_RATE = 44100
# DURATION_SEC = 8.0
# NUM_SAMPLES = 353280
# vae_path = './ext_weights/v1-44.pth'
# bigvgan_path = None
# mode = '44k'

synchformer_ckpt = './ext_weights/synchformer_state_dict.pth'

# MacBook 推荐：小 batch，降低 OOM 风险
BATCH_SIZE = 4
NUM_WORKERS = 4

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')

data_cfg = {
    'example': {
        'root': './training/example_videos',
        'subset_name': './training/example_video.tsv',
        'normalize_audio': True,
    },
}


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        log.info('Using MPS device.')
        return torch.device('mps')
    if torch.cuda.is_available():
        log.info('Using CUDA device.')
        return torch.device('cuda')
    log.warning('Neither MPS nor CUDA available, falling back to CPU.')
    return torch.device('cpu')


def clear_device_cache(device: torch.device):
    if device.type == 'mps':
        torch.mps.empty_cache()
    elif device.type == 'cuda':
        torch.cuda.empty_cache()


def setup_dataset(split: str):
    dataset = VGGSound(
        data_cfg[split]['root'],
        tsv_path=data_cfg[split]['subset_name'],
        sample_rate=SAMPLING_RATE,
        duration_sec=DURATION_SEC,
        audio_samples=NUM_SAMPLES,
        normalize_audio=data_cfg[split]['normalize_audio'],
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        shuffle=False,
        drop_last=False,
        collate_fn=error_avoidance_collate,
        pin_memory=False,
    )

    return dataset, loader


@torch.inference_mode()
def extract():
    parser = ArgumentParser(description='Extract video training latents for MacBook (MPS).')
    parser.add_argument('--latent_dir', type=Path, default='./training/example_output/video-latents')
    parser.add_argument('--output_dir', type=Path, default='./training/example_output/memmap')
    args = parser.parse_args()

    latent_dir = args.latent_dir
    output_dir = args.output_dir

    device = get_device()

    feature_extractor = FeaturesUtils(
        tod_vae_ckpt=vae_path,
        enable_conditions=True,
        bigvgan_vocoder_ckpt=bigvgan_path,
        synchformer_ckpt=synchformer_ckpt,
        mode=mode,
    ).eval().to(device)

    for split in data_cfg.keys():
        log.info('Extracting latents for split: %s', split)
        this_latent_dir = latent_dir / split
        this_latent_dir.mkdir(parents=True, exist_ok=True)

        dataset, loader = setup_dataset(split)
        log.info('Number of samples: %d', len(dataset))
        log.info('Number of batches: %d', len(loader))

        for curr_iter, data in enumerate(tqdm(loader)):
            output = {
                'id': data['id'],
                'caption': data['caption'],
            }

            audio = data['audio'].to(device)
            dist = feature_extractor.encode_audio(audio)
            output['mean'] = dist.mean.detach().cpu().transpose(1, 2)
            output['std'] = dist.std.detach().cpu().transpose(1, 2)

            clip_video = data['clip_video'].to(device)
            clip_features = feature_extractor.encode_video_with_clip(clip_video)
            output['clip_features'] = clip_features.detach().cpu()

            sync_video = data['sync_video'].to(device)
            sync_features = feature_extractor.encode_video_with_sync(sync_video)
            output['sync_features'] = sync_features.detach().cpu()

            caption = data['caption']
            text_features = feature_extractor.encode_text(caption)
            output['text_features'] = text_features.detach().cpu()

            torch.save(output, this_latent_dir / f'b{curr_iter}.pth')

            if curr_iter % 20 == 0:
                clear_device_cache(device)

        log.info('Extraction done for %s. Combining results.', split)

        used_id = set()
        list_of_ids_and_labels = []
        output_data = {
            'mean': [],
            'std': [],
            'clip_features': [],
            'sync_features': [],
            'text_features': [],
        }

        for t in tqdm(sorted(os.listdir(this_latent_dir))):
            if not t.endswith('.pth'):
                continue
            data = torch.load(this_latent_dir / t, weights_only=True)
            bs = len(data['id'])

            for bi in range(bs):
                this_id = data['id'][bi]
                this_caption = data['caption'][bi]
                if this_id in used_id:
                    log.warning('Duplicate id: %s', this_id)
                    continue

                list_of_ids_and_labels.append({'id': this_id, 'label': this_caption})
                used_id.add(this_id)
                output_data['mean'].append(data['mean'][bi])
                output_data['std'].append(data['std'][bi])
                output_data['clip_features'].append(data['clip_features'][bi])
                output_data['sync_features'].append(data['sync_features'][bi])
                output_data['text_features'].append(data['text_features'][bi])

        output_dir.mkdir(parents=True, exist_ok=True)
        output_df = pd.DataFrame(list_of_ids_and_labels)
        output_df.to_csv(output_dir / f'vgg-{split}.tsv', sep='\t', index=False)

        log.info('Output samples: %d', len(output_df))

        output_data = {k: torch.stack(v) for k, v in output_data.items()}
        td.TensorDict(output_data).memmap_(output_dir / f'vgg-{split}')


if __name__ == '__main__':
    extract()
