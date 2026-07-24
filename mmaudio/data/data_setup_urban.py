"""Single-dataset loaders for urban video-to-audio fine-tuning."""

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from mmaudio.data.extracted_vgg import ExtractedVGG


def make_video_dataset(tsv_path, memmap_dir, cfg):
    return ExtractedVGG(
        tsv_path=tsv_path,
        premade_mmap_dir=memmap_dir,
        data_dim=cfg.data_dim,
    )


def setup_training_datasets(cfg):
    dataset = make_video_dataset(
        cfg.data.ExtractedVGG.tsv,
        cfg.data.ExtractedVGG.memmap_dir,
        cfg,
    )
    sampler = DistributedSampler(dataset, shuffle=True, drop_last=True)
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        sampler=sampler,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=True,
        persistent_workers=(cfg.num_workers > 0),
    )
    return dataset, sampler, loader


def setup_val_datasets(cfg):
    val_dataset = make_video_dataset(
        cfg.data.ExtractedVGG_val.tsv,
        cfg.data.ExtractedVGG_val.memmap_dir,
        cfg,
    )
    test_dataset = make_video_dataset(
        cfg.data.ExtractedVGG_test.tsv,
        cfg.data.ExtractedVGG_test.memmap_dir,
        cfg,
    )
    common = {
        "batch_size": cfg.eval_batch_size,
        "shuffle": False,
        "num_workers": cfg.num_workers,
        "pin_memory": cfg.pin_memory,
        "drop_last": False,
        "persistent_workers": cfg.num_workers > 0,
    }
    return (
        val_dataset,
        DataLoader(val_dataset, **common),
        DataLoader(test_dataset, **common),
    )
