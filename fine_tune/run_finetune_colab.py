import os
import subprocess
from pathlib import Path
import yaml

# ==============================
# PATH CONFIG
# ==============================

REPO_ROOT = Path(".").resolve()

# 正式训练用 full memmap
MEMMAP_DIR = Path("./mmaudio_latents_stride8/training_latents")

TRAIN_TSV = Path("./fine_tune/training/video_train_ft.tsv")
VAL_TSV   = Path("./fine_tune/training/video_val_ft.tsv")
TEST_TSV  = Path("./fine_tune/training/video_test_ft.tsv")

FLOW_WEIGHTS = REPO_ROOT / "weights" / "mmaudio_small_16k.pth"

# ==============================
# TRAINING CONFIG
# ==============================

EXP_ID = "urban_ft_stride10_v1"
MODEL = "small_16k"

BATCH_SIZE = 8
LR = 2e-5
STEPS = 8000
WARMUP = 500

VAL_INTERVAL = 500
EVAL_INTERVAL = 2000
SAVE_EVAL_INTERVAL = 4000
SAVE_WEIGHTS_INTERVAL = 1000
SAVE_CHECKPOINT_INTERVAL = 1000

NUM_WORKERS = 2
AMP = True
COMPILE = False

# ==============================
# CHECK
# ==============================

required_paths = [
    REPO_ROOT,
    MEMMAP_DIR,
    TRAIN_TSV,
    VAL_TSV,
    TEST_TSV,
    FLOW_WEIGHTS,
]

print("=== Path check ===")
all_ok = True
for p in required_paths:
    ok = p.exists()
    print(f"{p}: {'OK' if ok else 'MISSING'}")
    if not ok:
        all_ok = False

if not all_ok:
    raise FileNotFoundError("Some required paths are missing.")

# ==============================
# UPDATE DATA CONFIG
# ==============================

config_file = REPO_ROOT / "config" / "data" / "base.yaml"

with open(config_file, "r") as f:
    cfg = yaml.safe_load(f)

cfg["ExtractedVGG"] = {
    "tsv": str(TRAIN_TSV),
    "memmap_dir": str(MEMMAP_DIR),
}

cfg["ExtractedVGG_val"] = {
    "tag": "val",
    "tsv": str(VAL_TSV),
    "memmap_dir": str(MEMMAP_DIR),
    "gt_cache": None,
    "output_subdir": "val",
}

cfg["ExtractedVGG_test"] = {
    "tag": "test",
    "tsv": str(TEST_TSV),
    "memmap_dir": str(MEMMAP_DIR),
    "gt_cache": None,
    "output_subdir": None,
}

with open(config_file, "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)

print("\nUpdated config:", config_file)

# ==============================
# RUN TRAINING
# ==============================

os.chdir(REPO_ROOT)

cmd = [
    "torchrun",
    "--standalone",
    "--nproc_per_node=1",
    "train.py",
    f"exp_id={EXP_ID}",
    f"model={MODEL}",
    f"weights={FLOW_WEIGHTS}",
    f"batch_size={BATCH_SIZE}",
    f"learning_rate={LR}",
    f"num_iterations={STEPS}",
    f"linear_warmup_steps={WARMUP}",
    f"val_interval={VAL_INTERVAL}",
    f"eval_interval={EVAL_INTERVAL}",
    f"save_eval_interval={SAVE_EVAL_INTERVAL}",
    f"save_weights_interval={SAVE_WEIGHTS_INTERVAL}",
    f"save_checkpoint_interval={SAVE_CHECKPOINT_INTERVAL}",
    f"num_workers={NUM_WORKERS}",
    f"amp={str(AMP)}",
    f"compile={str(COMPILE)}",
]

print("\nRunning command:")
print(" ".join(map(str, cmd)))

env = os.environ.copy()
env["OMP_NUM_THREADS"] = "2"

subprocess.run(cmd, check=True, env=env)