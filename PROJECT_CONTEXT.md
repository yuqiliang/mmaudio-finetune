# Expanded dataset preparation snapshot

Updated: 2026-10-07. This public snapshot contains code and aggregate evidence only; private manifests and receipts stay outside Git.

- **VERIFIED FACT:** The expanded `small_44k` media completion receipt covers exactly 7,740 canonical IDs (train 6,203 / val 780 / test 757; 85 source recordings), with `training_ready=false`. Receipt SHA256: `37a5764317ffe4685b9ac6bfffa495bfd3097b07dcb146e478b21cbb14f9dfe3`.
- **VERIFIED FACT:** Human Caption review is unfinished, confirmed by the user. The generated review template retains candidates and marks every row pending; it does not confer approval.
- **VERIFIED FACT:** The preparation code preserves the existing media identity/extraction modules byte for byte. It adds separately bound text extraction, portable five-modality bundles, destination bindings and SGE jobs.
- **PLANNED:** Finish Caption review/freeze and text extraction in Colab; assemble and verify the final bundle there; transfer from Drive directly to Myriad; run destination verification and checkpoint-resume smoke before formal training.
- **UNKNOWN / REQUIRES CONFIRMATION:** Myriad access exists, but the read-only SSH check timed out. Destination paths, quota, modules, Python/CUDA compatibility and GPU performance remain unverified.
- **HISTORICAL RESULT:** The separate 6,941-clip custom `small_16k` pilot is not a result for this expanded official workflow. Its files and code are preserved.

Start with [the migration runbook](docs/MYRIAD_MIGRATION.md) and [the local configuration template](config/myriad.example.json). Real data files, locally generated jobs, review sheets and receipts belong under ignored `outputs/` or external storage. No new transfer, GPU smoke, formal training or evaluation was run during preparation.
