# Expanded dataset preparation snapshot

Updated: 2026-10-07. This public snapshot contains code and aggregate evidence only; private manifests and receipts stay outside Git.

- **VERIFIED FACT:** The expanded `small_44k` media completion receipt covers exactly 7,740 canonical IDs (train 6,203 / val 780 / test 757; 85 source recordings), with `training_ready=false`. Receipt SHA256: `37a5764317ffe4685b9ac6bfffa495bfd3097b07dcb146e478b21cbb14f9dfe3`.
- **VERIFIED FACT:** Human Caption review is unfinished, confirmed by the user. The generated review template retains candidates and marks every row pending; it does not confer approval.
- **VERIFIED FACT:** The preparation code preserves the existing media identity/extraction modules byte for byte. It adds separately bound text extraction, portable five-modality bundles, destination bindings and SGE jobs.
- **VERIFIED FACT:** The user changed the compute allocation: **MMAudio training stays in Colab; Myriad is reserved for later perception embedding training.**
- **PLANNED:** Finish Caption review/freeze and text extraction in Colab; assemble and verify the final bundle; stage it to local `/content`; run a Drive-restore smoke; train in bounded Colab sessions with verified Drive recovery snapshots.
- **UNKNOWN / REQUIRES CONFIRMATION:** Actual Colab GPU memory/performance and the complete training environment are unvalidated. Myriad connection/storage details and the later perception embedding model, data, objective and labels remain to be defined for that separate stage.
- **HISTORICAL RESULT:** The separate 6,941-clip custom `small_16k` pilot is not a result for this expanded official workflow. Its files and code are preserved.

Start with [the Colab training runbook](docs/COLAB_OFFICIAL_TRAINING.md), [notebook](notebooks/colab_official_training.ipynb) and [configuration template](config/colab.example.json). The Myriad MMAudio scripts are an archived alternative and are not a perception embedding trainer. Real data files, locally generated jobs, review sheets and receipts belong under ignored `outputs/` or external storage. No new transfer, GPU smoke, formal training or evaluation was run during preparation.
