# Expanded workflow preparation log

## 2026-10-07 — Migration and training preparation

**VERIFIED FACT:** Revalidated the lightweight `MEDIA_7740_CPU_VERIFIED` completion receipt against the exact media plan, media run, all canonical IDs in extraction order and all 16 batch indices. Completion time recorded by the receipt: 2026-10-07 13:41:20 UTC. SHA256: `37a5764317ffe4685b9ac6bfffa495bfd3097b07dcb146e478b21cbb14f9dfe3`. This preparation did not re-download or re-audit all real tensor payloads.

**VERIFIED FACT:** Generated a private 7,740-row review template with zero approved rows. The user confirmed Caption human review is not yet complete. Candidate captions and original media artifacts were not overwritten.

**VERIFIED FACT:** Added CPU-tested contracts for Caption freezing, text-cache recovery, final memmap assembly and relocation, destination SHA256 validation, SGE config protection, and a two-process checkpoint probe. Synthetic CPU tensors test serialization and exact readback; they are not evidence of GPU numerical behavior. Captured the existing Colab media dependency snapshot as a reference, not as a validated Myriad training environment.

Validation: `python -B -m unittest discover -s tests -q` — **116 tests passed**, using local CPU PyTorch 2.6.0 / TensorDict 0.7.1. Python AST parsing, generated qsub `bash -n`, transfer-script syntax, and staged whitespace checks passed. The three media identity/extraction file hashes match [the preserved source snapshot](docs/media_source_snapshot.json).

**UNKNOWN / REQUIRES CONFIRMATION:** A read-only SSH connection to the configured Myriad host timed out. No remote environment installation, transfer or job submission occurred.

**PLANNED:** Follow [MYRIAD_MIGRATION.md](docs/MYRIAD_MIGRATION.md). Record a new evidence-linked entry after the real Caption freeze, text extraction, final assembly, destination verification, GPU smoke, formal training and final evaluation. Keep test evaluation separate from training/validation.
