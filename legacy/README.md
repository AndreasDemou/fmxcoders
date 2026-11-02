# Legacy files

Scripts kept for reference from earlier work (PolySAE experiments) that this codebase grew out of.
They are not needed to train or evaluate fmxcoders and are not maintained.

| File | What it was |
|---|---|
| `train_sae.py` | Earlier single-layer SAE training script (superseded by `../train_and_saebench.py`) |
| `train_and_saebench_old.py` | Previous version of `../train_and_saebench.py` |
| `run_commands.txt` | Command list for the PolySAE (single-layer, polynomial decoder) experiments |
| `debug_folding.py` | One-off debugging script for PolySAE decoder-norm folding |

Run them from the repository root (e.g. `python legacy/train_sae.py ...`) so that `sae_lens` resolves to this repo's fork.
