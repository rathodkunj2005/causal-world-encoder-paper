# Causal World Encoder

Real-data experiments and paper for a small-scale study of causal world-model design for agents.

What this repo contains
- `experiments/`: reproducible Python experiments
- `results/json/`: recorded numeric outputs from real runs
- `results/figures/`: figures generated from those runs
- `paper/main.tex`: LaTeX manuscript
- `scripts/download_data.sh`: fetches public datasets used here

Datasets
- LoCoMo text release from Snap Research
- BabyAI-KeyCorridorS3R1 offline trajectories
- BabyAI-UnlockPickup offline trajectories
- Live arXiv abstracts fetched at runtime for the workspace experiment

Main finding
- Twin-stream action/world decomposition helped held-out action generalization.
- Bounded workspace reconstruction substantially reduced prompt load.
- HTBS did not beat simpler sequence baselines in this implementation.
- Sparse commitment hit the budget target and isolated a stronger committed subset, but only weakly separated high- from low-change transitions.

How to reproduce
1. Download data:
   `bash scripts/download_data.sh`
2. Run experiments:
   `python3 experiments/run_real_experiments.py`
3. Build the paper:
   `cd paper && /Library/TeX/texbin/pdflatex -interaction=nonstopmode main.tex && /Library/TeX/texbin/bibtex main && /Library/TeX/texbin/pdflatex -interaction=nonstopmode main.tex && /Library/TeX/texbin/pdflatex -interaction=nonstopmode main.tex`

Notes
- The repo uses real data only. No mock results are included.
- Training was run on CPU for numerical stability.
- A fresh `.venv` install failed on this machine because the data volume ran out of free space while downloading large wheels (`torch` during `pip install -r requirements.txt`). The recorded experiments were run successfully with the already-installed system Python packages instead.
