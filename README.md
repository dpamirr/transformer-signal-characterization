# Statistical Characterization of Decoder-Only Transformers on Synthetic Signals

Research project, Goethe University Frankfurt, supervised by Prof. Dr. V. Ramesh.

## Setup

pip install torch numpy matplotlib scipy

## Experiments
- `multi_seed_all_signals.ipynb` — full 200-model experiment (4 signals × 5 noise × 10 seeds)
- `model_based_comparison.ipynb` — parametric fitting vs Transformer
- `teacher_forcing_test.ipynb` — learning vs generation failure analysis
- `surrogate_statistics.ipynb` — formal statistical testing (exploratory)
- `*_test.ipynb` — ablation studies (attention fix, normalization, dropout)

## Results
All results saved in `results/multi_seed/all_results.json`.
Trained models not included due to size — rerun notebooks to reproduce.
