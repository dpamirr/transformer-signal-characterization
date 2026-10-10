# Statistical Characterization of Decoder-Only Transformers on Synthetic Signals

Research project at Goethe University Frankfurt, supervised by Prof. Dr. Visvanathan Ramesh.

This repository studies how decoder-only Transformers behave on controlled synthetic time-series signals under increasing levels of Gaussian noise. The goal is to characterize predictive, generative, and structural fidelity under known data-generating conditions.

## Setup

```bash
pip install -r requirements.txt
```

Run the notebooks from anywhere inside the cloned repository. Paths are resolved relative to the repository root rather than to a specific user account or machine.

## Repository Structure

```text
.
├── models/
│   ├── sinusoid/
│   ├── triangle/
│   ├── square/
│   └── delta/
├── notebooks/
│   ├── experiments/
│   ├── analysis/
│   └── ablations/
├── plots/
├── results/
│   └── multi_seed/
├── requirements.txt
└── README.md
```

Model checkpoints are intentionally excluded because of their size. Experiment notebooks recreate them when needed.

## Main Experiments

- `notebooks/experiments/multi_seed_all_signals.ipynb`  
  Main experiment across 4 signal families, 5 noise levels, and 10 random seeds (`4 × 5 × 10 = 200` independently trained models).

- `notebooks/experiments/sinusoid_experiment.ipynb`  
  Baseline sinusoid experiment and Gaussian-noise sweep.

- `notebooks/experiments/triangle_experiment.ipynb`  
  Triangle-wave noise sweep.

- `notebooks/experiments/square_experiment.ipynb`  
  Square-wave noise sweep.

- `notebooks/experiments/delta_experiment.ipynb`  
  Impulse-like / delta-signal noise sweep.

## Analysis

- `notebooks/analysis/model_based_comparison.ipynb`  
  Comparison between Transformer forecasting and simple parametric signal models.

- `notebooks/analysis/teacher_forcing_test.ipynb`  
  Teacher-forced evaluation used to distinguish one-step prediction degradation from errors caused by autoregressive rollout.

- `notebooks/analysis/surrogate_statistics.ipynb`  
  Exploratory structural-fidelity analysis using statistics such as variance and spectral concentration.

- `notebooks/analysis/results_analysis.ipynb`  
  Aggregate analysis and visualization of the multi-seed experiment.

## Ablation Studies

The notebooks in `notebooks/ablations/` contain targeted tests of:

- corrected attention scaling
- fixed-range normalization
- dropout regularization

These experiments test whether implementation details, representation choices, or ordinary overfitting account for the degradation observed under noise.

## Experimental Overview

Synthetic signals:

- sinusoid
- triangle wave
- square wave
- impulse-like / delta signal

Gaussian noise levels:

```text
σ ∈ {0.0, 0.1, 0.2, 0.5, 1.0}
```

For each signal/noise combination in the main experiment, 10 independently trained Transformer models are evaluated.

The main model is a relatively small GPT-style decoder-only Transformer with:

- 4 Transformer blocks
- 4 attention heads
- model dimension `d_model = 128`
- feed-forward dimension `512`
- context length `512` tokens
- learned absolute positional embeddings
- digit-level next-token prediction

## Results

Main multi-seed results are stored in:

```text
results/multi_seed/all_results.json
```

Generated plots are stored in `plots/`.

## Reproducibility Notes

Trained model checkpoints are not included in the repository because of their size. Run the corresponding experiment notebooks to regenerate them. The expected checkpoint layout is documented in `models/README.md`.

The structural-fidelity experiments are exploratory and should not be interpreted as formal model-level statistical inference. Repeated stochastic rollouts from the same trained model measure generation variability, but they are not equivalent to independently trained model seeds.
