# Model checkpoints

Trained PyTorch checkpoints are intentionally not tracked because of their size.
Run the experiment notebooks to regenerate them. Expected locations are:

- `models/sinusoid/baseline_clean.pt` — clean baseline sinusoid model
- `models/sinusoid/sigma{noise}.pt` — sinusoid noise-sweep checkpoints
- `models/triangle/sigma{noise}.pt`
- `models/square/sigma{noise}.pt`
- `models/delta/sigma{noise}.pt`

where `{noise}` is one of `0.0`, `0.1`, `0.2`, `0.5`, or `1.0`.
