#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
print(f"GPU: {torch.cuda.get_device_name(0)}")


# In[2]:


block_size = 512
batch_size = 16
learning_rate = 0.001
weight_decay = 1e-3
dropout = 0.0
vocab_size = 10
n_head = 4
n_layer = 4
n_embd = 128

def discretize(y_norm, precision=3, base=10):
    tokens = []
    for val in y_norm:
        val = np.clip(val, 0, 1 - 1e-9)
        digits = []
        remaining = val
        for _ in range(precision):
            remaining *= base
            digit = int(remaining)
            digits.append(digit)
            remaining -= digit
        tokens.extend(digits)
    return tokens

def undiscretize(tokens, precision=3, base=10):
    values = []
    for i in range(0, len(tokens), precision):
        chunk = tokens[i:i+precision]
        if len(chunk) < precision:
            break
        val = 0
        for j, d in enumerate(chunk):
            val += d / (base ** (j + 1))
        values.append(val)
    return np.array(values)

def generate_signal(signal_type, A, omega, length, phase, dt=0.01):
    t = np.arange(length) * dt
    if signal_type == 'sinusoid':
        return A * np.sin(omega * t + phase)
    elif signal_type == 'triangle':
        return A * (2 / np.pi) * np.arcsin(np.sin(omega * t + phase))
    elif signal_type == 'square':
        return A * np.sign(np.sin(omega * t + phase))

def generate_training_data(signal_type, num_signals, signal_length, A, omega, noise_std=0.0, seed=None):
    if seed is not None:
        np.random.seed(seed)
    all_tokens = []
    for i in range(num_signals):
        phase = np.random.uniform(0, 2 * np.pi)
        y = generate_signal(signal_type, A, omega, signal_length, phase)
        if noise_std > 0:
            y += np.random.normal(0, noise_std, signal_length)
        y_range = A + 3 * noise_std if noise_std > 0 else A
        y_norm = (y - (-y_range)) / (2 * y_range)
        y_norm = np.clip(y_norm, 0, 1 - 1e-9)
        tokens = discretize(y_norm)
        all_tokens.append(torch.tensor(tokens, dtype=torch.long))
    return all_tokens, y_range

def get_batch(sequences, split='train'):
    n = int(0.9 * len(sequences))
    data = sequences[:n] if split == 'train' else sequences[n:]
    while True:
        seq_indices = torch.randint(len(data), (batch_size,))
        x_batch, y_batch = [], []
        for idx in seq_indices:
            sequence = data[idx]
            if len(sequence) >= block_size + 1:
                start_idx = torch.randint(0, len(sequence) - block_size, (1,))
                x_batch.append(sequence[start_idx:start_idx + block_size])
                y_batch.append(sequence[start_idx + 1:start_idx + block_size + 1])
        if len(x_batch) > 0:
            return torch.stack(x_batch).to(device), torch.stack(y_batch).to(device)

@torch.no_grad()
def estimate_loss(model, sequences):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(200)
        for k in range(200):
            X, Y = get_batch(sequences, split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

print("Functions defined.")


# In[3]:


class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * C ** -0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        return wei @ v

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), nn.GELU(),
            nn.Linear(4 * n_embd, n_embd), nn.Dropout(dropout))
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class SignalTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

print("Model defined.")


# In[ ]:


# Run all experiments: 3 signals × 5 noise levels × 10 seeds = 150 models

A = 1.0
omega = 2 * np.pi
signal_types = ['sinusoid', 'triangle', 'square']
noise_levels = [0.0, 0.1, 0.2, 0.5, 1.0]
num_seeds = 10
num_iterations = 20000

results_dir = Path(os.path.expanduser('~/research-project/results/multi_seed'))
results_dir.mkdir(parents=True, exist_ok=True)

all_results = {}

for sig_type in signal_types:
    all_results[sig_type] = {}

    for sigma in noise_levels:
        all_results[sig_type][sigma] = []

        for seed_idx in range(num_seeds):
            seed = 42 + seed_idx

            print(f"\n{sig_type} | σ={sigma} | seed {seed_idx+1}/{num_seeds} (seed={seed})")

            # Generate data with this seed
            torch.manual_seed(seed)
            signals, y_range = generate_training_data(
                sig_type, 500, 256, A, omega, noise_std=sigma, seed=seed)

            # Train
            model = SignalTransformer().to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

            for iter in range(num_iterations):
                xb, yb = get_batch(signals, 'train')
                logits, loss = model(xb, yb)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            # Final loss
            losses = estimate_loss(model, signals)
            train_loss = losses['train'].item()
            val_loss = losses['val'].item()

            # Generate
            model.eval()
            ctx_phase = 0.0
            y_ctx = generate_signal(sig_type, A, omega, 50, ctx_phase)
            y_ctx_norm = (y_ctx - (-y_range)) / (2 * y_range)
            y_ctx_norm = np.clip(y_ctx_norm, 0, 1 - 1e-9)
            ctx_tokens = discretize(y_ctx_norm)
            ctx_tensor = torch.tensor(ctx_tokens, dtype=torch.long).unsqueeze(0).to(device)

            with torch.no_grad():
                gen = model.generate(ctx_tensor, 600)

            gen_tokens = gen.tolist()[0]
            gen_values_norm = undiscretize(gen_tokens)
            gen_values = gen_values_norm * (2 * y_range) + (-y_range)
            generated = gen_values[50:]

            t_gen = np.arange(len(generated)) * 0.01
            y_true = generate_signal(sig_type, A, omega, len(generated), 0.0, dt=0.01)
            # Shift by context length
            t_shifted = np.arange(len(generated)) * 0.01 + 0.50
            y_true = generate_signal(sig_type, A, omega, len(t_shifted), 0.0, dt=0.01)
            # Recompute properly
            t_true = np.arange(50, 50 + len(generated)) * 0.01
            y_true = generate_signal(sig_type, A, omega, 50 + len(generated), 0.0)[50:]

            mae = np.mean(np.abs(y_true[:len(generated)] - generated[:len(y_true)]))

            all_results[sig_type][sigma].append({
                'seed': seed,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'mae': mae,
            })

            print(f"  train={train_loss:.4f} val={val_loss:.4f} MAE={mae:.4f}")

# Save results
save_data = {}
for sig_type in all_results:
    save_data[sig_type] = {}
    for sigma in all_results[sig_type]:
        save_data[sig_type][str(sigma)] = all_results[sig_type][sigma]

with open(results_dir / 'all_results.json', 'w') as f:
    json.dump(save_data, f, indent=2)

print("\n\nALL EXPERIMENTS COMPLETE!")
print("Results saved to ~/research-project/results/multi_seed/all_results.json")


# In[ ]:


# Summary table with mean ± std

print("RESULTS: Mean ± Std across 10 seeds")
print("=" * 80)

for sig_type in signal_types:
    print(f"\n--- {sig_type.upper()} ---")
    print(f"{'σ':<8} {'Train Loss':<20} {'Val Loss':<20} {'MAE':<20}")
    print("-" * 68)

    for sigma in noise_levels:
        runs = all_results[sig_type][sigma]
        train_losses = [r['train_loss'] for r in runs]
        val_losses = [r['val_loss'] for r in runs]
        maes = [r['mae'] for r in runs]

        print(f"{sigma:<8} "
              f"{np.mean(train_losses):.4f} ± {np.std(train_losses):.4f}   "
              f"{np.mean(val_losses):.4f} ± {np.std(val_losses):.4f}   "
              f"{np.mean(maes):.4f} ± {np.std(maes):.4f}")

print("\n\nCROSS-SIGNAL COMPARISON (MAE at σ=0)")
print("-" * 50)
for sig_type in signal_types:
    runs = all_results[sig_type][0.0]
    maes = [r['mae'] for r in runs]
    print(f"{sig_type:<12} {np.mean(maes):.4f} ± {np.std(maes):.4f}")

