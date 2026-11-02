#!/usr/bin/env python3
"""Debug PolySAE folding to see what's causing the discrepancy."""
import torch
from sae_lens.saes.polysae import PolyTopKTrainingSAE, PolyTopKTrainingSAEConfig

print("Creating test PolySAE with rescale_acts_by_decoder_norm=False...")
cfg = PolyTopKTrainingSAEConfig(
    d_in=768, d_sae=16384, k=64, 
    device='cpu', dtype='float32',
    rescale_acts_by_decoder_norm=False
)
sae = PolyTopKTrainingSAE(cfg)

torch.manual_seed(42)
x = torch.randn(2, 10, 768)

print("\n=== Before Folding ===")
with torch.no_grad():
    # Manually trace through
    sae_in = sae.process_sae_in(x)
    print(f"sae_in mean: {sae_in.mean().item():.6f}, std: {sae_in.std().item():.6f}")
    
    hidden_pre = sae_in @ sae.W_enc + sae.b_enc
    print(f"hidden_pre mean: {hidden_pre.mean().item():.6f}, std: {hidden_pre.std().item():.6f}")
    
    feature_acts = sae.activation_fn(hidden_pre)
    print(f"feature_acts sparsity: {(feature_acts != 0).float().mean().item():.4f}")
    print(f"feature_acts mean: {feature_acts.mean().item():.6f}, std: {feature_acts.std().item():.6f}")
    
    out_before = sae(x)
    print(f"Final output mean: {out_before.mean().item():.6f}, std: {out_before.std().item():.6f}")

print("\n=== Applying Folding ===")
scaling_factor = 2.0
print(f"Scaling W_enc by {scaling_factor}...")
print(f"W_enc norm before: {sae.W_enc.norm().item():.6f}")
sae.fold_activation_norm_scaling_factor(scaling_factor)
print(f"W_enc norm after: {sae.W_enc.norm().item():.6f}")
print(f"C1 norm after: {sae.C1.norm().item():.6f}")

print("\n=== After Folding ===")
with torch.no_grad():
    sae_in = sae.process_sae_in(x)
    print(f"sae_in mean: {sae_in.mean().item():.6f}, std: {sae_in.std().item():.6f}")
    
    hidden_pre = sae_in @ sae.W_enc + sae.b_enc
    print(f"hidden_pre mean: {hidden_pre.mean().item():.6f}, std: {hidden_pre.std().item():.6f}")
    
    feature_acts = sae.activation_fn(hidden_pre)
    print(f"feature_acts sparsity: {(feature_acts != 0).float().mean().item():.4f}")
    print(f"feature_acts mean: {feature_acts.mean().item():.6f}, std: {feature_acts.std().item():.6f}")
    
    out_after = sae(x)
    print(f"Final output mean: {out_after.mean().item():.6f}, std: {out_after.std().item():.6f}")

print("\n=== Comparison ===")
max_diff = (out_before - out_after).abs().max().item()
print(f"Max diff: {max_diff:.2e}")
print(f"Sparsity changed: {((feature_acts != 0).float().mean().item())}")
