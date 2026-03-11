import pytest
import torch
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.deq_layer import DEQLayer

def test_convergence():
    layer = DEQLayer(x_dim=8, latent_dim=32, hidden_dim=64, deq_max_iter=50, deq_tol=1e-4)
    x = torch.randn(16, 8)
    z_star, info = layer(x)
    assert info['residual_norm'] < 1e-4

def test_gradient_flow():
    layer = DEQLayer(x_dim=8, latent_dim=32, hidden_dim=64)
    x = torch.randn(16, 8, requires_grad=True)
    z_star, _ = layer(x)
    loss = z_star.sum()
    loss.backward()
    
    assert x.grad is not None
    assert torch.abs(x.grad).max() > 0
    # Also verify inner params got grad
    for k, v in layer.f_theta.named_parameters():
        if 'weight' in k:
            assert v.grad is not None

def test_spectral_norm():
    layer = DEQLayer(x_dim=8, latent_dim=32, hidden_dim=64)
    for m in layer.f_theta.modules():
        if isinstance(m, torch.nn.Linear):
            # Compute spectral norm of weight
            u, s, v = torch.svd(m.weight)
            # Standard spectral norm should be exactly ~1.0 if parametrizations work properly,
            # or bounded by 1. We test if it is <= 1.0 + eps
            assert s.max().item() <= 1.0 + 1e-2
