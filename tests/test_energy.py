import pytest
import torch
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.energy_net import EnergyNet

def test_energy_scalar():
    net = EnergyNet(x_dim=8, latent_dim=32, hidden_dim=64)
    x = torch.randn(16, 8)
    z = torch.randn(16, 32)
    energy = net(z, x)
    assert energy.shape == (16,)

def test_infer_z_decreases_energy():
    net = EnergyNet(x_dim=8, latent_dim=32, hidden_dim=64)
    x = torch.randn(16, 8)
    
    torch.manual_seed(42)
    z_random = torch.randn(16, 32)
    e_random = net(z_random, x).sum().item()
    
    torch.manual_seed(42)
    z_inferred = net.infer_z(x, n_steps=20, lr=0.1)
    e_inferred = net(z_inferred, x).sum().item()
    
    assert e_inferred < e_random

def test_contrastive_loss_positive():
    net = EnergyNet(x_dim=8, latent_dim=32, hidden_dim=64)
    x = torch.randn(16, 8)
    z_pos = torch.randn(16, 32)
    
    loss, _ = net.contrastive_loss(z_pos, x)
    assert loss.item() > 0
    
    # Must be differentiable
    loss.backward()
    has_grad = False
    for p in net.parameters():
        if p.grad is not None and torch.abs(p.grad).sum() > 0:
            has_grad = True
    assert has_grad
