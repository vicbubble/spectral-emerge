import torch
import pytest
import yaml
from src.models.full_model import SpectralEmergeModel
from src.models.spectral_reg import layer_spectral_penalty

def test_spectral_loss_has_gradient():
    """Spectral penalty must send non-zero gradients to DEQ parameters."""
    with open("configs/default.yaml", "r") as f:
        cfg = yaml.safe_load(f)
        
    model = SpectralEmergeModel(cfg)
    x = torch.randn(32, cfg['model']['x_dim'])

    outputs = model(x)
    # SpectralEmergeModel returns z_star, energy, x_rec, convergence_info
    z_star = outputs[0] if isinstance(outputs, tuple) else outputs['z_star']

    # Using tau=0.0 to ensure the relu passes gradients regardless of layer initialization
    loss = layer_spectral_penalty(model, tau=0.0)
    loss.backward()

    grad_norms = [
        p.grad.norm().item()
        for p in model.deq.f_theta.parameters()
        if p.grad is not None
    ]

    assert len(grad_norms) > 0, "No gradients reached DEQ parameters"
    assert max(grad_norms) > 1e-6, f"Gradients too small: {grad_norms}"

if __name__ == "__main__":
    test_spectral_loss_has_gradient()
    print("Test passed!")
