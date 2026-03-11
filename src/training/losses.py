import torch
import torch.nn as nn
import torch.nn.functional as F

def reconstruction_loss(x, x_rec):
    """Computes Mean Squared Error reconstruction loss."""
    return F.mse_loss(x_rec, x)

def total_loss(x, x_rec, z_star, energy, neg_energy, lambda_s, beta, spectral_fn):
    """
    Computes the total loss for the SpectralEmergeModel.
    
    Args:
        x: Input tensor
        x_rec: Reconstructed input tensor
        z_star: Fixed-point latent state
        energy: Energy of positive samples E(z*, x)
        neg_energy: Energy of negative samples E(z_neg, x)
        lambda_s: Weight for spectral regularization
        beta: Weight for sparsity loss
        spectral_fn: Callable taking (z_star) to compute spectral penalty
        
    Returns:
        total: Scalar tensor with the total loss
        loss_dict: Dictionary with individual loss components for logging
    """
    from ..models.spectral_reg import sparse_loss
    
    recon = reconstruction_loss(x, x_rec)
    
    # Noise contrastive estimation surrogate: softplus(E_pos - E_neg)
    energy_contrastive = F.softplus(energy - neg_energy).mean()
    
    spectral = spectral_fn(z_star)
    
    sparse = sparse_loss(z_star)
    
    total = recon + energy_contrastive + lambda_s * spectral + beta * sparse
    
    loss_dict = {
        'recon': recon.item(),
        'energy_contrastive': energy_contrastive.item(),
        'spectral': spectral.item(),
        'sparse': sparse.item(),
        'total': total.item()
    }
    
    return total, loss_dict
