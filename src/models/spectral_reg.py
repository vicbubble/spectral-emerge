import torch

def layer_spectral_penalty(model, tau: float = 1.0) -> torch.Tensor:
    """
    Penalizes large spectral norms of weight matrices inside the DEQ function.

    Uses short power iteration and remains differentiable with respect to weights.

    Args:
        model: SpectralEmergeModel instance.
        tau: Soft threshold; only norms above tau are penalized.
    Returns:
        Scalar penalty tensor.
    """
    device = next(model.parameters()).device
    penalty = torch.zeros(1, device=device)

    for name, param in model.deq.f_theta.named_parameters():
        if "weight" in name and param.dim() == 2:
            u = torch.randn(param.shape[0], 1, device=param.device)
            for _ in range(3):
                v = param.T @ u
                v = v / (v.norm() + 1e-8)
                u = param @ v
                u = u / (u.norm() + 1e-8)
            sigma = (u.T @ param @ v).squeeze()
            penalty = penalty + torch.relu(sigma - tau)

    return penalty

def vjp_spectral_loss(z_star: torch.Tensor,
                      f_partial,
                      n_probes: int = 8,
                      tau: float = 1.0) -> torch.Tensor:
    """
    Estimates a local Jacobian-norm surrogate for df/dz using autograd VJPs.

    Notes:
        - This computes vector-Jacobian products, not true JVPs.
        - f_partial must be a closure with x already fixed:
              f_partial(z) = f_theta(z, x_fixed)
        - Do NOT wrap f_partial in no_grad.

    Args:
        z_star: Tensor of shape (batch, d).
        f_partial: Callable mapping z -> f_theta(z, x_fixed).
        n_probes: Number of random probe vectors.
        tau: Soft threshold.
    Returns:
        Scalar penalty tensor.
    """
    penalty = torch.zeros(1, device=z_star.device)

    for _ in range(n_probes):
        v = torch.randn_like(z_star)
        v = v / (v.norm(dim=1, keepdim=True) + 1e-8)

        z_in = z_star.detach().requires_grad_(True)
        f_out = f_partial(z_in)

        vjp = torch.autograd.grad(
            outputs=f_out,
            inputs=z_in,
            grad_outputs=v,
            create_graph=True,
            retain_graph=True,
        )[0]

        sigma_v = vjp.norm(dim=1).mean()
        penalty = penalty + torch.relu(sigma_v - tau)

    return penalty / n_probes

def sparse_loss(z_star):
    """L1 norm penalty for sparsity."""
    return torch.abs(z_star).mean()

def mode_collapse_loss(z_star):
    """Variance penalty to prevent mode collapse."""
    var = torch.var(z_star, dim=0).mean()
    return 1.0 / (var + 1e-6)
