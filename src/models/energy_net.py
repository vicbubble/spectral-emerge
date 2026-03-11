import torch
import torch.nn as nn

class EnergyNet(nn.Module):
    """Energy network connecting latent states and inputs."""
    def __init__(self, x_dim, latent_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(x_dim + latent_dim),
            nn.Linear(x_dim + latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.latent_dim = latent_dim
        
    def forward(self, z, x):
        """Computes energy E(z, x)."""
        inp = torch.cat([z, x], dim=-1)
        return self.net(inp).squeeze(-1)
        
    def infer_z(self, x, n_steps=10, lr=0.1):
        """Gradient descent on z to minimize E(z, x)."""
        bsz = x.size(0)
        z = torch.randn(bsz, self.latent_dim, device=x.device, requires_grad=True)
        optimizer = torch.optim.SGD([z], lr=lr)
        
        for _ in range(n_steps):
            optimizer.zero_grad()
            energy = self(z, x).sum()
            energy.backward()
            optimizer.step()
            
        return z.detach()
        
    def contrastive_samples(self, z_pos, sigma):
        """Generates negative samples by adding Gaussian noise."""
        return z_pos + torch.randn_like(z_pos) * sigma
        
    def contrastive_loss(self, z_pos, x, sigma=0.1):
        """Implements noise contrastive estimation loss."""
        z_neg = self.contrastive_samples(z_pos, sigma)
        e_pos = self(z_pos, x)
        e_neg = self(z_neg, x)
        
        # Softplus formulation of contrastive loss: log(1 + exp(E_pos - E_neg))
        loss = torch.nn.functional.softplus(e_pos - e_neg).mean()
        return loss, e_neg
