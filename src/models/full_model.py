import torch
import torch.nn as nn
from .deq_layer import DEQLayer
from .energy_net import EnergyNet

class SpectralEmergeModel(nn.Module):
    """
    Full Spectral Emergent AI model comprising:
    - Encoder
    - Deep Equilibrium Layer (DEQ)
    - Energy Network
    - Decoder
    """
    def __init__(self, cfg):
        super().__init__()
        x_dim = cfg['model']['x_dim']
        z_dim = cfg['model']['latent_dim']
        h_dim = cfg['model']['hidden_dim']
        deq_max_iter = cfg['model']['deq_max_iter']
        deq_tol = float(cfg['model']['deq_tol'])
        
        # encoder: x -> z0 (MLP with ReLU)
        self.encoder = nn.Sequential(
            nn.Linear(x_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, z_dim)
        )
        
        # deq: DEQLayer - context is z0 thus context dim is z_dim
        self.deq = DEQLayer(x_dim=z_dim, latent_dim=z_dim, hidden_dim=h_dim, 
                            deq_max_iter=deq_max_iter, deq_tol=deq_tol)
        
        # energy: EnergyNet
        self.energy = EnergyNet(x_dim, z_dim, h_dim)
        
        # decoder: z* -> x_rec (MLP with ReLU, last layer linear)
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, x_dim)
        )
        
    def forward(self, x):
        """
        Computes forward pass.
        Returns: z_star, energy, x_rec, convergence_info
        """
        # Context for DEQ
        context = self.encoder(x)
        
        z_star, convergence_info = self.deq(context)
        
        energy_val = self.energy(z_star, x)
        
        x_rec = self.decoder(z_star)
        
        return z_star, energy_val, x_rec, convergence_info
    
    def compute_loss(self, x, lambda_s, beta, tau, spectral_fn=None):
        """
        Computes the total loss consisting of various regularizations.
        Returns: total_loss, loss_dict
        """
        from ..training.losses import total_loss
        
        z_star, energy, x_rec, convergence_info = self.forward(x)
        
        # Get negative samples and compute their energy
        z_neg = self.energy.contrastive_samples(z_star, sigma=0.1)
        neg_energy = self.energy(z_neg, x)
        
        if spectral_fn is None:
            from ..models.spectral_reg import layer_spectral_penalty
            spectral_fn = lambda z: layer_spectral_penalty(self, tau=tau)
            
        tot_loss, loss_dict = total_loss(x, x_rec, z_star, energy, neg_energy, lambda_s, beta, spectral_fn)
        return tot_loss, loss_dict, convergence_info
