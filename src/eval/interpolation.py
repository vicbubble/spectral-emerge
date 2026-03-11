import torch
import numpy as np

def slerp(z1, z2, t):
    """Spherical linear interpolation between two vectors."""
    # z1, z2 shapes: (D,)
    # t is scalar or a tensor of shapes matching broadcast
    omega = torch.acos(torch.clamp(torch.dot(z1/torch.norm(z1), z2/torch.norm(z2)), -1.0, 1.0))
    so = torch.sin(omega)
    if so == 0:
        return lerp(z1, z2, t)
    return torch.sin((1.0 - t)*omega) / so * z1 + torch.sin(t*omega) / so * z2

def lerp(z1, z2, t):
    """Linear interpolation between two vectors."""
    return (1.0 - t)*z1 + t * z2

def evaluate_interpolation(model, x1, x2, z1, z2, c1, c2, centers, steps=11, use_slerp=True):
    """
    Evaluates interpolation path.
    Args:
        model: Model with .decoder and .energy methods (or VQ-VAE)
        x1, x2: Original samples (for energy evaluation)
        z1, z2: Latent states
        c1, c2: Ground truth cluster IDs
        centers: Tensor of all cluster centers in x-space to evaluate semantic consistency
    """
    device = z1.device
    model.eval()
    
    ts = torch.linspace(0, 1, steps, device=device)
    z_path = []
    
    for t in ts:
        if use_slerp:
            z_t = slerp(z1, z2, t)
        else:
            z_t = lerp(z1, z2, t)
        z_path.append(z_t)
        
    z_path = torch.stack(z_path) # (steps, latent_dim)
    
    with torch.no_grad():
        if hasattr(model, 'decoder'):
            x_recs = model.decoder(z_path)
        else: # assuming signature like SpectralEmerge or VQVAE fallback
            # Spectral emerge has .decoder
            pass
            
        dx = torch.norm(x_recs[1:] - x_recs[:-1], dim=-1)
        smoothness = dx.mean().item()
        boundary_sharpness = dx.std().item()
        
        # Semantic consistency
        # x_recs shape (steps, x_dim), centers shape (K, x_dim)
        dists = torch.cdist(x_recs, centers) # (steps, K)
        closest_clusters = torch.argmin(dists, dim=-1)
        
        # valid step if closest cluster is either c1 or c2
        is_consistent = (closest_clusters == c1) | (closest_clusters == c2)
        semantic_consistency = is_consistent.float().mean().item()
        
        # Energy monotonicity
        if hasattr(model, 'energy'):
            # evaluate energy using x1 as reference
            e_vals = model.energy(z_path, x1.unsqueeze(0).expand(steps, -1))
            de = e_vals[1:] - e_vals[:-1]
            # check if monotone (all diffs same sign)
            is_monotone = torch.all(de >= 0) or torch.all(de <= 0)
            energy_monotone = 1.0 if is_monotone else 0.0
        else:
            energy_monotone = 0.0
            
    return {
        'smoothness': smoothness,
        'boundary_sharpness': boundary_sharpness,
        'semantic_consistency': semantic_consistency,
        'energy_monotone': energy_monotone
    }

def run_interpolation_benchmark():
    import json
    import os
    import sys
    import yaml
    import matplotlib.pyplot as plt
    
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from experiments.ablation import VQVAE, train_vqvae
    from src.training.trainer import Trainer
    
    # 1. Setup trainer and get spectral emerge model, data
    with open('configs/default.yaml', 'r') as f:
        cfg = yaml.safe_load(f)
    cfg['training']['epochs'] = 10 
    
    trainer = Trainer('configs/default.yaml')
    trainer.fit()
    model_se = trainer.model
    model_se.eval()
    
    # 2. Get VQ-VAE
    device = trainer.device
    model_vq = VQVAE(cfg['model']['x_dim'], cfg['model']['latent_dim'], cfg['model']['hidden_dim']).to(device)
    # Train simple vqvae
    loader = trainer.train_loader
    opt = torch.optim.Adam(model_vq.parameters(), lr=1e-3)
    for _ in range(5):
        for X, _ in loader:
            X = X.to(device)
            z_q, x_rec, vq_loss = model_vq(X)
            loss = torch.nn.functional.mse_loss(x_rec, X) + vq_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
    model_vq.eval()
    
    # get centers
    centers = []
    labels_list = []
    for X, y in trainer.val_loader:
        labels_list.append(y)
    
    # Approximate cluster centers from data
    data_x = trainer.val_loader.dataset.data
    data_y = trainer.val_loader.dataset.labels
    num_clusters = len(torch.unique(data_y))
    centers = torch.stack([data_x[data_y == i].mean(dim=0) for i in range(num_clusters)]).to(device)
    
    # 200 random pairs
    n_pairs = 200
    idx1 = torch.randint(0, len(data_x), (n_pairs,))
    idx2 = torch.randint(0, len(data_x), (n_pairs,))
    
    # Force different clusters
    for i in range(n_pairs):
        while data_y[idx1[i]] == data_y[idx2[i]]:
            idx2[i] = torch.randint(0, len(data_x), (1,)).item()
            
    x1s = data_x[idx1].to(device)
    x2s = data_x[idx2].to(device)
    c1s = data_y[idx1].to(device)
    c2s = data_y[idx2].to(device)
    
    results_se = {'smoothness': [], 'boundary_sharpness': [], 'semantic_consistency': [], 'energy_monotone': []}
    results_vq = {'smoothness': [], 'boundary_sharpness': [], 'semantic_consistency': [], 'energy_monotone': []}
    
    with torch.no_grad():
        z1_se, _, _, _ = model_se(x1s)
        z2_se, _, _, _ = model_se(x2s)
        z1_vq = model_vq.encoder(x1s)
        z1_vq, _ = model_vq.vq(z1_vq)
        z2_vq = model_vq.encoder(x2s)
        z2_vq, _ = model_vq.vq(z2_vq)
        
    for i in range(n_pairs):
        # build path tensors for SE and VQ
        z_path_se = []
        z_path_vq = []
        ts = torch.linspace(0, 1, 11, device=device)
        for t in ts:
            z_path_se.append(slerp(z1_se[i], z2_se[i], t))
            z_path_vq.append(lerp(z1_vq[i], z2_vq[i], t))
            
        z_path_se = torch.stack(z_path_se)
        z_path_vq = torch.stack(z_path_vq)
        
        with torch.no_grad():
            x_rec_se = model_se.decoder(z_path_se)
            x_rec_vq = model_vq.decoder(z_path_vq)
            
            # Evaluate SE
            dx = torch.norm(x_rec_se[1:] - x_rec_se[:-1], dim=-1)
            results_se['smoothness'].append(dx.mean().item())
            results_se['boundary_sharpness'].append(dx.std().item())
            
            dists = torch.cdist(x_rec_se, centers)
            closest = torch.argmin(dists, dim=-1)
            consist = ((closest == c1s[i]) | (closest == c2s[i])).float().mean().item()
            results_se['semantic_consistency'].append(consist)
            
            e_vals = model_se.energy(z_path_se, x1s[i].unsqueeze(0).expand(11, -1))
            de = e_vals[1:] - e_vals[:-1]
            if torch.all(de >= 0) or torch.all(de <= 0):
                results_se['energy_monotone'].append(1.0)
            else:
                results_se['energy_monotone'].append(0.0)
                
            # Evaluate VQ
            dx_vq = torch.norm(x_rec_vq[1:] - x_rec_vq[:-1], dim=-1)
            results_vq['smoothness'].append(dx_vq.mean().item())
            results_vq['boundary_sharpness'].append(dx_vq.std().item())
            
            dists_vq = torch.cdist(x_rec_vq, centers)
            closest_vq = torch.argmin(dists_vq, dim=-1)
            consist_vq = ((closest_vq == c1s[i]) | (closest_vq == c2s[i])).float().mean().item()
            results_vq['semantic_consistency'].append(consist_vq)
            results_vq['energy_monotone'].append(0.0)
            
    summary = {
        'spectral_emerge': {k: float(np.mean(v)) for k, v in results_se.items()},
        'vqvae': {k: float(np.mean(v)) for k, v in results_vq.items()}
    }
    
    os.makedirs('experiments/results/figures', exist_ok=True)
    with open('experiments/results/interpolation_benchmark.json', 'w') as f:
        json.dump(summary, f, indent=4)
        
    plt.figure(figsize=(8,6))
    plt.scatter(results_se['smoothness'], results_se['boundary_sharpness'], label='Spectral Emerge (SLERP)', alpha=0.6)
    plt.scatter(results_vq['smoothness'], results_vq['boundary_sharpness'], label='VQ-VAE (LERP)', alpha=0.6)
    plt.xlabel('Smoothness (lower = smoother)')
    plt.ylabel('Boundary Sharpness')
    plt.title('Interpolation: Smoothness vs Sharpness')
    plt.legend()
    plt.savefig('experiments/results/figures/interpolation_scatter.png')
    plt.close()
    
if __name__ == '__main__':
    run_interpolation_benchmark()
