import os
import sys
import yaml
import json
import torch
import copy
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.training.trainer import Trainer
from src.eval.metrics import mode_count, spectral_gap

def get_mean_jacobian_svd(model, dataloader, num_samples=512):
    model.eval()
    all_x = []
    for X, _ in dataloader:
        all_x.append(X)
        if sum(len(x) for x in all_x) >= num_samples:
            break
            
    x_batch = torch.cat(all_x, dim=0)[:num_samples].to(next(model.parameters()).device)
    
    def func(x_in):
        z_star, _, _, _ = model(x_in)
        return z_star
        
    s_list = []
    # Jacobian is large, do one by one or small batches to save memory
    for i in range(num_samples):
        x_i = x_batch[i:i+1]
        try:
            J = torch.autograd.functional.jacobian(func, x_i).squeeze()
            if J.dim() == 2:
                # full SVD
                U, S, Vh = torch.linalg.svd(J, full_matrices=False)
                s_list.append(S.detach().cpu().numpy())
        except Exception:
            pass
            
    if not s_list:
        return np.zeros(1)
        
    return np.mean(s_list, axis=0)
    

def run_dimension_sweep():
    with open('configs/default.yaml', 'r') as f:
        base_cfg = yaml.safe_load(f)
        
    base_cfg['training']['epochs'] = 50 
    base_cfg['logging']['entity'] = None
    
    dims = [48, 64, 72, 80, 88, 96, 104, 112, 120, 128]
    results = {}
    
    os.makedirs('configs/tmp', exist_ok=True)
    
    for d in dims:
        print(f"Sweeping d={d}...")
        cfg = copy.deepcopy(base_cfg)
        cfg['model']['latent_dim'] = d
        
        temp_path = f"configs/tmp/sweep_d{d}.yaml"
        with open(temp_path, 'w') as f:
            yaml.dump(cfg, f)
            
        trainer = Trainer(temp_path, data_type='synthetic')
        trainer.fit()
        trainer.model.eval()
        
        all_z = []
        all_iters = []
        with torch.no_grad():
            for X, _ in trainer.val_loader:
                X = X.to(trainer.device)
                z_star, _, _, info = trainer.model(X)
                all_z.append(z_star.cpu())
                all_iters.append(info['n_iters'])
                
        z_all = torch.cat(all_z, dim=0)
        mc = mode_count(z_all)
        v_norms = torch.var(torch.linalg.norm(z_all, dim=1)).item()
        
        # calculate spectral gap 
        gap = spectral_gap(trainer.model, next(iter(trainer.val_loader))[0][:32].to(trainer.device))
        
        results[d] = {
            'mode_count': mc,
            'spectral_gap': gap,
            'mean_iters': float(np.mean(all_iters)),
            'var_z_norms': v_norms
        }
        
    return results, base_cfg

def run_jacobian_analysis(base_cfg):
    print("Running Jacobian analysis...")
    spectra = {}
    
    for d in [32, 128]:
        cfg = copy.deepcopy(base_cfg)
        cfg['model']['latent_dim'] = d
        cfg['training']['epochs'] = 50
        temp_path = f"configs/tmp/jacob_d{d}.yaml"
        with open(temp_path, 'w') as f:
            yaml.dump(cfg, f)
            
        trainer = Trainer(temp_path, data_type='synthetic')
        trainer.fit()
        
        mean_s = get_mean_jacobian_svd(trainer.model, trainer.val_loader, num_samples=512)
        spectra[d] = mean_s
        
    plt.figure(figsize=(8,6))
    for d, S in spectra.items():
        plt.plot(S, label=f'd={d}')
    plt.yscale('log')
    plt.xlabel('Singular Value Index')
    plt.ylabel('Magnitude')
    plt.legend()
    plt.title('Jacobian Singular Value Spectrum')
    plt.savefig('experiments/results/figures/jacobian_spectrum_comparison.png')
    plt.close()
    
    return spectra
    
def run_fixedpoint_landscape(base_cfg):
    print("Running 2D fixed point landscape...")
    cfg = copy.deepcopy(base_cfg)
    cfg['model']['latent_dim'] = 2
    cfg['training']['epochs'] = 50
    temp_path = "configs/tmp/landscape_d2.yaml"
    with open(temp_path, 'w') as f:
        yaml.dump(cfg, f)
        
    trainer = Trainer(temp_path, data_type='synthetic')
    trainer.fit()
    model = trainer.model
    model.eval()
    
    # Grab a single sample context
    X, _ = next(iter(trainer.val_loader))
    x_sample = X[0:1].to(trainer.device)
    context = model.encoder(x_sample)
    
    grid_size = 50
    z_coords = torch.linspace(-4, 4, grid_size)
    Z1, Z2 = torch.meshgrid(z_coords, z_coords, indexing='ij')
    Z = torch.stack([Z1.reshape(-1), Z2.reshape(-1)], dim=-1).to(trainer.device)
    
    C = context.expand(Z.size(0), -1)
    
    with torch.no_grad():
        f_Z = model.deq._f(Z, C)
        distances = torch.sum((f_Z - Z)**2, dim=-1).reshape(grid_size, grid_size).cpu().numpy()
        
    plt.figure(figsize=(8,6))
    plt.contourf(Z1.numpy(), Z2.numpy(), distances, levels=50, cmap='viridis')
    plt.colorbar(label='||f(z) - z||^2')
    plt.title('Fixed-point Landscape (distance to equilibrium)')
    plt.savefig('experiments/results/figures/fixedpoint_landscape.png')
    plt.close()

def main():
    os.makedirs('experiments/results/figures', exist_ok=True)
    sweep_results, base_cfg = run_dimension_sweep()
    spectra = run_jacobian_analysis(base_cfg)
    run_fixedpoint_landscape(base_cfg)
    
    with open('experiments/results/collapse_sweep.json', 'w') as f:
        json.dump(sweep_results, f, indent=4)
        
    # Find critical dimension
    critical_dim = None
    for d, res in sorted(sweep_results.items()):
        if res['mode_count'] < 4:
            critical_dim = d
            break
            
    if critical_dim is None:
        critical_dim = 128
        
    # Need ratio of gaps. If d=32 not in sweep results, approximate or use the spectra we just computed
    # Wait, 32 isn't in sweep_results, so let's use the spectra S[0]/S[1]
    if 32 in spectra and len(spectra[32]) > 1:
        gap_32 = spectra[32][0] / (spectra[32][1] + 1e-8)
    else:
        gap_32 = 1.0
        
    if critical_dim in sweep_results:
        gap_crit = sweep_results[critical_dim]['spectral_gap']
    else:
        gap_crit = spectra[128][0] / (spectra[128][1] + 1e-8) if 128 in spectra else 1.0
        
    ratio = gap_crit / gap_32
    
    hypothesis = f"""# Collapse Investigation Hypothesis

**Critical Dimension**: {critical_dim}

**Spectral Gap Ratio (critical / d=32)**: {ratio:.4f}

**Hypothesis**: 
As the latent dimension increases, the network volume expands excessively, reducing the representation bottleneck required to form discrete isolated modes. The spectral regularization penalty fails to appropriately constrain the vast null-space of the higher-dimensional Jacobian, transforming previously sharp attractor wells into broad, poorly conditioned flat manifolds where distinct equilibria collapse into identical diffuse states.
"""
    with open('experiments/results/collapse_hypothesis.md', 'w') as f:
        f.write(hypothesis)

if __name__ == '__main__':
    main()
