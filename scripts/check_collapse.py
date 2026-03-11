import os
import torch
import numpy as np
import yaml
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import json

from src.models.full_model import SpectralEmergeModel
from src.data.synthetic import GaussianMixtureDataset
from src.training.trainer import Trainer

def train_model(model, cfg, device):
    trainer = Trainer("configs/stress_test.yaml", data_type='overlapping')
    trainer.cfg['training']['epochs'] = 20
    trainer.log_every = 999
    trainer.fit()
    model.load_state_dict(torch.load("experiments/checkpoints/best_model.pt", map_location=device))

def check_collapse():
    os.makedirs("experiments/results", exist_ok=True)
    os.makedirs("experiments/results/figures", exist_ok=True)

    # 1. Load config and data
    cfg_path = "configs/stress_test.yaml"
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Validation set of overlapping dataset
    cfg['data']['type'] = 'overlapping'
    val_loader = GaussianMixtureDataset.get_dataloader(cfg, 'val')
    
    # 2. Get un-trained or recently trained model?
    # Actually, the instructions say "Before applying any collapse fix, run this confirmation test"
    # The overlapping dataset was trained in Phase 2. Let's load that model if it exists.
    model = SpectralEmergeModel(cfg).to(device)
    model_path = "experiments/checkpoints/best_model_overlapping.pt"
    if os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            print("Loaded best_model_overlapping.pt for collapse check.")
        except Exception:
            print("Failed to load checkpoint. Training from scratch.")
            train_model(model, cfg, device)
    else:
        print("Model not found. Training for collapse check...")
        train_model(model, cfg, device)
        
    model.eval()
    
    z_stars = []
    with torch.no_grad():
        for X, _ in val_loader:
            X = X.to(device)
            out = model(X)
            # Output structure depends on model, assume it returns a dict or tuple
            z_s = out['z_star'] if isinstance(out, dict) else out[0]
            z_stars.append(z_s.cpu())
            
    z_stars = torch.cat(z_stars, dim=0).numpy() # (N, d)
    
    # 3. Fit GMM with BIC to estimate mode_count
    max_components = min(10, z_stars.shape[0] // 10)
    bics = []
    for k in range(1, max_components + 1):
        try:
            gmm = GaussianMixture(n_components=k, covariance_type='full', random_state=42)
            gmm.fit(z_stars)
            bics.append(gmm.bic(z_stars))
        except Exception:
            break
            
    if len(bics) > 0:
        mode_count = np.argmin(bics) + 1
    else:
        mode_count = 1
        
    # 4. Compute variance of ||z*|| across samples
    z_norms = np.linalg.norm(z_stars, axis=1)
    norm_var = np.var(z_norms)
    
    print(f"Estimated mode_count (BIC): {mode_count}")
    print(f"Variance of ||z*||: {norm_var:.6f}")
    
    is_collapsed = (mode_count == 1) and (norm_var < 0.05)
    print(f"Collapse Confirmed: {is_collapsed}")
    
    # 5. Plot histogram
    plt.figure(figsize=(8, 6))
    plt.hist(z_norms, bins=50, alpha=0.7, color='blue', edgecolor='black')
    plt.title(f"Distribution of ||z*|| (var={norm_var:.4f}, modes={mode_count})")
    plt.xlabel("||z*||")
    plt.ylabel("Frequency")
    plt.grid(True, alpha=0.3)
    plt.savefig("experiments/results/figures/zstar_norm_hist_phase3_initial.png")
    plt.close()
    
    # Save results
    results = {
        "mode_count": int(mode_count),
        "norm_variance": float(norm_var),
        "is_collapsed": bool(is_collapsed)
    }
    
    with open("experiments/results/collapse_check_initial.json", "w") as f:
        json.dump(results, f, indent=4)
        
    # Write a small file to signal if VICReg is needed
    with open("experiments/results/.collapse_flag", "w") as f:
        f.write(str(int(is_collapsed)))

if __name__ == "__main__":
    check_collapse()
