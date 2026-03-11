import os
import torch
from tqdm import tqdm
import wandb
import yaml

from ..models.full_model import SpectralEmergeModel
from ..data.synthetic import GaussianMixtureDataset
from ..data.timeseries import PhysioNetLoader

class Trainer:
    """Trainer class for Spectral Emergent AI model."""
    def __init__(self, cfg_path, data_type='synthetic'):
        with open(cfg_path, 'r') as f:
            self.cfg = yaml.safe_load(f)
            
        if data_type != 'synthetic':
            self.cfg['data']['type'] = data_type
            
        seed = self.cfg['training']['seed']
        torch.manual_seed(seed)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = SpectralEmergeModel(self.cfg).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=float(self.cfg['training']['lr']))
        
        dt = self.cfg['data']['type']
        if dt.startswith('synthetic') or dt in ['overlapping', 'hierarchical', 'nonstationary']: 
            loader_cls = GaussianMixtureDataset
        else:
            loader_cls = PhysioNetLoader
            
        self.train_loader = loader_cls.get_dataloader(self.cfg, 'train')
        self.val_loader = loader_cls.get_dataloader(self.cfg, 'val')
        self.test_loader = loader_cls.get_dataloader(self.cfg, 'test')
        
        self.use_wandb = False
        if self.cfg['logging'].get('entity'):
            try:
                wandb.init(
                    project=self.cfg['logging']['project'],
                    entity=self.cfg['logging']['entity'],
                    config=self.cfg
                )
                self.use_wandb = True
            except Exception as e:
                print(f"Failed to init wandb: {e}")
                
        self.log_every = self.cfg['logging'].get('log_every', 10)
        self.lambda_s = float(self.cfg['training']['lambda_spectral'])
        self.beta = float(self.cfg['training']['beta_sparse'])
        self.tau = float(self.cfg['training']['tau_spectral'])

    def run_epoch(self, split):
        """Runs a single training or validation epoch."""
        is_train = split == 'train'
        if is_train:
            self.model.train()
            loader = self.train_loader
        else:
            self.model.eval()
            loader = self.val_loader if split == 'val' else self.test_loader
            
        total_metrics = {'recon': 0.0, 'energy_contrastive': 0.0, 'spectral': 0.0, 
                         'sparse': 0.0, 'total': 0.0, 'n_iters': 0.0}
        n_batches = 0
        
        pbar = tqdm(loader, desc=f"{split.capitalize()} Epoch")
        
        for i, (X, _) in enumerate(pbar):
            X = X.to(self.device)
            
            with torch.set_grad_enabled(is_train):
                from ..models.spectral_reg import layer_spectral_penalty
                spectral_fn = lambda z: layer_spectral_penalty(self.model, tau=self.tau)
                loss, loss_dict, info = self.model.compute_loss(X, self.lambda_s, self.beta, self.tau, spectral_fn)
                
                if is_train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
            
            for k, v in loss_dict.items():
                total_metrics[k] += v
            total_metrics['n_iters'] += info['n_iters']
            n_batches += 1
            
            pbar.set_postfix({'loss': loss.item(), 'iters': info['n_iters']})
            
            if is_train and i % self.log_every == 0 and self.use_wandb:
                wandb.log({
                    **{f"train_step/{k}": v for k, v in loss_dict.items()},
                    "train_step/n_iters": info['n_iters']
                })
                
        for k in total_metrics:
            total_metrics[k] /= n_batches
            
        if self.use_wandb:
            wandb.log({f"{split}_epoch/{k}": v for k, v in total_metrics.items()})
            
        return total_metrics

    def fit(self):
        """Main training loop."""
        epochs = self.cfg['training']['epochs']
        best_val_recon = float('inf')
        
        os.makedirs("experiments/checkpoints", exist_ok=True)
        
        for epoch in range(epochs):
            print(f"--- Epoch {epoch+1}/{epochs} ---")
            
            train_metrics = self.run_epoch('train')
            val_metrics = self.run_epoch('val')
            
            if val_metrics['recon'] < best_val_recon:
                best_val_recon = val_metrics['recon']
                print(f"New best val recon: {best_val_recon:.4f}. Saving...")
                torch.save(self.model.state_dict(), "experiments/checkpoints/best_model.pt")
                
        if self.use_wandb:
            wandb.finish()
