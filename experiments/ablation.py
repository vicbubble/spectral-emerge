import os
import sys
import json
import torch
import torch.nn as nn
import yaml
import copy
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.training.trainer import Trainer
from src.eval.metrics import mode_count
from src.data.synthetic import GaussianMixtureDataset

def run_temp_trainer(cfg, run_name):
    os.makedirs("configs/tmp", exist_ok=True)
    temp_path = f"configs/tmp/temp_{run_name}.yaml"
    with open(temp_path, 'w') as f:
        yaml.dump(cfg, f)
    trainer = Trainer(temp_path)
    trainer.fit()
    
    trainer.model.eval()
    all_z = []
    with torch.no_grad():
        for X, _ in trainer.val_loader:
            z_star, _, _, _ = trainer.model(X.to(trainer.device))
            all_z.append(z_star.cpu())
    z_all = torch.cat(all_z, dim=0)
    mc = mode_count(z_all)
    
    if os.path.exists(temp_path):
        os.remove(temp_path)
    return mc

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        self.embeddings.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)
        
    def forward(self, inputs):
        flat_inputs = inputs.view(-1, self.embedding_dim)
        distances = (torch.sum(flat_inputs**2, dim=1, keepdim=True) 
                    + torch.sum(self.embeddings.weight**2, dim=1)
                    - 2 * torch.matmul(flat_inputs, self.embeddings.weight.t()))
                    
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        quantized = torch.matmul(encodings, self.embeddings.weight).view(inputs.shape)
        
        e_latent_loss = torch.mean((quantized.detach() - inputs)**2)
        q_latent_loss = torch.mean((quantized - inputs.detach())**2)
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        
        quantized = inputs + (quantized - inputs).detach()
        return quantized, loss

class VQVAE(nn.Module):
    def __init__(self, x_dim, latent_dim, hidden_dim, default_k=10):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(x_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, latent_dim))
        self.vq = VectorQuantizer(default_k, latent_dim)
        self.decoder = nn.Sequential(nn.Linear(latent_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, x_dim))
        
    def forward(self, x):
        z = self.encoder(x)
        z_q, vq_loss = self.vq(z)
        x_rec = self.decoder(z_q)
        return z_q, x_rec, vq_loss

def train_vqvae(cfg):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = VQVAE(cfg['model']['x_dim'], cfg['model']['latent_dim'], cfg['model']['hidden_dim'], default_k=cfg['data']['n_clusters']).to(device)
    loader = GaussianMixtureDataset.get_dataloader(cfg, 'train')
    val_loader = GaussianMixtureDataset.get_dataloader(cfg, 'val')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    epochs = 10
    for epoch in range(epochs):
        model.train()
        for X, _ in loader:
            X = X.to(device)
            z_q, x_rec, vq_loss = model(X)
            recon_loss = torch.nn.functional.mse_loss(x_rec, X)
            loss = recon_loss + vq_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
    model.eval()
    all_z = []
    with torch.no_grad():
        for X, _ in val_loader:
            z_q, _, _ = model(X.to(device))
            all_z.append(z_q.cpu())
    z_all = torch.cat(all_z, dim=0)
    return mode_count(z_all)

class FeedForwardBaseline(nn.Module):
    def __init__(self, x_dim, latent_dim, hidden_dim):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(x_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, latent_dim))
        self.ff = nn.Sequential(nn.Linear(latent_dim + x_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, latent_dim), nn.Tanh())
        self.decoder = nn.Sequential(nn.Linear(latent_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, x_dim))
        
    def forward(self, x):
        z0 = self.encoder(x)
        z_star = self.ff(torch.cat([z0, x], dim=-1))
        x_rec = self.decoder(z_star)
        return z_star, x_rec

def train_ff(cfg):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FeedForwardBaseline(cfg['model']['x_dim'], cfg['model']['latent_dim'], cfg['model']['hidden_dim']).to(device)
    loader = GaussianMixtureDataset.get_dataloader(cfg, 'train')
    val_loader = GaussianMixtureDataset.get_dataloader(cfg, 'val')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    epochs = 10
    for epoch in range(epochs):
        model.train()
        for X, _ in loader:
            X = X.to(device)
            z_star, x_rec = model(X)
            loss = torch.nn.functional.mse_loss(x_rec, X) + 0.01 * torch.abs(z_star).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
    model.eval()
    all_z = []
    with torch.no_grad():
        for X, _ in val_loader:
            z_star, _ = model(X.to(device))
            all_z.append(z_star.cpu())
    z_all = torch.cat(all_z, dim=0)
    return mode_count(z_all)

def main():
    with open('configs/default.yaml', 'r') as f:
        base_cfg = yaml.safe_load(f)
        
    base_cfg['training']['epochs'] = 10  # Speed up ablation tests 
    base_cfg['logging']['entity'] = None
    
    results = {}
    os.makedirs('experiments/results/figures', exist_ok=True)
    
    # Exp 1: Lambda Sweep
    print("Running Exp 1: Lambda Sweep")
    lambdas = [0.0, 0.001, 0.01, 0.1, 1.0, 10.0]
    lambda_mcs = []
    for lam in lambdas:
        cfg = copy.deepcopy(base_cfg)
        cfg['training']['lambda_spectral'] = lam
        mc = run_temp_trainer(cfg, f"lam_{lam}")
        lambda_mcs.append(mc)
    results['lambda_sweep'] = {'lambdas': lambdas, 'mode_counts': lambda_mcs}
    
    plt.figure()
    plt.plot(lambdas, lambda_mcs, marker='o')
    plt.xscale('symlog', linthresh=0.001)
    plt.xlabel('Lambda Spectral')
    plt.ylabel('Mode Count')
    plt.title('Effect of Spectral Regularization on Mode Emergence')
    plt.savefig('experiments/results/figures/exp1_lambda_sweep.png')
    plt.close()
    
    # Exp 2: DEQ vs Feedforward
    print("Running Exp 2: DEQ vs Feedforward")
    ff_mc = train_ff(base_cfg)
    deq_mc = run_temp_trainer(base_cfg, "deq_baseline")
    results['deq_vs_ff'] = {'deq_mode_count': deq_mc, 'ff_mode_count': ff_mc}
    
    plt.figure()
    plt.bar(['DEQ', 'Feedforward'], [deq_mc, ff_mc])
    plt.ylabel('Mode Count')
    plt.title('DEQ vs Feedforward')
    plt.savefig('experiments/results/figures/exp2_deq_vs_ff.png')
    plt.close()
    
    # Exp 3: Latent Dim Sweep
    print("Running Exp 3: Latent Dim Sweep")
    dims = [8, 16, 32, 64, 128]
    dim_mcs = []
    for d in dims:
        cfg = copy.deepcopy(base_cfg)
        cfg['model']['latent_dim'] = d
        mc = run_temp_trainer(cfg, f"dim_{d}")
        dim_mcs.append(mc)
    results['dim_sweep'] = {'dims': dims, 'mode_counts': dim_mcs}
    
    plt.figure()
    plt.plot(dims, dim_mcs, marker='o')
    plt.xlabel('Latent Dim')
    plt.ylabel('Mode Count')
    plt.title('Effect of Latent Dimension')
    plt.savefig('experiments/results/figures/exp3_dim_sweep.png')
    plt.close()
    
    # Exp 4: VQ-VAE Baseline
    print("Running Exp 4: VQ-VAE Baseline")
    vq_mc = train_vqvae(base_cfg)
    results['vqvae_baseline'] = {'vqvae_mode_count': vq_mc, 'deq_mode_count': deq_mc}
    
    with open('experiments/results/ablation_summary.json', 'w') as f:
        json.dump(results, f, indent=4)
        
    print("Ablation studies complete. Results saved.")

if __name__ == '__main__':
    main()
