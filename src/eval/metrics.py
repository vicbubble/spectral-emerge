import torch
import numpy as np
from sklearn.metrics import silhouette_score as sk_silhouette
from sklearn.mixture import GaussianMixture
from scipy.spatial.distance import directed_hausdorff

def silhouette_score(z_star, labels):
    """Computes Silhouette score from sklearn."""
    if isinstance(z_star, torch.Tensor):
        z_star = z_star.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
        
    if len(np.unique(labels)) <= 1 or len(z_star) <= 1:
        return 0.0
    return float(sk_silhouette(z_star, labels))

def mode_count(z_star, max_components=10):
    """Fits GMMs to find BIC-optimal number of components."""
    if isinstance(z_star, torch.Tensor):
        z_star = z_star.detach().cpu().numpy()
        
    best_bic = np.inf
    best_k = 1
    
    # Avoid errors if n_samples < max_components
    max_k = min(max_components, len(z_star))
    
    for k in range(1, max_k + 1):
        gmm = GaussianMixture(n_components=k, covariance_type='full', random_state=42)
        try:
            gmm.fit(z_star)
            bic = gmm.bic(z_star)
            if bic < best_bic:
                best_bic = bic
                best_k = k
        except ValueError:
            pass
            
    return best_k

def mode_stability(z_star_run1, z_star_run2):
    """Computes Hausdorff distance between cluster centroids of two runs."""
    if isinstance(z_star_run1, torch.Tensor):
        z_star_run1 = z_star_run1.detach().cpu().numpy()
    if isinstance(z_star_run2, torch.Tensor):
        z_star_run2 = z_star_run2.detach().cpu().numpy()
        
    k1 = mode_count(z_star_run1)
    g1 = GaussianMixture(n_components=k1, random_state=42).fit(z_star_run1)
    
    k2 = mode_count(z_star_run2)
    g2 = GaussianMixture(n_components=k2, random_state=42).fit(z_star_run2)
    
    d1 = directed_hausdorff(g1.means_, g2.means_)[0]
    d2 = directed_hausdorff(g2.means_, g1.means_)[0]
    return float(max(d1, d2))

def spectral_gap(model, x_sample):
    """Ratio sigma_1/sigma_2 of mean Jacobian singular values."""
    is_training = model.training
    model.eval()
    
    def func(x_in):
        z_star, _, _, _ = model(x_in)
        return z_star
        
    batch_size = x_sample.size(0)
    gaps = []
    
    for i in range(batch_size):
        x_i = x_sample[i:i+1]
        try:
            J = torch.autograd.functional.jacobian(func, x_i)
            # shape: (1, z_dim, 1, x_dim)
            J = J.view(J.size(1), J.size(3))
            
            _, S, _ = torch.svd(J)
            if len(S) > 1:
                gap = (S[0] / (S[1] + 1e-8)).item()
                gaps.append(gap)
        except RuntimeError:
            pass
            
    if is_training:
        model.train()
        
    return float(np.mean(gaps)) if gaps else 1.0

def reconstruction_mse(x, x_rec):
    """MSE between input and reconstruction."""
    return torch.nn.functional.mse_loss(x_rec, x).item()

def mmd_score(z_star, z_prior, bandwidths=[0.1, 1.0, 10.0]):
    """Maximum Mean Discrepancy with RBF kernel."""
    def rbf(x, y, sigma):
        dists = torch.cdist(x, y, p=2)**2
        return torch.exp(-dists / (2 * sigma**2))
        
    xx = 0.0
    yy = 0.0
    xy = 0.0
    for sigma in bandwidths:
        xx += rbf(z_star, z_star, sigma).mean()
        yy += rbf(z_prior, z_prior, sigma).mean()
        xy += rbf(z_star, z_prior, sigma).mean()
        
    mmd = xx + yy - 2 * xy
    return mmd.item()
