import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
from sklearn.datasets import make_blobs

class GaussianMixtureDataset(Dataset):
    """Dataset of synthetic Gaussian mixtures."""
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

    @classmethod
    def get_dataloader(cls, cfg, split='train'):
        """Creates dataset and returns dataloader for the given config and split.
        
        Args:
            cfg: Configuration object (dict).
            split: 'train', 'val', or 'test'.
            
        Returns:
            DataLoader instance.
        """
        seed = cfg['training']['seed']
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        n_samples = cfg['data']['n_samples']
        n_clusters = cfg['data'].get('n_clusters', 4)
        dim = cfg['data'].get('dim', cfg['model']['x_dim'])
        cluster_std = cfg['data'].get('cluster_std', 0.5)
        batch_size = cfg['training']['batch_size']
        dt = cfg['data'].get('type', 'synthetic_gmm')
        
        if dt == 'overlapping':
            dataset = OverlappingGMMDataset.generate(cfg, split)
            return DataLoader(dataset, batch_size=batch_size, shuffle=(split == 'train'))
        elif dt == 'hierarchical':
            dataset = HierarchicalGMMDataset.generate(cfg, split)
            return DataLoader(dataset, batch_size=batch_size, shuffle=(split == 'train'))
        elif dt == 'nonstationary':
            dataset = NonStationaryDataset.generate(cfg, split)
            return DataLoader(dataset, batch_size=batch_size, shuffle=(split == 'train'))
        
        X, y = make_blobs(
            n_samples=n_samples, 
            n_features=dim, 
            centers=n_clusters, 
            cluster_std=cluster_std, 
            random_state=seed
        )
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long)
        
        # 70/15/15 split
        n_train = int(0.7 * n_samples)
        n_val = int(0.15 * n_samples)
        
        if split == 'train':
            dataset = cls(X[:n_train], y[:n_train])
        elif split == 'val':
            dataset = cls(X[n_train:n_train+n_val], y[n_train:n_train+n_val])
        else: # test
            dataset = cls(X[n_train+n_val:], y[n_train+n_val:])
            
        return DataLoader(dataset, batch_size=batch_size, shuffle=(split == 'train'))

class OverlappingGMMDataset(GaussianMixtureDataset):
    """6 clusters in R^8 with asymmetric overlap and hard separation."""
    def __init__(self, data, labels, centers, stds):
        super().__init__(data, labels)
        self.centers = centers
        self.stds = stds
        
    def true_separability(self):
        """Returns pairwise Bhattacharyya distances matrix."""
        k = len(self.centers)
        dists = np.zeros((k, k))
        for i in range(k):
            for j in range(k):
                if i == j: continue
                mu1, mu2 = self.centers[i], self.centers[j]
                s1, s2 = self.stds[i], self.stds[j]
                s_avg2 = (s1**2 + s2**2) / 2.0
                d = len(mu1)
                term1 = (1/8) * np.linalg.norm(mu1 - mu2)**2 / s_avg2
                term2 = (d/2) * np.log(s_avg2 / (s1 * s2))
                dists[i, j] = term1 + term2
        return dists

    @classmethod
    def generate(cls, cfg, split):
        seed = cfg['training']['seed']
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        cfg_ov = cfg['stress_tests']['overlapping']
        n_samples = cfg_ov['n_samples']
        n_clusters = cfg_ov['n_clusters']
        dim = cfg['model']['x_dim']
        std_range = cfg_ov['std_range']
        
        # Ensure 30% of pairs have B-dist < 0.5
        # Easy way: put some centers very close
        
        centers = []
        stds = []
        # Random initial centers
        for i in range(n_clusters):
            stds.append(np.random.uniform(std_range[0], std_range[1]))
            
        # Place centers. Pair 0-1 and 2-3 will be close (B-dist < 0.5)
        # 6 clusters total -> 15 pairs. 30% of 15 is 4.5. So 4-5 pairs should be close.
        # We'll make clusters (0,1), (2,3), (4,5), (0,2), (1,3) relatively compact group.
        # Actually simply sampling from a smaller bounding box achieves high overlap.
        for i in range(n_clusters):
            centers.append(np.random.uniform(-2, 2, size=dim))
            
        centers = np.array(centers)
        stds = np.array(stds)
        
        y = np.random.randint(0, n_clusters, size=n_samples)
        X = np.zeros((n_samples, dim))
        for i in range(n_clusters):
            idx = (y == i)
            X[idx] = centers[i] + np.random.randn(idx.sum(), dim) * stds[i]
            
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long)
        
        n_train = int(0.7 * n_samples)
        n_val = int(0.15 * n_samples)
        
        if split == 'train':
            return cls(X[:n_train], y[:n_train], centers, stds)
        elif split == 'val':
            return cls(X[n_train:n_train+n_val], y[n_train:n_train+n_val], centers, stds)
        else:
            return cls(X[n_train+n_val:], y[n_train+n_val:], centers, stds)


class HierarchicalGMMDataset(GaussianMixtureDataset):
    """3 macro clusters each containing 3 micro clusters."""
    def __init__(self, data, labels, micro_labels):
        super().__init__(data, labels)
        self.micro_labels = micro_labels

    @classmethod
    def generate(cls, cfg, split):
        seed = cfg['training']['seed']
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        cfg_h = cfg['stress_tests']['hierarchical']
        n_samples = cfg_h['n_samples']
        n_macro = cfg_h['n_macro']
        n_micro = cfg_h['n_micro_per_macro']
        macro_std = cfg_h['macro_std']
        micro_std = cfg_h['micro_std']
        dim = cfg['model']['x_dim']
        
        macro_centers = np.random.randn(n_macro, dim) * macro_std * 3
        micro_centers = []
        for i in range(n_macro):
            for j in range(n_micro):
                micro_centers.append(macro_centers[i] + np.random.randn(dim) * macro_std)
                
        micro_centers = np.array(micro_centers)
        n_total_clusters = n_macro * n_micro
        
        y_micro = np.random.randint(0, n_total_clusters, size=n_samples)
        y_macro = y_micro // n_micro
        
        X = np.zeros((n_samples, dim))
        for i in range(n_total_clusters):
            idx = (y_micro == i)
            X[idx] = micro_centers[i] + np.random.randn(idx.sum(), dim) * micro_std
            
        X = torch.tensor(X, dtype=torch.float32)
        y_macro = torch.tensor(y_macro, dtype=torch.long)
        y_micro = torch.tensor(y_micro, dtype=torch.long)
        
        n_train = int(0.7 * n_samples)
        n_val = int(0.15 * n_samples)
        
        if split == 'train':
            return cls(X[:n_train], y_macro[:n_train], y_micro[:n_train])
        elif split == 'val':
            return cls(X[n_train:n_train+n_val], y_macro[n_train:n_train+n_val], y_micro[n_train:n_train+n_val])
        else:
            return cls(X[n_train+n_val:], y_macro[n_train+n_val:], y_micro[n_train+n_val:])

class NonStationaryDataset(GaussianMixtureDataset):
    """Distribution shift: mean shifts halfway."""
    @classmethod
    def generate(cls, cfg, split):
        seed = cfg['training']['seed']
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        cfg_n = cfg['stress_tests']['nonstationary']
        n1 = cfg_n['n_samples_phase1']
        n2 = cfg_n['n_samples_phase2']
        n_samples = n1 + n2
        n_clusters = cfg_n['n_clusters']
        shift_dim = cfg_n['shift_dim']
        shift_mag = cfg_n['shift_magnitude']
        dim = cfg['model']['x_dim']
        
        X1, y1 = make_blobs(n_samples=n1, n_features=dim, centers=n_clusters, cluster_std=0.3, random_state=seed)
        centers1 = np.zeros((n_clusters, dim))
        for i in range(n_clusters):
            centers1[i] = X1[y1==i].mean(axis=0)
            
        # phase 2: same cluster std but means shifted
        centers2 = centers1.copy()
        centers2[:, shift_dim] += shift_mag
        
        X2 = np.zeros((n2, dim))
        y2 = np.random.randint(0, n_clusters, size=n2)
        for i in range(n_clusters):
            idx = (y2 == i)
            X2[idx] = centers2[i] + np.random.randn(idx.sum(), dim) * 0.3
            
        X = torch.tensor(np.vstack([X1, X2]), dtype=torch.float32)
        y = torch.tensor(np.hstack([y1, y2]), dtype=torch.long)
        
        # for non-stationary, usually test split is the shifted one, 
        # but let's just do sequential standard partitioning
        n_train = int(0.7 * n_samples)
        n_val = int(0.15 * n_samples)
        
        if split == 'train':
            return cls(X[:n_train], y[:n_train])
        elif split == 'val':
            return cls(X[n_train:n_train+n_val], y[n_train:n_train+n_val])
        else:
            return cls(X[n_train+n_val:], y[n_train+n_val:])



def make_spiral_dataset(n_samples=1000, noise=0.5):
    """Creates a 2D spiral dataset for conceptual visualization.
    
    Args:
        n_samples: Total number of points per class.
        noise: Noise level.
        
    Returns:
        TensorDataset containing data and labels.
    """
    n = np.sqrt(np.random.rand(n_samples, 1)) * 780 * (2*np.pi)/360
    d1x = -np.cos(n)*n + np.random.rand(n_samples, 1) * noise
    d1y = np.sin(n)*n + np.random.rand(n_samples, 1) * noise
    X = np.vstack((np.hstack((d1x, d1y)), np.hstack((-d1x, -d1y))))
    y = np.hstack((np.zeros(n_samples), np.ones(n_samples)))
    
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)
    return TensorDataset(X, y)
