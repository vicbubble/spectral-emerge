import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import warnings

try:
    import wfdb
    WFDB_AVAILABLE = True
except ImportError:
    WFDB_AVAILABLE = False
    warnings.warn(
        "wfdb not installed. Run: pip install wfdb\n"
        "ECG experiments will use SYNTHETIC FALLBACK — not valid for publication.",
        stacklevel=2
    )

def _synthetic_fallback(x_dim: int = 187,
                        n_clusters: int = 5,
                        n_samples: int = 2000,
                        seed: int = 42):
    """
    Honest fallback when wfdb is unavailable.

    Generates 5 synthetic classes with distinct sinusoidal patterns and noise.
    Labels are synthetic analogs of AAMI classes:
    0=N, 1=S, 2=V, 3=F, 4=Q.

    Args:
        x_dim: Segment length.
        n_clusters: Number of synthetic classes.
        n_samples: Total number of samples.
        seed: Random seed.
    Returns:
        Tuple (segments, labels).
    """
    np.random.seed(seed)
    t = np.linspace(0, 1, x_dim)
    freqs = [1.0, 1.5, 2.0, 2.5, 3.0]

    segments, labels = [], []
    per_class = n_samples // n_clusters

    for cls, freq in enumerate(freqs[:n_clusters]):
        base = np.sin(2 * np.pi * freq * t)
        noise = np.random.randn(per_class, x_dim) * 0.3
        segs = base[None, :] + noise
        segments.append(segs)
        labels.extend([cls] * per_class)

    segments = np.vstack(segments).astype(np.float32)
    labels = np.array(labels, dtype=np.int64)
    return segments, labels


AAMI_MAPPING = {
    'N': 0, 'L': 0, 'R': 0, 'e': 0, 'j': 0, # N class
    'A': 1, 'a': 1, 'J': 1, 'S': 1,         # S class
    'V': 2, 'E': 2,                         # V class
    'F': 3,                                 # F class
    '/': 4, 'f': 4, 'Q': 4                  # Q class
}

class ECGPreprocessor:
    def bandpass_filter(self, signal, lowcut=0.5, highcut=40.0, fs=360.0):
        # Butterworth bandpass filter — removes baseline wander and HF noise
        from scipy.signal import butter, sosfiltfilt
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        sos = butter(4, [low, high], btype='band', output='sos')
        return sosfiltfilt(sos, signal)
    
    def segment_beats(self, signal, annotations, symbols, window=187):
        # Segment around R-peaks from annotations
        # Each segment: 90 samples before R-peak, 97 after
        segments = []
        labels = []
        for i, peak in enumerate(annotations):
            if peak >= 90 and peak + 97 < len(signal):
                sym = symbols[i]
                if sym in AAMI_MAPPING:
                    segments.append(signal[peak-90:peak+97])
                    labels.append(AAMI_MAPPING[sym])
        if len(segments) == 0:
            return np.array([]), np.array([])
        return np.array(segments), np.array(labels)
    
    def normalize(self, segments):
        # Per-segment: subtract mean, divide by std
        # Clip to [-5, 5] to handle artifacts
        if len(segments) == 0:
            return segments
        means = segments.mean(axis=1, keepdims=True)
        stds = segments.std(axis=1, keepdims=True)
        stds[stds == 0] = 1.0
        norm = (segments - means) / stds
        return np.clip(norm, -5.0, 5.0)

class ECGDataset(Dataset):
    """Dataset for ECG segments."""
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
        
    def get_class_balance(self):
        counts = torch.bincount(self.labels, minlength=5)
        return {i: c.item() for i, c in enumerate(counts)}


class PhysioNetLoader:
    """Loader for MIT-BIH Arrhythmia Dataset."""
    
    @classmethod
    def get_dataloader(cls, cfg, split='train'):
        """Creates dataset and returns dataloader for the given config and split.
        
        Args:
            cfg: Configuration object.
            split: 'train', 'val', or 'test'.
            
        Returns:
            DataLoader instance.
        """
        seed = cfg['training']['seed']
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        batch_size = cfg['training']['batch_size']
        
        if not WFDB_AVAILABLE:
            cls.data_source = "synthetic_fallback"
            return cls._fallback_dataloader(cfg, split)
            
        cls.data_source = "physionet"
        dl_dir = './physionet_data'
        os.makedirs(dl_dir, exist_ok=True)
        
        
        # Override records if specified in config
        records = cfg['data'].get('records', ['100', '101'])
        
        preprocessor = ECGPreprocessor()
        
        windows = []
        labels = []
        
        try:
            for rec in records:
                rec_str = str(rec)
                if not os.path.exists(os.path.join(dl_dir, f"{rec_str}.dat")):
                    wfdb.dl_database('mitdb', dl_dir=dl_dir, records=[rec_str])
                
                record = wfdb.rdrecord(os.path.join(dl_dir, rec_str))
                ann = wfdb.rdann(os.path.join(dl_dir, rec_str), 'atr')
                
                sig = record.p_signal[:, 0] # First channel
                sig = preprocessor.bandpass_filter(sig)
                
                seg, lab = preprocessor.segment_beats(sig, ann.sample, ann.symbol)
                
                if len(seg) > 0:
                    windows.append(seg)
                    labels.append(lab)
                    
        except Exception as e:
            warnings.warn(f"Failed to load PhysioNet data via wfdb: {e}. Falling back to synthetic.")
            return cls._fallback_dataloader(cfg, split)
            
        if len(windows) == 0:
            warnings.warn("No windows extracted from PhysioNet. Falling back to synthetic.")
            return cls._fallback_dataloader(cfg, split)
            
        X = np.concatenate(windows, axis=0)
        y = np.concatenate(labels, axis=0)
        X = preprocessor.normalize(X)
        
        idx = np.random.permutation(len(X))
        X = X[idx]
        y = y[idx]
        
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long)
        
        n_samples = len(X)
        n_train = int(0.7 * n_samples)
        n_val = int(0.15 * n_samples)
        
        if split == 'train':
            dataset = ECGDataset(X[:n_train], y[:n_train])
        elif split == 'val':
            dataset = ECGDataset(X[n_train:n_train+n_val], y[n_train:n_train+n_val])
        else: # test
            dataset = ECGDataset(X[n_train+n_val:], y[n_train+n_val:])
            
        return DataLoader(dataset, batch_size=batch_size, shuffle=(split == 'train'))
        
    @classmethod
    def _fallback_dataloader(cls, cfg, split):
        """Synthetic fallback dataset when wfdb fails."""
        n_samples = cfg['data'].get('n_samples', 2000)
        batch_size = cfg['training']['batch_size']
        seed = cfg['training']['seed']
        
        X, y = _synthetic_fallback(n_samples=n_samples, seed=seed)
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long)
        
        # shuffle
        idx = torch.randperm(len(X))
        X = X[idx]
        y = y[idx]
        
        n_train = int(0.7 * n_samples)
        n_val = int(0.15 * n_samples)
        
        if split == 'train':
            dataset = ECGDataset(X[:n_train], y[:n_train])
        elif split == 'val':
            dataset = ECGDataset(X[n_train:n_train+n_val], y[n_train:n_train+n_val])
        else:
            dataset = ECGDataset(X[n_train+n_val:], y[n_train+n_val:])
            
        return DataLoader(dataset, batch_size=batch_size, shuffle=(split == 'train'))


class OrderedBeatDataset(Dataset):
    """
    Returns all beats of a single patient record in original temporal order.
    One item = one beat.

    Args:
        record_id: MIT-BIH record ID, e.g. 100 or "100"
        cfg: config dict (uses data.cache_dir)
        preprocessor: ECGPreprocessor (created if None)

    Returns per item: (x: float32 tensor, label: int64 tensor, idx: int)
    Order is always preserved. If fewer than 100 beats survive, dataset len = 0.
    """
    def __init__(self, record_id, cfg, preprocessor=None):
        self.record_id = str(record_id)
        self.cfg = cfg
        self.preprocessor = preprocessor or ECGPreprocessor()
        self.segments = np.array([], dtype=np.float32)
        self.labels = np.array([], dtype=np.int64)
        self._load()

    def _load(self):
        if not WFDB_AVAILABLE:
            warnings.warn(
                f"wfdb not available — OrderedBeatDataset for record {self.record_id} "
                "cannot load real PhysioNet data.",
                stacklevel=2
            )
            return
        cache_dir = self.cfg.get('data', {}).get('cache_dir', './physionet_data') \
                    if isinstance(self.cfg, dict) else './physionet_data'
        os.makedirs(cache_dir, exist_ok=True)
        rec_path = os.path.join(cache_dir, self.record_id)
        try:
            if not os.path.exists(rec_path + '.dat'):
                wfdb.dl_database('mitdb', dl_dir=cache_dir, records=[self.record_id])
            record = wfdb.rdrecord(rec_path)
            ann = wfdb.rdann(rec_path, 'atr')
            sig = record.p_signal[:, 0]
            sig = self.preprocessor.bandpass_filter(sig)
            segs, labs = self.preprocessor.segment_beats(sig, ann.sample, ann.symbol)
            if len(segs) < 100:
                warnings.warn(
                    f"Record {self.record_id}: only {len(segs)} beats — "
                    "fewer than 100, dataset will be empty.",
                    stacklevel=2
                )
                return
            segs = self.preprocessor.normalize(segs)
            self.segments = segs.astype(np.float32)
            self.labels = labs.astype(np.int64)
        except Exception as e:
            warnings.warn(f"Failed to load record {self.record_id}: {e}", stacklevel=2)

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, index):
        x = torch.tensor(self.segments[index], dtype=torch.float32)
        label = torch.tensor(self.labels[index], dtype=torch.long)
        return x, label, index


def get_ordered_loader(record_id, cfg, shuffle=False):
    """
    Returns a DataLoader preserving temporal beat order.
    shuffle=True raises ValueError immediately.
    """
    if shuffle:
        raise ValueError(
            "get_ordered_loader: shuffle=True is not permitted. "
            "Temporal evaluation requires preserved beat order."
        )
    dataset = OrderedBeatDataset(record_id, cfg)
    batch_size = min(cfg['training']['batch_size'], 64)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
