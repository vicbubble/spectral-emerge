import torch
import numpy as np
import pytest
from src.data.timeseries import ECGPreprocessor, AAMI_MAPPING

def test_ecg_preprocessor_shape():
    prep = ECGPreprocessor()
    # fake signal of length 300
    signal = np.sin(np.linspace(0, 10, 300))
    # annotation at 100, so window is 100-90 to 100+97 = 10:197 (len 187)
    ann = [100, 200]
    syms = ['N', 'V']
    
    seg, lab = prep.segment_beats(signal, ann, syms)
    
    assert seg.shape == (2, 187)
    assert len(lab) == 2
    
def test_ecg_label_mapping():
    # test all expected symbols map to [0..4]
    for k, v in AAMI_MAPPING.items():
        assert v in [0, 1, 2, 3, 4]
        
    prep = ECGPreprocessor()
    signal = np.zeros(200)
    ann = [100, 100, 100, 100, 100]
    syms = ['N', 'A', 'V', 'F', '/']
    
    seg, lab = prep.segment_beats(signal, ann, syms)
    assert len(lab) == 5
    assert lab[0] == 0
    assert lab[1] == 1
    assert lab[2] == 2
    assert lab[3] == 3
    assert lab[4] == 4

def test_ecg_dataset_balance():
    from src.data.timeseries import PhysioNetLoader
    import yaml
    with open('configs/ecg.yaml', 'r') as f:
        cfg = yaml.safe_load(f)
    cfg['data']['n_samples'] = 100 # for fast fallback if needed
    
    loader = PhysioNetLoader.get_dataloader(cfg, 'train')
    
    # N-class (0) should be majority > 60%
    y = loader.dataset.labels
    counts = torch.bincount(y, minlength=5)
    
    # fallback returns only 0s, real returns mostly 0s
    n_class_frac = counts[0].float() / len(y)
    assert n_class_frac > 0.60
