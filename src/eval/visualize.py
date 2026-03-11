import os
import matplotlib.pyplot as plt
import seaborn as sns
import umap
import torch
import numpy as np

def plot_latent_umap(z_star, labels, title, save_path):
    """Plots UMAP projection of latent states."""
    if isinstance(z_star, torch.Tensor):
        z_star = z_star.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
        
    reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
    embedding = reducer.fit_transform(z_star)
    
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=embedding[:, 0], y=embedding[:, 1], hue=labels, palette="deep", s=10)
    plt.title(title)
    plt.savefig(save_path)
    plt.close()

def plot_energy_landscape_2d(model, x_sample, save_path):
    """Plots 2D energy landscape for a given sample, assuming latent_dim=2."""
    if model.deq.latent_dim != 2:
        return
        
    is_training = model.training
    model.eval()
    
    grid_size = 50
    z1 = torch.linspace(-5, 5, grid_size)
    z2 = torch.linspace(-5, 5, grid_size)
    Z1, Z2 = torch.meshgrid(z1, z2, indexing='ij')
    Z = torch.stack([Z1.reshape(-1), Z2.reshape(-1)], dim=-1).to(x_sample.device)
    
    # Use just the first sample
    X = x_sample[0:1].expand(Z.size(0), -1)
    
    with torch.no_grad():
        energy = model.energy(Z, X).reshape(grid_size, grid_size).cpu().numpy()
        
    plt.figure(figsize=(8, 6))
    plt.contourf(Z1.numpy(), Z2.numpy(), energy, levels=50, cmap='viridis')
    plt.colorbar(label='Energy')
    plt.title("Energy Landscape")
    plt.savefig(save_path)
    plt.close()
    
    if is_training:
        model.train()

def plot_singular_value_spectrum(model, x_sample, save_path):
    """Plots the mean singular value spectrum of the Jacobian."""
    is_training = model.training
    model.eval()
    
    def func(x_in):
        z_star, _, _, _ = model(x_in)
        return z_star
        
    b = x_sample.size(0)
    all_s = []
    
    for i in range(b):
        x_i = x_sample[i:i+1]
        try:
            J = torch.autograd.functional.jacobian(func, x_i).squeeze()
            if J.numel() > 0:
                _, S, _ = torch.svd(J)
                all_s.append(S.detach().cpu().numpy())
        except RuntimeError:
            pass
            
    if all_s:
        mean_s = np.mean(all_s, axis=0)
        plt.figure(figsize=(8,6))
        plt.plot(mean_s, marker='o')
        plt.yscale('log')
        plt.title('Singular Value Spectrum of dz*/dx')
        plt.xlabel('Index')
        plt.ylabel('Singular Value')
        plt.savefig(save_path)
        plt.close()
        
    if is_training:
        model.train()

def plot_mode_histogram(z_star, save_path):
    """Plots a histogram of the norms of z*."""
    if isinstance(z_star, torch.Tensor):
        z_star = z_star.detach().cpu().numpy()
        
    norms = np.linalg.norm(z_star, axis=1)
    plt.figure(figsize=(8,6))
    sns.histplot(norms, bins=30, kde=True)
    plt.title('Distribution of ||z*||')
    plt.xlabel('Norm')
    plt.savefig(save_path)
    plt.close()

def plot_loss_curves(history_dict, save_path):
    """Plots loss over epochs."""
    plt.figure(figsize=(10,6))
    for k, v in history_dict.items():
        plt.plot(v, label=k)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training Loss Curves')
    plt.savefig(save_path)
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 4 VISUALIZATIONS
# ─────────────────────────────────────────────────────────────────────────────

def plot_patient_timeline(labels_seq, anomaly_scores, z_star_seq, iter_seq,
                          true_transitions, change_points, record_id, save_path):
    """
    4-panel figure with beat index on x-axis.

    Panel 1: True class label per beat (categorical bar)
    Panel 2: Anomaly score + sustained transitions + detected change points
    Panel 3: ||z*|| over time, colored by true class
    Panel 4: DEQ iterations per beat + mean line
    """
    import matplotlib.colors as mcolors

    T = len(labels_seq)
    beat_idx = np.arange(T)
    z_norms = np.linalg.norm(z_star_seq, axis=1)
    mean_iters = iter_seq.mean()

    cmap = plt.cm.get_cmap('tab10', 5)

    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    fig.suptitle(f'Patient Timeline — Record {record_id}', fontsize=13)

    # Panel 1: class label strip
    ax = axes[0]
    for t in range(T):
        ax.axvspan(t - 0.5, t + 0.5, color=cmap(int(labels_seq[t])), alpha=0.7)
    handles = [plt.Rectangle((0, 0), 1, 1, color=cmap(i)) for i in range(5)]
    ax.legend(handles, ['N', 'S', 'V', 'F', 'Q'], loc='upper right', fontsize=7, ncol=5)
    ax.set_ylabel('True Class')
    ax.set_yticks([])

    # Panel 2: anomaly score
    ax = axes[1]
    ax.plot(beat_idx, anomaly_scores, color='steelblue', lw=0.8, label='Anomaly score')
    for tt in true_transitions:
        ax.axvline(tt, color='black', ls='--', lw=1.0, alpha=0.7, label='True transition')
    for cp in change_points:
        ax.axvline(cp, color='red', ls=':', lw=1.0, alpha=0.8, label='Detected change')
    handles, lbls = ax.get_legend_handles_labels()
    unique = dict(zip(lbls, handles))
    ax.legend(unique.values(), unique.keys(), fontsize=7)
    ax.set_ylabel('Anomaly score')
    ax.set_ylim(-0.05, 1.05)

    # Panel 3: ||z*|| colored by class
    ax = axes[2]
    for cls in range(5):
        mask = labels_seq == cls
        ax.scatter(beat_idx[mask], z_norms[mask], s=2, color=cmap(cls), alpha=0.6)
    ax.set_ylabel('||z*||')

    # Panel 4: DEQ iterations
    ax = axes[3]
    ax.plot(beat_idx, iter_seq, color='darkorange', lw=0.7, alpha=0.8)
    ax.axhline(mean_iters, color='black', ls='--', lw=1.0, label=f'Mean={mean_iters:.1f}')
    ax.set_ylabel('DEQ iters')
    ax.set_xlabel('Beat index')
    ax.legend(fontsize=7)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    plt.savefig(save_path, dpi=100)
    plt.close()


def plot_auroc_curve(labels, scores, record_id, score_type, save_path):
    """
    ROC curve with AUROC in legend, random diagonal, optimal F1 point.
    """
    from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve
    import warnings as _warnings

    binary = (labels != 0).astype(int)
    if len(np.unique(binary)) < 2:
        return

    auroc = roc_auc_score(binary, scores)
    fpr, tpr, _ = roc_curve(binary, scores)

    prec, rec, thresh_pr = precision_recall_curve(binary, scores)
    f1s = 2 * prec * rec / (prec + rec + 1e-12)
    best_i = np.argmax(f1s)

    # Find fpr/tpr for optimal threshold
    from sklearn.metrics import roc_curve as _rc
    fpr_all, tpr_all, thresh_roc = _rc(binary, scores)
    # find threshold closest to thresh_pr[best_i]
    if len(thresh_pr) > best_i:
        target_thresh = thresh_pr[best_i]
        t_idx = np.argmin(np.abs(thresh_roc - target_thresh))
        opt_fpr = fpr_all[t_idx]
        opt_tpr = tpr_all[t_idx]
    else:
        opt_fpr, opt_tpr = float('nan'), float('nan')

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, lw=1.5, label=f'AUROC = {auroc:.3f}')
    ax.plot([0, 1], [0, 1], 'k--', lw=0.8, label='Random')
    if not np.isnan(opt_fpr):
        ax.scatter([opt_fpr], [opt_tpr], color='red', zorder=5, label=f'Opt F1={f1s[best_i]:.3f}')
    ax.set_xlabel('FPR')
    ax.set_ylabel('TPR')
    ax.set_title(f'ROC — Record {record_id} [{score_type}]')
    ax.legend(fontsize=8)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    plt.savefig(save_path, dpi=100)
    plt.close()


def plot_latent_trajectory_2d(z_star_seq, labels_seq, change_points, record_id, save_path):
    """
    2D projection of latent trajectory:
    - UMAP if available; fallback to PCA (logged explicitly)
    - Color by true class
    - Consecutive beats connected with thin gray lines
    - Detected change points marked with red X
    """
    try:
        import umap as _umap
        reducer = _umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
        z_2d = reducer.fit_transform(z_star_seq)
        method = 'UMAP'
    except ImportError:
        import logging as _logging
        _logging.getLogger(__name__).warning(
            "umap not available — falling back to PCA for trajectory plot."
        )
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2, random_state=42)
        z_2d = pca.fit_transform(z_star_seq)
        method = 'PCA (UMAP fallback)'

    cmap = plt.cm.get_cmap('tab10', 5)
    T = len(z_2d)

    fig, ax = plt.subplots(figsize=(8, 6))
    # Gray lines connecting consecutive beats
    for t in range(T - 1):
        ax.plot([z_2d[t, 0], z_2d[t + 1, 0]], [z_2d[t, 1], z_2d[t + 1, 1]],
                color='lightgray', lw=0.5, alpha=0.5)

    # Points colored by class
    for cls in range(5):
        mask = labels_seq == cls
        ax.scatter(z_2d[mask, 0], z_2d[mask, 1], s=8, color=cmap(cls),
                   alpha=0.7, label=f'Class {cls}')

    # Change points
    if len(change_points) > 0:
        ax.scatter(z_2d[change_points, 0], z_2d[change_points, 1],
                   marker='X', color='red', s=80, zorder=5, label='Change point')

    ax.set_title(f'Latent Trajectory [{method}] — Record {record_id}')
    ax.legend(fontsize=7, markerscale=1.5)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    plt.savefig(save_path, dpi=100)
    plt.close()

