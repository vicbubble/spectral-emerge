import os
import sys
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
import logging
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.training.trainer import Trainer
from src.data.timeseries import PhysioNetLoader
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import json

def align_clusters_to_classes(z_star, true_labels, n_modes):
    """
    Use Hungarian algorithm (scipy.optimize.linear_sum_assignment) to find
    the optimal bijection between emergent modes and AAMI classes.
    Returns: alignment accuracy, confusion matrix, alignment_dict
    """
    z_star_np = z_star.detach().cpu().numpy()
    true_labels_np = true_labels.detach().cpu().numpy()
    
    kmeans = KMeans(n_clusters=n_modes, n_init=10, random_state=42)
    clusters = kmeans.fit_predict(z_star_np)
    
    unique_true = np.unique(true_labels_np)
    num_classes = len(unique_true) # typically 5
    
    # Cost matrix: -overlap
    cost_matrix = np.zeros((n_modes, num_classes))
    for i in range(n_modes):
        for j in range(num_classes):
            cls_j = unique_true[j] # actually 0 to 4
            cost_matrix[i, j] = -np.sum((clusters == i) & (true_labels_np == cls_j))
            
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    alignment_dict = {r: int(unique_true[c]) for r, c in zip(row_ind, col_ind)}
    
    # Map any unassigned clusters to majority class to avoid KeyError
    for r in range(n_modes):
        if r not in alignment_dict:
            alignment_dict[r] = 0 # default to Normal
            
    aligned_clusters = np.array([alignment_dict[c] for c in clusters])
    accuracy = np.mean(aligned_clusters == true_labels_np)
    
    cm = np.zeros((5, 5))
    for t, p in zip(true_labels_np, aligned_clusters):
        cm[t, p] += 1
        
    return accuracy, cm, alignment_dict, z_star_np, aligned_clusters

def main():
    with open('configs/ecg.yaml', 'r') as f:
        cfg = yaml.safe_load(f)
        
    trainer = Trainer('configs/ecg.yaml', data_type='ecg_physionet')
    
    # Check if we successfully loaded ecg data physically
    # A bit hard to know if it's fallback, but we can check the dataset class
    model_path = 'experiments/checkpoints/best_model.pt'
    if os.path.exists(model_path):
        trainer.model.load_state_dict(torch.load(model_path, map_location=trainer.device, weights_only=True))
        print(f"Loaded pre-trained model from {model_path}")
    else:
        trainer.fit()
    trainer.model.eval()
    
    print(f"\n[DATA SOURCE: {PhysioNetLoader.data_source}]")
    if PhysioNetLoader.data_source == "synthetic_fallback":
        print("WARNING: results below are NOT from real ECG and must not be reported as PhysioNet results.\n")
    
    all_X = []
    all_z = []
    all_y = []
    all_iters = []
    
    with torch.no_grad():
        for X, y in trainer.val_loader:
            X = X.to(trainer.device)
            all_X.append(X.cpu())
            z_star, _, _, info = trainer.model(X)
            all_z.append(z_star.cpu())
            all_y.append(y.cpu())
            all_iters.append(info['n_iters'])
            
    if len(all_X) > 0:
        X_all = torch.cat(all_X, dim=0)
        X_all = X_all.view(X_all.size(0), -1).numpy()
    else:
        X_all = np.array([])
    z_all = torch.cat(all_z, dim=0)
    y_all = torch.cat(all_y, dim=0)
    
    n_modes = 5 # Given we expect 5 AAMI classes
    accuracy, cm, align_dict, z_np, aligned_clusters = align_clusters_to_classes(z_all, y_all, n_modes)
    
    # Calculate metrics
    y_np = y_all.numpy()
    ari = adjusted_rand_score(y_np, aligned_clusters)
    nmi = normalized_mutual_info_score(y_np, aligned_clusters)
    purity = np.sum(np.amax(cm, axis=0)) / np.sum(cm)
    
    random_preds = np.random.randint(0, n_modes, size=len(y_np))
    random_ari = adjusted_rand_score(y_np, random_preds)
    
    kmeans_baseline = KMeans(n_clusters=n_modes, n_init=10, random_state=42)
    kmeans_preds = kmeans_baseline.fit_predict(X_all)
    kmeans_ari = adjusted_rand_score(y_np, kmeans_preds)
    
    unique_true_classes = len(np.unique(y_np))
    unique_pred_clusters = len(np.unique(aligned_clusters))
    
    # Save metrics JSON
    metrics_dict = {
        "ARI": float(ari),
        "NMI": float(nmi),
        "purity": float(purity),
        "random_baseline_ARI": float(random_ari),
        "kmeans_baseline_ARI": float(kmeans_ari),
        "unique_true_classes": int(unique_true_classes),
        "unique_pred_clusters": int(unique_pred_clusters),
        "data_source": PhysioNetLoader.data_source
    }
    with open('experiments/results/ecg_metrics_phase3.json', 'w') as f:
        json.dump(metrics_dict, f, indent=4)
    
    if accuracy < 0.35:
        logging.warning("Alignment accuracy < 0.35, retraining with higher lambda_spectral")
        trainer.model.lambda_spectral = 0.02
        trainer.fit() # retrain 
        all_z, all_y, all_iters = [], [], []
        with torch.no_grad():
            for X, y in trainer.val_loader:
                X = X.to(trainer.device)
                z_star, _, _, info = trainer.model(X)
                all_z.append(z_star.cpu())
                all_y.append(y.cpu())
                all_iters.append(info['n_iters'])
        z_all = torch.cat(all_z, dim=0)
        y_all = torch.cat(all_y, dim=0)
        accuracy, cm, align_dict, z_np, aligned_clusters = align_clusters_to_classes(z_all, y_all, n_modes)

    if accuracy < 0.45:
        logging.warning(f"Soft threshold failed: alignment_accuracy={accuracy:.4f} < 0.45")
    elif accuracy > 0.60:
        logging.info(f"Strong result: alignment_accuracy={accuracy:.4f} > 0.60")
        
    n_class_idx = 0
    if np.sum(cm[n_class_idx, :]) > 0:
        n_purity = cm[n_class_idx, n_class_idx] / np.sum(cm[:, n_class_idx] + 1e-9)
        if n_purity > 0.75:
            logging.info("Model correctly isolates normal beats (N-class purity > 0.75)")
    
    # TSNE side by side
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    # subset for faster TSNE
    if len(z_np) > 2000:
        idx = np.random.choice(len(z_np), 2000, replace=False)
        z_tsne_input = z_np[idx]
        y_tsne_true = y_all.numpy()[idx]
        y_tsne_pred = aligned_clusters[idx]
    else:
        z_tsne_input = z_np
        y_tsne_true = y_all.numpy()
        y_tsne_pred = aligned_clusters
        
    z_2d = tsne.fit_transform(z_tsne_input)
    
    os.makedirs('experiments/results/figures', exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.scatter(z_2d[:, 0], z_2d[:, 1], c=y_tsne_true, cmap='tab10', s=5, alpha=0.7)
    ax1.set_title("True AAMI Classes")
    ax2.scatter(z_2d[:, 0], z_2d[:, 1], c=y_tsne_pred, cmap='tab10', s=5, alpha=0.7)
    ax2.set_title("Emergent Cluster Assignments")
    plt.savefig('experiments/results/figures/ecg_latent_tsne_phase3.png')
    plt.close()
    
    # summary
    with open('experiments/results/ecg_summary_phase3.md', 'w') as f:
        y_np = y_all.numpy()
        classes, counts = np.unique(y_np, return_counts=True)
        stats_str = ", ".join([f"Class {c}: {count}" for c, count in zip(classes, counts)])
        
        # confuse class pairs
        # exclude diagonal
        np.fill_diagonal(cm, 0)
        # get top 3 confused
        flat_indices = np.argsort(cm.flatten())[::-1]
        top_3 = flat_indices[:3]
        confused_str = ""
        for idx in top_3:
            r, c = np.unravel_index(idx, cm.shape)
            confused_str += f"- True Class {r} predicted as {c}: {int(cm[r, c])} times\n"
            
        # least confused 
        least_3 = flat_indices[-3:]
        least_str = ""
        for idx in least_3:
            r, c = np.unravel_index(idx, cm.shape)
            least_str += f"- True Class {r} predicted as {c}: {int(cm[r, c])} times\n"
            
        title_prefix = "SYNTHETIC FALLBACK " if PhysioNetLoader.data_source == "synthetic_fallback" else ""
        
        f.write(f"# {title_prefix}ECG Evaluation Summary\n\n")
        f.write(f"**Data Source**: {PhysioNetLoader.data_source}\n\n")
        f.write(f"**Dataset stats (n_samples per class)**: {stats_str}\n\n")
        f.write(f"**Model convergence (mean iterations)**: {np.mean(all_iters):.2f}\n\n")
        f.write(f"**ARI**: {ari:.4f}\n")
        f.write(f"**NMI**: {nmi:.4f}\n")
        f.write(f"**Purity**: {purity:.4f}\n")
        f.write(f"**Random Baseline ARI**: {random_ari:.4f}\n")
        f.write(f"**K-Means Baseline ARI**: {kmeans_ari:.4f}\n")
        f.write(f"**Unique True Classes**: {unique_true_classes}\n")
        f.write(f"**Unique Pred Clusters**: {unique_pred_clusters}\n\n")
        f.write(f"## Most Confused Class Pairs\n{confused_str}\n")
        f.write(f"## Least Confused Class Pairs\n{least_str}\n")

    # Save confusion matrix plot 
    plt.figure()
    plt.imshow(cm, cmap='Blues')
    plt.colorbar()
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Cluster")
    plt.ylabel("True Class")
    plt.savefig('experiments/results/figures/ecg_confusion_matrix_phase3.png')
    plt.close()
        
if __name__ == '__main__':
    main()
