import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
from lda_decoding_utils import (
    reshape_for_lda, compute_lda, decode_accuracy_plot,
    per_orientation_accuracy_plot, real_vs_shuffled_plot
)
from rf_recon.rf_func import sig_label


def fit_gaussian_3d(data):
    mu = data.mean(axis=0)
    sigmas = np.sqrt(data.var(axis=0))
    sigmas = np.clip(sigmas, 1e-6, None)
    return [*mu, *sigmas, 1.0]


def create_sphere_mesh(center, radius, resolution=20):
    phi = np.linspace(0, 2 * np.pi, resolution)
    theta = np.linspace(0, np.pi, resolution)
    phi, theta = np.meshgrid(phi, theta)
    x = center[0] + radius[0] * np.sin(theta) * np.cos(phi)
    y = center[1] + radius[1] * np.sin(theta) * np.sin(phi)
    z = center[2] + radius[2] * np.cos(theta)
    return x, y, z


def visualize_lda_decoding_with_spheres(npz_file, cv_folds=5, random_state=42, show_cv_figure=False):
    npz_file = Path(npz_file)
    session_folder = npz_file.parent
    animal_id = session_folder.parent.name
    session_id = session_folder.name
    embed = session_folder / "embedding"
    embed.mkdir(exist_ok=True)

    d = np.load(npz_file, allow_pickle=True)
    R = d['all_units_responses']
    oris = np.array(d['unique_orientation'])
    quals = np.array(d['unit_qualities'])
    mask = quals != 'noise'
    n_good = mask.sum()
    print(f"Number of good units: {n_good}")

    R = R[mask]

    X, y = reshape_for_lda(R, oris)
    le = LabelEncoder()
    y_cls = le.fit_transform(y)

    clf, X_lda, scores, skf = compute_lda(X, y_cls, cv_folds=cv_folds, random_state=random_state)

    # Gaussian spheres - only create if show_cv_figure is True
    if show_cv_figure:
        unique = np.unique(y_cls)
        cmap = plt.get_cmap('tab10')
        for i, ori_enc in enumerate(unique):
            ori_val = le.inverse_transform([ori_enc])[0]
            idx = np.where(y_cls == ori_enc)[0]
            pts = X_lda[idx, :3]
            params = fit_gaussian_3d(pts)

            fig = plt.figure(figsize=(7, 6))
            ax = fig.add_subplot(111, projection='3d')

            xs, ys, zs = create_sphere_mesh(params[:3], params[3:6], resolution=30)
            ax.plot_wireframe(xs, ys, zs, color='red', linewidth=1.0, rcount=20, ccount=20)
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], color=cmap(i / len(unique)), alpha=0.3, s=20)

            ax.set_xlabel("LDA 1", fontsize=16)
            ax.set_ylabel("LDA 2", fontsize=16)
            ax.set_zlabel("LDA 3", fontsize=16)
            ax.set_title(f"Animal: {animal_id}, Session: {session_id}\nOrientation: {ori_val}°", fontsize=18, pad=20)
            ax.tick_params(axis='both', which='major', labelsize=14)
            fig.savefig(embed / f"ori_{ori_val}_scatter_sphere.png", dpi=300, bbox_inches='tight')
            plt.close(fig)

    # Separate CV figure
    fig_cv = plt.figure(figsize=(8, 6))
    ax_cv = fig_cv.add_subplot(111)
    decode_accuracy_plot(scores, ax_cv)
    ax_cv.tick_params(axis='both', which='major', labelsize=14)
    ax_cv.set_xlabel(ax_cv.get_xlabel(), fontsize=16)
    ax_cv.set_ylabel(ax_cv.get_ylabel(), fontsize=16)
    ax_cv.set_title(ax_cv.get_title(), fontsize=18)
    plt.tight_layout()
    fig_cv.savefig(embed / "LDA_CV_accuracy.png", dpi=300, bbox_inches='tight')
    plt.close(fig_cv)

    # Separate overlap matrix figure
    fig_overlap = plt.figure(figsize=(8, 6))
    ax_overlap = fig_overlap.add_subplot(111)
    plot_overlap_heatmap(X_lda, y_cls, le, ax_overlap)
    plt.tight_layout()
    fig_overlap.savefig(embed / "LDA_overlap_matrix.png", dpi=300, bbox_inches='tight')
    plt.close(fig_overlap)

    # Main summary figure with 4 subplots (custom layout)
    fig = plt.figure(figsize=(16, 12))
    # Create custom grid: top 2 plots bigger, bottom 2 plots smaller
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 2], hspace=0.3, wspace=0.2)
    ax1 = fig.add_subplot(gs[0, 0], projection='3d')  # LDA scatter (bigger)
    ax2 = fig.add_subplot(gs[0, 1], projection='3d')  # LDA ellipsoids (bigger)
    ax3 = fig.add_subplot(gs[1, 0])  # Per-orientation accuracy (smaller)
    ax4 = fig.add_subplot(gs[1, 1])  # Real vs shuffled (smaller)

    # LDA scatter plot (top left)
    unique = np.unique(y_cls)
    cmap = plt.get_cmap('tab10')
    for i, ori_enc in enumerate(unique):
        idx = np.where(y_cls == ori_enc)[0]
        ori_val = le.inverse_transform([ori_enc])[0]
        ax1.scatter(X_lda[idx, 0], X_lda[idx, 1], X_lda[idx, 2],
                    color=cmap(i / len(unique)), label=f"{ori_val}°", alpha=0.6)
    ax1.set_title("LDA Projection (3D)", fontsize=18)
    ax1.set_xlabel("LDA1", fontsize=16)
    ax1.set_ylabel("LDA2", fontsize=16)
    ax1.set_zlabel("LDA3", fontsize=16)
    ax1.legend(title="Orientation", loc="best", fontsize=12, title_fontsize=14)
    ax1.set_xticklabels([])  # Hide x tick labels
    ax1.set_yticklabels([])  # Hide y tick labels
    ax1.set_zticklabels([])  # Hide z tick labels

    # LDA ellipsoids plot (top right)
    plot_ellipsoids(X_lda, y_cls, le, ax2)
    ax2.set_xticklabels([])  # Hide x tick labels
    ax2.set_yticklabels([])  # Hide y tick labels
    ax2.set_zticklabels([])  # Hide z tick labels
    
    # Per-orientation accuracy (bottom left)
    per_orientation_accuracy_plot_fixed(X, y_cls, clf, skf, le, ax3)
    
    # Real vs shuffled (bottom right)
    real_vs_shuffled_plot(X, y_cls, clf, skf, ax4)
    ax4.tick_params(axis='both', which='major', labelsize=14)
    ax4.set_xlabel(ax4.get_xlabel(), fontsize=16)
    ax4.set_ylabel(ax4.get_ylabel(), fontsize=16)
    ax4.set_title(ax4.get_title(), fontsize=18)

    fig.suptitle(f"Animal: {animal_id}, Session: {session_id}\nUnits number: {n_good}, LDA CV Accuracy: {scores.mean():.3f} ± {scores.std():.3f}", 
                 fontsize=20, y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(embed / "LDA_summary_main.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"Saved all figures in {embed}")
    return scores


def per_orientation_accuracy_plot_fixed(X, y_cls, clf, skf, le, ax):
    """
    Modified version of per_orientation_accuracy_plot with proper x-axis labels
    """
    from sklearn.metrics import accuracy_score
    
    # Get predictions for each fold
    y_pred = cross_val_predict(clf, X, y_cls, cv=skf)
    
    # Calculate per-orientation accuracy
    unique_oris = np.unique(y_cls)
    ori_accuracies = []
    ori_labels = []
    
    for ori_enc in unique_oris:
        ori_val = le.inverse_transform([ori_enc])[0]
        ori_labels.append(f"{ori_val}°")
        
        # Get indices for this orientation
        ori_mask = y_cls == ori_enc
        ori_acc = accuracy_score(y_cls[ori_mask], y_pred[ori_mask])
        ori_accuracies.append(ori_acc)
    
    # Create the plot
    bars = ax.bar(range(len(ori_labels)), ori_accuracies, alpha=0.7)
    ax.set_xlabel("Orientation", fontsize=16)
    ax.set_ylabel("Accuracy", fontsize=16)
    ax.set_title("Per-Orientation Decoding Accuracy", fontsize=18)
    ax.set_xticks(range(len(ori_labels)))
    ax.set_xticklabels(ori_labels, rotation=45, ha='right', fontsize=14)
    ax.tick_params(axis='y', which='major', labelsize=14)
    ax.set_ylim(0, 1)
    
    # Add value labels on bars
    for i, (bar, acc) in enumerate(zip(bars, ori_accuracies)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{acc:.3f}', ha='center', va='bottom', fontsize=12)


def plot_ellipsoids(X_lda, y_cls, le, ax):
    """
    Plot ellipsoids representing the distribution of each orientation in LDA space
    """
    import pandas as pd
    
    # Compute means and stds for each orientation
    stats = []
    for ori_enc in np.unique(y_cls):
        ori_val = le.inverse_transform([ori_enc])[0]
        pts = X_lda[y_cls == ori_enc]
        mu = pts.mean(axis=0)
        sigma = pts.std(axis=0)
        stats.append({
            'orientation': ori_val,
            'mean_x': mu[0], 'mean_y': mu[1], 'mean_z': mu[2],
            'std_x': sigma[0], 'std_y': sigma[1], 'std_z': sigma[2],
        })
    
    df = pd.DataFrame(stats).sort_values('orientation')
    cmap = plt.get_cmap('tab10')
    num = len(df)
    
    # Plot ellipsoids
    for i, row in df.reset_index().iterrows():
        center = row[['mean_x', 'mean_y', 'mean_z']].values
        radius = row[['std_x', 'std_y', 'std_z']].values
        ori = row['orientation']
        
        xs, ys, zs = create_sphere_mesh(center, radius)
        color = cmap(i / num)
        
        ax.plot_surface(xs, ys, zs,
                       facecolor=color, edgecolor='k', linewidth=0.2,
                       alpha=0.25, shade=True)
        ax.scatter(*center, color=color, s=50, label=f"{ori}°")
    
    ax.set_xlabel("LDA 1", fontsize=16)
    ax.set_ylabel("LDA 2", fontsize=16)
    ax.set_zlabel("LDA 3", fontsize=16)
    ax.set_title("LDA Orientation Ellipsoids", fontsize=18)
    # Removed legend for ellipsoids
    ax.tick_params(axis='both', which='major', labelsize=12)


def plot_overlap_heatmap(X_lda, y_cls, le, ax):
    """
    Plot heatmap of ellipsoid overlaps between orientations
    """
    import pandas as pd
    
    # Compute means and stds for each orientation
    stats = []
    for ori_enc in np.unique(y_cls):
        ori_val = le.inverse_transform([ori_enc])[0]
        pts = X_lda[y_cls == ori_enc]
        mu = pts.mean(axis=0)
        sigma = pts.std(axis=0)
        stats.append({
            'orientation': ori_val,
            'mean_x': mu[0], 'mean_y': mu[1], 'mean_z': mu[2],
            'std_x': sigma[0], 'std_y': sigma[1], 'std_z': sigma[2],
        })
    
    df = pd.DataFrame(stats).sort_values('orientation')
    
    # Compute pairwise overlaps
    centers = df[['mean_x', 'mean_y', 'mean_z']].values
    radii = df[['std_x', 'std_y', 'std_z']].values
    n = len(df)
    N = 50000  # Reduced for faster computation
    overlap_frac = np.zeros((n, n))
    
    for i in range(n):
        c1, r1 = centers[i], radii[i]
        for j in range(n):
            if i == j:
                overlap_frac[i, j] = 1.0
                continue
            c2, r2 = centers[j], radii[j]
            mins = np.minimum(c1 - r1, c2 - r2)
            maxs = np.maximum(c1 + r1, c2 + r2)
            span = maxs - mins
            pts = np.random.rand(N, 3) * span + mins
            
            in1 = (((pts - c1) / r1) ** 2).sum(axis=1) <= 1
            in2 = (((pts - c2) / r2) ** 2).sum(axis=1) <= 1
            count1 = in1.sum()
            inter = np.logical_and(in1, in2).sum()
            
            overlap_frac[i, j] = inter / count1 if count1 > 0 else 0.0
    
    # Plot heatmap
    im = ax.imshow(overlap_frac, vmin=0, vmax=1, cmap='viridis')
    
    # Annotate cells
    for i in range(n):
        for j in range(n):
            val = overlap_frac[i, j]
            ax.text(j, i, f"{val:.2f}",
                   ha='center', va='center',
                   color='white' if val > 0.5 else 'black',
                   fontsize=10)
    
    # Format axes
    labels = df['orientation'].astype(str).tolist()
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=14)
    ax.set_yticklabels(labels, fontsize=14)
    ax.set_xlabel("Orientation", fontsize=16)
    ax.set_ylabel("Orientation", fontsize=16)
    ax.set_title("Ellipsoid Overlap Fraction", fontsize=18)
    
    # Add colorbar
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label('Overlap Fraction', fontsize=14)
    cbar.ax.tick_params(labelsize=12)


if __name__=='__main__':
    # npz = r'\\10.129.151.108\xieluanlabs\xl_cl\code\sortout\CnL22\250531_204108\static_grating_responses.npz'
    npz = '/Volumes/xieluanlabs/xl_cl/code/sortout/CnL22/250523_142619/static_grating_responses.npz'
    
    # Default: don't show CV figure
    visualize_lda_decoding_with_spheres(npz, show_cv_figure=False)
    
    # To show CV figure, use:
    # visualize_lda_decoding_with_spheres(npz, show_cv_figure=True)