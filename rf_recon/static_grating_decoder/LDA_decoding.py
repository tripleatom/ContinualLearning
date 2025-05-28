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


def visualize_lda_decoding_with_spheres(npz_file, cv_folds=5, random_state=42):
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

    # Gaussian spheres
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

        ax.set_xlabel("LDA 1"); ax.set_ylabel("LDA 2"); ax.set_zlabel("LDA 3")
        ax.set_title(f"Animal: {animal_id}, Session: {session_id}\nOrientation: {ori_val}°", fontsize=16, pad=20)
        fig.savefig(embed / f"ori_{ori_val}_scatter_sphere.png")
        plt.close(fig)

    # Summary figure
    fig = plt.figure(figsize=(14, 10))
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222, projection='3d')
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)

    decode_accuracy_plot(scores, ax1)

    for i, ori_enc in enumerate(unique):
        idx = np.where(y_cls == ori_enc)[0]
        ori_val = le.inverse_transform([ori_enc])[0]
        ax2.scatter(X_lda[idx, 0], X_lda[idx, 1], X_lda[idx, 2],
                    color=cmap(i / len(unique)), label=f"{ori_val}°", alpha=0.6)
    ax2.set(title="LDA Projection (3D)", xlabel="LDA1", ylabel="LDA2", zlabel="LDA3")
    ax2.legend(title="Orientation", loc="best", fontsize=8)

    per_orientation_accuracy_plot(X, y_cls, clf, skf, le, ax3)
    real_vs_shuffled_plot(X, y_cls, clf, skf, ax4)

    fig.suptitle(f"Animal: {animal_id}, Session: {session_id}\nUnits number: {n_good}, LDA CV Accuracy: {scores.mean():.3f} ± {scores.std():.3f}", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(embed / "LDA_summary_with_spheres.png", dpi=300)
    plt.close(fig)

    print(f"Saved all figures in {embed}")
    return scores

if __name__=='__main__':
    npz = r'\\10.129.151.108\xieluanlabs\xl_cl\code\sortout\CnL30\250521_114006\static_grating_responses.npz'
    visualize_lda_decoding_with_spheres(npz)
