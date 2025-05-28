import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
from lda_decoding_utils import reshape_for_lda, compute_lda, decode_accuracy_plot, per_orientation_accuracy_plot, real_vs_shuffled_plot

def visualize_lda_decoding(npz_file, cv_folds=5, random_state=42, OSI_threshold=0.0, z_noise_threshold=0.0, show_fig=True):
    npz_file = Path(npz_file)
    session_folder = npz_file.parent
    animal_id = session_folder.parent.name
    session_id = session_folder.name

    data = np.load(npz_file, allow_pickle=True)
    all_units_responses = data['all_units_responses']
    unique_orientation = data['unique_orientation']
    unit_qualities = np.array(data['unit_qualities'])
    metrics_file = npz_file.parent / "single_static_grating_tuning_metrics.npz"
    data = np.load(metrics_file, allow_pickle=True)
    all_shank_info = data["all_shank_info"].item()

    all_gOSI = []
    all_z_noise = []
    for shank_id, units_dict in all_shank_info.items():
        for unit_id, metrics_dict in units_dict.items():
            all_gOSI.append(metrics_dict['gOSI'])
            all_z_noise.append(metrics_dict['z_noise'])

    all_gOSI = np.array(all_gOSI)
    all_z_noise = np.array(all_z_noise)
    valid_units_mask = (unit_qualities != 'noise') & (all_gOSI > OSI_threshold) & (all_z_noise < z_noise_threshold)
    all_units_responses = all_units_responses[valid_units_mask, ...]

    n_units, n_ori, n_phase, n_sf, n_repeats = all_units_responses.shape

    X, y = reshape_for_lda(all_units_responses, unique_orientation)
    le = LabelEncoder()
    y_cls = le.fit_transform(y)

    clf, X_lda, scores, skf = compute_lda(X, y_cls, cv_folds=cv_folds, random_state=random_state)

    fig = plt.figure(figsize=(14, 10))
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222, projection='3d')
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)

    decode_accuracy_plot(scores, ax1)

    cmap = plt.get_cmap('tab10')
    unique_orientations_encoded = np.unique(y_cls)
    for i, ori_enc in enumerate(unique_orientations_encoded):
        idx = np.where(y_cls == ori_enc)[0]
        color = cmap(i / len(unique_orientations_encoded))
        ori_val = le.inverse_transform([ori_enc])[0]
        ax2.scatter(X_lda[idx, 0], X_lda[idx, 1], X_lda[idx, 2], color=color, label=f"{ori_val}°", alpha=0.7, s=10)
    ax2.set_xlabel("LDA 1")
    ax2.set_ylabel("LDA 2")
    ax2.set_zlabel("LDA 3")
    ax2.set_title("LDA Projection (3D)")
    ax2.legend(title="Orientation", fontsize=8)

    per_orientation_accuracy_plot(X, y_cls, clf, skf, le, ax3)

    real_vs_shuffled_plot(X, y_cls, clf, skf, ax4)

    fig.suptitle(f"LDA Decoding | {animal_id} - {session_id} | OSI>{OSI_threshold}, z-noise<{z_noise_threshold:.2f} | Remaining Units: {n_units}\nLDA CV Accuracy: {scores.mean():.3f} ± {scores.std():.3f}", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    embedding_folder = session_folder / "embedding"
    embedding_folder.mkdir(parents=True, exist_ok=True)
    save_path_combined = embedding_folder / f"LDA_summary_OSI_{OSI_threshold}_znoise_{z_noise_threshold:.2f}.png"
    plt.savefig(save_path_combined, dpi=300)

    print(f"Combined LDA figure saved to {save_path_combined}")
    print(f"Mean decoding accuracy: {scores.mean():.3f} ± {scores.std():.3f} (real)")

    if show_fig:
        plt.show()
    else:
        plt.close()

    return scores


if __name__ == '__main__':
    OSI_threshold = 0
    z_noise_threshold = 0.5
    npz_file_path = rf"\\10.129.151.108\xieluanlabs\xl_cl\code\sortout\CnL38\250515_173450\static_grating_responses.npz"
    visualize_lda_decoding(npz_file_path, OSI_threshold=OSI_threshold, z_noise_threshold=z_noise_threshold, show_fig=False)