import numpy as np
import matplotlib.pyplot as plt
from rf_recon.static_grating_decoder.LDA_filter_gOSI import visualize_lda_decoding
from pathlib import Path
# Simulated results from calling visualize_lda_decoding with different z_noise_thresholds
z_noise_thresholds = np.linspace(0.1, 0.8, 10)
simulated_accuracies = []
npz_file_path = rf"\\10.129.151.108\xieluanlabs\xl_cl\code\sortout\CnL38\250515_173450\static_grating_responses.npz"
npz_file_path = Path(npz_file_path)

for z_thresh in z_noise_thresholds:
    # Simulate slightly higher accuracy when stricter threshold is used
    simulated_cv_scores = visualize_lda_decoding(npz_file_path, z_noise_threshold=z_thresh, show_fig=False)
    simulated_accuracies.append(np.mean(simulated_cv_scores))

# Plotting
plt.figure(figsize=(8, 5))
plt.plot(z_noise_thresholds, simulated_accuracies, marker='o', linewidth=2)
plt.xlabel("z-noise Threshold")
plt.ylabel("Mean Decoding Accuracy")
plt.title("Effect of z-noise Threshold on LDA Decoding Accuracy")
plt.grid(True)
plt.tight_layout()
out_path = npz_file_path.parent / "embedding" / "LDA_diff_znoise_thres.png"
plt.savefig(out_path)
plt.show()
