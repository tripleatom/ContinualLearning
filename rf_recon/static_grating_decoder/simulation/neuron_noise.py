import numpy as np
import matplotlib.pyplot as plt
from grating_lda import simulate_grating_lda


# Sweep over neuron numbers and noise std values
neuron_range = np.arange(20, 121, 20)
noise_range = np.logspace(np.log10(0.1), np.log10(3), 20)

# Store accuracy values
accuracy_matrix = np.zeros((len(neuron_range), len(noise_range)))

for i, n in enumerate(neuron_range):
    for j, noise in enumerate(noise_range):
        acc = simulate_grating_lda(n_neurons=n, noise_std=noise, n_trials_per_angle=750, plot_confusion=False)
        accuracy_matrix[i, j] = acc

# Plotting
plt.figure(figsize=(12, 6))
for i, n in enumerate(neuron_range):
    plt.plot(noise_range, accuracy_matrix[i], label=f'{n} neurons')
plt.xlabel("Noise Std Dev")
plt.ylabel("LDA Accuracy")
plt.title("LDA Decoding Accuracy vs Noise Level for Different Neuron Counts")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
plt.tight_layout()
plt.grid(True)
plt.show()
