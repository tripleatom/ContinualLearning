import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import pandas as pd

def simulate_grating_lda(n_neurons=40, noise_std=1.0, n_trials_per_angle=750, plot_confusion=True):
    # --- Fixed parameters ---
    angles = np.arange(0, 180 - 1e-3, 22.5)  # 0 to 180, step 22.5 -> 8 angles
    tuning_width = 40  # std of the Gaussian tuning
    np.random.seed(0)  # for reproducibility

    # --- Generate tuning curves ---
    preferred_angles = np.random.uniform(0, 180, n_neurons)

    def gaussian_tuning(x, mu, sigma):
        return np.exp(-0.5 * ((x - mu) % 180)**2 / sigma**2)

    tuning_curves = np.array([
        gaussian_tuning(angles, mu, tuning_width)
        for mu in preferred_angles
    ]).T  # shape: (n_angles, n_neurons)

    # --- Simulate responses ---
    X, y = [], []
    for i, angle in enumerate(angles):
        mean_response = tuning_curves[i]
        responses = np.tile(mean_response, (n_trials_per_angle, 1))
        noisy_responses = responses + np.random.normal(0, noise_std, responses.shape)
        X.append(noisy_responses)
        y.extend([angle] * n_trials_per_angle)

    X = np.vstack(X)
    y = np.array(y)

    # --- Label encoding ---
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # --- LDA decoding ---
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=1)
    clf = LDA()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    # --- Plot confusion matrix ---
    if plot_confusion:
        cm = confusion_matrix(y_test, y_pred)
        class_labels = label_encoder.inverse_transform(np.arange(len(angles)))
        cm_df = pd.DataFrame(cm, index=class_labels, columns=class_labels)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(f"LDA Confusion Matrix (Acc: {acc:.2%})")
        plt.tight_layout()
        plt.show()

    return acc

