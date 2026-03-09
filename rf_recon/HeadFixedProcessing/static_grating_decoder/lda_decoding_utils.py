import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import LabelEncoder
from pathlib import Path

def reshape_for_lda(R, oris):
    n_o, n_p, n_s, n_r = R.shape[1:]
    X = R.transpose(1, 2, 3, 4, 0).reshape(-1, R.shape[0])
    y = np.repeat(oris, n_p * n_s * n_r)
    return X, y

def compute_lda(X, y_cls, cv_folds=5, random_state=42):
    clf = LinearDiscriminantAnalysis()
    skf = StratifiedKFold(cv_folds, shuffle=True, random_state=random_state)
    scores = cross_val_score(clf, X, y_cls, cv=skf)
    clf.fit(X, y_cls)
    X_lda = clf.transform(X)
    return clf, X_lda, scores, skf

def decode_accuracy_plot(cv_scores, ax):
    ax.bar(range(1, len(cv_scores) + 1), cv_scores, color='skyblue', edgecolor='k')
    ax.set_xlabel("CV Fold")
    ax.set_ylabel("Accuracy")
    ax.set_title("Cross-Validated Accuracy")
    ax.set_ylim([0, 1])

def per_orientation_accuracy_plot(X, y_cls, clf, skf, le, ax):
    y_pred = cross_val_predict(clf, X, y_cls, cv=skf)
    unique = np.unique(y_cls)
    accs, labels = [], []
    for ori_enc in unique:
        idx = np.where(y_cls == ori_enc)[0]
        accs.append((y_pred[idx] == y_cls[idx]).mean())
        labels.append(le.inverse_transform([ori_enc])[0])
    ax.bar(labels, accs, color='lightgreen', edgecolor='k')
    ax.set_xlabel("Orientation (°)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Per-Orientation Accuracy")
    ax.set_ylim([0, 1])

from scipy.stats import ttest_ind
from rf_recon.rf_func import sig_label  # Assumes you already use this in other parts

def real_vs_shuffled_plot(X, y_cls, clf, skf, ax):
    scores_real = cross_val_score(clf, X, y_cls, cv=skf)
    X_shuffled = X[np.random.permutation(len(X))]
    scores_shuffled = cross_val_score(clf, X_shuffled, y_cls, cv=skf)

    means = [scores_real.mean(), scores_shuffled.mean()]
    stds = [scores_real.std(), scores_shuffled.std()]

    ax.bar(['Real', 'Shuffled'], means, yerr=stds,
           color=['dodgerblue', 'salmon'], edgecolor='k', capsize=8)
    ax.set_ylabel("Accuracy")
    ax.set_ylim([0, 1])
    ax.set_title("Real vs Shuffled Accuracy")

    # Significance testing
    t_stat, p_val = ttest_ind(scores_real, scores_shuffled)
    label = sig_label(p_val)  # Converts p-value to string significance label

    # Position the label slightly above the tallest bar
    max_bar = max(m + e for m, e in zip(means, stds))
    h = max_bar + 0.02
    ax.plot([0, 0, 1, 1], [h - 0.01, h, h, h - 0.01], lw=1.2, color='k')
    ax.text(0.5, h + 0.01, label, ha='center', va='bottom', fontsize=12)

    return scores_real, scores_shuffled

