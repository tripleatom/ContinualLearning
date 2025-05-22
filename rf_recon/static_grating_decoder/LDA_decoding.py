import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score, StratifiedKFold, cross_val_predict
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from scipy.optimize import curve_fit
from scipy.stats import ttest_ind
from rf_recon.rf_func import sig_label

def fit_gaussian_3d(data):
    mu     = data.mean(axis=0)
    sigmas = np.sqrt(data.var(axis=0))
    sigmas = np.clip(sigmas, 1e-6, None)  # avoid zeros
    return [*mu, *sigmas, 1.0]

def create_sphere_mesh(center, radius, resolution=20):
    phi = np.linspace(0, 2*np.pi, resolution)
    theta = np.linspace(0, np.pi, resolution)
    phi, theta = np.meshgrid(phi, theta)
    x = center[0] + radius[0] * np.sin(theta) * np.cos(phi)
    y = center[1] + radius[1] * np.sin(theta) * np.sin(phi)
    z = center[2] + radius[2] * np.cos(theta)
    return x, y, z

def visualize_lda_decoding(npz_file, cv_folds=5, random_state=42):
    npz_file = Path(npz_file)
    session_folder = npz_file.parent
    animal_id = session_folder.parent.name
    session_id = session_folder.name
    folder   = npz_file.parent
    embed    = folder / "embedding"
    embed.mkdir(exist_ok=True)

    d      = np.load(npz_file, allow_pickle=True)
    R      = d['all_units_responses'] # (n_u, n_o, n_p, n_s, n_r)
    oris   = np.array(d['unique_orientation'])
    quals  = np.array(d['unit_qualities'])
    mask   = quals != 'noise'
    R      = R[mask]
    n_o, n_p, n_s, n_r = R.shape[1:]

    # — RESHAPE FOR LDA —
    X      = R.transpose(1,2,3,4,0).reshape(-1, R.shape[0]) # (n_o, n_p, n_s, n_r, n_u) -> (n_o*n_p*n_s*n_r, n_u)
    y      = np.repeat(oris, n_p*n_s*n_r) # repeat orientation for each pixel
    le     = LabelEncoder()
    y_cls  = le.fit_transform(y) # encode orientations as integers

    # — pre‑CV per‑ori scatter + sphere —
    clf_vis   = LinearDiscriminantAnalysis().fit(X, y_cls) # fit LDA model
    X_lda_vis = clf_vis.transform(X)
    unique    = np.unique(y_cls)
    num       = len(unique)
    gaussian_params = {}

    for i, ori_enc in enumerate(unique):
        ori_val = le.inverse_transform([ori_enc])[0]
        idx     = np.where(y_cls==ori_enc)[0]
        pts     = X_lda_vis[idx,:3]
        params  = fit_gaussian_3d(pts)
        gaussian_params[ori_val] = params

        fig = plt.figure(figsize=(7,6))
        ax  = fig.add_subplot(111, projection='3d')

        # generate sphere mesh
        xs, ys, zs = create_sphere_mesh(params[:3], params[3:6], resolution=30)

        # option: bright red wireframe
        ax.plot_wireframe(
            xs, ys, zs,
            color='red',      # high contrast
            linewidth=1.0,    # slightly thicker lines
            rcount=20,        # resolution of the wireframe
            ccount=20
        )

        # then plot the points on top
        ax.scatter(pts[:,0], pts[:,1], pts[:,2],
                color='blue',  # or your tab10 color
                alpha=0.2,     # semi‑transparent so wireframe shows
                s=20)


        ax.set_xlabel("LDA 1"); ax.set_ylabel("LDA 2"); ax.set_zlabel("LDA 3")
        ax.set_title(f"Animal: {animal_id}, Session: {session_id}\nOrientation: {ori_val}°", fontsize=16, pad=20)
        out = embed / f"ori_{ori_val}_scatter_sphere.png"
        fig.savefig(out); plt.close(fig)
        print("Saved:", out)

    # — CV accuracy & projection —
    clf = LinearDiscriminantAnalysis()
    skf = StratifiedKFold(cv_folds, shuffle=True, random_state=random_state)
    scores = cross_val_score(clf, X, y_cls, cv=skf)

    clf.fit(X,y_cls)
    X_lda = clf.transform(X)

    fig = plt.figure(figsize=(15,8))
    gs  = fig.add_gridspec(1,2, width_ratios=[1,2])
    a1  = fig.add_subplot(gs[0])
    a2  = fig.add_subplot(gs[1], projection='3d')

    a1.bar(range(1,cv_folds+1), scores, color='skyblue', edgecolor='k')
    a1.set(title=f"LDA CV Accuracy", xlabel="Fold", ylabel="Accuracy", ylim=(0,1))

    cmap = plt.get_cmap('tab10')
    for i, ori_enc in enumerate(unique):
        idx    = np.where(y_cls==ori_enc)[0]
        clr    = cmap(i/num)
        ori_val= le.inverse_transform([ori_enc])[0]
        a2.scatter(
            X_lda[idx,0], X_lda[idx,1], X_lda[idx,2],
            color=clr, label=f"{ori_val}°", alpha=0.6
        )

    a2.set(title=f"LDA Projection", xlabel="LDA1", ylabel="LDA2", zlabel="LDA3")
    a2.legend(title="Ori", loc="best")
    fig.suptitle(f"Animal: {animal_id}, Session: {session_id}\nLDA CV Accuracy: {scores.mean():.3f} ± {scores.std():.3f}", fontsize=16)

    out1 = embed / "LDA_decoding.png"
    fig.savefig(out1); plt.close(fig)
    print("Saved:", out1)

    # — per‐orientation CV acc —
    y_pred = cross_val_predict(clf, X, y_cls, cv=skf)
    per   = []
    labs  = []
    for ori_enc in unique:
        idx = np.where(y_cls==ori_enc)[0]
        per.append((y_pred[idx]==y_cls[idx]).mean())
        labs.append(le.inverse_transform([ori_enc])[0])
    fig, ax = plt.subplots(figsize=(7,5))
    ax.bar(labs, per, color='lightgreen', edgecolor='k')
    ax.set_title(f"Animal: {animal_id}, Session: {session_id}\nPer-Orientation CV Accuracy", fontsize=16, pad=20)
    plt.tight_layout()
    out3 = embed / "LDA_per_orientation_accuracy.png"
    fig.savefig(out3); plt.close(fig)
    print("Saved:", out3)


    # 1. compute real vs. shuffled cross‑val scores
    scores = cross_val_score(clf, X, y_cls, cv=skf)
    Xs = X[np.random.permutation(len(X))]
    shuffled_scores = cross_val_score(clf, Xs, y_cls, cv=skf)

    # 2. t‑test and annotation
    t_stat, p = ttest_ind(scores, shuffled_scores)
    print(f"t-statistic: {t_stat:.3f}, p-value: {p:.3e}")
    label = sig_label(p)

    # 3. plot
    fig, ax = plt.subplots(figsize=(5,5), dpi=300)
    means = [scores.mean(), shuffled_scores.mean()]
    sems  = [scores.std(ddof=1)/np.sqrt(len(scores)),
            shuffled_scores.std(ddof=1)/np.sqrt(len(shuffled_scores))]

    ax.bar(['Real','Shuffled'], means, yerr=sems, capsize=5, edgecolor='k')

    # 4. draw significance line just above the higher bar
    y_max = max(m + e for m, e in zip(means, sems))
    h = y_max + 0.02                      # line height
    baroff = 0.01                         # indentation
    ax.plot(
        [0, 0, 1, 1],
        [h-baroff, h, h, h-baroff],
        lw=1, color='k'
    )
    ax.text(0.5, h, label, ha='center', va='bottom', fontsize=12)

    # 5. finalize
    ax.set_ylim(0, max(1, h + 0.05))
    ax.set_title(
        f"Animal: {animal_id}, Session: {session_id}\n"
        "Real vs Shuffled Decoding Accuracy",
        fontsize=16, pad=20
    )
    fig.tight_layout()
    out4 = embed / "LDA_real_vs_shuffled.png"
    fig.savefig(out4)
    plt.close(fig)
    print("Saved:", out4)


    print(f"Overall CV: {scores.mean():.3f} ± {scores.std():.3f}")
    return scores

if __name__=='__main__':
    npz = r'\\10.129.151.108\xieluanlabs\xl_cl\code\sortout\CnL22\250406_002515\static_grating_responses.npz'
    visualize_lda_decoding(npz)
