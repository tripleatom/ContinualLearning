import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import LabelEncoder

def create_sphere_mesh(center, radius, resolution=30):
    """Return an ellipsoid mesh (X, Y, Z) for given center & radii."""
    phi = np.linspace(0, 2*np.pi, resolution)
    theta = np.linspace(0, np.pi, resolution)
    phi, theta = np.meshgrid(phi, theta)
    x = center[0] + radius[0] * np.sin(theta) * np.cos(phi)
    y = center[1] + radius[1] * np.sin(theta) * np.sin(phi)
    z = center[2] + radius[2] * np.cos(theta)
    return x, y, z

# — PARAMETERS —
npz_file   = Path(r'\\10.129.151.108\xieluanlabs\xl_cl\code\sortout\CnL30\250521_114006\static_grating_responses.npz')
animal_id = npz_file.parent.parent.name
session_id = npz_file.parent.name

embed_dir  = npz_file.parent / "embedding"
embed_dir.mkdir(exist_ok=True)

# — LOAD & FILTER —
data       = np.load(npz_file, allow_pickle=True)
R          = data['all_units_responses']
orient     = np.array(data['unique_orientation'])
quality    = np.array(data['unit_qualities'])
R          = R[quality != 'noise']

# — RESHAPE FOR LDA —
n_units, n_ori, n_ph, n_sf, n_rep = R.shape
n_trials = n_ori * n_ph * n_sf * n_rep
X        = R.transpose(1,2,3,4,0).reshape(n_trials, n_units)
y        = np.repeat(orient, n_ph * n_sf * n_rep)

le    = LabelEncoder().fit(y)
y_cls = le.transform(y)

# — LDA PROJECTION —
lda   = LinearDiscriminantAnalysis().fit(X, y_cls)
X_lda = lda.transform(X)[:, :3]

# — COMPUTE MEANS & STDS —
stats = []
for ori_enc in np.unique(y_cls):
    ori_val = le.inverse_transform([ori_enc])[0]
    pts     = X_lda[y_cls==ori_enc]
    mu      = pts.mean(axis=0)
    sigma   = pts.std(axis=0)
    stats.append({
        'orientation': ori_val,
        'mean_x': mu[0], 'mean_y': mu[1], 'mean_z': mu[2],
        'std_x': sigma[0], 'std_y': sigma[1], 'std_z': sigma[2],
    })

df = pd.DataFrame(stats).sort_values('orientation').round(3)
print(df.to_string(index=False))

# — PLOT TABLE —
fig, ax = plt.subplots(figsize=(10, df.shape[0]*0.3 + 1))
ax.axis('off')
tbl = ax.table(cellText=df.values,
               colLabels=df.columns,
               cellLoc='center', loc='center')
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
tbl.scale(1, 1.2)
plt.title(f"{animal_id} {session_id} \n LDA Orientation Means & Stds", fontsize=12, pad=20)
plt.tight_layout()
fig.savefig(embed_dir/"orientation_means_stds.png", dpi=300, bbox_inches='tight')
plt.close(fig)

# — PLOT ALL ELLIPSOIDS —
fig = plt.figure(figsize=(10,8))
ax  = fig.add_subplot(111, projection='3d')
cmap = plt.get_cmap('tab10')
num  = len(df)

for i, row in df.reset_index().iterrows():
    center = row[['mean_x','mean_y','mean_z']].values
    radius = row[['std_x','std_y','std_z']].values
    ori     = row['orientation']
    xs, ys, zs = create_sphere_mesh(center, radius)
    color = cmap(i/num)
    ax.plot_surface(xs, ys, zs,
                    facecolor=color, edgecolor='k', linewidth=0.2,
                    alpha=0.25, shade=True)
    ax.scatter(*center, color=color, s=50, label=f"{ori}°")

ax.set_xlabel("LDA 1"); ax.set_ylabel("LDA 2"); ax.set_zlabel("LDA 3")
ax.set_title(f"{animal_id} {session_id} \n LDA Orientation Ellipsoids", pad=20, fontsize=12)
ax.legend(title="Orientation", bbox_to_anchor=(1.05,1), loc='upper left')
plt.tight_layout()
fig.savefig(embed_dir/"all_orientations_ellipsoids.png", dpi=300, bbox_inches='tight')
plt.close(fig)

# — PAIRWISE OVERLAP: INTERSECTION / VOL(i) ——
centers      = df[['mean_x','mean_y','mean_z']].values
radii        = df[['std_x','std_y','std_z']].values
n            = len(df)
N            = 200_000
overlap_frac = np.zeros((n,n))

for i in range(n):
    c1, r1 = centers[i], radii[i]
    for j in range(n):
        if i == j:
            overlap_frac[i,j] = 1.0
            continue
        c2, r2 = centers[j], radii[j]
        mins    = np.minimum(c1-r1, c2-r2)
        maxs    = np.maximum(c1+r1, c2+r2)
        span    = maxs - mins
        pts     = np.random.rand(N,3) * span + mins

        in1     = (((pts - c1)/r1)**2).sum(axis=1) <= 1
        in2     = (((pts - c2)/r2)**2).sum(axis=1) <= 1
        count1  = in1.sum()
        inter   = np.logical_and(in1,in2).sum()

        overlap_frac[i,j] = inter/count1 if count1>0 else 0.0

# — HEATMAP OF FRACTIONAL OVERLAP ——
fig, ax = plt.subplots(figsize=(8,6))
im = ax.imshow(overlap_frac, vmin=0, vmax=1, cmap='viridis')

# annotate every cell
for i in range(n):
    for j in range(n):
        val = overlap_frac[i,j]
        ax.text(j, i, f"{val:.2f}",
                ha='center', va='center',
                color='white' if val>0.5 else 'black',
                fontsize=9)

# axis labels
labels = df['orientation'].astype(str).tolist()
ax.set_xticks(np.arange(n)); ax.set_yticks(np.arange(n))
ax.set_xticklabels(labels, rotation=45, ha='right')
ax.xaxis.tick_top()
ax.xaxis.set_label_position('top')
ax.set_yticklabels(labels)

# grid & spines
ax.set_xticks(np.arange(n+1)-.5, minor=True)
ax.set_yticks(np.arange(n+1)-.5, minor=True)
ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)
for spine in ax.spines.values():
    spine.set_visible(False)

# colorbar
cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('Overlap / Vol(i)', fontsize=11)
cbar.ax.tick_params(labelsize=9)

ax.set_title(f"{animal_id} {session_id} \n Ellipsoid Overlap Fraction", fontsize=12, pad=20)
plt.tight_layout()
fig.savefig(embed_dir/"ellipsoid_overlap_fraction.png", dpi=300, bbox_inches='tight')
plt.close(fig)
