import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from rec2nwb.preproc_func import parse_session_info
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from load_unit_fr import process_static_grating_responses

rec_folder = Path(input("Please enter the full path to the recording folder: ").strip())
stimdata_file = Path(input("Please enter the full path to the .mat file: ").strip())
grating_response_file = process_static_grating_responses(rec_folder, stimdata_file, overwrite=False)

animal_id, session_id, folder_name = parse_session_info(rec_folder)
code_folder = Path(__file__).parent.parent.parent
session_folder = code_folder / rf"sortout/{animal_id}/{session_id}"

print("Loading data from:", grating_response_file)

# Load the data from the npz file
data = np.load(grating_response_file, allow_pickle=True)

# Extract arrays from the file
all_units_responses = data['all_units_responses']
unit_info = data['unit_info']
stim_orientation = data['stim_orientation']
stim_phase = data['stim_phase']
stim_spatialFreq = data['stim_spatialFreq']
unique_orientation = data['unique_orientation']
unique_phase = data['unique_phase']
unique_spatialFreq = data['unique_spatialFreq']
static_grating_rising_edges = data['static_grating_rising_edges']
digInFreq = data['digInFreq']

print("all_units_responses shape:", all_units_responses.shape)
print("Number of units:", all_units_responses.shape[0])
print("Unique orientations:", unique_orientation)
print("Unique phases:", unique_phase)
print("Unique spatial frequencies:", unique_spatialFreq)

# Create an "embedding" folder in the session folder to save images
embedding_folder = Path(session_folder) / 'embedding'
embedding_folder.mkdir(parents=True, exist_ok=True)
print("Embedding images will be saved in:", embedding_folder)


#TODO: see if the code is wrong...
# ------------------------------------------------------------------
# Reshape data for embedding
# all_units_responses shape: (n_units, n_ori, n_phase, n_sf, n_repeats)
n_units, n_ori, n_phase, n_sf, n_rep = all_units_responses.shape

# Reshape so that each trial's multi-neuron firing rate vector is one row.
# Order of trials: (i_ori, i_phase, i_sf, i_rep) then n_units is last.
X = np.transpose(all_units_responses, (1, 2, 3, 4, 0))  # now shape: (n_ori, n_phase, n_sf, n_rep, n_units)
X = X.reshape(-1, n_units)  # shape: (n_trials, n_units)

# Build an array of orientation labels for each trial (order must match X)
all_oris = []
for i_ori in range(n_ori):
    for i_phase in range(n_phase):
        for i_sf in range(n_sf):
            for i_rep in range(n_rep):
                all_oris.append(unique_orientation[i_ori])
all_oris = np.array(all_oris)  # shape: (n_trials,)

# Use a discrete colormap with high contrast (Set1)
unique_oris = np.unique(all_oris)
colors = plt.cm.Set1(np.linspace(0, 1, len(unique_oris)))

# ------------------------ PCA Embedding ------------------------
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plt.figure(figsize=(7, 6))
for i, ori in enumerate(unique_oris):
    mask = (all_oris == ori)
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1],
                color=colors[i],
                label=f"{ori:.1f}°",
                s=10, alpha=0.8)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA of All-Neuron Firing Rates')
plt.legend(title="Orientation")
plt.tight_layout()
pca_file = embedding_folder / 'pca_embedding.png'
plt.savefig(pca_file)
plt.close()
print("Saved PCA embedding to:", pca_file)

# ------------------------ t-SNE Embedding ------------------------
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X)

plt.figure(figsize=(7, 6))
for i, ori in enumerate(unique_oris):
    mask = (all_oris == ori)
    plt.scatter(X_tsne[mask, 0], X_tsne[mask, 1],
                color=colors[i],
                label=f"{ori:.1f}°",
                s=10, alpha=0.8)
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.title('t-SNE of All-Neuron Firing Rates')
plt.legend(title="Orientation")
plt.tight_layout()
tsne_file = embedding_folder / 'tsne_embedding.png'
plt.savefig(tsne_file)
plt.close()
print("Saved t-SNE embedding to:", tsne_file)

# ------------------------ UMAP Embedding ------------------------
umap_embedder = umap.UMAP(n_components=2, random_state=42)
X_umap = umap_embedder.fit_transform(X)

plt.figure(figsize=(7, 6))
for i, ori in enumerate(unique_oris):
    mask = (all_oris == ori)
    plt.scatter(X_umap[mask, 0], X_umap[mask, 1],
                color=colors[i],
                label=f"{ori:.1f}°",
                s=10, alpha=0.8)
plt.xlabel('UMAP Dimension 1')
plt.ylabel('UMAP Dimension 2')
plt.title('UMAP of All-Neuron Firing Rates')
plt.legend(title="Orientation")
plt.tight_layout()
umap_file = embedding_folder / 'umap_embedding.png'
plt.savefig(umap_file)
plt.close()
print("Saved UMAP embedding to:", umap_file)