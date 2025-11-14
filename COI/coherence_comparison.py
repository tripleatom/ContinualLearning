import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

# === CONFIGURATION ===
# List of pickle files to compare
pkl_files = [
    r"\\10.129.151.108\xieluanlabs\xl_spinal_cord_electrode\CoI\CoI11\251103\3cms\data_251103_163841\data_251103_163841_brain_spinal_coherence.pkl",
    r"\\10.129.151.108\xieluanlabs\xl_spinal_cord_electrode\CoI\CoI11\251103\3cms_2\data_251103_165302\data_251103_165302_brain_spinal_coherence.pkl",
    r"\\10.129.151.108\xieluanlabs\xl_spinal_cord_electrode\CoI\CoI11\251103\data_251103_165957\data_251103_165957_brain_spinal_coherence.pkl",
]

# Labels for each session (will be extracted from filenames if None)
session_labels = ['3cm', '3cm_2', 'static']  # Or provide custom labels like ['Session 1', 'Session 2', 'Session 3']

# Output folder for comparison plots
output_folder = r"\\10.129.151.108\xieluanlabs\xl_spinal_cord_electrode\CoI\CoI11\251103\coherence_comparison"

# Plotting parameters
plot_params = {
    'figsize': (12, 8),
    'dpi': 150,
    'colors': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'],  # Up to 6 sessions
    'linewidth': 2,
    'alpha': 0.7,
    'freq_range': (0.5, 100),  # Frequency range to plot
    'log_freq': False,  # Use log scale for frequency axis
}

# === LOAD ALL PICKLE FILES ===
print("Loading pickle files...")
sessions_data = []
session_names = []

for i, pkl_file in enumerate(pkl_files):
    pkl_path = Path(pkl_file)
    
    if not pkl_path.exists():
        print(f"WARNING: File not found: {pkl_file}")
        continue
    
    print(f"\nLoading: {pkl_path.name}")
    
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    sessions_data.append(data)
    
    # Extract session name
    if session_labels is None:
        session_name = pkl_path.stem.replace('_brain_spinal_coherence', '')
    else:
        session_name = session_labels[i]
    
    session_names.append(session_name)
    
    print(f"  Session: {session_name}")
    print(f"  Number of pairs: {len(data['coherence'])}")
    print(f"  Frequency points: {len(data['frequencies'])}")

n_sessions = len(sessions_data)
print(f"\nTotal sessions loaded: {n_sessions}")

if n_sessions == 0:
    print("ERROR: No valid pickle files found!")
    exit(1)

# === VERIFY COMPATIBILITY ===
print("\nVerifying data compatibility...")

# Check if all sessions have same frequency array
ref_freqs = sessions_data[0]['frequencies']
for i, data in enumerate(sessions_data[1:], 1):
    if not np.allclose(ref_freqs, data['frequencies']):
        print(f"WARNING: Session {i} has different frequency array!")

# === BUILD PAIR INDEX ===
print("\nBuilding pair index across sessions...")

# Create a mapping of (brain_shank, brain_ch, spinal_shank, spinal_ch) -> list of session indices
pair_to_sessions = {}

for session_idx, data in enumerate(sessions_data):
    for pair_idx in range(len(data['coherence'])):
        pair_key = (
            data['brain_shank'][pair_idx],
            data['brain_channel_id'][pair_idx],
            data['spinal_shank'][pair_idx],
            data['spinal_channel_id'][pair_idx]
        )
        
        if pair_key not in pair_to_sessions:
            pair_to_sessions[pair_key] = {}
        
        pair_to_sessions[pair_key][session_idx] = pair_idx

print(f"Total unique pairs: {len(pair_to_sessions)}")

# Filter to only pairs that exist in ALL sessions
complete_pairs = {k: v for k, v in pair_to_sessions.items() if len(v) == n_sessions}
print(f"Pairs present in all {n_sessions} sessions: {len(complete_pairs)}")

if len(complete_pairs) == 0:
    print("ERROR: No pairs found in all sessions!")
    exit(1)

# === CREATE OUTPUT FOLDER ===
output_path = Path(output_folder)
output_path.mkdir(parents=True, exist_ok=True)
print(f"\nSaving comparison plots to: {output_path}")

# === PLOT EACH PAIR ===
print(f"\nGenerating {len(complete_pairs)} comparison plots...")

for pair_key, session_pair_indices in tqdm(complete_pairs.items(), desc="Plotting pairs"):
    brain_shank, brain_ch, spinal_shank, spinal_ch = pair_key
    
    # Create figure
    fig, ax = plt.subplots(figsize=plot_params['figsize'])
    
    # Plot coherence for each session
    for session_idx in range(n_sessions):
        pair_idx = session_pair_indices[session_idx]
        data = sessions_data[session_idx]
        
        freqs = data['frequencies']
        coherence = data['coherence'][pair_idx]
        
        # Apply frequency range filter
        freq_mask = (freqs >= plot_params['freq_range'][0]) & \
                    (freqs <= plot_params['freq_range'][1])
        freqs_plot = freqs[freq_mask]
        coherence_plot = coherence[freq_mask]
        
        # Plot
        color = plot_params['colors'][session_idx % len(plot_params['colors'])]
        ax.plot(freqs_plot, coherence_plot, 
               color=color, 
               linewidth=plot_params['linewidth'],
               alpha=plot_params['alpha'],
               label=session_names[session_idx])
    
    # Formatting
    ax.set_xlabel('Frequency (Hz)', fontsize=12)
    ax.set_ylabel('Coherence', fontsize=12)
    ax.set_title(f'Brain Sh{brain_shank} Ch{brain_ch} - Spinal Sh{spinal_shank} Ch{spinal_ch}',
                fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', framealpha=0.9)
    
    if plot_params['log_freq']:
        ax.set_xscale('log')
    
    # Add horizontal line at 0.5 for reference
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3, linewidth=1)
    
    plt.tight_layout()
    
    # Save figure
    filename = f"coherence_brain_sh{brain_shank}_ch{brain_ch:03d}_spinal_sh{spinal_shank}_ch{spinal_ch:03d}.png"
    save_path = output_path / filename
    plt.savefig(save_path, dpi=plot_params['dpi'], bbox_inches='tight')
    plt.close()

print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
print(f"Sessions compared: {n_sessions}")
for i, name in enumerate(session_names):
    print(f"  {i+1}. {name}")
print(f"\nTotal pairs plotted: {len(complete_pairs)}")
print(f"Output folder: {output_path}")
print(f"\n{'='*60}")

# === CREATE SUMMARY STATISTICS ===
print("\nCalculating summary statistics...")

# Average coherence across all pairs for each session
summary_data = {
    'session_names': session_names,
    'frequencies': ref_freqs,
    'mean_coherence': [],
    'std_coherence': [],
    'sem_coherence': [],
}

for session_idx in range(n_sessions):
    data = sessions_data[session_idx]
    
    # Get coherence for all pairs in this session
    all_coherence = []
    for pair_key in complete_pairs.keys():
        pair_idx = pair_to_sessions[pair_key][session_idx]
        all_coherence.append(data['coherence'][pair_idx])
    
    all_coherence = np.array(all_coherence)
    
    summary_data['mean_coherence'].append(np.mean(all_coherence, axis=0))
    summary_data['std_coherence'].append(np.std(all_coherence, axis=0))
    summary_data['sem_coherence'].append(np.std(all_coherence, axis=0) / np.sqrt(len(all_coherence)))

# Plot summary comparison
fig, ax = plt.subplots(figsize=(12, 8))

for session_idx in range(n_sessions):
    freqs = summary_data['frequencies']
    mean_coh = summary_data['mean_coherence'][session_idx]
    sem_coh = summary_data['sem_coherence'][session_idx]
    
    # Apply frequency range filter
    freq_mask = (freqs >= plot_params['freq_range'][0]) & \
                (freqs <= plot_params['freq_range'][1])
    freqs_plot = freqs[freq_mask]
    mean_plot = mean_coh[freq_mask]
    sem_plot = sem_coh[freq_mask]
    
    color = plot_params['colors'][session_idx % len(plot_params['colors'])]
    
    # Plot mean
    ax.plot(freqs_plot, mean_plot, 
           color=color,
           linewidth=plot_params['linewidth'],
           label=session_names[session_idx])
    
    # Plot SEM as shaded region
    ax.fill_between(freqs_plot, 
                    mean_plot - sem_plot,
                    mean_plot + sem_plot,
                    color=color,
                    alpha=0.2)

ax.set_xlabel('Frequency (Hz)', fontsize=12)
ax.set_ylabel('Coherence', fontsize=12)
ax.set_title('Average Brain-Spinal Coherence Across All Pairs', fontsize=14, fontweight='bold')
ax.set_ylim([0, 1])
ax.grid(True, alpha=0.3)
ax.legend(loc='best', framealpha=0.9)
ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3, linewidth=1)

if plot_params['log_freq']:
    ax.set_xscale('log')

plt.tight_layout()
summary_path = output_path / "summary_average_coherence.png"
plt.savefig(summary_path, dpi=plot_params['dpi'], bbox_inches='tight')
plt.close()

print(f"Summary plot saved: {summary_path}")

# Save summary statistics
summary_pkl_path = output_path / "summary_statistics.pkl"
with open(summary_pkl_path, 'wb') as f:
    pickle.dump(summary_data, f)

print(f"Summary statistics saved: {summary_pkl_path}")
print("\nDone!")