# passive_two_grating_task.py
from psychopy import visual, core, event
import csv, random, time
import os

# -----------------------
# Experiment parameters
# -----------------------
win_fullscreen    = True
screen_bg_color   = [-1, -1, -1]     # black
stim_duration_s   = 0.25              # on-screen time per trial
iti_duration_s    = 0.5              # black screen between trials
grating_oris_deg  = (0.0, 45.0)      # two orientations to present
n_trials          = 10               # number of trials
random_seed       = 42

# Layout (PIXELS)
win_size_pix      = (1440, 900)
eccentricity_pix  = 300              # horizontal offset of patch centers from screen center

# Stimulus appearance (PIXELS & cycles/pixel)
# You can set different SF and SIZE for the two gratings:
grating_sfs_cyc_per_pix   = (0.006, 0.006)   # left-stim type vs right-stim type (cycles/pixel)
grating_sizes_pix         = (500, 500)       # diameter (pixels) for those two grating types
contrast                  = 1.0
phase                     = 0.0

# -----------------------
# Save name

clock = core.Clock()
run_id = time.strftime("%Y%m%d_%H%M%S")
save_dir = os.path.join(os.path.dirname(__file__), "logs")  # creates "logs" inside the same folder as this script
os.makedirs(save_dir, exist_ok=True)  # make the folder if it doesn’t exist

# Build full save path
log_path = os.path.join(save_dir, f"two_grating_passive_{run_id}.csv")


# -----------------------
# Setup
# -----------------------
random.seed(random_seed)
win = visual.Window(
    size=win_size_pix,
    fullscr=win_fullscreen,
    units="pix",
    color=screen_bg_color,
    allowGUI=False
)

# Build the two GratingStims once; we’ll update their ori/sf/size each trial
left_grat = visual.GratingStim(
    win=win, mask="circle", size=grating_sizes_pix[0], sf=grating_sfs_cyc_per_pix[0],
    ori=0.0, phase=phase, contrast=contrast, pos=(-eccentricity_pix, 0)
)
right_grat = visual.GratingStim(
    win=win, mask="circle", size=grating_sizes_pix[1], sf=grating_sfs_cyc_per_pix[1],
    ori=0.0, phase=phase, contrast=contrast, pos=(+eccentricity_pix, 0)
)

# Prebuild trials: randomize which grating type (A vs B) goes left/right
# “Type A” = (ori=grating_oris_deg[0], sf=grating_sfs_cyc_per_pix[0], size=grating_sizes_pix[0])
# “Type B” = (ori=grating_oris_deg[1], sf=grating_sfs_cyc_per_pix[1], size=grating_sizes_pix[1])
trials = []
for i in range(n_trials):
    if random.random() < 0.5:
        left_type, right_type = "A", "B"
    else:
        left_type, right_type = "B", "A"
    trials.append({"trial_index": i + 1, "left_type": left_type, "right_type": right_type})

# -----------------------
# Run
# -----------------------

with open(log_path, "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["trial_index",
                    "left_type","right_type",
                    "left_ori","right_ori",
                    "left_sf","right_sf",
                    "left_size","right_size",
                    "stim_on_s","stim_off_s"]
    )
    writer.writeheader()

    # Initial ITI
    win.flip()
    core.wait(iti_duration_s)

    for tr in trials:
        if "escape" in event.getKeys(): break

        # Assign parameters by type
        if tr["left_type"] == "A":
            left_ori  = grating_oris_deg[0]
            left_sf   = grating_sfs_cyc_per_pix[0]
            left_size = grating_sizes_pix[0]
        else:
            left_ori  = grating_oris_deg[1]
            left_sf   = grating_sfs_cyc_per_pix[1]
            left_size = grating_sizes_pix[1]

        if tr["right_type"] == "A":
            right_ori  = grating_oris_deg[0]
            right_sf   = grating_sfs_cyc_per_pix[0]
            right_size = grating_sizes_pix[0]
        else:
            right_ori  = grating_oris_deg[1]
            right_sf   = grating_sfs_cyc_per_pix[1]
            right_size = grating_sizes_pix[1]

        # Update BOTH stimuli every trial (ori, sf, size)
        left_grat.ori  = left_ori
        left_grat.sf   = left_sf
        left_grat.size = left_size

        right_grat.ori  = right_ori
        right_grat.sf   = right_sf
        right_grat.size = right_size

        # Draw & show
        left_grat.draw()
        right_grat.draw()
        stim_on = clock.getTime()
        win.flip()
        core.wait(stim_duration_s)

        # ITI (black)
        win.flip()
        stim_off = clock.getTime()
        core.wait(iti_duration_s)

        # Log
        tr.update({
            "left_ori": left_ori, "right_ori": right_ori,
            "left_sf": left_sf, "right_sf": right_sf,
            "left_size": left_size, "right_size": right_size,
            "stim_on_s": stim_on, "stim_off_s": stim_off
        })
        writer.writerow(tr)

win.close()
core.quit()
