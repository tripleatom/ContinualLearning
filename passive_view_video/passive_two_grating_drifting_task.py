from psychopy import visual, core, event, monitors
import csv, random, time, os, math

# =========================
# 1) Experiment parameters
# =========================
win_fullscreen     = True
screen_bg_color    = [0, 0, 0]        # black
stim_duration_s    = 1.0              # on-screen time per trial
iti_duration_s     = 0.5              # black screen between trials
grating_oris_deg   = (45.0, 135.0)    # orientations to present
n_trials           = 50
random_seed        = 42

# Spatial and temporal parameters
grating_sfs_cpd    = (0.08, 0.08)     # spatial frequency in cycles/degree
grating_sizes_deg  = (100.0, 100.0)   # diameter (deg)
eccentricity_deg   = 70.0             # horizontal eccentricity
contrast           = 1.0
start_phase        = 0.0

# 🔧 Temporal frequency (drift speed) in Hz per orientation
tf_hz_map = {
    45.0: 2.0,      # drift speed for 45° gratings
    135.0: 1.0      # drift speed for 135° gratings
}

# =========================
# 2) Screen geometry
# =========================
screen2_width_mm   = 520.0
screen2_height_mm  = 520.0
screen2_res_px     = (1440, 900)
view_dist_mm       = 200.0

# =========================
# 3) Deg↔Pix conversion
# =========================
def pixels_per_degree(view_dist_mm: float, px_per_mm: float) -> float:
    mm_per_deg = 2.0 * view_dist_mm * math.tan(math.radians(0.5))
    return px_per_mm * mm_per_deg

px_per_mm_x = screen2_res_px[0] / screen2_width_mm
px_per_mm_y = screen2_res_px[1] / screen2_height_mm
px_per_mm   = 0.5 * (px_per_mm_x + px_per_mm_y)
PPD = pixels_per_degree(view_dist_mm, px_per_mm)

grating_sfs_cyc_per_pix = tuple(sf_cpd / PPD for sf_cpd in grating_sfs_cpd)
grating_sizes_pix       = tuple(int(round(sz_deg * PPD)) for sz_deg in grating_sizes_deg)
eccentricity_pix        = int(round(eccentricity_deg * PPD))

print(f"[Info] pixels/degree (PPD): {PPD:.2f}")
print(f"[Info] SF cyc/pix: {grating_sfs_cyc_per_pix}")
print(f"[Info] Sizes (pix): {grating_sizes_pix}, Eccentricity (pix): {eccentricity_pix}")

# =========================
# 4) Save path
# =========================
clock = core.Clock()
run_id = time.strftime("%Y%m%d_%H%M%S")
save_dir = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(save_dir, exist_ok=True)
log_path = os.path.join(save_dir, f"two_grating_passive_drifting_{run_id}.csv")

# =========================
# 5) Monitor profile
# =========================
width_cm = screen2_width_mm / 10.0
dist_cm  = view_dist_mm / 10.0
mon = monitors.Monitor("Screen2")
mon.setWidth(width_cm)
mon.setDistance(dist_cm)
mon.setSizePix(screen2_res_px)

# =========================
# 6) Window
# =========================
random.seed(random_seed)
win = visual.Window(
    size=screen2_res_px,
    fullscr=win_fullscreen,
    units="pix",
    screen=1,
    winType="pyglet",
    monitor=mon,
    color=screen_bg_color,
    allowGUI=False,
    multiSample=True,
    checkTiming=False,
    numSamples=8
)
win.recordFrameIntervals = False

# =========================
# 7) Sync patch
# =========================
width, height = win.size
sync_width_px   = 70
sync_height_px  = 60
sync_margin_px  = 10
sync_patch_x = (width / 2)  - (sync_width_px / 2)  - sync_margin_px
sync_patch_y = (height / 2) - (sync_height_px / 2) - sync_margin_px

sync_patch = visual.Rect(
    win,
    width=sync_width_px,
    height=sync_height_px,
    pos=(sync_patch_x, sync_patch_y),
    fillColor=[-1, -1, -1],
    lineColor=None,
    units='pix',
    autoLog=False
)

# =========================
# 8) Gratings
# =========================
left_grat = visual.GratingStim(
    win=win, mask="circle",
    size=grating_sizes_pix[0],
    sf=grating_sfs_cyc_per_pix[0],
    ori=0.0, phase=start_phase, contrast=contrast,
    pos=(-eccentricity_pix, 0),
    interpolate=True, texRes=512
)
right_grat = visual.GratingStim(
    win=win, mask="circle",
    size=grating_sizes_pix[1],
    sf=grating_sfs_cyc_per_pix[1],
    ori=0.0, phase=start_phase, contrast=contrast,
    pos=(+eccentricity_pix, 0),
    interpolate=True, texRes=512
)

# =========================
# 9) Trial list
# =========================
trials = []
for i in range(n_trials):
    if random.random() < 0.5:
        trials.append({"trial_index": i + 1, "left_type": "A", "right_type": "B"})
    else:
        trials.append({"trial_index": i + 1, "left_type": "B", "right_type": "A"})

# =========================
# 10) Helper for drift direction
# =========================
def drift_sign_for_ori(ori_deg: float) -> int:
    # Flip sign so opposite diagonals (45° vs 135°) drift oppositely
    return 1 if math.cos(math.radians(ori_deg)) >= 0 else -1

# =========================
# 11) Run
# =========================
with open(log_path, "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "trial_index",
            "left_type","right_type",
            "left_ori","right_ori",
            "left_tf_hz","right_tf_hz",
            "stim_on_s","stim_off_s"
        ]
    )
    writer.writeheader()

    sync_patch.fillColor = [-1, -1, -1]
    sync_patch.draw()
    win.flip()
    core.wait(iti_duration_s)

    for tr in trials:
        if "escape" in event.getKeys():
            break

        # Assign parameters
        if tr["left_type"] == "A":
            left_ori = grating_oris_deg[0]
        else:
            left_ori = grating_oris_deg[1]

        if tr["right_type"] == "A":
            right_ori = grating_oris_deg[0]
        else:
            right_ori = grating_oris_deg[1]

        # Set TFs based on orientation and drift direction
        left_tf  = tf_hz_map[left_ori]  * drift_sign_for_ori(left_ori)
        right_tf = tf_hz_map[right_ori] * drift_sign_for_ori(right_ori)

        # Update gratings
        left_grat.ori = left_ori
        right_grat.ori = right_ori

        # ---------------- Stimulus ON (drifting) ----------------
        stim_clock = core.Clock()
        left_grat.phase = start_phase
        right_grat.phase = start_phase

        left_grat.draw(); right_grat.draw()
        sync_patch.fillColor = [1, 1, 1]
        sync_patch.draw()
        stim_on = win.flip()

        while stim_clock.getTime() < stim_duration_s:
            if "escape" in event.getKeys():
                break
            t = stim_clock.getTime()
            left_grat.phase  = (start_phase + left_tf  * t) % 1.0
            right_grat.phase = (start_phase + right_tf * t) % 1.0
            left_grat.draw(); right_grat.draw()
            sync_patch.fillColor = [1, 1, 1]
            sync_patch.draw()
            win.flip()

        # ---------------- Stimulus OFF ----------------
        sync_patch.fillColor = [-1, -1, -1]
        sync_patch.draw()
        win.flip()
        stim_off = clock.getTime()

        core.wait(iti_duration_s)

        tr.update({
            "left_type": tr["left_type"], "right_type": tr["right_type"],
            "left_ori": left_ori, "right_ori": right_ori,
            "left_tf_hz": left_tf, "right_tf_hz": right_tf,
            "stim_on_s": stim_on, "stim_off_s": stim_off
        })
        writer.writerow(tr)

win.close()
core.quit()
