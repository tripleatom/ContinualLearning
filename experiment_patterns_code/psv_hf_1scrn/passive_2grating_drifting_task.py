from psychopy import visual, core, event, monitors
import csv, random, time, os, math

# =========================
# 1) Experiment parameters
# =========================
win_fullscreen     = True
screen_bg_color    = [0, 0, 0]
stim_duration_s    = 1.0
iti_duration_s     = 0.5
n_trials           = 600
random_seed        = 42

# Orientation pairs (flexible)
orientation_pairs = [
    (45.0, 135.0),
    (0.0, 90.0)
]

# Spatial & temporal parameters
grating_sfs_cpd    = (0.08, 0.08)
grating_sizes_deg  = (100.0, 100.0)
eccentricity_deg   = 70.0
contrast           = 1.0
start_phase        = 0.0
tf_hz              = 4.0  # drift speed (same for all gratings)

# =========================
# 1.5) Save file configuration
# =========================
ori_label   = "_".join([f"{int(a)}-{int(b)}" for a,b in orientation_pairs])
save_prefix = f"CnL42_drifting_{ori_label}"
run_id      = time.strftime("%Y%m%d_%H%M%S")
save_dir    = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(save_dir, exist_ok=True)
log_path    = os.path.join(save_dir, f"{save_prefix}_{run_id}.csv")

print(f"[Info] Saving log file to:\n  {log_path}")

# =========================
# 2) Screen geometry (same as static)
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
PPD         = pixels_per_degree(view_dist_mm, px_per_mm)

grating_sfs_cyc_per_pix = tuple(sf_cpd / PPD for sf_cpd in grating_sfs_cpd)
grating_sizes_pix       = tuple(int(round(sz_deg * PPD)) for sz_deg in grating_sizes_deg)
eccentricity_pix        = int(round(eccentricity_deg * PPD))

# =========================
# 4) Monitor profile
# =========================
width_cm = screen2_width_mm / 10.0
dist_cm  = view_dist_mm / 10.0
mon = monitors.Monitor("Screen2")
mon.setWidth(width_cm)
mon.setDistance(dist_cm)
mon.setSizePix(screen2_res_px)

# =========================
# 5) Window
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

# One global clock for BOTH on/off times (same timebase)
global_clock = core.Clock()
global_clock.reset()

# =========================
# 6) Sync patch (same as static)
# =========================
width, height = win.size
sync_patch = visual.Rect(
    win,
    width=100, height=100,
    pos=((width/2)-(100/2)-10, (height/2)-(100/2)-10),
    fillColor=[-1,-1,-1],
    lineColor=None,
    units="pix",
)

# =========================
# 7) Gratings (same look as static)
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
# 8) Balanced trial builder
# =========================
def build_trials_balanced(n_trials, orientation_pairs):
    """Return randomized trials balanced across orientation pairs and sides."""
    if n_trials % (2 * len(orientation_pairs)) != 0:
        raise ValueError("n_trials must be divisible by 2 × number of orientation pairs.")

    per_pair_total = n_trials // len(orientation_pairs)
    per_side = per_pair_total // 2
    trials = []
    idx = 0

    for a, b in orientation_pairs:
        for _ in range(per_side):
            trials.append({"trial_index": idx, "left_ori": a, "right_ori": b, "pair_id": f"{a}-{b}"}); idx += 1
            trials.append({"trial_index": idx, "left_ori": b, "right_ori": a, "pair_id": f"{a}-{b}"}); idx += 1

    random.shuffle(trials)
    for i, tr in enumerate(trials):
        tr["trial_index"] = i
    return trials

trials = build_trials_balanced(n_trials, orientation_pairs)

# =========================
# 9) Drift direction helper
# =========================
def drift_sign_for_ori(ori_deg: float) -> int:
    """Make diagonals drift in opposite directions."""
    return 1 if math.cos(math.radians(ori_deg)) >= 0 else -1

# =========================
# 10) Run experiment
# =========================
with open(log_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "trial_index", "pair_id",
        "left_ori", "right_ori",
        "left_tf_hz", "right_tf_hz",
        "stim_on_s", "stim_off_s"
    ])
    writer.writeheader()

    # Initial ITI (no logging row)
    sync_patch.fillColor = [-1,-1,-1]
    sync_patch.draw()
    win.flip()
    core.wait(iti_duration_s)

    for tr in trials:
        if "escape" in event.getKeys():
            break

        left_ori, right_ori = tr["left_ori"], tr["right_ori"]
        left_grat.ori, right_grat.ori = left_ori, right_ori

        left_tf  = tf_hz * drift_sign_for_ori(left_ori)
        right_tf = tf_hz * drift_sign_for_ori(right_ori)

        stim_clock = core.Clock()
        left_grat.phase = start_phase
        right_grat.phase = start_phase

        # Stim ON
        left_grat.draw(); right_grat.draw()
        sync_patch.fillColor = [1,1,1]; sync_patch.draw()
        win.flip()
        stim_on = global_clock.getTime()   # <- SAME timebase as stim_off

        # Drifting period
        while stim_clock.getTime() < stim_duration_s:
            if "escape" in event.getKeys():
                break
            t = stim_clock.getTime()
            left_grat.phase  = (start_phase + left_tf  * t) % 1.0
            right_grat.phase = (start_phase + right_tf * t) % 1.0
            left_grat.draw(); right_grat.draw()
            sync_patch.fillColor = [1,1,1]; sync_patch.draw()
            win.flip()

        # Stim OFF
        sync_patch.fillColor = [-1,-1,-1]; sync_patch.draw()
        win.flip()
        stim_off = global_clock.getTime()  # <- SAME timebase as stim_on

        # ITI
        core.wait(iti_duration_s)

        writer.writerow({
            "trial_index": tr["trial_index"],
            "pair_id": tr["pair_id"],
            "left_ori": left_ori,
            "right_ori": right_ori,
            "left_tf_hz": left_tf,
            "right_tf_hz": right_tf,
            "stim_on_s": stim_on,
            "stim_off_s": stim_off
        })

win.close()
core.quit()
