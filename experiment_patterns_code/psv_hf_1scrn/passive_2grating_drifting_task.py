# -------------------------------------------------------------------------
# Two drifting sinusoidal gratings on left/right, with the SAME save format
# as the static script. Each CSV row has stim_on_s and stim_off_s using a
# single global timebase. Orientation pairs are balanced and randomized.
# -------------------------------------------------------------------------

from psychopy import visual, core, event, monitors
import psychopy.logging as logging
import csv, random, time, os, math
import numpy as np

# Quiet console
logging.console.setLevel(logging.ERROR)

# =========================
# 1) Experiment parameters
# =========================
win_fullscreen     = True
screen_bg_color    = [0, 0, 0]
stim_duration_s    = 1.0
iti_duration_s     = 0.5
n_trials           = 600        # total across all orientation pairs
random_seed        = 42

# Orientation pairs (L,R). Each pair will be shown equally often with sides swapped.
orientation_pairs = [
    (45.0, 135.0),
    (0.0, 90.0),
]

# Grating parameters
grating_sfs_cpd    = (0.08, 0.08)
grating_sizes_deg  = (100.0, 100.0)
eccentricity_deg   = 70.0
contrast           = 1.0
start_phase        = 0.0
tf_hz              = 4.0   # drift speed (magnitude, in cycles/s)

# =========================
# 1.5) Save paths
# =========================
run_id   = time.strftime("%Y%m%d_%H%M%S")
save_dir = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(save_dir, exist_ok=True)
log_path = os.path.join(save_dir, f"two_grating_passive_drifting_{run_id}.csv")

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
PPD         = pixels_per_degree(view_dist_mm, px_per_mm)

grating_sfs_cyc_per_pix = tuple(sf_cpd / PPD for sf_cpd in grating_sfs_cpd)
grating_sizes_pix       = tuple(int(round(sz_deg * PPD)) for sz_deg in grating_sizes_deg)
eccentricity_pix        = int(round(eccentricity_deg * PPD))

print(f"[Info] pixels/degree (PPD): {PPD:.2f}")
print(f"[Info] SF cyc/pix: {grating_sfs_cyc_per_pix}")
print(f"[Info] Sizes (pix): {grating_sizes_pix}, Eccentricity (pix): {eccentricity_pix}")

# =========================
# 5) Monitor & window
# =========================
width_cm = screen2_width_mm / 10.0
dist_cm  = view_dist_mm / 10.0
mon = monitors.Monitor("Screen2")
mon.setWidth(width_cm)
mon.setDistance(dist_cm)
mon.setSizePix(screen2_res_px)

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

# Global clock for on/off timestamps (same timebase)
global_clock = core.Clock()
global_clock.reset()

# =========================
# 7) Sync patch (top-right)
# =========================
width, height = win.size
sync_width_px   = 100
sync_height_px  = 100
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
# 9) Balanced trial builder (same logic as static)
# =========================
def build_trials_from_pairs(n_trials, orientation_pairs):
    """
    Build balanced randomized trial list for any number of orientation pairs.
    Each pair is shown equally often, with sides swapped half the time.
    """
    n_pairs = len(orientation_pairs)
    if n_trials % (2 * n_pairs) != 0:
        raise ValueError("n_trials must be divisible by 2 × number of orientation pairs.")

    trials = []
    idx = 0
    per_pair_total = n_trials // n_pairs      # trials per pair
    per_side = per_pair_total // 2            # left/right swap per pair

    for pair in orientation_pairs:
        lo, ro = pair
        for _ in range(per_side):             # as given
            trials.append({"trial_index": idx, "left_ori": lo, "right_ori": ro, "pair_id": f"{lo}-{ro}"})
            idx += 1
        for _ in range(per_side):             # swapped
            trials.append({"trial_index": idx, "left_ori": ro, "right_ori": lo, "pair_id": f"{lo}-{ro}"})
            idx += 1

    random.shuffle(trials)
    for i, tr in enumerate(trials):
        tr["trial_index"] = i
    return trials

trials = build_trials_from_pairs(n_trials, orientation_pairs)
print(f"[Info] Generated {len(trials)} trials using {len(orientation_pairs)} orientation pairs.")

# =========================
# 9.5) Save per-run task metadata (same format + drift info)
# =========================
left_ori_seq  = [tr["left_ori"]  for tr in trials]
right_ori_seq = [tr["right_ori"] for tr in trials]
pair_id_seq   = [tr["pair_id"]   for tr in trials]
taskmeta_path = os.path.join(
    save_dir, f"{os.path.splitext(os.path.basename(log_path))[0]}_taskmeta.npz"
)
np.savez_compressed(
    taskmeta_path,
    run_id=run_id,
    task_name="two_grating_passive_drifting",
    csv_path=log_path,
    stimulus_duration_s=float(stim_duration_s),
    iti_duration_s=float(iti_duration_s),
    n_trials=int(n_trials),
    orientation_pairs=np.array(orientation_pairs, dtype=float),
    left_ori_seq=np.array(left_ori_seq, dtype=float),
    right_ori_seq=np.array(right_ori_seq, dtype=float),
    pair_id_seq=np.array(pair_id_seq, dtype=object),
    grating_sfs_cpd=np.array(grating_sfs_cpd, dtype=float),
    grating_sizes_deg=np.array(grating_sizes_deg, dtype=float),
    eccentricity_deg=float(eccentricity_deg),
    contrast=float(contrast),
    start_phase=float(start_phase),
    temporal_freq_hz=float(tf_hz),
    PPD=float(PPD),
    screen2_width_mm=float(screen2_width_mm),
    screen2_height_mm=float(screen2_height_mm),
    screen2_res_px=np.array(screen2_res_px, dtype=int),
    view_dist_mm=float(view_dist_mm),
)
print(f"[Info] Wrote task metadata: {taskmeta_path}")

# =========================
# 10) Drift direction helper
# =========================
def drift_sign_for_ori(ori_deg: float) -> int:
    """
    Decide drift direction based on orientation.
    This does NOT change the orientation itself; it only flips the sign
    of the temporal frequency along that orientation.
    """
    return 1 if math.cos(math.radians(ori_deg)) >= 0 else -1

# =========================
# 11) Run experiment (drifting, static-style logging)
# =========================
with open(log_path, "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "trial_index", "pair_id", "left_ori", "right_ori",
            "stim_on_s", "stim_off_s", "contrast", "spatial_freq"
        ]
    )
    writer.writeheader()

    # Initial ITI (no logging)
    sync_patch.fillColor = [-1, -1, -1]
    sync_patch.draw()
    win.flip()
    core.wait(iti_duration_s)

    abort = False

    for tr in trials:
        if "escape" in event.getKeys():
            abort = True
        if abort:
            break

        # Assign orientations (this is the actual grating orientation on screen)
        left_grat.ori  = tr["left_ori"]
        right_grat.ori = tr["right_ori"]

        # Temporal frequencies (signed) for drift direction
        left_tf  = tf_hz * drift_sign_for_ori(tr["left_ori"])
        right_tf = tf_hz * drift_sign_for_ori(tr["right_ori"])

        # Reset phase and local timer
        left_grat.phase  = start_phase
        right_grat.phase = start_phase
        stim_clock = core.Clock()

        # ---------------- Stimulus ON ----------------
        left_grat.draw()
        right_grat.draw()
        sync_patch.fillColor = [1, 1, 1]
        sync_patch.draw()
        win.flip()
        stim_on = global_clock.getTime()   # same timebase for ON/OFF

        # Drifting during stim_duration_s
        while stim_clock.getTime() < stim_duration_s:
            if "escape" in event.getKeys():
                abort = True
                break

            t = stim_clock.getTime()
            # phase evolves in cycles; ori stays fixed at the requested angle
            left_grat.phase  = (start_phase + left_tf  * t) % 1.0
            right_grat.phase = (start_phase + right_tf * t) % 1.0

            left_grat.draw()
            right_grat.draw()
            sync_patch.fillColor = [1, 1, 1]
            sync_patch.draw()
            win.flip()

        # ---------------- Stimulus OFF ----------------
        sync_patch.fillColor = [-1, -1, -1]
        sync_patch.draw()
        win.flip()
        stim_off = global_clock.getTime()

        if abort:
            break

        # ITI
        core.wait(iti_duration_s)

        # Log: same fields as static version
        writer.writerow({
            "trial_index": tr["trial_index"],
            "pair_id": tr["pair_id"],
            "left_ori": tr["left_ori"],
            "right_ori": tr["right_ori"],
            "stim_on_s": stim_on,
            "stim_off_s": stim_off,
            "contrast": contrast,
            "spatial_freq": grating_sfs_cpd[0]  # assuming both gratings share SF
        })

# Finish
win.close()
core.quit()
