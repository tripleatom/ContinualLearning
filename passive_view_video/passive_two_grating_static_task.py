from psychopy import visual, core, event, monitors
import psychopy.logging as logging
import csv, random, time, os, math

# Quiet down console chatter (optional)
logging.console.setLevel(logging.ERROR)

# =========================
# 1) Experiment parameters
# =========================
win_fullscreen     = True
screen_bg_color    = [0, 0, 0]        # black
stim_duration_s    = 1.0              # on-screen time per trial
iti_duration_s     = 0.5              # black screen between trials
grating_oris_deg   = (45.0, 135.0)      # two orientations to present (bar tilt)
n_trials           = 600
random_seed        = 42

# Give parameters in DEGREES (visual angle):
grating_sfs_cpd    = (0.08, 0.08)     # spatial frequency (LEFT type, RIGHT type) in cycles/degree
grating_sizes_deg  = (100.0, 100.0)   # diameter (deg) for the two grating types
eccentricity_deg   = 70.0             # horizontal eccentricity of patch centers (deg)

contrast           = 1.0
start_phase        = 0.0

# =========================
# 2) Screen 2 geometry (edit to your setup)
# =========================
# Physical size of the visible area (millimeters):
screen2_width_mm   = 520.0
screen2_height_mm  = 520.0
# Pixel resolution of the PsychoPy window (same as the physical display if fullscreen):
screen2_res_px     = (1440, 900)      # (width, height) in pixels
# Eye-to-screen distance (millimeters):
view_dist_mm       = 200.0            # e.g., 200 mm = 20 cm

# =========================
# 3) Deg↔Pix conversion
# =========================
def pixels_per_degree(view_dist_mm: float, px_per_mm: float) -> float:
    """Return pixels/degree given viewing distance (mm) and pixels/mm."""
    mm_per_deg = 2.0 * view_dist_mm * math.tan(math.radians(0.5))
    return px_per_mm * mm_per_deg

px_per_mm_x = screen2_res_px[0] / screen2_width_mm
px_per_mm_y = screen2_res_px[1] / screen2_height_mm
px_per_mm   = 0.5 * (px_per_mm_x + px_per_mm_y)  # average if not exactly square pixels

PPD = pixels_per_degree(view_dist_mm, px_per_mm)  # pixels/degree

# Convert deg-based parameters to PsychoPy's 'pix' units:
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
log_path = os.path.join(save_dir, f"CnL42_45_135_two_grating_passive_static_{run_id}.csv")

# =========================
# 5) Monitor profile (prevents PsychoPy from 'measuring' screen size)
# =========================
width_cm = screen2_width_mm / 10.0
dist_cm  = view_dist_mm / 10.0

mon = monitors.Monitor("Screen2")      # arbitrary name for this profile
mon.setWidth(width_cm)                 # width in cm
mon.setDistance(dist_cm)               # distance in cm
mon.setSizePix(screen2_res_px)         # pixel resolution

# =========================
# 6) Window
# =========================
random.seed(random_seed)
win = visual.Window(
    size=screen2_res_px,
    fullscr=win_fullscreen,
    units="pix",
    screen=1,               # 0=primary, 1=external (adjust if needed)
    winType="pyglet",       # try "glfw" if preferred/installed
    monitor=mon,            # <-- pass the defined monitor to avoid measurement
    color=screen_bg_color,
    allowGUI=False,
    multiSample=True,
    checkTiming=False,
    numSamples=8
)
win.recordFrameIntervals = False

# =========================
# 7) Rectangular sync patch (top-right)
# =========================
width, height = win.size
sync_width_px   = 100       # horizontal size
sync_height_px  = 100       # vertical size (shorter for rectangular look)
sync_margin_px  = 10       # margin from screen edges

sync_patch_x = (width / 2)  - (sync_width_px / 2)  - sync_margin_px
sync_patch_y = (height / 2) - (sync_height_px / 2) - sync_margin_px

sync_patch = visual.Rect(
    win,
    width=sync_width_px,
    height=sync_height_px,
    pos=(sync_patch_x, sync_patch_y),
    fillColor=[-1, -1, -1],  # OFF/black by default
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
    sf=grating_sfs_cyc_per_pix[0],    # cycles/pixel (derived from cyc/deg)
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
# 9) Trial list (A vs B)
# =========================
# --- 10) Run (only these edits are new) ---
with open(log_path, "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "trial_index",
            "left_type","right_type",
            "left_ori","right_ori",
            "left_sf_cpd","right_sf_cpd",
            "left_size_deg","right_size_deg",
            "left_sf_cyc_per_pix","right_sf_cyc_per_pix",
            "left_size_pix","right_size_pix",
            "stim_on_s","stim_off_s",
            # >>> added fields <<<
            "left_phase_deg","right_phase_deg",   # Stimdata-style 'phase'
            "t_trial_s",                          # Stimdata 't_trial'
            "iti_s",                              # convenience
            "ppd"                                 # pixels-per-degree used
        ]
    )
    writer.writeheader()

    # Initial ITI (sync OFF/black)
    sync_patch.fillColor = [-1, -1, -1]
    sync_patch.draw()
    win.flip()
    core.wait(iti_duration_s)

    for tr in trials:
        if "escape" in event.getKeys():
            break

        # ... (unchanged parameter assignment & drawing code above) ...

        # ---------------- Stimulus ON ----------------
        stim_clock = core.Clock()
        t_trial = stim_duration_s

        left_grat.draw(); right_grat.draw()
        sync_patch.fillColor = [1, 1, 1]
        sync_patch.draw()
        stim_on = win.flip()                 # flip time (sec)

        while stim_clock.getTime() < t_trial:
            if "escape" in event.getKeys():
                break
            left_grat.draw(); right_grat.draw()
            sync_patch.fillColor = [1, 1, 1]
            sync_patch.draw()
            win.flip()

        # ---------------- Stimulus OFF ----------------
        sync_patch.fillColor = [-1, -1, -1]
        sync_patch.draw()
        win.flip()
        stim_off = clock.getTime()           # same timebase as stim_on

        # ITI
        core.wait(iti_duration_s)

        # Log trial (only the bottom part changed)
        tr.update({
            "left_ori": left_ori, "right_ori": right_ori,
            "left_sf_cpd": left_sf_cpd, "right_sf_cpd": right_sf_cpd,
            "left_size_deg": left_size_deg, "right_size_deg": right_size_deg,
            "left_sf_cyc_per_pix": left_sf_cpp, "right_sf_cyc_per_pix": right_sf_cpp,
            "left_size_pix": left_size_pix, "right_size_pix": right_size_pix,
            "stim_on_s": stim_on, "stim_off_s": stim_off,

            # >>> added fields <<<
            "left_phase_deg":  float(start_phase),   # static gratings -> same phase
            "right_phase_deg": float(start_phase),
            "t_trial_s":       float(t_trial),       # equals stim_duration_s
            "iti_s":           float(iti_duration_s),
            "ppd":             float(PPD),
        })
        writer.writerow(tr)


