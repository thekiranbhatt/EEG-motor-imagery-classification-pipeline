"""All settings live here so you can change them in one place."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "results" / "figures"
METRIC_DIR = ROOT / "results" / "metrics"

# Subjects to analyse (the dataset has 109, start small).
SUBJECTS = [1, 2, 3, 4, 5]

# Runs 4, 8, 12 = imagined left fist vs imagined right fist.
# (Runs 6, 10, 14 are fists vs feet, runs 3, 7, 11 are real movement.)
RUNS = [4, 8, 12]

# Band-pass filter. 7-30 Hz covers the mu (8-12 Hz) and beta (13-30 Hz)
# rhythms, which are the ones that change during motor imagery.
L_FREQ = 7.0
H_FREQ = 30.0

# Epoch window around the cue (seconds). The long window is used for plots.
TMIN, TMAX = -1.0, 4.0
# Shorter window after the cue, used for the classifier.
CLF_TMIN, CLF_TMAX = 0.5, 3.5

N_CSP_COMPONENTS = 6
CV_FOLDS = 5
RANDOM_STATE = 42
USE_MOTOR_ONLY = True

for d in (DATA_DIR, FIG_DIR, METRIC_DIR):
    d.mkdir(parents=True, exist_ok=True)
