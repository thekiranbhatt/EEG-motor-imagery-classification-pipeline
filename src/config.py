"""Project settings."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "results" / "figures"
METRIC_DIR = ROOT / "results" / "metrics"

# Subjects to analyse.
SUBJECTS = [1, 2, 3, 4, 5]

# Runs 4, 8, and 12 contain imagined left- and right-hand movements.
RUNS = [4, 8, 12]

# Signal range used for the analysis, in hertz.
L_FREQ = 7.0
H_FREQ = 30.0

# Trial window around the cue, in seconds.
TMIN, TMAX = -1.0, 4.0
# Part of each trial used to train and evaluate the model.
CLF_TMIN, CLF_TMAX = 0.5, 3.5

N_CSP_COMPONENTS = 6
CV_FOLDS = 5
RANDOM_STATE = 42
USE_MOTOR_ONLY = True

for d in (DATA_DIR, FIG_DIR, METRIC_DIR):
    d.mkdir(parents=True, exist_ok=True)
