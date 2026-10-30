"""Step 1: download and load the raw EEG files for one subject."""
import mne
from mne.datasets import eegbci
from mne.io import concatenate_raws, read_raw_edf

from config import DATA_DIR, RUNS


def load_subject(subject: int) -> mne.io.BaseRaw:
    """Download (first time only) and join the motor imagery runs of a subject."""
    paths = eegbci.load_data(subject, RUNS, path=str(DATA_DIR), update_path = False) 
    raws = [read_raw_edf(p, preload=True, verbose=False) for p in paths]
    raw = concatenate_raws(raws)

    # Channel names in the files look like "C3.." , this cleans them to "C3".
    eegbci.standardize(raw)
    # Give each channel a 3D position so we can draw scalp maps later.
    raw.set_montage(mne.channels.make_standard_montage("standard_1005"))
    return raw
