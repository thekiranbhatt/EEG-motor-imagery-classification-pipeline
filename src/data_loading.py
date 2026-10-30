"""Load EEG recordings for one subject."""
import mne
from mne.datasets import eegbci
from mne.io import concatenate_raws, read_raw_edf

from config import DATA_DIR, RUNS


def load_subject(subject: int) -> mne.io.BaseRaw:
    """Download and join the selected recording runs."""
    paths = eegbci.load_data(subject, RUNS, path=str(DATA_DIR), update_path = False) 
    raws = [read_raw_edf(p, preload=True, verbose=False) for p in paths]
    raw = concatenate_raws(raws)

    # Standardize channel names.
    eegbci.standardize(raw)
    # Add sensor positions for scalp maps.
    raw.set_montage(mne.channels.make_standard_montage("standard_1005"))
    return raw
