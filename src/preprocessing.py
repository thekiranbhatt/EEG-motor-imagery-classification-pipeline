"""Step 2: clean the signal and cut it into labelled trials (epochs)."""
import mne

from config import L_FREQ, H_FREQ, TMIN, TMAX


def filter_raw(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    """Band-pass filter to keep only the mu and beta rhythms."""
    raw = raw.copy()
    raw.filter(L_FREQ, H_FREQ, fir_design="firwin",
               skip_by_annotation="edge", verbose=False)
    return raw


def make_epochs(raw: mne.io.BaseRaw) -> mne.Epochs:
    """Cut the recording into trials.

    In these runs the annotations are:
      T0 = rest, T1 = left fist, T2 = right fist.
    We keep only T1 and T2.
    """
    events, _ = mne.events_from_annotations(
        raw, event_id=dict(T1=2, T2=3), verbose=False)
    picks = mne.pick_types(raw.info, eeg=True, exclude="bads")
    epochs = mne.Epochs(
        raw, events, event_id=dict(left=2, right=3),
        tmin=TMIN, tmax=TMAX, picks=picks,
        baseline=None, preload=True, verbose=False)
    return epochs
