"""Filter recordings and split them into trials."""
import mne

from config import L_FREQ, H_FREQ, TMIN, TMAX


def filter_raw(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    """Keep signal changes between the configured frequency limits."""
    raw = raw.copy()
    raw.filter(L_FREQ, H_FREQ, fir_design="firwin",
               skip_by_annotation="edge", verbose=False)
    return raw


def make_epochs(raw: mne.io.BaseRaw) -> mne.Epochs:
    """Split the recording into left- and right-hand trials.

    T0 marks rest, T1 the left hand, and T2 the right hand.
    """
    events, _ = mne.events_from_annotations(
        raw, event_id=dict(T1=2, T2=3), verbose=False)
    picks = mne.pick_types(raw.info, eeg=True, exclude="bads")
    epochs = mne.Epochs(
        raw, events, event_id=dict(left=2, right=3),
        tmin=TMIN, tmax=TMAX, picks=picks,
        baseline=None, preload=True, verbose=False)
    return epochs
