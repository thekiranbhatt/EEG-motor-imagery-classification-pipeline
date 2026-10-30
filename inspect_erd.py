
"""
Inspect motor-imagery ERD without generating plots.

Calculates percentage power change from baseline
for mu (8-13 Hz) and beta (13-30 Hz) bands.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np
import pandas as pd

from data_loading import load_subject
from preprocessing import make_epochs
from config import SUBJECTS


MOTOR_CHANNELS = {
    "FC3", "FC1", "FCZ", "FC2", "FC4",
    "C5", "C3", "C1", "CZ", "C2", "C4", "C6",
    "CP3", "CP1", "CPZ", "CP2", "CP4",
}

BANDS = {
    "mu": (8, 13),
    "beta": (13, 30),
}

BASELINE = (-1.0, -0.2)
TASK = (0.5, 3.5)



def calculate_erd(epochs, band_name, low, high):
    """Calculate channel-wise ERD percentage for each class.

    Input epochs have already been filtered into the relevant band.
    """

    results = []

    for condition in ("left", "right"):
        condition_epochs = epochs[condition]

        baseline_epochs = condition_epochs.copy().crop(
            tmin=BASELINE[0],
            tmax=BASELINE[1],
        )

        task_epochs = condition_epochs.copy().crop(
            tmin=TASK[0],
            tmax=TASK[1],
        )

        baseline_data = baseline_epochs.get_data()
        task_data = task_epochs.get_data()

        baseline_power = np.mean(baseline_data ** 2, axis=(0, 2))
        task_power = np.mean(task_data ** 2, axis=(0, 2))

        erd = (
            (task_power - baseline_power)
            / np.maximum(baseline_power, 1e-20)
        ) * 100

        for channel, value in zip(epochs.ch_names, erd):
            results.append({
                "band": band_name,
                "condition": condition,
                "channel": channel,
                "erd_percent": value,
            })

    return results


def run_subject(subject):
    print(f"\n{'=' * 55}")
    print(f"SUBJECT {subject}")
    print(f"{'=' * 55}")

    raw = load_subject(subject)
    raw.load_data()

    all_results = []

    for band_name, (low, high) in BANDS.items():
        # Filter the continuous recording before epoching.
        band_raw = raw.copy()
        band_raw.filter(
            l_freq=low,
            h_freq=high,
            picks="eeg",
            verbose="ERROR",
        )

        epochs = make_epochs(band_raw)

        keep = [
            ch for ch in epochs.ch_names
            if ch.upper() in MOTOR_CHANNELS
        ]

        if not keep:
            raise ValueError("No motor channels found.")

        epochs.pick(keep)

        all_results.extend(
            calculate_erd(epochs, band_name, low, high)
        )

    df = pd.DataFrame(all_results)

    for band in BANDS:
        print(f"\n--- {band.upper()} BAND ---")

        band_df = df[df["band"] == band]

        for condition in ("left", "right"):
            subset = band_df[
                band_df["condition"] == condition
            ].sort_values("erd_percent")

            print(f"\n{condition.upper()} IMAGERY")
            print(
                subset[["channel", "erd_percent"]]
                .to_string(
                    index=False,
                    formatters={
                        "erd_percent": lambda x: f"{x:8.2f}%"
                    },
                )
            )

    return df


def main():
    subjects = [int(s) for s in sys.argv[1:]] or SUBJECTS

    all_subjects = []

    for subject in subjects:
        try:
            df = run_subject(subject)
            df["subject"] = subject
            all_subjects.append(df)

        except Exception as e:
            print(f"Subject {subject} failed: {e}")

    if all_subjects:
        combined = pd.concat(all_subjects, ignore_index=True)

        output = Path("results/metrics/erd_summary.csv")
        output.parent.mkdir(parents=True, exist_ok=True)

        combined.to_csv(output, index=False)

        print(f"\nERD results saved to: {output.resolve()}")


if __name__ == "__main__":
    main()
