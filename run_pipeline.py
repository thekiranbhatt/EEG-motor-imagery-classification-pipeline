"""Run the analysis for default subjects or the numbers provided.

Examples:
    python run_pipeline.py
    python run_pipeline.py 1 7 12
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import pandas as pd

from config import FIG_DIR, METRIC_DIR, SUBJECTS, USE_MOTOR_ONLY
from data_loading import load_subject
from model import evaluate, fit_final, get_data
from preprocessing import filter_raw, make_epochs
from visualize import (
    plot_confusion,
    plot_csp_patterns,
    plot_erd,
    plot_psd,
    plot_summary,
)


def run_subject(subject: int) -> dict:
    print(f"\n=== Subject {subject} ===")
    tag = f"S{subject:03d}"

    raw = load_subject(subject)
    plot_psd(raw, FIG_DIR / f"{tag}_psd_raw.png", f"{tag} raw")

    raw = filter_raw(raw)
    plot_psd(
        raw,
        FIG_DIR / f"{tag}_psd_filtered.png",
        f"{tag} filtered 7-30 Hz",
    )

    epochs = make_epochs(raw)

    if USE_MOTOR_ONLY:
        motor = {
            "FC3", "FC1", "FCZ", "FC2", "FC4",
            "C5", "C3", "C1", "CZ", "C2", "C4", "C6",
            "CP3", "CP1", "CPZ", "CP2", "CP4",
        }

        keep = [
            ch for ch in epochs.ch_names
            if ch.upper() in motor
        ]

        if not keep:
            raise ValueError("No motor-related channels found.")

        print(f"Using {len(keep)} motor channels")
        epochs.pick(keep)

    else:
        print(f"Using all {len(epochs.ch_names)} channels")

    print(
        f"{len(epochs)} trials "
        f"({len(epochs['left'])} left, {len(epochs['right'])} right)"
    )

    plot_erd(epochs, FIG_DIR / f"{tag}_erd.png")

    X, y = get_data(epochs)
    res = evaluate(X, y)

    print(res["report"])
    print(f"Accuracy={res['accuracy']:.3f}  AUC={res['auc']:.3f}")

    final = fit_final(X, y)
    train_acc = final.score(X, y)

    print(
        f"Train accuracy={train_acc:.3f} "
        f"(CV accuracy={res['accuracy']:.3f})"
    )

    plot_confusion(
        res["confusion_matrix"],
        FIG_DIR / f"{tag}_confusion.png",
        f"{tag} (acc {res['accuracy']:.2f})",
    )

    plot_csp_patterns(
        final.named_steps["csp"],
        epochs.info,
        FIG_DIR / f"{tag}_csp_patterns.png",
    )

    return {
        "subject": subject,
        "n_trials": len(y),
        "train_accuracy": train_acc,
        "accuracy": res["accuracy"],
        "auc": res["auc"],
    }


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    METRIC_DIR.mkdir(parents=True, exist_ok=True)

    subjects = [int(s) for s in sys.argv[1:]] or SUBJECTS

    rows = []

    for s in subjects:
        try:
            rows.append(run_subject(s))
        except Exception as e:
            print(f"Subject {s} failed: {e}")

    if not rows:
        print("No subject finished.")
        return

    df = pd.DataFrame(rows)
    df.to_csv(METRIC_DIR / "summary.csv", index=False)

    plot_summary(df, FIG_DIR / "summary_accuracy.png")

    print("\n", df.to_string(index=False))
    print(f"\nMean accuracy: {df['accuracy'].mean():.3f}")
    print(f"Figures: {FIG_DIR}")
    print(f"Metrics: {METRIC_DIR / 'summary.csv'}")


if __name__ == "__main__":
    main()
