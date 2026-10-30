"""Create and save analysis plots."""
import matplotlib
matplotlib.use("Agg")  # no pop-up windows, just save files
import matplotlib.pyplot as plt
import numpy as np


def plot_psd(raw, path, title):
    """Plot the recording's power spectrum."""
    fig = raw.compute_psd(fmax=60, picks="eeg", verbose=False).plot(
        show=False, average=True)
    fig.suptitle(title)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_erd(epochs, path):
    """Plot mu-band power at C3 and C4 for each hand.

    Values show percentage change from the period before the cue.
    """
    ep = epochs.copy().pick(["C3", "C4"]).filter(8, 12, verbose=False)
    ep.apply_hilbert(envelope=True)
    times = ep.times
    base = (times >= -1.0) & (times < 0.0)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, ch in zip(axes, ["C3", "C4"]):
        ci = ep.ch_names.index(ch)
        for label, color in [("left", "tab:blue"), ("right", "tab:red")]:
            power = ep[label].get_data(copy=True)[:, ci, :] ** 2
            mean_power = power.mean(axis=0)
            erd = (mean_power / mean_power[base].mean() - 1) * 100
            ax.plot(times, erd, color=color, label=f"{label} imagery")
        ax.axvline(0, color="k", ls="--", lw=0.8)
        ax.axhline(0, color="gray", lw=0.5)
        ax.set_title(ch)
        ax.set_xlabel("Time (s)")
    axes[0].set_ylabel("Mu power change vs baseline (%)")
    axes[0].legend()
    fig.suptitle("Event-related desynchronisation (mu band)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_csp_patterns(csp, info, path):
    """Plot the sensor patterns used by the classifier."""
    fig = csp.plot_patterns(info, components=list(range(csp.n_components)),
                            ch_type="eeg", units="Pattern (AU)",
                            size=1.5, show=False)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_confusion(cm, path, title):
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cm, cmap="Blues")
    labels = ["left", "right"]
    ax.set_xticks([0, 1], labels)
    ax.set_yticks([0, 1], labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_summary(df, path):
    """Plot accuracy for each subject."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(df["subject"].astype(str), df["accuracy"], color="tab:blue")
    ax.axhline(0.5, color="red", ls="--", label="chance (50%)")
    ax.axhline(df["accuracy"].mean(), color="k", ls=":",
               label=f"mean ({df['accuracy'].mean():.2f})")
    ax.set_ylim(0, 1)
    ax.set_xlabel("Subject")
    ax.set_ylabel("Cross-validated accuracy")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
