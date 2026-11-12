# Recognizing Imagined Hand Movements from Brain Signals

This project uses electroencephalography (EEG) recordings to explore whether
brain activity can help distinguish between imagining a left-hand movement and
imagining a right-hand movement. EEG records small changes in electrical
activity at the surface of the head. The recordings come from the PhysioNet
EEG Motor Movement/Imagery dataset.

The program downloads the recordings it needs, prepares the signals, trains a
model to distinguish the two types of imagined movement, and saves charts and
a results summary. This is an experiment using recorded data; its results do
not establish how well the method would work for every person or in everyday
use.

## Project files

```text
bci-motor-imagery/
├── README.md
├── requirements.txt       Packages used by the project
├── run_pipeline.py        Runs the analysis
├── inspect_erd.py         Prints mu- and beta-band power changes
├── src/
│   ├── config.py          Settings such as subjects, sessions, and time ranges
│   ├── data_loading.py    Downloads and opens the recordings
│   ├── preprocessing.py   Filters the signals and divides them into trials
│   ├── model.py           Trains and evaluates the model
│   └── visualize.py       Creates the charts
├── data/                  Downloaded recordings
└── results/
    ├── figures/           Saved charts
    └── metrics/           Results summary
```

## Requirements and setup

Use Python 3.10 or newer. From the project folder, install the required
packages and run the analysis:

```bash
pip install -r requirements.txt
python run_pipeline.py
```

By default, the program processes subjects 1 through 5, as listed in
`src/config.py`. To process specific subjects, provide their numbers when
starting the program:

```bash
python run_pipeline.py 1 7 12
```

The first run downloads the required recording files from PhysioNet. The files
are saved in the `data` folder and can be reused on later runs.

## The dataset

Each subject completed several recording sessions. This project uses sessions
4, 8, and 12, in which the subject saw a cue on the left or right side of a
screen and imagined opening and closing the corresponding fist. The recordings
also include rest periods; the analysis keeps the left- and right-hand trials
and leaves the rest periods out.

There are a limited number of trials for each subject, so individual results
can vary. The model's score should be interpreted as an estimate from these
recordings, not as a guarantee of performance in a new setting.

## Analysis steps

### 1. Load the recordings

`src/data_loading.py` downloads and opens the three selected sessions for each
subject, then joins them into one recording. It standardizes the sensor names
and adds their positions on a standard head map so the charts can show where
the signals were recorded.

### 2. Filter the signal

`src/preprocessing.py` keeps signal changes between 7 and 30 hertz. A hertz is
one cycle per second. This range includes the mu and beta rhythms, patterns of
brain activity that can change during imagined movement. Filtering reduces
slower changes and faster activity that are outside the range this analysis
uses.

### 3. Divide the recording into trials

The program cuts the continuous recording into short sections around each
left- or right-hand cue. Each section starts one second before the cue and ends
four seconds after it. The program uses the labels in the recording to identify
the left- and right-hand trials.

### 4. Choose sensors and inspect signal changes

By default, the analysis uses 17 sensors near the areas of the head involved
in hand movement. This can be changed in `src/config.py`. The program also
saves power spectrum charts before and after filtering. These charts show how
much signal activity appears at different speeds and help check the effect of
filtering.

The event-related desynchronisation (ERD) chart shows how mu-band activity at
sensors C3 and C4 changes compared with the period before the cue. In general,
imagining a right-hand movement may reduce activity at C3, and imagining a
left-hand movement may reduce activity at C4. The chart helps show whether
these patterns appear in the recordings; it does not by itself prove that the
model can identify the imagined movement.

### 5. Train the model

The model uses two steps. Common Spatial Patterns (CSP) combines signals from
different sensors to find patterns that differ between left- and right-hand
trials. The program uses six such patterns. Linear Discriminant Analysis
(LDA) then uses those patterns to choose between the two labels.

For training, the model uses the part of each trial from 0.5 to 3.5 seconds
after the cue. This leaves out the period just before the cue and the earliest
response to it.

### 6. Check the model

The program uses five-fold cross-validation: it divides the trials into five
groups, trains on four groups, and checks its predictions on the remaining
group. It repeats this so every group is used for checking. The CSP step is
re-learned each time using only the training groups, which prevents information
from the checked trials from influencing training.

The results include:

- **Accuracy:** the share of checked trials labelled correctly.
- **ROC AUC:** a score describing how well the model separates the two labels
  across different decision thresholds. A score near 0.5 is similar to
  chance-level separation; a score closer to 1 indicates better separation.
- **Confusion matrix:** counts of correct and incorrect left- and right-hand
  predictions.
- **Precision and recall:** additional measures of how reliable and complete
  the predictions are for each hand.

The program also fits a final model using all available trials for each
subject. This model is used to create the sensor-pattern chart; its training
accuracy is not the cross-validated score and should not be used as an estimate
of performance on unseen trials.

### 7. Review the charts

The sensor-pattern charts show which areas contributed to the patterns learned
by the model. The summary chart compares each subject's cross-validated
accuracy with a 50% reference line. This reference assumes the two labels are
equally represented; the results file also reports the number of trials for
each subject, which can help when interpreting the comparison.

## Results

After the program finishes, the output is saved in the `results` folder:

- `results/metrics/summary.csv` contains one row per subject, with the number
  of trials, training accuracy, cross-validated accuracy, and ROC AUC.
- `results/figures/` contains the signal, ERD, confusion-matrix, sensor-pattern,
  and accuracy-summary charts.
- `results/metrics/summary_all_channels.csv` has the same table for the
  all-channels comparison described below.
- `results/metrics/erd_summary.csv` has the mu- and beta-band ERD values
  from `inspect_erd.py`.

On subjects 1-5 (45 trials each), five-fold cross-validated accuracy with the
17 motor-area channels was 80.0%, 84.4%, 44.4%, 57.8% and 51.1% (mean 63.6%;
mean ROC AUC 0.69). Two subjects are clearly above chance and three are not.
With 45 trials per subject, the 95% chance range is roughly 35%-65%.

### Channel selection

Using all available channels gave a mean cross-validated accuracy of 60.9%
(mean ROC AUC 0.67), against 63.6% with the 17 motor channels. The difference
is much smaller than the subject-level uncertainty (about +/-7.5 points with
45 trials each), so motor channels were chosen on neuroscientific grounds, not
because they scored higher. All-channel models fit the training data almost
perfectly (91-100% training accuracy), a sign of overfitting. Settings were
explored on these same subjects, so scores may be slightly optimistic.

## Possible next steps

- Add sessions 6, 10, and 14 to compare imagined hand movements with imagined
  foot movements.
- Compare other model choices, such as Support Vector Machines or logistic
  regression, with the current LDA model.
- Try a narrower signal range or compare several ranges.
- Exclude unusually noisy trials before training and compare the results.

The subject list, recording sessions, signal range, trial timing, and other
settings are in `src/config.py`.