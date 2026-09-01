---
title: Convert raw accelerometry to NPY
subtitle: Turn one raw wrist recording into quality-controlled daily arrays ready for Sensori.
order: 1
level: Beginner
duration: 5 min
tutorial_previous:
  url: /tutorials/
  title: Browse all tutorials
tutorial_next:
  url: /tutorials/minute-level-har/
  title: Recognise activities minute by minute
steps:
  - id: what-you-need
    label: What you need
  - id: convert-one-recording
    label: Convert a recording
  - id: what-the-script-does
    label: Understand the pipeline
  - id: check-the-result
    label: Check the result
---

## What you need

Sensori requires Python 3.13. Clone the repository, create an isolated
environment and install the package:

```bash
git clone https://github.com/OxWearables/Sensori.git
cd Sensori

conda create -n sensori python=3.13 pip
conda activate sensori
pip install -e .
```

The script accepts `.cwa`, `.gt3x`, `.bin` and `.csv` files, with optional
`.gz` compression. A CSV must contain `time,x,y,z`: timestamps must be unique
and increasing, and XYZ acceleration must be finite and expressed in g.

- Note: CWA, GT3X and BIN files need Java 8 or newer, while CSV files do not need Java.

## Convert one recording

From the repository root, run the command that matches your input. For a device
file:

```bash
python scripts/get_npy.py \
  --file raw/participant_001.cwa.gz \
  --output processed
```

The CSV sampling rate is inferred from its timestamps and must be greater than
10 Hz. If you know the nominal rate, you can supply it explicitly with, for
example, `--input-sample-rate 100`.

The input filename becomes the output-directory name by default:

```text
processed/
└── participant_001/
    ├── day_0.npy
    ├── day_1.npy
    ├── info.json
    └── wear_duration.csv
```

If you have no recording to hand, `tutorials/nhanes_preprocessing.ipynb` runs
this same parser on public data: it downloads NHANES participant archives from
the CDC, combines their hourly files into one CSV and writes the same
`day_*.npy` layout, one call per participant.

## What the script does

The preprocessing path is fixed for compatibility with the released model:

```text
gravity calibration → 5 Hz low-pass filter → 10 Hz resampling
→ non-wear detection → calendar-day segmentation → quality control → NPY
```

A day is retained only when it contains a complete, finite 24-hour signal and
at least 22 hours of wear. The recording must also pass reading, calibration
and filtering checks, contain fewer than 10 interruptions, have mean ENMO no
greater than 200 mg, and contain at least one eligible day. Non-wear is used
for quality control; the retained signal is not replaced with zeros or missing
values.

Eligible days are written chronologically as `float32` XYZ acceleration in g.
Every array has shape `(2880, 300, 3)`: 2,880 consecutive 30-second windows,
300 samples per window at 10 Hz, and three axes.

## Check the result

Check one output before starting inference:

```python
import numpy as np

day = np.load("processed/participant_001/day_0.npy", allow_pickle=False)

assert day.shape == (2880, 300, 3)
assert day.dtype == np.float32
assert np.isfinite(day).all()
```

Things to watch out for:

- The script processes one recording per command. Run it again for each input.
- Partial calendar days are skipped. `day_0` is not a date; use
  `wear_duration.csv` to map output files to dates and inspect exclusion reasons.
- Exit code `3` means preprocessing completed but strict quality control found
  no eligible day. Inspect `info.json` for the failed criteria.
- A matching completed run is skipped. Use `--overwrite` only after inspecting
  the target; it replaces the parser-generated metadata and `day_*.npy` files.
- Do not change the scientific constants when preparing data for the released
  model.
