# Kaggle Tabular Leaderboard Baselines

This folder contains a reusable boosting ensemble baseline for Kaggle tabular competitions, plus two ready-to-run leaderboard profiles.

## Files

- `tabular_boosting_ensemble.py`: core trainer (LGBM + XGB + CAT + OOF blend)
- `tabular_boosting_conservative.py`: stable profile (faster, lower variance)
- `tabular_boosting_aggressive.py`: leaderboard profile (higher complexity + multi-seed)
- `requirements.txt`: python dependencies

## Install

```bash
pip install -r requirements.txt
```

## Common usage

Use your competition folder as `--data-dir` (must include `train.csv` and `test.csv`; `sample_submission.csv` recommended).

### 1) Base profile (balanced)

```bash
python tabular_boosting_ensemble.py --data-dir . --target target --id-column id --output submission.csv
```

### 2) Conservative profile (stable)

```bash
python tabular_boosting_conservative.py --data-dir . --target target --id-column id
```

Default output: `submission_conservative.csv`

### 3) Aggressive profile (stronger leaderboard push)

```bash
python tabular_boosting_aggressive.py --data-dir . --target target --id-column id
```

Default output: `submission_aggressive.csv`

## Override defaults

Wrapper scripts append your custom CLI args at the end, so your values override defaults.

Example:

```bash
python tabular_boosting_aggressive.py --data-dir . --target target --id-column id --n-splits 8 --output my_submit.csv
```

## Outputs

Each run writes:

- submission CSV (`--output`)
- metrics report JSON (`<output>.metrics.json`) with CV scores, fold metrics, seeds, and params
