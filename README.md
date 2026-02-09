# Kaggle Tabular Leaderboard Baselines

[简体中文说明](README.zh-CN.md)

A production-ready starter for Kaggle tabular competitions using a 3-model boosting ensemble:

- LightGBM
- XGBoost
- CatBoost

The trainer supports out-of-fold (OOF) validation, weighted blending, multi-seed averaging, and auto submission generation.

## What is included

- `tabular_boosting_ensemble.py` - core trainer with full CLI options
- `tabular_boosting_conservative.py` - stable/faster preset
- `tabular_boosting_aggressive.py` - stronger leaderboard preset
- `requirements.txt` - dependencies

## Quick start

### 1) Install dependencies

```bash
pip install -r requirements.txt
```

### 2) Prepare data

Put competition files under one folder (usually project root):

- `train.csv`
- `test.csv`
- `sample_submission.csv` (recommended)

### 3) Run one profile

Balanced (base script):

```bash
python tabular_boosting_ensemble.py --data-dir . --target target --id-column id --output submission.csv
```

Conservative preset:

```bash
python tabular_boosting_conservative.py --data-dir . --target target --id-column id
```

Aggressive preset:

```bash
python tabular_boosting_aggressive.py --data-dir . --target target --id-column id
```

## Preset comparison

| Preset | Goal | CV | Seeds | Complexity |
|---|---|---:|---:|---|
| Conservative | Stable score and faster iteration | 5-fold | 1 | Lower |
| Aggressive | Better LB potential | 10-fold | 3 | Higher |
| Base | Customizable middle ground | 5-fold (default) | 1 (default) | Medium |

## Outputs

Each run writes:

- submission file (`--output`)
- metrics report (`<output>.metrics.json`) containing:
  - fold scores per model
  - overall OOF score per model + ensemble
  - used seeds and main hyperparameters

## Useful CLI options (core script)

```bash
python tabular_boosting_ensemble.py \
  --data-dir . \
  --target target \
  --id-column id \
  --n-splits 5 \
  --seed 42 \
  --seed-list 42,2024,3407 \
  --weights 0.4,0.35,0.25 \
  --learning-rate 0.02 \
  --n-estimators 5000 \
  --early-stopping-rounds 250 \
  --output submission.csv
```

Notes:

- Wrapper scripts (`conservative`/`aggressive`) append your extra args at runtime, so your custom values override preset defaults.
- If `--target` or `--id-column` is omitted, the script tries to infer them.

## Typical tuning path

1. Start with conservative profile and validate CV stability.
2. Switch to aggressive profile for leaderboard pushes.
3. Tune blend weights and tree depth based on OOF and LB gap.
4. Add feature engineering and custom CV strategy (group/time split if needed).

## Recommended environment

- Python 3.10+
- 16GB+ RAM for medium/large tabular datasets
- CPU-only is supported; GPU is optional
