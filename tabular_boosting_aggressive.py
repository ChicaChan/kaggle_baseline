from __future__ import annotations

import subprocess
import sys
from pathlib import Path

DEFAULT_ARGS = [
    "--n-splits",
    "10",
    "--weights",
    "0.35,0.35,0.30",
    "--learning-rate",
    "0.015",
    "--n-estimators",
    "8000",
    "--early-stopping-rounds",
    "400",
    "--lgb-num-leaves",
    "128",
    "--lgb-min-child-samples",
    "25",
    "--lgb-subsample",
    "0.90",
    "--lgb-colsample",
    "0.90",
    "--xgb-max-depth",
    "8",
    "--xgb-subsample",
    "0.90",
    "--xgb-colsample",
    "0.85",
    "--xgb-min-child-weight",
    "1.5",
    "--cat-depth",
    "8",
    "--cat-l2-leaf-reg",
    "4.0",
    "--seed-list",
    "42,2024,3407",
    "--output",
    "submission_aggressive.csv",
]


def main() -> None:
    base_script = Path(__file__).with_name("tabular_boosting_ensemble.py")
    cmd = [sys.executable, str(base_script), *DEFAULT_ARGS, *sys.argv[1:]]
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
