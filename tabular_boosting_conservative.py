from __future__ import annotations

import subprocess
import sys
from pathlib import Path

DEFAULT_ARGS = [
    "--n-splits",
    "5",
    "--weights",
    "0.45,0.35,0.20",
    "--learning-rate",
    "0.03",
    "--n-estimators",
    "3000",
    "--early-stopping-rounds",
    "200",
    "--lgb-num-leaves",
    "31",
    "--lgb-min-child-samples",
    "60",
    "--lgb-subsample",
    "0.75",
    "--lgb-colsample",
    "0.75",
    "--xgb-max-depth",
    "5",
    "--xgb-subsample",
    "0.75",
    "--xgb-colsample",
    "0.70",
    "--xgb-min-child-weight",
    "3.0",
    "--cat-depth",
    "6",
    "--cat-l2-leaf-reg",
    "8.0",
    "--seed-list",
    "42",
    "--output",
    "submission_conservative.csv",
]


def main() -> None:
    base_script = Path(__file__).with_name("tabular_boosting_ensemble.py")
    cmd = [sys.executable, str(base_script), *DEFAULT_ARGS, *sys.argv[1:]]
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
