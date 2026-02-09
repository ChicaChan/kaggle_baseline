# Repository Guidelines

## Project Structure & Module Organization
This repository is intentionally flat and script-driven:
- `tabular_boosting_ensemble.py`: core CLI pipeline (feature prep, CV training, blending, submission, metrics).
- `tabular_boosting_conservative.py` and `tabular_boosting_aggressive.py`: preset wrappers around the core script.
- `requirements.txt`: runtime dependencies.
- `README.md` and `README.zh-CN.md`: usage documentation.

Competition data is not versioned. Place `train.csv`, `test.csv`, and optionally `sample_submission.csv` in the working directory (or set `--data-dir`). Outputs are generated locally (for example `submission.csv` and `submission.csv.metrics.json`).

## Build, Test, and Development Commands
- `python -m pip install -r requirements.txt` - install dependencies.
- `python tabular_boosting_ensemble.py --data-dir . --target target --id-column id --output submission.csv` - run the base profile.
- `python tabular_boosting_conservative.py --data-dir . --target target --id-column id` - run fast/stable preset.
- `python tabular_boosting_aggressive.py --data-dir . --target target --id-column id` - run higher-compute preset.
- `python -m py_compile tabular_boosting_ensemble.py tabular_boosting_conservative.py tabular_boosting_aggressive.py` - quick syntax smoke check before PR.

## Coding Style & Naming Conventions
Use Python 3.10+ conventions with 4-space indentation and type hints where present. Follow existing naming patterns:
- `snake_case` for functions, variables, and CLI flags.
- `PascalCase` for classes/dataclasses.

Keep functions single-purpose (data preparation, evaluation, model training are separated today). Prefer extending existing inference helpers (`infer_target_column`, `infer_id_column`) over hardcoding competition-specific logic.

## Testing Guidelines
Use `pytest` for new tests and place them under `tests/`.
- File names: `test_<module>.py`
- Test names: `test_<behavior>`

Test public helpers, edge cases, and failures (for example missing target/id inference, seed parsing, and classification vs regression paths). Mock expensive training when possible. For changed code, aim for >=80% coverage:
- `python -m pytest --cov=. --cov-report=term-missing`

## Commit & Pull Request Guidelines
Current history uses imperative, sentence-case subjects (for example: `Add ...`, `Improve ...`). Keep subject lines concise (about 72 chars or less) and scope each commit to one concern.

PRs should include: purpose, key changes, exact run command, and a short metrics summary from `*.metrics.json`. Link the related issue/ticket when available and note dataset assumptions (`target`, `id`, CV strategy).

## Security & Configuration Tips
Never commit Kaggle credentials, raw competition data, or sensitive submissions. Keep local artifacts out of git, and avoid printing secrets in logs or error traces.
