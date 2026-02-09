from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor, Pool
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.metrics import log_loss, mean_squared_error, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier, XGBRegressor

warnings.filterwarnings("ignore")


@dataclass
class PreparedData:
    train_oh: pd.DataFrame
    test_oh: pd.DataFrame
    train_cat: pd.DataFrame
    test_cat: pd.DataFrame
    cat_feature_indices: list[int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Kaggle tabular boosting ensemble baseline")
    parser.add_argument("--data-dir", type=Path, default=Path("."), help="Directory with train/test/sample_submission")
    parser.add_argument("--train-file", default="train.csv")
    parser.add_argument("--test-file", default="test.csv")
    parser.add_argument("--sample-file", default="sample_submission.csv")
    parser.add_argument("--target", default=None, help="Target column name")
    parser.add_argument("--id-column", default=None, help="ID column name")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seed-list", default="", help="Optional extra seeds, e.g. 42,2024,3407")
    parser.add_argument("--output", type=Path, default=Path("submission.csv"))
    parser.add_argument("--weights", default="0.4,0.35,0.25", help="lgb,xgb,cat weights")

    parser.add_argument("--learning-rate", type=float, default=0.02)
    parser.add_argument("--n-estimators", type=int, default=5000)
    parser.add_argument("--early-stopping-rounds", type=int, default=250)

    parser.add_argument("--lgb-num-leaves", type=int, default=64)
    parser.add_argument("--lgb-min-child-samples", type=int, default=40)
    parser.add_argument("--lgb-subsample", type=float, default=0.8)
    parser.add_argument("--lgb-colsample", type=float, default=0.8)
    parser.add_argument("--lgb-reg-alpha", type=float, default=0.1)
    parser.add_argument("--lgb-reg-lambda", type=float, default=0.2)

    parser.add_argument("--xgb-max-depth", type=int, default=6)
    parser.add_argument("--xgb-subsample", type=float, default=0.85)
    parser.add_argument("--xgb-colsample", type=float, default=0.75)
    parser.add_argument("--xgb-min-child-weight", type=float, default=2.0)
    parser.add_argument("--xgb-reg-alpha", type=float, default=0.05)
    parser.add_argument("--xgb-reg-lambda", type=float, default=1.0)

    parser.add_argument("--cat-depth", type=int, default=7)
    parser.add_argument("--cat-l2-leaf-reg", type=float, default=6.0)

    return parser.parse_args()


def infer_id_column(train: pd.DataFrame, test: pd.DataFrame, sample: pd.DataFrame | None, provided: str | None) -> str | None:
    if provided:
        return provided
    shared = [col for col in train.columns if col in test.columns]
    if sample is not None and not sample.empty and sample.columns[0] in shared:
        return sample.columns[0]
    for candidate in ["id", "ID", "Id", "index"]:
        if candidate in shared:
            return candidate
    for col in shared:
        if "id" in col.lower():
            return col
    return None


def infer_target_column(train: pd.DataFrame, sample: pd.DataFrame | None, id_col: str | None, provided: str | None) -> str:
    if provided:
        if provided not in train.columns:
            raise ValueError(f"Provided target '{provided}' not found in train columns")
        return provided

    if sample is not None:
        target_candidates = [col for col in sample.columns if col != id_col]
        if len(target_candidates) == 1 and target_candidates[0] in train.columns:
            return target_candidates[0]

    for candidate in ["target", "label", "class", "y", "response", "Survived", "stroke", "heart_disease"]:
        if candidate in train.columns:
            return candidate

    raise ValueError("Could not infer target column. Pass --target explicitly.")


def infer_task(y: pd.Series) -> str:
    if y.dtype == "O" or str(y.dtype).startswith("category"):
        return "classification"
    if y.nunique(dropna=True) <= 20:
        return "classification"
    return "regression"


def parse_seeds(primary_seed: int, seed_list: str) -> list[int]:
    seeds = [primary_seed]
    if seed_list.strip():
        for token in seed_list.split(","):
            value = int(token.strip())
            if value not in seeds:
                seeds.append(value)
    return seeds


def prepare_features(train: pd.DataFrame, test: pd.DataFrame, feature_columns: list[str]) -> PreparedData:
    train_x = train[feature_columns].copy()
    test_x = test[feature_columns].copy()

    cat_cols = train_x.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    num_cols = [col for col in feature_columns if col not in cat_cols]

    for col in cat_cols:
        train_x[col] = train_x[col].astype("string").fillna("__MISSING__")
        test_x[col] = test_x[col].astype("string").fillna("__MISSING__")

    if num_cols:
        medians = train_x[num_cols].median(numeric_only=True)
        train_x[num_cols] = train_x[num_cols].fillna(medians)
        test_x[num_cols] = test_x[num_cols].fillna(medians)

    combined = pd.concat([train_x, test_x], axis=0, ignore_index=True)
    if cat_cols:
        combined = pd.get_dummies(combined, columns=cat_cols, dummy_na=False)

    train_oh = combined.iloc[: len(train_x)].reset_index(drop=True)
    test_oh = combined.iloc[len(train_x) :].reset_index(drop=True)
    cat_feature_indices = [train_x.columns.get_loc(col) for col in cat_cols]

    return PreparedData(
        train_oh=train_oh,
        test_oh=test_oh,
        train_cat=train_x.reset_index(drop=True),
        test_cat=test_x.reset_index(drop=True),
        cat_feature_indices=cat_feature_indices,
    )


def evaluate(y_true: np.ndarray, pred: np.ndarray, task: str, n_classes: int) -> float:
    if task == "regression":
        return float(np.sqrt(mean_squared_error(y_true, pred)))
    if n_classes == 2:
        if len(np.unique(y_true)) < 2:
            return float("nan")
        return float(roc_auc_score(y_true, pred))
    clipped = np.clip(pred, 1e-9, 1 - 1e-9)
    clipped = clipped / clipped.sum(axis=1, keepdims=True)
    return float(log_loss(y_true, clipped))


def allocate_prediction_arrays(train_len: int, test_len: int, task: str, n_classes: int) -> tuple[np.ndarray, ...]:
    if task == "classification" and n_classes > 2:
        oof_lgb = np.zeros((train_len, n_classes), dtype=np.float64)
        oof_xgb = np.zeros((train_len, n_classes), dtype=np.float64)
        oof_cat = np.zeros((train_len, n_classes), dtype=np.float64)
        pred_lgb = np.zeros((test_len, n_classes), dtype=np.float64)
        pred_xgb = np.zeros((test_len, n_classes), dtype=np.float64)
        pred_cat = np.zeros((test_len, n_classes), dtype=np.float64)
    else:
        oof_lgb = np.zeros(train_len, dtype=np.float64)
        oof_xgb = np.zeros(train_len, dtype=np.float64)
        oof_cat = np.zeros(train_len, dtype=np.float64)
        pred_lgb = np.zeros(test_len, dtype=np.float64)
        pred_xgb = np.zeros(test_len, dtype=np.float64)
        pred_cat = np.zeros(test_len, dtype=np.float64)
    return oof_lgb, oof_xgb, oof_cat, pred_lgb, pred_xgb, pred_cat


def train_lgb(
    args: argparse.Namespace,
    task: str,
    n_classes: int,
    model_seed: int,
    xtr: pd.DataFrame,
    ytr: np.ndarray,
    xva: pd.DataFrame,
    yva: np.ndarray,
    xte: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    if task == "classification":
        model = LGBMClassifier(
            objective="binary" if n_classes == 2 else "multiclass",
            n_estimators=args.n_estimators,
            learning_rate=args.learning_rate,
            num_leaves=args.lgb_num_leaves,
            subsample=args.lgb_subsample,
            colsample_bytree=args.lgb_colsample,
            min_child_samples=args.lgb_min_child_samples,
            reg_alpha=args.lgb_reg_alpha,
            reg_lambda=args.lgb_reg_lambda,
            random_state=model_seed,
            n_jobs=-1,
        )
        if n_classes > 2:
            model.set_params(num_class=n_classes)
        model.fit(
            xtr,
            ytr,
            eval_set=[(xva, yva)],
            eval_metric="auc" if n_classes == 2 else "multi_logloss",
            callbacks=[lgb.early_stopping(args.early_stopping_rounds, verbose=False)],
        )
        va_pred = model.predict_proba(xva)
        te_pred = model.predict_proba(xte)
        if n_classes == 2:
            va_pred = va_pred[:, 1]
            te_pred = te_pred[:, 1]
        return va_pred, te_pred

    model = LGBMRegressor(
        objective="regression",
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        num_leaves=args.lgb_num_leaves,
        subsample=args.lgb_subsample,
        colsample_bytree=args.lgb_colsample,
        min_child_samples=args.lgb_min_child_samples,
        reg_alpha=args.lgb_reg_alpha,
        reg_lambda=args.lgb_reg_lambda,
        random_state=model_seed,
        n_jobs=-1,
    )
    model.fit(
        xtr,
        ytr,
        eval_set=[(xva, yva)],
        eval_metric="l2",
        callbacks=[lgb.early_stopping(args.early_stopping_rounds, verbose=False)],
    )
    return model.predict(xva), model.predict(xte)


def train_xgb(
    args: argparse.Namespace,
    task: str,
    n_classes: int,
    model_seed: int,
    xtr: pd.DataFrame,
    ytr: np.ndarray,
    xva: pd.DataFrame,
    yva: np.ndarray,
    xte: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    if task == "classification":
        model = XGBClassifier(
            objective="binary:logistic" if n_classes == 2 else "multi:softprob",
            n_estimators=args.n_estimators,
            learning_rate=args.learning_rate,
            max_depth=args.xgb_max_depth,
            subsample=args.xgb_subsample,
            colsample_bytree=args.xgb_colsample,
            min_child_weight=args.xgb_min_child_weight,
            reg_alpha=args.xgb_reg_alpha,
            reg_lambda=args.xgb_reg_lambda,
            tree_method="hist",
            eval_metric="auc" if n_classes == 2 else "mlogloss",
            random_state=model_seed,
            n_jobs=-1,
        )
        if n_classes > 2:
            model.set_params(num_class=n_classes)
        model.fit(
            xtr,
            ytr,
            eval_set=[(xva, yva)],
            verbose=False,
        )
        va_pred = model.predict_proba(xva)
        te_pred = model.predict_proba(xte)
        if n_classes == 2:
            va_pred = va_pred[:, 1]
            te_pred = te_pred[:, 1]
        return va_pred, te_pred

    model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        max_depth=args.xgb_max_depth,
        subsample=args.xgb_subsample,
        colsample_bytree=args.xgb_colsample,
        min_child_weight=args.xgb_min_child_weight,
        reg_alpha=args.xgb_reg_alpha,
        reg_lambda=args.xgb_reg_lambda,
        tree_method="hist",
        eval_metric="rmse",
        random_state=model_seed,
        n_jobs=-1,
    )
    model.fit(
        xtr,
        ytr,
        eval_set=[(xva, yva)],
        verbose=False,
    )
    return model.predict(xva), model.predict(xte)


def train_cat(
    args: argparse.Namespace,
    task: str,
    n_classes: int,
    model_seed: int,
    xtr: pd.DataFrame,
    ytr: np.ndarray,
    xva: pd.DataFrame,
    yva: np.ndarray,
    xte: pd.DataFrame,
    cat_feature_indices: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    train_pool = Pool(xtr, ytr, cat_features=cat_feature_indices)
    valid_pool = Pool(xva, yva, cat_features=cat_feature_indices)
    test_pool = Pool(xte, cat_features=cat_feature_indices)

    if task == "classification":
        model = CatBoostClassifier(
            loss_function="Logloss" if n_classes == 2 else "MultiClass",
            eval_metric="AUC" if n_classes == 2 else "MultiClass",
            iterations=args.n_estimators,
            learning_rate=args.learning_rate,
            depth=args.cat_depth,
            l2_leaf_reg=args.cat_l2_leaf_reg,
            random_seed=model_seed,
            od_type="Iter",
            od_wait=args.early_stopping_rounds,
            verbose=False,
        )
        model.fit(train_pool, eval_set=valid_pool, use_best_model=True)
        va_pred = model.predict_proba(valid_pool)
        te_pred = model.predict_proba(test_pool)
        if n_classes == 2:
            va_pred = va_pred[:, 1]
            te_pred = te_pred[:, 1]
        return va_pred, te_pred

    model = CatBoostRegressor(
        loss_function="RMSE",
        eval_metric="RMSE",
        iterations=args.n_estimators,
        learning_rate=args.learning_rate,
        depth=args.cat_depth,
        l2_leaf_reg=args.cat_l2_leaf_reg,
        random_seed=model_seed,
        od_type="Iter",
        od_wait=args.early_stopping_rounds,
        verbose=False,
    )
    model.fit(train_pool, eval_set=valid_pool, use_best_model=True)
    return model.predict(valid_pool), model.predict(test_pool)


def fold_msg(task: str, n_classes: int, fold: int, lgb_score: float, xgb_score: float, cat_score: float) -> str:
    if task == "regression":
        return f"Fold {fold} | LGB RMSE={lgb_score:.5f} | XGB RMSE={xgb_score:.5f} | CAT RMSE={cat_score:.5f}"
    if n_classes == 2:
        return f"Fold {fold} | LGB AUC={lgb_score:.5f} | XGB AUC={xgb_score:.5f} | CAT AUC={cat_score:.5f}"
    return f"Fold {fold} | LGB logloss={lgb_score:.5f} | XGB logloss={xgb_score:.5f} | CAT logloss={cat_score:.5f}"


def main() -> None:
    args = parse_args()

    train = pd.read_csv(args.data_dir / args.train_file)
    test = pd.read_csv(args.data_dir / args.test_file)
    sample_path = args.data_dir / args.sample_file
    sample = pd.read_csv(sample_path) if sample_path.exists() else None

    id_col = infer_id_column(train, test, sample, args.id_column)
    target_col = infer_target_column(train, sample, id_col, args.target)

    feature_columns = [col for col in train.columns if col != target_col]
    if id_col and id_col in feature_columns:
        feature_columns.remove(id_col)

    y_raw = train[target_col]
    task = infer_task(y_raw)

    label_encoder = None
    if task == "classification":
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y_raw.astype(str))
        n_classes = int(len(label_encoder.classes_))
    else:
        y = y_raw.astype(float).to_numpy()
        n_classes = 1

    prepared = prepare_features(train, test, feature_columns)
    seeds = parse_seeds(args.seed, args.seed_list)

    if task == "classification":
        splitter = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
        split_iter = splitter.split(prepared.train_oh, y)
    else:
        splitter = KFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
        split_iter = splitter.split(prepared.train_oh)

    oof_lgb, oof_xgb, oof_cat, pred_lgb, pred_xgb, pred_cat = allocate_prediction_arrays(
        len(train),
        len(test),
        task,
        n_classes,
    )

    fold_scores: dict[str, list[float]] = {"lgb": [], "xgb": [], "cat": []}

    for fold, (tr_idx, va_idx) in enumerate(split_iter, start=1):
        xtr_oh = prepared.train_oh.iloc[tr_idx]
        xva_oh = prepared.train_oh.iloc[va_idx]
        xtr_cat = prepared.train_cat.iloc[tr_idx]
        xva_cat = prepared.train_cat.iloc[va_idx]
        ytr = y[tr_idx]
        yva = y[va_idx]

        if task == "classification" and n_classes > 2:
            lgb_va_acc = np.zeros((len(va_idx), n_classes), dtype=np.float64)
            xgb_va_acc = np.zeros((len(va_idx), n_classes), dtype=np.float64)
            cat_va_acc = np.zeros((len(va_idx), n_classes), dtype=np.float64)
            lgb_te_acc = np.zeros((len(test), n_classes), dtype=np.float64)
            xgb_te_acc = np.zeros((len(test), n_classes), dtype=np.float64)
            cat_te_acc = np.zeros((len(test), n_classes), dtype=np.float64)
        else:
            lgb_va_acc = np.zeros(len(va_idx), dtype=np.float64)
            xgb_va_acc = np.zeros(len(va_idx), dtype=np.float64)
            cat_va_acc = np.zeros(len(va_idx), dtype=np.float64)
            lgb_te_acc = np.zeros(len(test), dtype=np.float64)
            xgb_te_acc = np.zeros(len(test), dtype=np.float64)
            cat_te_acc = np.zeros(len(test), dtype=np.float64)

        for seed_idx, model_seed in enumerate(seeds, start=1):
            seeded = model_seed + fold
            lgb_va, lgb_te = train_lgb(
                args,
                task,
                n_classes,
                seeded,
                xtr_oh,
                ytr,
                xva_oh,
                yva,
                prepared.test_oh,
            )
            xgb_va, xgb_te = train_xgb(
                args,
                task,
                n_classes,
                seeded,
                xtr_oh,
                ytr,
                xva_oh,
                yva,
                prepared.test_oh,
            )
            cat_va, cat_te = train_cat(
                args,
                task,
                n_classes,
                seeded,
                xtr_cat,
                ytr,
                xva_cat,
                yva,
                prepared.test_cat,
                prepared.cat_feature_indices,
            )

            lgb_va_acc += lgb_va / len(seeds)
            xgb_va_acc += xgb_va / len(seeds)
            cat_va_acc += cat_va / len(seeds)
            lgb_te_acc += lgb_te / len(seeds)
            xgb_te_acc += xgb_te / len(seeds)
            cat_te_acc += cat_te / len(seeds)

            print(f"Fold {fold} seed {seed_idx}/{len(seeds)} done (seed={seeded})")

        oof_lgb[va_idx] = lgb_va_acc
        oof_xgb[va_idx] = xgb_va_acc
        oof_cat[va_idx] = cat_va_acc

        pred_lgb += lgb_te_acc / args.n_splits
        pred_xgb += xgb_te_acc / args.n_splits
        pred_cat += cat_te_acc / args.n_splits

        lgb_score = evaluate(yva, lgb_va_acc, task, n_classes)
        xgb_score = evaluate(yva, xgb_va_acc, task, n_classes)
        cat_score = evaluate(yva, cat_va_acc, task, n_classes)

        fold_scores["lgb"].append(lgb_score)
        fold_scores["xgb"].append(xgb_score)
        fold_scores["cat"].append(cat_score)

        print(fold_msg(task, n_classes, fold, lgb_score, xgb_score, cat_score))

    weights = [float(value.strip()) for value in args.weights.split(",")]
    if len(weights) != 3:
        raise ValueError("--weights must contain exactly 3 comma-separated values")
    weight_sum = sum(weights)
    w_lgb, w_xgb, w_cat = [value / weight_sum for value in weights]

    oof_ens = oof_lgb * w_lgb + oof_xgb * w_xgb + oof_cat * w_cat
    pred_ens = pred_lgb * w_lgb + pred_xgb * w_xgb + pred_cat * w_cat

    final_lgb = evaluate(y, oof_lgb, task, n_classes)
    final_xgb = evaluate(y, oof_xgb, task, n_classes)
    final_cat = evaluate(y, oof_cat, task, n_classes)
    final_ens = evaluate(y, oof_ens, task, n_classes)

    metric_name = "RMSE" if task == "regression" else ("AUC" if n_classes == 2 else "logloss")
    print(f"CV {metric_name} | LGB={final_lgb:.5f} | XGB={final_xgb:.5f} | CAT={final_cat:.5f} | ENS={final_ens:.5f}")

    if sample is not None:
        submission = sample.copy()
        target_columns = [col for col in submission.columns if col != id_col] if id_col else list(submission.columns)
        if task == "classification" and n_classes > 2 and len(target_columns) == n_classes:
            for idx, col in enumerate(target_columns):
                submission[col] = pred_ens[:, idx]
        else:
            if len(target_columns) != 1:
                raise ValueError(
                    "Expected one target column in sample_submission for regression/binary classification. "
                    f"Found: {target_columns}"
                )
            submission[target_columns[0]] = pred_ens
        if id_col and id_col in test.columns:
            submission[id_col] = test[id_col].values
    else:
        submission = pd.DataFrame({"prediction": pred_ens})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.output, index=False)

    report = {
        "task": task,
        "target": target_col,
        "id_column": id_col,
        "feature_count": len(feature_columns),
        "seeds": seeds,
        "fold_scores": fold_scores,
        "cv_metric": metric_name,
        "cv": {
            "lgb": final_lgb,
            "xgb": final_xgb,
            "cat": final_cat,
            "ensemble": final_ens,
        },
        "weights": {"lgb": w_lgb, "xgb": w_xgb, "cat": w_cat},
        "params": {
            "learning_rate": args.learning_rate,
            "n_estimators": args.n_estimators,
            "early_stopping_rounds": args.early_stopping_rounds,
            "lgb": {
                "num_leaves": args.lgb_num_leaves,
                "min_child_samples": args.lgb_min_child_samples,
                "subsample": args.lgb_subsample,
                "colsample": args.lgb_colsample,
            },
            "xgb": {
                "max_depth": args.xgb_max_depth,
                "subsample": args.xgb_subsample,
                "colsample": args.xgb_colsample,
                "min_child_weight": args.xgb_min_child_weight,
            },
            "cat": {
                "depth": args.cat_depth,
                "l2_leaf_reg": args.cat_l2_leaf_reg,
            },
        },
        "output": str(args.output),
    }

    report_path = args.output.with_suffix(".metrics.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Saved submission to: {args.output}")
    print(f"Saved metrics report to: {report_path}")


if __name__ == "__main__":
    main()
