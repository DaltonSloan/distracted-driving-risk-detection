"""Comprehensive modeling pipeline with temporal-style features, diverse
ensembles, and semi-supervised fine-tuning."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from xgboost import XGBClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report


DATA_DIR = Path(__file__).resolve().parents[3] / "data"

NUMERIC_FEATURES = [
    "observation_hour",
    "speed",
    "rpm",
    "acceleration",
    "throttle_position",
    "engine_temperature",
    "engine_load_value",
    "heart_rate",
    "current_weather",
    "visibility",
    "precipitation",
    "accidents_onsite",
    "design_speed",
    "accidents_time",
]

SEQUENCE_COLUMNS = [
    "speed",
    "rpm",
    "acceleration",
    "heart_rate",
    "visibility",
    "precipitation",
]


def build_base_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy().reset_index(drop=True)
    for col in NUMERIC_FEATURES:
        data[col] = data[col].astype(float)

    # Binary flag for label provenance (train uses expert labels).
    if "label_source" in data.columns:
        data["label_source_flag"] = (
            data["label_source"].eq("expert_verified").astype(float)
        )
        data = data.drop(columns=["label_source"])

    hour_rad = data["observation_hour"] / 24.0 * (2.0 * np.pi)
    data["hour_sin"] = np.sin(hour_rad)
    data["hour_cos"] = np.cos(hour_rad)

    data["speed_to_design"] = data["speed"] / (data["design_speed"] + 1.0)
    data["speed_minus_design"] = data["speed"] - data["design_speed"]
    data["speed_pct_of_rpm"] = data["speed"] / (data["rpm"] + 1.0)
    data["speed_visibility_ratio"] = data["speed"] / (data["visibility"] + 1.0)
    data["speed_precip_ratio"] = data["speed"] / (data["precipitation"] + 1.0)
    data["speed_squared"] = data["speed"] ** 2

    data["abs_acceleration"] = np.abs(data["acceleration"])
    data["accel_throttle"] = data["acceleration"] * data["throttle_position"]
    data["accel_speed_ratio"] = data["acceleration"] / (data["speed"] + 1.0)

    data["rpm_per_speed"] = data["rpm"] / (data["speed"] + 1.0)
    data["rpm_per_load"] = data["rpm"] / (data["engine_load_value"] + 1.0)
    data["rpm_temp_ratio"] = data["rpm"] / (data["engine_temperature"] + 1.0)

    data["engine_temp_minus_load"] = (
        data["engine_temperature"] - data["engine_load_value"]
    )
    data["engine_load_ratio"] = data["engine_load_value"] / (
        data["engine_temperature"] + 1.0
    )
    data["throttle_load_ratio"] = data["throttle_position"] / (
        data["engine_load_value"] + 1.0
    )
    data["throttle_temp_ratio"] = data["throttle_position"] / (
        data["engine_temperature"] + 1.0
    )

    data["heart_rate_speed_ratio"] = data["heart_rate"] / (data["speed"] + 1.0)
    data["heart_rate_accel"] = data["heart_rate"] * (data["abs_acceleration"] + 1.0)
    data["stress_visibility"] = data["heart_rate"] * (
        data["abs_acceleration"] + 1.0
    ) / (data["visibility"] + 1.0)
    data["heart_over_design"] = data["heart_rate"] / (data["design_speed"] + 1.0)

    accidents_sum = data["accidents_onsite"] + data["accidents_time"]
    data["accidents_total"] = accidents_sum
    data["precip_accident"] = data["precipitation"] * accidents_sum
    data["accident_hour_ratio"] = accidents_sum / (data["observation_hour"] + 1.0)

    data["visibility_precip_ratio"] = data["visibility"] / (
        data["precipitation"] + 1.0
    )
    data["precip_speed"] = data["precipitation"] * data["speed"]
    data["weather_visibility"] = data["current_weather"] * data["visibility"]
    data["weather_precip"] = data["current_weather"] * data["precipitation"]
    data["design_visibility_ratio"] = data["design_speed"] / (
        data["visibility"] + 1.0
    )

    data["is_raining"] = (data["precipitation"] > 0).astype(float)
    data["low_visibility"] = (data["visibility"] < 5).astype(float)
    data["high_speeding"] = (data["speed"] > data["design_speed"]).astype(float)

    for col in NUMERIC_FEATURES:
        data[f"{col}_rank"] = data[col].rank(pct=True).astype(float)

    return data


def add_sequence_features(data: pd.DataFrame) -> pd.DataFrame:
    seq_df = data.copy()
    seq_df["row_idx"] = np.arange(len(seq_df))
    proxy = (
        seq_df["design_speed"].round().astype(int).astype(str)
        + "_"
        + seq_df["accidents_onsite"].round().astype(int).astype(str)
        + "_"
        + seq_df["observation_hour"].round().astype(int).astype(str)
    )
    seq_df["session_proxy"] = proxy
    seq_df = seq_df.sort_values(["session_proxy", "row_idx"]).copy()

    grouped = seq_df.groupby("session_proxy", sort=False)
    for col in SEQUENCE_COLUMNS:
        seq_df[f"{col}_lag1"] = grouped[col].shift(1)
        seq_df[f"{col}_lag2"] = grouped[col].shift(2)
        seq_df[f"{col}_delta1"] = seq_df[col] - seq_df[f"{col}_lag1"]
        seq_df[f"{col}_rolling_mean3"] = (
            grouped[col]
            .rolling(window=3, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )
        seq_df[f"{col}_rolling_std3"] = (
            grouped[col]
            .rolling(window=3, min_periods=1)
            .std()
            .reset_index(level=0, drop=True)
        )

    seq_df = seq_df.sort_values("row_idx").drop(columns=["session_proxy", "row_idx"])
    return seq_df.fillna(0.0)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    base = build_base_features(df)
    enriched = add_sequence_features(base)
    return enriched.fillna(0.0)


def train_stack_models(
    X: pd.DataFrame, y: np.ndarray, X_test: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, dict[str, list[float]]]:
    n_classes = len(np.unique(y))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = {
        "hgb": np.zeros((len(X), n_classes)),
        "lgb": np.zeros((len(X), n_classes)),
        "xgb": np.zeros((len(X), n_classes)),
    }
    fold_scores: dict[str, list[float]] = {k: [] for k in oof_preds}

    models_template = {
        "hgb": HistGradientBoostingClassifier(
            learning_rate=0.06,
            max_iter=350,
            max_depth=9,
            max_leaf_nodes=63,
            min_samples_leaf=20,
            l2_regularization=0.2,
            early_stopping=False,
            random_state=7,
        ),
        "lgb": lgb.LGBMClassifier(
            objective="multiclass",
            num_class=n_classes,
            learning_rate=0.045,
            n_estimators=1100,
            num_leaves=68,
            max_depth=-1,
            subsample=0.85,
            subsample_freq=1,
            colsample_bytree=0.85,
            reg_alpha=0.2,
            reg_lambda=0.4,
            min_child_samples=28,
            random_state=11,
            n_jobs=-1,
            verbosity=-1,
        ),
        "xgb": XGBClassifier(
            objective="multi:softprob",
            num_class=n_classes,
            learning_rate=0.05,
            n_estimators=420,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.85,
            reg_lambda=1.0,
            reg_alpha=0.1,
            min_child_weight=2,
            gamma=0.0,
            tree_method="hist",
            random_state=17,
            n_jobs=-1,
            verbosity=0,
            eval_metric="mlogloss",
        ),
    }

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        for name, template in models_template.items():
            model = template.__class__(**template.get_params())
            if name == "lgb":
                model.fit(
                    X_train,
                    y_train,
                    eval_set=[(X_val, y_val)],
                    eval_metric="multi_logloss",
                    callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)],
                )
            else:
                model.fit(X_train, y_train)
            probas = model.predict_proba(X_val)
            oof_preds[name][val_idx] = probas
            preds = probas.argmax(axis=1)
            acc = accuracy_score(y_val, preds)
            fold_scores[name].append(acc)
            print(f"Fold {fold} {name} accuracy={acc:.4f}")

    meta_X = np.hstack([oof_preds[name] for name in ["hgb", "lgb", "xgb"]])
    meta_model = LogisticRegression(
        max_iter=400,
        C=2.0,
        multi_class="multinomial",
        solver="lbfgs",
    )
    meta_model.fit(meta_X, y)
    meta_preds = meta_model.predict(meta_X)
    print(
        "Meta-model OOF accuracy",
        round(accuracy_score(y, meta_preds), 4),
    )
    print(classification_report(y + 1, meta_preds + 1, digits=3))

    final_models = {
        name: template.__class__(**template.get_params())
        for name, template in models_template.items()
    }
    test_meta_parts = []
    for name, model in final_models.items():
        model.fit(X, y)
        probas = model.predict_proba(X_test)
        test_meta_parts.append(probas)
    test_meta = np.hstack(test_meta_parts)
    test_stack_probas = meta_model.predict_proba(test_meta)
    return test_stack_probas, meta_X, fold_scores


def semi_supervised_lightgbm(
    X: pd.DataFrame,
    y: np.ndarray,
    X_test: pd.DataFrame,
    stack_test_probas: np.ndarray,
    threshold: float = 0.97,
) -> np.ndarray:
    confidences = stack_test_probas.max(axis=1)
    mask = confidences >= threshold
    pseudo_X = X_test[mask].reset_index(drop=True)
    pseudo_y = stack_test_probas[mask].argmax(axis=1)
    print(f"Pseudo-label count: {len(pseudo_y)} (threshold={threshold})")

    params = dict(
        objective="multiclass",
        num_class=4,
        learning_rate=0.032,
        n_estimators=1400,
        num_leaves=88,
        max_depth=-1,
        subsample=0.9,
        subsample_freq=1,
        colsample_bytree=0.9,
        reg_alpha=0.3,
        reg_lambda=0.6,
        min_child_samples=24,
        random_state=2026,
        n_jobs=-1,
        verbosity=-1,
    )

    pseudo_weight = 0.3
    skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        X_train = pd.concat([X.iloc[train_idx], pseudo_X], axis=0).reset_index(drop=True)
        y_train = np.concatenate([y[train_idx], pseudo_y])
        weights = np.concatenate(
            [np.ones(len(train_idx)), np.full(len(pseudo_y), pseudo_weight)]
        )
        model = lgb.LGBMClassifier(**params)
        model.fit(X_train, y_train, sample_weight=weights)
        preds = model.predict(X.iloc[val_idx])
        print(f"Semi-supervised fold {fold} acc={accuracy_score(y[val_idx], preds):.4f}")

    full_X = pd.concat([X, pseudo_X], axis=0).reset_index(drop=True)
    full_y = np.concatenate([y, pseudo_y])
    full_weights = np.concatenate(
        [np.ones(len(y)), np.full(len(pseudo_y), pseudo_weight)]
    )
    final_model = lgb.LGBMClassifier(**params)
    final_model.fit(full_X, full_y, sample_weight=full_weights)
    final_probas = final_model.predict_proba(X_test)
    return final_probas


def main() -> None:
    train_df = pd.read_csv(DATA_DIR / "kaggle_train.csv")
    test_df = pd.read_csv(DATA_DIR / "kaggle_test.csv")

    y = train_df["risk_level"].astype(int).values - 1
    train_features = engineer_features(train_df.drop(columns=["risk_level"]))
    test_features = engineer_features(test_df)

    stack_test_probas, _, fold_scores = train_stack_models(
        train_features, y, test_features
    )
    print("Fold accuracies per model:")
    for name, scores in fold_scores.items():
        print(name, [round(s, 4) for s in scores])

    semi_probas = semi_supervised_lightgbm(
        train_features, y, test_features, stack_test_probas, threshold=0.97
    )

    blended = 0.65 * semi_probas + 0.35 * stack_test_probas
    preds = blended.argmax(axis=1) + 1

    submission = pd.DataFrame({"id": np.arange(len(preds)), "risk_level": preds})
    out_path = Path("driving_risk_submission_v8.csv")
    submission.to_csv(out_path, index=False)
    print("Saved submission to", out_path.resolve())
    print("Prediction distribution:\n", submission["risk_level"].value_counts().sort_index())


if __name__ == "__main__":
    main()
