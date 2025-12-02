"""
Semi-supervised XGBoost for driving risk
---------------------------------------
Start from the tuned single XGBoost, then pseudo-label high-confidence
test rows to adapt toward test distribution. Outputs submission CSV.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


BASE_FEATURES = [
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

TARGET = "risk_level"
DATA_DIR = Path(__file__).resolve().parents[3] / "data"


def load_data(
    train_path: str | Path | None = None, test_path: str | Path | None = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_file = DATA_DIR / "kaggle_train.csv" if train_path is None else Path(train_path)
    test_file = DATA_DIR / "kaggle_test.csv" if test_path is None else Path(test_path)
    return pd.read_csv(train_file), pd.read_csv(test_file)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    for col in BASE_FEATURES:
        data[col] = data[col].astype(float)

    data["label_source_flag"] = (
        data.get("label_source", pd.Series(0, index=data.index)).eq("expert_verified").astype(float)
    )
    data = data.drop(columns=["label_source"], errors="ignore")

    hour_rad = data["observation_hour"] / 24.0 * (2.0 * np.pi)
    data["hour_sin"] = np.sin(hour_rad)
    data["hour_cos"] = np.cos(hour_rad)

    data["speed_to_design"] = data["speed"] / (data["design_speed"] + 1e-3)
    data["speed_minus_design"] = data["speed"] - data["design_speed"]
    data["speed_visibility_ratio"] = data["speed"] / (data["visibility"] + 1.0)
    data["speed_precip_ratio"] = data["speed"] / (data["precipitation"] + 1.0)

    data["throttle_load_ratio"] = data["throttle_position"] / (data["engine_load_value"] + 1.0)
    data["rpm_per_speed"] = data["rpm"] / (data["speed"] + 1.0)
    data["rpm_per_load"] = data["rpm"] / (data["engine_load_value"] + 1.0)
    data["rpm_temp_ratio"] = data["rpm"] / (data["engine_temperature"] + 1.0)

    data["engine_stress"] = data["engine_load_value"] * data["engine_temperature"]
    data["engine_load_ratio"] = data["engine_load_value"] / (data["engine_temperature"] + 1.0)

    data["heart_rate_dev"] = data["heart_rate"] - data["heart_rate"].mean()
    data["heart_over_speed"] = data["heart_rate"] / (data["speed"] + 1.0)
    data["heart_over_design"] = data["heart_rate"] / (data["design_speed"] + 1.0)
    data["stress_proxy"] = data["heart_rate"] * (data["acceleration"].abs() + 1.0) / (data["visibility"] + 1.0)

    data["visibility_precip_ratio"] = data["visibility"] / (data["precipitation"] + 1.0)
    data["precip_speed"] = data["precipitation"] * data["speed"]
    data["accel_throttle"] = data["acceleration"] * data["throttle_position"]

    data["env_risk"] = data["current_weather"] * (data["precipitation"] + 1.0) / (data["visibility"] + 1.0)
    data["hist_weather_risk"] = data["accidents_onsite"] * (data["precipitation"] + 1.0)
    data["total_accidents"] = data["accidents_onsite"] + data["accidents_time"]

    return data.replace([np.inf, -np.inf], 0.0).fillna(0.0)


def make_model(num_classes: int = 4, seed: int = 42) -> XGBClassifier:
    return XGBClassifier(
        objective="multi:softprob",
        num_class=num_classes,
        learning_rate=0.06,
        n_estimators=900,
        max_depth=7,
        subsample=0.95,
        colsample_bytree=0.9,
        min_child_weight=1,
        reg_lambda=1.0,
        reg_alpha=0.15,
        gamma=0.05,
        tree_method="hist",
        eval_metric="mlogloss",
        random_state=seed,
    )


def train_and_pseudo_label(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    proba_thresh: float = 0.99,
    pseudo_weight: float = 0.25,
) -> Tuple[XGBClassifier, List[str]]:
    # Engineer
    train_proc = engineer_features(train_df)
    test_proc = engineer_features(test_df)
    X_train_full = train_proc.drop(columns=[TARGET])
    y_full = train_proc[TARGET].astype(int) - 1
    features = X_train_full.columns.tolist()

    # Base model for validation & pseudo-labels
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_full, y_full, test_size=0.2, stratify=y_full, random_state=42
    )
    base_model = make_model(seed=42)
    base_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    val_preds = base_model.predict(X_val)
    val_acc = accuracy_score(y_val, val_preds)
    print("\n=== BASE VALIDATION ===")
    print(f"Accuracy: {val_acc:.4f}")
    print(classification_report(y_val + 1, val_preds + 1, digits=4))
    print(confusion_matrix(y_val + 1, val_preds + 1))

    test_probs = base_model.predict_proba(test_proc[features])
    confidences = test_probs.max(axis=1)
    mask = confidences >= proba_thresh
    pseudo_X = test_proc.loc[mask, features]
    pseudo_y = test_probs[mask].argmax(axis=1)
    print(f"Pseudo-labeled rows: {mask.sum()} at threshold {proba_thresh}")

    # Weighted retrain with pseudo-labels
    X_aug = pd.concat([X_train_full, pseudo_X], axis=0).reset_index(drop=True)
    y_aug = np.concatenate([y_full, pseudo_y])
    weights = np.concatenate([np.ones(len(y_full)), np.full(len(pseudo_y), pseudo_weight)])

    final_model = make_model(seed=99)
    final_model.fit(X_aug, y_aug, sample_weight=weights, verbose=False)
    return final_model, features


def predict_submission(model: XGBClassifier, feature_order: List[str], df_test: pd.DataFrame) -> pd.DataFrame:
    proc = engineer_features(df_test)
    X_test = proc.reindex(columns=feature_order, fill_value=0.0)
    preds = model.predict(X_test) + 1
    return pd.DataFrame({"id": np.arange(len(preds)), "risk_level": preds})


if __name__ == "__main__":
    train_df, test_df = load_data()
    model, feat_order = train_and_pseudo_label(train_df, test_df, proba_thresh=0.992, pseudo_weight=0.2)
    submission = predict_submission(model, feat_order, test_df)
    out_path = "driving_risk_submission_pseudo.csv"
    submission.to_csv(out_path, index=False)
    print(f"Saved submission to {out_path}")
