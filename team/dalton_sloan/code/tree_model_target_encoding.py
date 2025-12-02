import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression

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

DATA_DIR = Path(__file__).resolve().parents[3] / "data"

TARGET_ENCODING_COLS = [
    "observation_hour",
    "current_weather",
    "design_speed",
    "accidents_onsite",
]


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(DATA_DIR / "kaggle_train.csv")
    test = pd.read_csv(DATA_DIR / "kaggle_test.csv")
    return train, test


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["speed_to_design"] = df["speed"] / (df["design_speed"].replace(0, np.nan) + 1e-3)
    df["speed_minus_design"] = df["speed"] - df["design_speed"]
    df["throttle_load_ratio"] = df["throttle_position"] / (df["engine_load_value"] + 1.0)
    df["rpm_per_speed"] = df["rpm"] / (df["speed"] + 1.0)
    df["rpm_per_load"] = df["rpm"] / (df["engine_load_value"] + 1.0)
    df["engine_stress"] = df["engine_load_value"] * df["engine_temperature"]
    df["heart_rate_dev"] = df["heart_rate"] - df["heart_rate"].mean()
    df["stress_proxy"] = df["heart_rate"] * (df["acceleration"].abs() + 1.0) / (df["visibility"] + 1.0)
    df["env_risk"] = df["current_weather"] * (df["precipitation"] + 1.0) / (df["visibility"] + 1.0)
    df["hist_weather_risk"] = df["accidents_onsite"] * (df["precipitation"] + 1.0)
    df["total_accidents"] = df["accidents_onsite"] + df["accidents_time"]
    df["hour_sin"] = np.sin(2 * np.pi * df["observation_hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["observation_hour"] / 24.0)
    df["visibility_precip_ratio"] = df["visibility"] / (df["precipitation"] + 1.0)
    df["precip_speed"] = df["precipitation"] * df["speed"]
    df["accel_throttle"] = df["acceleration"] * df["throttle_position"]
    df["speed_visibility_ratio"] = df["speed"] / (df["visibility"] + 1.0)
    df["rpm_temp_ratio"] = df["rpm"] / (df["engine_temperature"] + 1.0)
    df["engine_load_ratio"] = df["engine_load_value"] / (df["engine_temperature"] + 1.0)
    df["heart_over_speed"] = df["heart_rate"] / (df["speed"] + 1.0)
    df["heart_over_design"] = df["heart_rate"] / (df["design_speed"] + 1.0)
    return df.replace([np.inf, -np.inf], 0.0)


def compute_target_encoding(train: pd.DataFrame, test: pd.DataFrame, cols: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train = train.copy()
    test = test.copy()
    y = train["risk_level"].astype(float)
    global_mean = y.mean()

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for col in cols:
        te_name = f"te_{col}"
        train[te_name] = 0.0
        for tr_idx, val_idx in skf.split(train, train["risk_level"]):
            fold = train.iloc[tr_idx]
            mapping = fold.groupby(col)["risk_level"].mean()
            train.iloc[val_idx, train.columns.get_loc(te_name)] = train.iloc[val_idx][col].map(mapping).fillna(global_mean)

        full_mapping = train.groupby(col)["risk_level"].mean()
        test[te_name] = test[col].map(full_mapping).fillna(global_mean)
    return train, test


def domain_weights(train: pd.DataFrame, test: pd.DataFrame, feature_cols: List[str]) -> np.ndarray:
    combo = pd.concat([train[feature_cols], test[feature_cols]], axis=0).reset_index(drop=True)
    indicator = np.concatenate([np.ones(len(train)), np.zeros(len(test))])
    scaler = StandardScaler()
    combo_scaled = scaler.fit_transform(combo.fillna(0))

    clf = LogisticRegression(max_iter=1000)
    clf.fit(combo_scaled, indicator)
    train_probs = clf.predict_proba(combo_scaled[: len(train)])[:, 1]
    weights = (1 - train_probs) / (train_probs + 1e-3)
    weights = np.clip(weights, 0.2, 5.0)
    return weights


def train_lightgbm(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[float]]:
    y = train_df["risk_level"].astype(int)
    feature_cols = [col for col in train_df.columns if col not in {"risk_level", "label_source"}]
    X = train_df[feature_cols]
    X_test = test_df[feature_cols]

    sample_weights = domain_weights(train_df, test_df, BASE_FEATURES)

    params = dict(
        objective="multiclass",
        num_class=4,
        learning_rate=0.035,
        n_estimators=1800,
        num_leaves=80,
        subsample=0.9,
        subsample_freq=1,
        colsample_bytree=0.9,
        reg_alpha=0.3,
        reg_lambda=0.6,
        min_child_samples=24,
        random_state=42,
        n_jobs=-1,
    )
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores: List[float] = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        model = lgb.LGBMClassifier(**params)
        model.fit(
            X.iloc[train_idx],
            y.iloc[train_idx],
            sample_weight=sample_weights[train_idx],
            eval_set=[(X.iloc[val_idx], y.iloc[val_idx])],
            eval_metric="multi_logloss",
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)],
        )
        preds = model.predict(X.iloc[val_idx])
        acc = accuracy_score(y.iloc[val_idx], preds)
        cv_scores.append(acc)
        print(f"Fold {fold} accuracy: {acc:.4f}")

    final_model = lgb.LGBMClassifier(**params)
    final_model.fit(X, y, sample_weight=sample_weights)
    test_preds = final_model.predict(X_test)
    return test_preds, y.values, cv_scores


def main() -> None:
    train_df, test_df = load_data()
    train_df = add_engineered_features(train_df)
    test_df = add_engineered_features(test_df)

    train_df, test_df = compute_target_encoding(train_df, test_df, TARGET_ENCODING_COLS)

    test_preds, _, scores = train_lightgbm(train_df, test_df)
    print("CV scores:", [round(s, 4) for s in scores])
    print("Mean CV accuracy:", round(float(np.mean(scores)), 4))

    submission = pd.DataFrame({"id": np.arange(len(test_preds)), "risk_level": test_preds})
    out_path = Path("driving_risk_submission_te.csv")
    submission.to_csv(out_path, index=False)
    print(f"Saved submission to {out_path.resolve()}")


if __name__ == "__main__":
    main()
