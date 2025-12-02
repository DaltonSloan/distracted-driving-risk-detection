import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import lightgbm as lgb


DATA_DIR = Path(__file__).resolve().parents[3] / "data"

CLUSTER_FEATURES = [
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

ROLLING_COLUMNS = ["speed", "rpm", "acceleration", "visibility", "heart_rate", "precipitation"]


def load_data(train_path: str | Path | None, test_path: str | Path | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_file = DATA_DIR / "kaggle_train.csv" if train_path is None else Path(train_path)
    test_file = DATA_DIR / "kaggle_test.csv" if test_path is None else Path(test_path)
    train = pd.read_csv(train_file)
    test = pd.read_csv(test_file)
    return train, test


def build_clusters(train: pd.DataFrame, test: pd.DataFrame, n_clusters: int = 160) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = train.copy()
    test = test.copy()
    train["is_train"] = 1
    test["is_train"] = 0
    combo = pd.concat([train, test], axis=0).reset_index(drop=True)
    combo["row_index"] = np.arange(len(combo))

    cluster_data = combo[CLUSTER_FEATURES + ["row_index"]].fillna(0.0)
    scaler = StandardScaler()
    cluster_scaled = scaler.fit_transform(cluster_data)

    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=42,
        batch_size=512,
        max_iter=200,
        n_init=20,
    )
    combo["pseudo_session_id"] = kmeans.fit_predict(cluster_scaled)

    train_clustered = combo.loc[combo["is_train"] == 1].drop(columns=["is_train"]).reset_index(drop=True)
    test_clustered = combo.loc[combo["is_train"] == 0].drop(columns=["is_train"]).reset_index(drop=True)
    return train_clustered, test_clustered


def add_pseudo_session_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    session_group = df.groupby("pseudo_session_id", sort=False)

    df["session_len"] = session_group["pseudo_session_id"].transform("count")
    df["session_pos"] = session_group.cumcount()
    df["session_pos_frac"] = df["session_pos"] / df["session_len"].clip(lower=1)
    df["session_rev_pos"] = df["session_len"] - df["session_pos"] - 1

    for col in ROLLING_COLUMNS:
        mean_name = f"{col}_session_mean"
        std_name = f"{col}_session_std"
        delta_name = f"{col}_delta"
        df[mean_name] = session_group[col].transform("mean")
        df[std_name] = session_group[col].transform("std").fillna(0.0)
        df[delta_name] = df[col] - df[mean_name]

        roll_mean_name = f"{col}_roll_mean_5"
        roll_std_name = f"{col}_roll_std_5"
        df[roll_mean_name] = (
            session_group[col]
            .rolling(window=5, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )
        df[roll_std_name] = (
            session_group[col]
            .rolling(window=5, min_periods=1)
            .std()
            .reset_index(level=0, drop=True)
            .fillna(0.0)
        )

    df["session_visibility_drop"] = (
        session_group["visibility"].transform("max") - session_group["visibility"].transform("min")
    )
    df["session_speed_var"] = session_group["speed"].transform("var").fillna(0.0)

    df = df.fillna(0.0)
    return df


def train_lightgbm(train_df: pd.DataFrame, feature_cols: list[str]) -> tuple[lgb.LGBMClassifier, list[float]]:
    X = train_df[feature_cols]
    y = train_df["risk_level"].astype(int)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    params = dict(
        objective="multiclass",
        num_class=4,
        learning_rate=0.04,
        n_estimators=1200,
        num_leaves=72,
        max_depth=-1,
        subsample=0.9,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_alpha=0.2,
        reg_lambda=0.4,
        min_child_samples=25,
        n_jobs=-1,
        random_state=42,
    )

    cv_scores: list[float] = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        model = lgb.LGBMClassifier(**params)
        model.fit(
            X.iloc[train_idx],
            y.iloc[train_idx],
            eval_set=[(X.iloc[val_idx], y.iloc[val_idx])],
            eval_metric="multi_logloss",
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)],
        )
        preds = model.predict(X.iloc[val_idx])
        acc = accuracy_score(y.iloc[val_idx], preds)
        cv_scores.append(acc)
        print(f"Fold {fold} accuracy: {acc:.4f}")

    final_model = lgb.LGBMClassifier(**params)
    final_model.fit(X, y)
    return final_model, cv_scores


def main() -> None:
    train_df, test_df = load_data(None, None)
    train_clustered, test_clustered = build_clusters(train_df, test_df, n_clusters=180)
    train_enriched = add_pseudo_session_features(train_clustered)
    test_enriched = add_pseudo_session_features(test_clustered)

    # Remove columns not for modeling
    drop_cols = ["label_source", "row_index"] if "label_source" in train_enriched.columns else ["row_index"]
    train_enriched = train_enriched.drop(columns=[col for col in drop_cols if col in train_enriched.columns])
    test_enriched = test_enriched.drop(columns=[col for col in drop_cols if col in test_enriched.columns])

    feature_cols = [col for col in train_enriched.columns if col not in {"risk_level"}]

    model, scores = train_lightgbm(train_enriched, feature_cols)
    print("CV accuracy scores:", [round(s, 4) for s in scores])
    print("Mean CV accuracy:", round(float(np.mean(scores)), 4))

    test_preds = model.predict(test_enriched[feature_cols])
    submission = pd.DataFrame({"id": np.arange(len(test_preds)), "risk_level": test_preds})
    out_path = Path("driving_risk_submission_cluster.csv")
    submission.to_csv(out_path, index=False)
    print(f"Saved submission to {out_path.resolve()}")


if __name__ == "__main__":
    main()
