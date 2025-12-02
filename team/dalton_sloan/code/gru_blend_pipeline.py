import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression

# Ensure deterministic CPU behaviour
torch.set_num_threads(1)
torch.manual_seed(42)

DATA_DIR = Path(__file__).resolve().parents[3] / "data"

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

TARGET_ENCODING_COLS = ["observation_hour", "current_weather", "design_speed", "accidents_onsite"]

SEQ_FEATURES = [
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
    # engineered ratios / signals to give the GRU more context
    "speed_to_design",
    "speed_minus_design",
    "throttle_load_ratio",
    "rpm_per_speed",
    "rpm_per_load",
    "engine_stress",
    "heart_rate_dev",
    "stress_proxy",
    "visibility_precip_ratio",
    "hour_sin",
    "hour_cos",
    "precip_speed",
    "accel_throttle",
]

SEQ_LEN = 48


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
        for train_idx, val_idx in skf.split(train, train["risk_level"]):
            fold = train.iloc[train_idx]
            mapping = fold.groupby(col)["risk_level"].mean()
            train.iloc[val_idx, train.columns.get_loc(te_name)] = (
                train.iloc[val_idx][col].map(mapping).fillna(global_mean)
            )
        full_mapping = train.groupby(col)["risk_level"].mean()
        test[te_name] = test[col].map(full_mapping).fillna(global_mean)
    return train, test


def domain_weights(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    combo = pd.concat([train[BASE_FEATURES], test[BASE_FEATURES]], axis=0).reset_index(drop=True)
    indicator = np.concatenate([np.ones(len(train)), np.zeros(len(test))])
    scaler = StandardScaler()
    combo_scaled = scaler.fit_transform(combo.fillna(0))
    clf = LogisticRegression(max_iter=1000)
    clf.fit(combo_scaled, indicator)
    train_probs = clf.predict_proba(combo_scaled[: len(train)])[:, 1]
    weights = (1 - train_probs) / (train_probs + 1e-3)
    weights = np.clip(weights, 0.2, 5.0)
    return weights


def train_lightgbm_with_probs(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[float]]:
    feature_cols = [col for col in train_df.columns if col not in {"risk_level", "label_source"}]
    X = train_df[feature_cols]
    y = train_df["risk_level"].astype(int) - 1
    X_test = test_df[feature_cols]

    sample_weights = domain_weights(train_df, test_df)
    params = dict(
        objective="multiclass",
        num_class=4,
        learning_rate=0.045,
        n_estimators=2200,
        num_leaves=96,
        subsample=0.9,
        subsample_freq=1,
        colsample_bytree=0.9,
        reg_alpha=0.25,
        reg_lambda=0.55,
        min_child_samples=20,
        random_state=42,
        n_jobs=-1,
    )

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros((len(train_df), 4), dtype=np.float32)
    test_probs = np.zeros((len(test_df), 4), dtype=np.float32)
    scores: List[float] = []

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
        oof[val_idx] = model.predict_proba(X.iloc[val_idx])
        fold_pred = model.predict(X.iloc[val_idx])
        acc = accuracy_score(y.iloc[val_idx], fold_pred)
        scores.append(acc)
        test_probs += model.predict_proba(X_test) / skf.n_splits
        print(f"LightGBM fold {fold} accuracy: {acc:.4f}")

    return oof, test_probs, scores


def infer_sessions(df: pd.DataFrame) -> np.ndarray:
    session_ids = []
    session_id = 0
    prev = None
    cols = [
        "speed",
        "visibility",
        "precipitation",
        "design_speed",
        "accidents_onsite",
        "observation_hour",
    ]
    for row in df[cols].itertuples(index=False):
        if prev is None:
            session_ids.append(session_id)
            prev = row
            continue
        speed_jump = abs(row[0] - prev[0]) > 60 and min(row[0], prev[0]) < 5
        visibility_jump = abs(row[1] - prev[1]) > 6
        precip_jump = abs(row[2] - prev[2]) > 6
        design_jump = abs(row[3] - prev[3]) > 25
        accidents_jump = abs(row[4] - prev[4]) > 15
        hour_gap = (row[5] + 24 - prev[5]) % 24 > 6
        if speed_jump or visibility_jump or precip_jump or design_jump or accidents_jump or hour_gap:
            session_id += 1
        session_ids.append(session_id)
        prev = row
    return np.array(session_ids)


def build_sequences(features: np.ndarray, session_ids: np.ndarray, seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
    n_samples, feat_dim = features.shape
    sequences = np.zeros((n_samples, seq_len, feat_dim), dtype=np.float32)
    lengths = np.zeros(n_samples, dtype=np.int32)
    session_to_indices: dict[int, List[int]] = {}
    for idx, sid in enumerate(session_ids):
        session_to_indices.setdefault(int(sid), []).append(idx)
    for indices in session_to_indices.values():
        for pos, idx in enumerate(indices):
            start = max(0, pos - seq_len + 1)
            seq_idx = indices[start : pos + 1]
            length = len(seq_idx)
            lengths[idx] = length
            sequences[idx, -length:, :] = features[seq_idx]
    return sequences, lengths


def prepare_sequence_data(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    scaler = StandardScaler()
    scaler.fit(pd.concat([train_df[SEQ_FEATURES], test_df[SEQ_FEATURES]], axis=0))
    train_scaled = scaler.transform(train_df[SEQ_FEATURES]).astype(np.float32)
    test_scaled = scaler.transform(test_df[SEQ_FEATURES]).astype(np.float32)

    train_sequences, train_lengths = build_sequences(train_scaled, train_df["session_id"].values, SEQ_LEN)
    test_sequences, test_lengths = build_sequences(test_scaled, test_df["session_id"].values, SEQ_LEN)
    return train_sequences, train_lengths, test_sequences, test_lengths


class SequenceDataset(Dataset):
    def __init__(self, sequences: np.ndarray, labels: np.ndarray | None = None) -> None:
        self.sequences = sequences
        self.labels = labels

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int):
        seq = torch.from_numpy(self.sequences[idx])
        if self.labels is None:
            return seq
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return seq, label


class GRUClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.15) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True,
        )
        self.fc = nn.Linear(hidden_dim * 2, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, hidden = self.gru(x)
        hidden_cat = torch.cat([hidden[-2], hidden[-1]], dim=1)
        return self.fc(hidden_cat)


def train_gru_model(train_df: pd.DataFrame, test_df: pd.DataFrame, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[float]]:
    train_sequences, _, test_sequences, _ = prepare_sequence_data(train_df, test_df)
    device = torch.device("cpu")
    oof = np.zeros((len(train_df), 4), dtype=np.float32)
    test_probs = np.zeros((len(test_df), 4), dtype=np.float32)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores: List[float] = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(train_sequences, y), start=1):
        model = GRUClassifier(input_dim=train_sequences.shape[2]).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        train_data = SequenceDataset(train_sequences[train_idx], y[train_idx])
        val_data = SequenceDataset(train_sequences[val_idx], y[val_idx])
        train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=128)

        best_state = None
        best_loss = float("inf")
        patience = 3
        wait = 0
        for epoch in range(15):
            model.train()
            for seq_batch, labels in train_loader:
                seq_batch = seq_batch.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                logits = model(seq_batch)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

            model.eval()
            val_losses = []
            with torch.no_grad():
                for seq_batch, labels in val_loader:
                    seq_batch = seq_batch.to(device)
                    labels = labels.to(device)
                    logits = model(seq_batch)
                    loss = criterion(logits, labels)
                    val_losses.append(loss.item())
            mean_loss = float(np.mean(val_losses))
            if mean_loss < best_loss - 1e-4:
                best_loss = mean_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        val_loader = DataLoader(val_data, batch_size=128)
        val_probs = []
        val_preds = []
        with torch.no_grad():
            for seq_batch, labels in val_loader:
                seq_batch = seq_batch.to(device)
                logits = model(seq_batch)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                val_probs.append(probs)
                val_preds.append(probs.argmax(axis=1))
        val_probs = np.vstack(val_probs)
        oof[val_idx] = val_probs
        fold_acc = accuracy_score(y[val_idx], np.concatenate(val_preds))
        scores.append(fold_acc)
        print(f"GRU fold {fold} accuracy: {fold_acc:.4f}")

        test_loader = DataLoader(SequenceDataset(test_sequences), batch_size=128)
        fold_test_probs = []
        with torch.no_grad():
            for seq_batch in test_loader:
                seq_batch = seq_batch.to(device)
                logits = model(seq_batch)
                fold_test_probs.append(torch.softmax(logits, dim=1).cpu().numpy())
        test_probs += np.vstack(fold_test_probs) / skf.n_splits

    return oof, test_probs, scores


def add_session_ids(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df["session_id"] = infer_sessions(train_df)
    test_df["session_id"] = infer_sessions(test_df)
    return train_df, test_df


def main() -> None:
    train_df, test_df = load_data()

    tree_train = add_engineered_features(train_df)
    tree_test = add_engineered_features(test_df)
    tree_train, tree_test = compute_target_encoding(tree_train, tree_test, TARGET_ENCODING_COLS)

    tree_oof, tree_test_probs, tree_scores = train_lightgbm_with_probs(tree_train, tree_test)
    print("LightGBM CV accuracies:", [round(s, 4) for s in tree_scores])
    print("LightGBM mean CV accuracy:", round(float(np.mean(tree_scores)), 4))

    seq_train_base = add_engineered_features(train_df)
    seq_test_base = add_engineered_features(test_df)
    seq_train, seq_test = add_session_ids(seq_train_base, seq_test_base)
    y = train_df["risk_level"].astype(int) - 1
    gru_oof, gru_test_probs, gru_scores = train_gru_model(seq_train, seq_test, y.values)
    print("GRU CV accuracies:", [round(s, 4) for s in gru_scores])
    print("GRU mean CV accuracy:", round(float(np.mean(gru_scores)), 4))

    best_weight = 0.0
    best_acc = 0.0
    weights = np.linspace(0.0, 1.0, 101)
    for w in weights:
        blended = w * gru_oof + (1 - w) * tree_oof
        preds = blended.argmax(axis=1)
        acc = accuracy_score(y, preds)
        if acc > best_acc:
            best_acc = acc
            best_weight = w
    print(f"Best blend weight (GRU share): {best_weight:.2f} with OOF accuracy {best_acc:.4f}")

    final_probs = best_weight * gru_test_probs + (1 - best_weight) * tree_test_probs
    final_preds = final_probs.argmax(axis=1) + 1
    submission = pd.DataFrame({"id": np.arange(len(final_preds)), "risk_level": final_preds})
    out_path = Path("driving_risk_submission_gru_blend.csv")
    submission.to_csv(out_path, index=False)
    print(f"Saved submission to {out_path.resolve()}")


if __name__ == "__main__":
    main()
