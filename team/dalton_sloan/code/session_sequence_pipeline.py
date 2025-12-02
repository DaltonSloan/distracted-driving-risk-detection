"""Advanced modeling pipeline with pseudo-session reconstruction, sequence
models, diverse ensembles, and staged semi-supervised learning."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import lightgbm as lgb
from xgboost import XGBClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

DATA_DIR = Path(__file__).resolve().parents[3] / "data"


NUMERIC_COLUMNS = [
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

SEQ_COLUMNS = [
    "speed",
    "rpm",
    "acceleration",
    "throttle_position",
    "engine_load_value",
    "heart_rate",
    "visibility",
    "precipitation",
    "design_speed",
    "accidents_time",
]


def infer_sessions(df: pd.DataFrame) -> pd.Series:
    session_ids: List[int] = []
    session_id = 0
    prev = None
    for row in df.itertuples(index=False):
        if prev is None:
            session_ids.append(session_id)
            prev = row
            continue
        speed_jump = abs(row.speed - prev.speed) > 45 and min(row.speed, prev.speed) < 5
        visibility_jump = abs(row.visibility - prev.visibility) > 6
        precip_jump = abs(row.precipitation - prev.precipitation) > 3
        design_jump = abs(row.design_speed - prev.design_speed) > 20
        accidents_jump = abs(row.accidents_onsite - prev.accidents_onsite) > 15
        hour_wrap = (row.observation_hour + 24 - prev.observation_hour) % 24 > 6
        if (
            speed_jump
            or visibility_jump
            or precip_jump
            or design_jump
            or accidents_jump
            or hour_wrap
        ):
            session_id += 1
        session_ids.append(session_id)
        prev = row
    return pd.Series(session_ids, index=df.index, name="session_id")


def add_session_features(df: pd.DataFrame, session_ids: pd.Series) -> pd.DataFrame:
    df = df.copy()
    df["session_id"] = session_ids.values
    df["session_pos"] = df.groupby("session_id").cumcount()
    df["session_len"] = df.groupby("session_id")["session_id"].transform("count")
    df["session_pos_frac"] = df["session_pos"] / df["session_len"].clip(lower=1)
    df["session_rev_pos"] = df["session_len"] - df["session_pos"] - 1

    rolling_windows = [3, 5, 9, 15, 30, 50]
    for window in rolling_windows:
        grouped = df.groupby("session_id", sort=False)
        for col in [
            "speed",
            "acceleration",
            "heart_rate",
            "visibility",
            "precipitation",
            "rpm",
        ]:
            roll = (
                grouped[col]
                .rolling(window=window, min_periods=1)
                .mean()
                .reset_index(level=0, drop=True)
            )
            df[f"{col}_roll_mean_{window}"] = roll
            roll_std = (
                grouped[col]
                .rolling(window=window, min_periods=1)
                .std()
                .reset_index(level=0, drop=True)
            )
            df[f"{col}_roll_std_{window}"] = roll_std.fillna(0.0)

    df["speed_cumsum"] = df.groupby("session_id")["speed"].cumsum()
    df["accel_cumsum"] = df.groupby("session_id")["acceleration"].cumsum()
    df["heart_rate_cumsum"] = df.groupby("session_id")["heart_rate"].cumsum()
    df["precip_cumsum"] = df.groupby("session_id")["precipitation"].cumsum()

    df = df.fillna(0.0)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    base = df.copy().reset_index(drop=True)
    for col in NUMERIC_COLUMNS:
        base[col] = base[col].astype(float)
    if "label_source" in base.columns:
        base["label_source_flag"] = base["label_source"].eq("expert_verified").astype(float)
        base = base.drop(columns=["label_source"])
    session_ids = infer_sessions(df)
    enriched = add_session_features(base, session_ids)
    hour_rad = enriched["observation_hour"] / 24.0 * (2.0 * np.pi)
    enriched["hour_sin"] = np.sin(hour_rad)
    enriched["hour_cos"] = np.cos(hour_rad)
    enriched["speed_design_ratio"] = enriched["speed"] / (enriched["design_speed"] + 1.0)
    enriched["heart_rate_speed_ratio"] = enriched["heart_rate"] / (enriched["speed"] + 1.0)
    enriched["visibility_precip_ratio"] = enriched["visibility"] / (
        enriched["precipitation"] + 1.0
    )
    enriched["rpm_load_ratio"] = enriched["rpm"] / (enriched["engine_load_value"] + 1.0)
    enriched["stress_proxy"] = enriched["heart_rate"] * (
        enriched["acceleration"].abs() + 1.0
    ) / (enriched["visibility"] + 1.0)
    enriched = enriched.fillna(0.0)
    return enriched


def train_tree_stack(
    X: pd.DataFrame, y: np.ndarray, X_test: pd.DataFrame
) -> Tuple[np.ndarray, Dict[str, List[float]]]:
    n_classes = len(np.unique(y))
    skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
    oof = {name: np.zeros((len(X), n_classes)) for name in ["hgb", "lgb", "xgb"]}
    fold_scores: Dict[str, List[float]] = {name: [] for name in oof}

    templates = {
        "hgb": HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=250,
            max_leaf_nodes=64,
            max_depth=10,
            min_samples_leaf=20,
            l2_regularization=0.2,
            early_stopping=False,
            random_state=7,
        ),
        "lgb": lgb.LGBMClassifier(
            objective="multiclass",
            num_class=n_classes,
            learning_rate=0.05,
            n_estimators=500,
            num_leaves=80,
            colsample_bytree=0.85,
            subsample=0.9,
            subsample_freq=1,
            reg_lambda=0.4,
            reg_alpha=0.2,
            min_child_samples=24,
            random_state=11,
            n_jobs=-1,
            verbosity=-1,
        ),
        "xgb": XGBClassifier(
            objective="multi:softprob",
            num_class=n_classes,
            learning_rate=0.05,
            n_estimators=200,
            max_depth=7,
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
        for name, template in templates.items():
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
            oof[name][val_idx] = probas
            preds = probas.argmax(axis=1)
            fold_scores[name].append(accuracy_score(y_val, preds))
            print(f"Tree fold {fold} {name} acc={fold_scores[name][-1]:.4f}")

    meta_X = np.hstack([oof[name] for name in ["hgb", "lgb", "xgb"]])
    meta_model = LogisticRegression(
        max_iter=300,
        C=2.0,
        multi_class="multinomial",
        solver="lbfgs",
    )
    meta_model.fit(meta_X, y)
    print("Stack meta-model OOF accuracy", accuracy_score(y, meta_model.predict(meta_X)))
    print(classification_report(y + 1, meta_model.predict(meta_X) + 1, digits=3))

    final_models = {
        name: template.__class__(**template.get_params()) for name, template in templates.items()
    }
    test_meta_parts = []
    for name, model in final_models.items():
        model.fit(X, y)
        test_meta_parts.append(model.predict_proba(X_test))
    test_meta = np.hstack(test_meta_parts)
    stack_test_probas = meta_model.predict_proba(test_meta)
    return stack_test_probas, fold_scores


def build_sequences(
    features: np.ndarray, session_ids: np.ndarray, seq_len: int
) -> Tuple[np.ndarray, np.ndarray]:
    sequences = np.zeros((len(features), seq_len, features.shape[1]), dtype=np.float32)
    lengths = np.zeros(len(features), dtype=np.int64)
    session_to_indices: Dict[int, List[int]] = {}
    for idx, sid in enumerate(session_ids):
        session_to_indices.setdefault(int(sid), []).append(idx)
    for sid, indices in session_to_indices.items():
        for pos, idx in enumerate(indices):
            start = max(0, pos - seq_len + 1)
            seq_indices = indices[start : pos + 1]
            length = len(seq_indices)
            lengths[idx] = length
            sequences[idx, seq_len - length :, :] = features[seq_indices]
    return sequences, lengths


class SequenceDataset(Dataset):
    def __init__(
        self,
        sequences: np.ndarray,
        lengths: np.ndarray,
        targets: np.ndarray | None,
    ) -> None:
        self.sequences = sequences
        self.lengths = lengths
        self.targets = targets

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int):
        seq = self.sequences[idx]
        length = self.lengths[idx]
        if self.targets is None:
            return seq, length
        return seq, length, self.targets[idx]


class GRUClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_dim * 2, 4)

    def forward(self, seq: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        packed = nn.utils.rnn.pack_padded_sequence(
            seq, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, hidden = self.gru(packed)
        if isinstance(hidden, torch.Tensor):
            hidden_cat = torch.cat([hidden[-2], hidden[-1]], dim=1)
        else:
            hidden_cat = torch.cat([hidden[0][-2], hidden[0][-1]], dim=1)
        hidden_cat = self.dropout(hidden_cat)
        return self.fc(hidden_cat)


def train_gru_sequences(
    train_sequences: np.ndarray,
    train_lengths: np.ndarray,
    y: np.ndarray,
    test_sequences: np.ndarray,
    test_lengths: np.ndarray,
    epochs: int = 3,
    batch_size: int = 64,
    lr: float = 1e-3,
) -> np.ndarray:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GRUClassifier(input_dim=train_sequences.shape[2]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    rng = np.random.RandomState(42)
    indices = np.arange(len(train_sequences))
    rng.shuffle(indices)
    split = int(0.85 * len(indices))
    train_idx = indices[:split]
    val_idx = indices[split:]

    train_data = SequenceDataset(
        train_sequences[train_idx], train_lengths[train_idx], y[train_idx]
    )
    val_data = SequenceDataset(
        train_sequences[val_idx], train_lengths[val_idx], y[val_idx]
    )
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)

    for epoch in range(epochs):
        model.train()
        for seq, lengths, targets in train_loader:
            seq = seq.to(device)
            lengths = lengths.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            logits = model(seq, lengths)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for seq, lengths, targets in val_loader:
                seq = seq.to(device)
                lengths = lengths.to(device)
                targets = targets.to(device)
                logits = model(seq, lengths)
                loss = criterion(logits, targets)
                val_losses.append(loss.item())
        print(
            f"GRU epoch {epoch+1}/{epochs} val loss={np.mean(val_losses):.4f}", flush=True
        )

    test_loader = DataLoader(
        SequenceDataset(test_sequences, test_lengths, None), batch_size=batch_size
    )
    preds = []
    model.eval()
    with torch.no_grad():
        for seq, lengths in test_loader:
            seq = seq.to(device)
            lengths = lengths.to(device)
            logits = model(seq, lengths)
            preds.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.vstack(preds)


def staged_pseudo_training(
    X_train: pd.DataFrame,
    y: np.ndarray,
    X_test: pd.DataFrame,
    base_probas: np.ndarray,
    thresholds: Sequence[Tuple[float, float]] = ((0.985, 0.5), (0.95, 0.3), (0.9, 0.15)),
) -> np.ndarray:
    pseudo_frames = []
    pseudo_labels = []
    pseudo_weights = []
    for thresh, weight in thresholds:
        mask = base_probas.max(axis=1) >= thresh
        count = int(mask.sum())
        if count == 0:
            continue
        pseudo_frames.append(X_test[mask].copy())
        pseudo_labels.append(base_probas[mask].argmax(axis=1))
        pseudo_weights.append(np.full(count, weight))
        print(f"Pseudo stage threshold {thresh} added {count} samples with weight {weight}")

    if pseudo_frames:
        pseudo_X = pd.concat(pseudo_frames, axis=0).reset_index(drop=True)
        pseudo_y = np.concatenate(pseudo_labels)
        sample_weights = np.concatenate(pseudo_weights)
        X_aug = pd.concat([X_train, pseudo_X], axis=0).reset_index(drop=True)
        y_aug = np.concatenate([y, pseudo_y])
        weights_aug = np.concatenate([np.ones(len(y)), sample_weights])
    else:
        X_aug = X_train.copy()
        y_aug = y.copy()
        weights_aug = np.ones(len(y))

    params = dict(
        objective="multiclass",
        num_class=4,
        learning_rate=0.03,
        n_estimators=1000,
        num_leaves=96,
        max_depth=-1,
        subsample=0.9,
        subsample_freq=1,
        colsample_bytree=0.9,
        reg_alpha=0.4,
        reg_lambda=0.7,
        min_child_samples=20,
        random_state=2027,
        n_jobs=-1,
        verbosity=-1,
    )

    model = lgb.LGBMClassifier(**params)
    model.fit(X_aug, y_aug, sample_weight=weights_aug)
    final_preds = model.predict_proba(X_test)
    return final_preds


def main() -> None:
    train_df = pd.read_csv(DATA_DIR / "kaggle_train.csv")
    test_df = pd.read_csv(DATA_DIR / "kaggle_test.csv")

    y = train_df["risk_level"].astype(int).values - 1
    train_features = engineer_features(train_df.drop(columns=["risk_level"]))
    test_features = engineer_features(test_df)
    print("Finished feature engineering", flush=True)

    tree_test_probas, fold_scores = train_tree_stack(train_features, y, test_features)
    print("Tree fold accuracies:", flush=True)
    for name, scores in fold_scores.items():
        print(name, [round(s, 4) for s in scores], flush=True)

    scaler = StandardScaler()
    seq_train = scaler.fit_transform(train_features[SEQ_COLUMNS])
    seq_test = scaler.transform(test_features[SEQ_COLUMNS])
    print("Built sequence matrices", flush=True)

    train_sequences, train_lengths = build_sequences(
        seq_train.astype(np.float32), train_features["session_id"].values.astype(int), seq_len=48
    )
    test_sequences, test_lengths = build_sequences(
        seq_test.astype(np.float32), test_features["session_id"].values.astype(int), seq_len=48
    )

    gru_test_probas = train_gru_sequences(
        train_sequences, train_lengths, y, test_sequences, test_lengths
    )
    print("Finished GRU training", flush=True)

    blended_base = 0.6 * tree_test_probas + 0.4 * gru_test_probas
    semi_preds = staged_pseudo_training(
        train_features, y, test_features, blended_base
    )
    print("Completed staged pseudo-label training", flush=True)

    final_blend = 0.6 * semi_preds + 0.4 * gru_test_probas
    final_preds = final_blend.argmax(axis=1) + 1

    submission = pd.DataFrame({"id": np.arange(len(final_preds)), "risk_level": final_preds})
    out_path = Path("driving_risk_submission_v9.csv")
    submission.to_csv(out_path, index=False)
    print("Saved submission to", out_path.resolve(), flush=True)
    print("Prediction distribution:\n", submission["risk_level"].value_counts().sort_index(), flush=True)


if __name__ == "__main__":
    main()
