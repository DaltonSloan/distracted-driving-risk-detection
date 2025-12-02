"""High-capacity hybrid pipeline that couples advanced feature engineering,
stacked tree ensembles with semi-supervised fine-tuning, and a contextual
GRU sequence model. Designed to push leaderboard performance beyond 0.92."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import lightgbm as lgb
from xgboost import XGBClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


np.random.seed(42)
torch.manual_seed(42)

DATA_DIR = Path(__file__).resolve().parents[3] / "data"


BASE_NUMERIC_COLUMNS = [
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

TARGET_ENCODING_COLUMNS = [
    "observation_hour",
    "current_weather",
    "design_speed",
    "accidents_onsite",
]

SESSION_ROLL_COLUMNS = [
    "speed",
    "acceleration",
    "heart_rate",
    "visibility",
    "precipitation",
    "rpm",
]

SEQ_FEATURES = [
    "speed",
    "rpm",
    "acceleration",
    "throttle_position",
    "engine_load_value",
    "engine_temperature",
    "heart_rate",
    "current_weather",
    "visibility",
    "precipitation",
    "accidents_onsite",
    "design_speed",
    "accidents_time",
    "speed_to_design",
    "speed_minus_design",
    "rpm_per_speed",
    "throttle_load_ratio",
    "stress_proxy",
]

CONTEXT_FEATURES = [
    "speed_to_design",
    "speed_minus_design",
    "rpm_per_speed",
    "rpm_per_load",
    "engine_stress",
    "heart_rate_dev",
    "stress_proxy",
    "visibility_precip_ratio",
    "session_pos_frac",
    "session_rev_pos",
    "session_len",
    "speed_roll_mean_5",
    "acceleration_roll_std_5",
    "hour_sin",
    "hour_cos",
]

SEQ_LEN = 64


@dataclass
class PseudoSource:
    features: pd.DataFrame
    probabilities: np.ndarray
    weight: float


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    train = pd.read_csv(DATA_DIR / "kaggle_train.csv")
    test = pd.read_csv(DATA_DIR / "kaggle_test.csv")
    unlabeled_path = DATA_DIR / "kaggle_full_unlabeled_data.csv"
    unlabeled = None
    if unlabeled_path.exists():
        unlabeled = pd.read_csv(unlabeled_path)
        if "risk_level" in unlabeled.columns:
            unlabeled = unlabeled.drop(columns=["risk_level"])
        if "Unnamed: 0" in unlabeled.columns:
            unlabeled = unlabeled.drop(columns=["Unnamed: 0"])
    return train, test, unlabeled


def ensure_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    data = df.copy()
    for col in columns:
        if col not in data.columns:
            data[col] = 0.0
    return data


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
        precip_jump = abs(row.precipitation - prev.precipitation) > 4
        design_jump = abs(row.design_speed - prev.design_speed) > 20
        accidents_jump = abs(row.accidents_onsite - prev.accidents_onsite) > 12
        hour_gap = (row.observation_hour + 24 - prev.observation_hour) % 24 > 6
        if speed_jump or visibility_jump or precip_jump or design_jump or accidents_jump or hour_gap:
            session_id += 1
        session_ids.append(session_id)
        prev = row
    return pd.Series(session_ids, index=df.index, name="session_id")


def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    for col in BASE_NUMERIC_COLUMNS:
        data[col] = data[col].astype(float)

    hour_angle = data["observation_hour"] / 24.0 * (2.0 * np.pi)
    data["hour_sin"] = np.sin(hour_angle)
    data["hour_cos"] = np.cos(hour_angle)

    data["speed_to_design"] = data["speed"] / (data["design_speed"] + 1.0)
    data["speed_minus_design"] = data["speed"] - data["design_speed"]
    data["speed_plus_design"] = data["speed"] + data["design_speed"]
    data["speed_visibility_ratio"] = data["speed"] / (data["visibility"] + 1.0)
    data["speed_precip_ratio"] = data["speed"] / (data["precipitation"] + 1.0)

    data["rpm_per_speed"] = data["rpm"] / (data["speed"] + 1.0)
    data["rpm_per_load"] = data["rpm"] / (data["engine_load_value"] + 1.0)
    data["rpm_temp_ratio"] = data["rpm"] / (data["engine_temperature"] + 1.0)

    data["throttle_load_ratio"] = data["throttle_position"] / (data["engine_load_value"] + 1.0)
    data["throttle_temp_ratio"] = data["throttle_position"] / (data["engine_temperature"] + 1.0)

    data["engine_stress"] = data["engine_load_value"] * data["engine_temperature"]
    data["engine_temp_minus_load"] = data["engine_temperature"] - data["engine_load_value"]

    data["acceleration_abs"] = data["acceleration"].abs()
    data["accel_throttle"] = data["acceleration"] * data["throttle_position"]
    data["accel_speed_ratio"] = data["acceleration"] / (data["speed"] + 1.0)

    accidents_total = data["accidents_onsite"] + data["accidents_time"]
    data["accidents_total"] = accidents_total
    data["accident_hour_ratio"] = accidents_total / (data["observation_hour"] + 1.0)
    data["precip_accidents"] = data["precipitation"] * accidents_total

    data["heart_rate_dev"] = data["heart_rate"] - data["heart_rate"].median()
    data["heart_rate_speed_ratio"] = data["heart_rate"] / (data["speed"] + 1.0)
    data["heart_over_design"] = data["heart_rate"] / (data["design_speed"] + 1.0)

    data["visibility_precip_ratio"] = data["visibility"] / (data["precipitation"] + 1.0)
    data["precip_speed"] = data["precipitation"] * data["speed"]
    data["weather_visibility"] = data["current_weather"] * data["visibility"]
    data["weather_precip"] = data["current_weather"] * data["precipitation"]

    data["stress_proxy"] = data["heart_rate"] * (data["acceleration"].abs() + 1.0) / (data["visibility"] + 1.0)

    data["is_raining"] = (data["precipitation"] > 0).astype(float)
    data["low_visibility"] = (data["visibility"] < 4).astype(float)
    data["high_speeding"] = (data["speed"] > data["design_speed"]).astype(float)

    for col in BASE_NUMERIC_COLUMNS:
        data[f"{col}_rank"] = data[col].rank(pct=True).astype(float)

    return data


def add_session_features(df: pd.DataFrame, session_ids: pd.Series) -> pd.DataFrame:
    data = df.copy()
    data["session_id"] = session_ids.values
    data = data.sort_index().copy()
    data["session_pos"] = data.groupby("session_id").cumcount()
    data["session_len"] = data.groupby("session_id")["session_id"].transform("count")
    data["session_pos_frac"] = data["session_pos"] / data["session_len"].clip(lower=1)
    data["session_rev_pos"] = data["session_len"] - data["session_pos"] - 1

    grouped = data.groupby("session_id", sort=False)
    for col in SESSION_ROLL_COLUMNS:
        lag1 = grouped[col].shift(1)
        lag2 = grouped[col].shift(2)
        data[f"{col}_lag1"] = lag1
        data[f"{col}_lag2"] = lag2
        data[f"{col}_diff1"] = data[col] - lag1
        for window in (3, 5, 9, 15):
            roll_mean = (
                grouped[col]
                .rolling(window=window, min_periods=1)
                .mean()
                .reset_index(level=0, drop=True)
            )
            data[f"{col}_roll_mean_{window}"] = roll_mean
            roll_std = (
                grouped[col]
                .rolling(window=window, min_periods=1)
                .std()
                .reset_index(level=0, drop=True)
            )
            data[f"{col}_roll_std_{window}"] = roll_std.fillna(0.0)
        data[f"{col}_session_mean"] = grouped[col].transform("mean")
        data[f"{col}_session_std"] = grouped[col].transform("std").fillna(0.0)

    data["speed_cumsum"] = grouped["speed"].cumsum()
    data["accel_cumsum"] = grouped["acceleration"].cumsum()
    data["heart_cumsum"] = grouped["heart_rate"].cumsum()
    data["precip_cumsum"] = grouped["precipitation"].cumsum()

    data = data.fillna(0.0)
    return data


def engineer_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
    keep_cols = [col for col in df.columns if col in BASE_NUMERIC_COLUMNS or col == "label_source"]
    base = df[keep_cols].copy().reset_index(drop=True)
    base = ensure_columns(base, BASE_NUMERIC_COLUMNS)
    if "label_source" in base.columns:
        base["label_source_flag"] = base["label_source"].eq("expert_verified").astype(float)
        base = base.drop(columns=["label_source"])
    else:
        base["label_source_flag"] = 0.0
    session_ids = infer_sessions(base)
    enriched = add_basic_features(base)
    enriched = add_session_features(enriched, session_ids)
    enriched = enriched.fillna(0.0)
    return enriched, session_ids.values


def compute_target_encoding(
    train_df: pd.DataFrame,
    target: np.ndarray,
    other_dfs: Sequence[pd.DataFrame],
    cols: Sequence[str],
) -> Tuple[pd.DataFrame, List[pd.DataFrame]]:
    train = train_df.copy()
    others = [df.copy() for df in other_dfs]
    y = target.astype(float)
    global_mean = float(y.mean())
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for col in cols:
        te_col = f"te_{col}"
        train[te_col] = 0.0
        for train_idx, val_idx in skf.split(train, target):
            fold = train.iloc[train_idx]
            mapping = fold.groupby(col)["risk_level"].mean()
            mapped = train.iloc[val_idx][col].map(mapping).fillna(global_mean)
            train.iloc[val_idx, train.columns.get_loc(te_col)] = mapped
        full_mapping = train.groupby(col)["risk_level"].mean()
        for other in others:
            other[te_col] = other[col].map(full_mapping).fillna(global_mean)
    return train, others


def domain_weights(train_df: pd.DataFrame, test_df: pd.DataFrame) -> np.ndarray:
    features = BASE_NUMERIC_COLUMNS + [
        "speed_to_design",
        "speed_minus_design",
        "rpm_per_speed",
        "rpm_per_load",
        "stress_proxy",
    ]
    train_feats = train_df[features].fillna(0.0)
    test_feats = test_df[features].fillna(0.0)
    combo = pd.concat([train_feats, test_feats], axis=0).reset_index(drop=True)
    indicator = np.concatenate([np.ones(len(train_feats)), np.zeros(len(test_feats))])
    scaler = StandardScaler()
    combo_scaled = scaler.fit_transform(combo)
    clf = LogisticRegression(max_iter=1000, multi_class="ovr")
    clf.fit(combo_scaled, indicator)
    train_probs = clf.predict_proba(combo_scaled[: len(train_feats)])[:, 1]
    weights = (1 - train_probs) / (train_probs + 1e-3)
    return np.clip(weights, 0.2, 5.0)


def train_tree_stack(
    X: pd.DataFrame,
    y: np.ndarray,
    X_test: pd.DataFrame,
    X_unlabeled: pd.DataFrame | None,
    sample_weights: np.ndarray,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray | None,
    Dict[str, List[float]],
]:
    n_classes = len(np.unique(y))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    base_templates = {
        "hgb": HistGradientBoostingClassifier(
            learning_rate=0.055,
            max_iter=350,
            max_depth=9,
            max_leaf_nodes=64,
            min_samples_leaf=20,
            l2_regularization=0.2,
            early_stopping=False,
            random_state=7,
        ),
        "lgb": lgb.LGBMClassifier(
            objective="multiclass",
            num_class=n_classes,
            learning_rate=0.04,
            n_estimators=1400,
            num_leaves=72,
            subsample=0.9,
            subsample_freq=1,
            colsample_bytree=0.85,
            reg_alpha=0.2,
            reg_lambda=0.45,
            min_child_samples=24,
            random_state=11,
            n_jobs=-1,
            verbosity=-1,
        ),
        "xgb": XGBClassifier(
            objective="multi:softprob",
            num_class=n_classes,
            learning_rate=0.05,
            n_estimators=400,
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

    oof = {name: np.zeros((len(X), n_classes)) for name in base_templates}
    fold_scores: Dict[str, List[float]] = {name: [] for name in base_templates}
    test_preds = {name: np.zeros((len(X_test), n_classes)) for name in base_templates}
    unl_preds = (
        {name: np.zeros((len(X_unlabeled), n_classes)) for name in base_templates}
        if X_unlabeled is not None
        else None
    )

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        w_train = sample_weights[train_idx]
        for name, template in base_templates.items():
            model = clone(template)
            if name == "lgb":
                model.fit(
                    X_train,
                    y_train,
                    sample_weight=w_train,
                    eval_set=[(X_val, y_val)],
                    eval_metric="multi_logloss",
                    callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)],
                )
            elif name == "xgb":
                model.fit(
                    X_train,
                    y_train,
                    sample_weight=w_train,
                    verbose=False,
                )
            else:
                model.fit(X_train, y_train, sample_weight=w_train)
            probas = model.predict_proba(X_val)
            oof[name][val_idx] = probas
            preds = probas.argmax(axis=1)
            acc = accuracy_score(y_val, preds)
            fold_scores[name].append(acc)
            print(f"Fold {fold} {name} accuracy={acc:.4f}")
            test_preds[name] += model.predict_proba(X_test) / skf.n_splits
            if unl_preds is not None and X_unlabeled is not None:
                unl_preds[name] += model.predict_proba(X_unlabeled) / skf.n_splits

    meta_X = np.hstack([oof[name] for name in ["hgb", "lgb", "xgb"]])
    meta_model = LogisticRegression(max_iter=600, C=2.0, multi_class="multinomial", solver="lbfgs")
    meta_model.fit(meta_X, y)
    meta_preds = meta_model.predict(meta_X)
    print("Tree stack meta OOF accuracy:", round(accuracy_score(y, meta_preds), 4))
    print(classification_report(y + 1, meta_preds + 1, digits=3))

    test_stack_input = np.hstack([test_preds[name] for name in ["hgb", "lgb", "xgb"]])
    stack_test = meta_model.predict_proba(test_stack_input)

    if unl_preds is not None and X_unlabeled is not None:
        unl_stack_input = np.hstack([unl_preds[name] for name in ["hgb", "lgb", "xgb"]])
        stack_unlabeled = meta_model.predict_proba(unl_stack_input)
    else:
        stack_unlabeled = None

    stack_oof = meta_model.predict_proba(meta_X)
    return stack_oof, stack_test, stack_unlabeled, fold_scores


def train_semi_supervised_lightgbm(
    X: pd.DataFrame,
    y: np.ndarray,
    X_test: pd.DataFrame,
    pseudo_sources: Sequence[PseudoSource],
) -> Tuple[np.ndarray, np.ndarray, List[float]]:
    n_classes = len(np.unique(y))
    params = dict(
        objective="multiclass",
        num_class=n_classes,
        learning_rate=0.032,
        n_estimators=1600,
        num_leaves=84,
        max_depth=-1,
        subsample=0.9,
        subsample_freq=1,
        colsample_bytree=0.9,
        reg_alpha=0.3,
        reg_lambda=0.6,
        min_child_samples=24,
        random_state=105,
        n_jobs=-1,
        verbosity=-1,
    )
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros((len(X), n_classes))
    test_probs = np.zeros((len(X_test), n_classes))
    fold_scores: List[float] = []

    pseudo_features: List[pd.DataFrame] = []
    pseudo_labels: List[np.ndarray] = []
    pseudo_weights: List[np.ndarray] = []
    for source in pseudo_sources:
        probs = source.probabilities
        if probs is None or len(probs) == 0:
            continue
        confidences = probs.max(axis=1)
        mask = confidences >= 0.985
        if source.weight < 0.2:
            mask = confidences >= 0.995
        if not np.any(mask):
            continue
        pseudo_features.append(source.features.iloc[mask].reset_index(drop=True))
        pseudo_labels.append(probs[mask].argmax(axis=1))
        pseudo_weights.append(np.full(mask.sum(), source.weight))
        print(f"Pseudo source retained {mask.sum()} rows with weight {source.weight}")

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        X_train = [X.iloc[train_idx].reset_index(drop=True)] + pseudo_features
        y_train_list = [y[train_idx]] + pseudo_labels
        weights_list = [np.ones(len(train_idx))] + pseudo_weights
        X_fold = pd.concat(X_train, axis=0).reset_index(drop=True)
        y_fold = np.concatenate(y_train_list)
        w_fold = np.concatenate(weights_list)
        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_fold,
            y_fold,
            sample_weight=w_fold,
            eval_set=[(X.iloc[val_idx], y[val_idx])],
            eval_metric="multi_logloss",
            callbacks=[lgb.early_stopping(150), lgb.log_evaluation(0)],
        )
        val_probas = model.predict_proba(X.iloc[val_idx])
        oof[val_idx] = val_probas
        preds = val_probas.argmax(axis=1)
        acc = accuracy_score(y[val_idx], preds)
        fold_scores.append(acc)
        print(f"Semi-supervised fold {fold} acc={acc:.4f}")
        test_probs += model.predict_proba(X_test) / skf.n_splits

    return oof, test_probs, fold_scores


def build_sequences(features: np.ndarray, session_ids: np.ndarray, seq_len: int) -> np.ndarray:
    n_samples, feat_dim = features.shape
    sequences = np.zeros((n_samples, seq_len, feat_dim), dtype=np.float32)
    session_map: Dict[int, List[int]] = {}
    for idx, sid in enumerate(session_ids):
        session_map.setdefault(int(sid), []).append(idx)
    for indices in session_map.values():
        for pos, idx in enumerate(indices):
            start = max(0, pos - seq_len + 1)
            seq_idx = indices[start : pos + 1]
            length = len(seq_idx)
            sequences[idx, -length:, :] = features[seq_idx]
    return sequences


def prepare_sequence_inputs(
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
    train_sessions: np.ndarray,
    test_sessions: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    seq_cols = [col for col in SEQ_FEATURES if col in train_features.columns]
    ctx_cols = [col for col in CONTEXT_FEATURES if col in train_features.columns]

    scaler = StandardScaler()
    scaler.fit(pd.concat([train_features[seq_cols], test_features[seq_cols]], axis=0))
    train_seq_scaled = scaler.transform(train_features[seq_cols]).astype(np.float32)
    test_seq_scaled = scaler.transform(test_features[seq_cols]).astype(np.float32)

    if ctx_cols:
        ctx_scaler = StandardScaler()
        ctx_scaler.fit(pd.concat([train_features[ctx_cols], test_features[ctx_cols]], axis=0))
        train_ctx = ctx_scaler.transform(train_features[ctx_cols]).astype(np.float32)
        test_ctx = ctx_scaler.transform(test_features[ctx_cols]).astype(np.float32)
    else:
        train_ctx = np.zeros((len(train_features), 0), dtype=np.float32)
        test_ctx = np.zeros((len(test_features), 0), dtype=np.float32)

    train_sequences = build_sequences(train_seq_scaled, train_sessions, SEQ_LEN)
    test_sequences = build_sequences(test_seq_scaled, test_sessions, SEQ_LEN)
    return train_sequences, train_ctx, test_sequences, test_ctx


class SeqCtxDataset(Dataset):
    def __init__(self, sequences: np.ndarray, contexts: np.ndarray, labels: np.ndarray | None = None) -> None:
        self.sequences = sequences
        self.contexts = contexts
        self.labels = labels

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int):
        seq = torch.from_numpy(self.sequences[idx])
        ctx = torch.from_numpy(self.contexts[idx])
        if self.labels is None:
            return seq, ctx
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return seq, ctx, label


class HybridGRUClassifier(nn.Module):
    def __init__(self, input_dim: int, context_dim: int, hidden_dim: int = 128, num_layers: int = 3) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.25,
            bidirectional=True,
        )
        self.context_bn = nn.BatchNorm1d(context_dim) if context_dim > 0 else None
        fusion_dim = hidden_dim * 2 + context_dim
        self.fc = nn.Sequential(
            nn.Linear(fusion_dim, 192),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(192, 4),
        )

    def forward(self, seq: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        _, hidden = self.gru(seq)
        rep = torch.cat([hidden[-2], hidden[-1]], dim=1)
        if self.context_bn is not None:
            ctx = self.context_bn(ctx)
        fused = torch.cat([rep, ctx], dim=1)
        return self.fc(fused)


def train_hybrid_gru(
    train_sequences: np.ndarray,
    train_contexts: np.ndarray,
    test_sequences: np.ndarray,
    test_contexts: np.ndarray,
    y: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, List[float]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros((len(train_sequences), 4), dtype=np.float32)
    test_probs = np.zeros((len(test_sequences), 4), dtype=np.float32)
    scores: List[float] = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(train_sequences, y), start=1):
        model = HybridGRUClassifier(train_sequences.shape[2], train_contexts.shape[1]).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.6, patience=2
        )
        criterion = nn.CrossEntropyLoss()

        train_data = SeqCtxDataset(train_sequences[train_idx], train_contexts[train_idx], y[train_idx])
        val_data = SeqCtxDataset(train_sequences[val_idx], train_contexts[val_idx], y[val_idx])
        train_loader = DataLoader(train_data, batch_size=96, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=192)

        best_state = None
        best_loss = float("inf")
        patience = 4
        wait = 0
        for epoch in range(20):
            model.train()
            for seq_batch, ctx_batch, labels in train_loader:
                seq_batch = seq_batch.to(device)
                ctx_batch = ctx_batch.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                logits = model(seq_batch, ctx_batch)
                loss = criterion(logits, labels)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            model.eval()
            val_losses: List[float] = []
            val_preds: List[np.ndarray] = []
            with torch.no_grad():
                for seq_batch, ctx_batch, labels in val_loader:
                    seq_batch = seq_batch.to(device)
                    ctx_batch = ctx_batch.to(device)
                    labels = labels.to(device)
                    logits = model(seq_batch, ctx_batch)
                    loss = criterion(logits, labels)
                    val_losses.append(loss.item())
                    val_preds.append(torch.softmax(logits, dim=1).cpu().numpy())
            mean_loss = float(np.mean(val_losses))
            scheduler.step(mean_loss)
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
        val_loader = DataLoader(val_data, batch_size=192)
        val_probs = []
        val_pred_labels = []
        with torch.no_grad():
            for seq_batch, ctx_batch, labels in val_loader:
                seq_batch = seq_batch.to(device)
                ctx_batch = ctx_batch.to(device)
                logits = model(seq_batch, ctx_batch)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                val_probs.append(probs)
                val_pred_labels.append(probs.argmax(axis=1))
        fold_probs = np.vstack(val_probs)
        oof[val_idx] = fold_probs
        fold_acc = accuracy_score(y[val_idx], np.concatenate(val_pred_labels))
        scores.append(fold_acc)
        print(f"GRU fold {fold} accuracy: {fold_acc:.4f}")

        test_loader = DataLoader(SeqCtxDataset(test_sequences, test_contexts), batch_size=256)
        fold_test_probs = []
        with torch.no_grad():
            for seq_batch, ctx_batch in test_loader:
                seq_batch = seq_batch.to(device)
                ctx_batch = ctx_batch.to(device)
                logits = model(seq_batch, ctx_batch)
                fold_test_probs.append(torch.softmax(logits, dim=1).cpu().numpy())
        test_probs += np.vstack(fold_test_probs) / skf.n_splits

    return oof, test_probs, scores


def search_best_weight(a: np.ndarray, b: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    best_w, best_acc = 0.0, 0.0
    for w in np.linspace(0, 1, 41):
        blended = w * a + (1 - w) * b
        preds = blended.argmax(axis=1)
        acc = accuracy_score(y, preds)
        if acc > best_acc:
            best_acc = acc
            best_w = float(w)
    return best_w, best_acc


def main() -> None:
    train_df, test_df, unlabeled_df = load_data()
    y = train_df["risk_level"].astype(int).values - 1

    train_base = train_df.drop(columns=["risk_level"]).reset_index(drop=True)
    train_features, train_sessions = engineer_features(train_base)
    test_features, test_sessions = engineer_features(test_df)
    unl_features = None
    unl_sessions = None
    if unlabeled_df is not None:
        unl_features, unl_sessions = engineer_features(unlabeled_df)
    print(
        "Feature shapes - train:{}, test:{}, unlabeled:{}".format(
            train_features.shape,
            test_features.shape,
            None if unl_features is None else unl_features.shape,
        )
    )

    tree_train = train_features.drop(columns=["session_id"], errors="ignore")
    tree_test = test_features.drop(columns=["session_id"], errors="ignore")
    tree_unlabeled = (
        unl_features.drop(columns=["session_id"], errors="ignore") if unl_features is not None else None
    )

    tree_train_te, [tree_test_te, tree_unlabeled_te] = compute_target_encoding(
        tree_train.assign(risk_level=y + 1),
        y,
        [tree_test, tree_unlabeled if tree_unlabeled is not None else tree_test.copy()],
        TARGET_ENCODING_COLUMNS,
    )
    tree_test = tree_test_te
    if tree_unlabeled is not None:
        tree_unlabeled = tree_unlabeled_te
    print("Finished target encoding")

    sample_w = domain_weights(tree_train_te, tree_test)
    print("Computed domain weights")
    stack_oof, stack_test, stack_unlabeled, fold_scores = train_tree_stack(
        tree_train_te.drop(columns=["risk_level"], errors="ignore"),
        y,
        tree_test,
        tree_unlabeled,
        sample_w,
    )
    print("Tree stack fold accuracies:")
    for name, scores in fold_scores.items():
        print(name, [round(s, 4) for s in scores])

    pseudo_sources = [PseudoSource(tree_test, stack_test, 0.3)]
    if tree_unlabeled is not None and stack_unlabeled is not None:
        pseudo_sources.append(PseudoSource(tree_unlabeled, stack_unlabeled, 0.15))

    semi_oof, semi_test, semi_scores = train_semi_supervised_lightgbm(
        tree_train_te.drop(columns=["risk_level"], errors="ignore"),
        y,
        tree_test,
        pseudo_sources,
    )
    print("Semi-supervised fold accuracies:", [round(s, 4) for s in semi_scores])

    tree_weight, tree_acc = search_best_weight(stack_oof, semi_oof, y)
    print(f"Best tree ensemble weight (stack share): {tree_weight:.2f}, accuracy={tree_acc:.4f}")
    tree_oof = tree_weight * stack_oof + (1 - tree_weight) * semi_oof
    tree_test_probs = tree_weight * stack_test + (1 - tree_weight) * semi_test

    train_sequences, train_ctx, test_sequences, test_ctx = prepare_sequence_inputs(
        train_features, test_features, train_sessions, test_sessions
    )
    print("Sequence arrays ready")
    gru_oof, gru_test, gru_scores = train_hybrid_gru(
        train_sequences,
        train_ctx,
        test_sequences,
        test_ctx,
        y,
    )
    print("GRU CV accuracies:", [round(s, 4) for s in gru_scores])

    final_weight, final_acc = search_best_weight(gru_oof, tree_oof, y)
    print(f"Best GRU weight: {final_weight:.2f}, blended OOF accuracy={final_acc:.4f}")

    final_probs = final_weight * gru_test + (1 - final_weight) * tree_test_probs
    final_preds = final_probs.argmax(axis=1) + 1

    submission = pd.DataFrame({"id": np.arange(len(final_preds)), "risk_level": final_preds})
    out_path = Path("driving_risk_submission_hybrid.csv")
    submission.to_csv(out_path, index=False)
    print("Saved submission to", out_path.resolve())
    print("Prediction distribution:\n", submission["risk_level"].value_counts().sort_index())


if __name__ == "__main__":
    main()
