# Dalton Sloan

This folder holds my work for the project plus my modeling scripts.

## What’s here
- `code/` — my scripts (XGBoost baselines, stacking/sequence experiments, EDA).
- `artifacts/` — saved model (`risk_detection_model.pkl`) and stack probabilities (`stack_test_probas.npz`).
- This README — summary of what I tried, what worked, and what failed.

## Experiments (timeline)
- Target-encoded LightGBM (`tree_model_target_encoding.py`): decent baseline but plateaued.
- Session heuristics + trees (`session_sequence_pipeline.py`, `cluster_session_pipeline.py`): noisy CV, feature bloat.
- Sequence/stacked hybrids (`full_temporal_pipeline.py`, `gru_blend_pipeline.py`, `advanced_hybrid_pipeline.py`): heavy training; GRU probabilities added variance; no leaderboard lift.
- Semi-supervised XGBoost (`semi_supervised_xgb.py`): small gain but sensitive to pseudo-label threshold; risk of reinforcing noise.
- Final engineered XGBoost (`risk_model.py`): chosen model. Feature ratios (speed/design, stress proxies, cyclic hour) and domain-drift robustness. Stable ~0.92 validation and reproducible without GPU.

## How to run my final model
1) Install deps: `pip install pandas numpy scikit-learn xgboost joblib`
2) Ensure data files are in `data/` (`kaggle_train.csv`, `kaggle_test.csv`, optional `kaggle_full_unlabeled_data.csv`).
3) From repo root: `python team/dalton_sloan/code/risk_model.py`
   - Saves `risk_detection_model.pkl` and `test_predictions.csv`

## Notes for teammates
- Feel free to reference or copy from `code/` if useful.
- Keep your own work in `team/<your_name>/code/` and document in your README.
