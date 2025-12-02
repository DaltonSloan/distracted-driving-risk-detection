# Distracted Driving Risk Detection

We predict a driver’s risk level from on-vehicle telemetry, weather, and historical accident context. The repo holds multiple modeling attempts plus the submissions for the provided train/test split (`data/kaggle_train.csv`, `data/kaggle_test.csv`).

## Project overview
- **Goal:** Map each row of sensor and context data to a discrete `risk_level` (1–4). Primary focus is high validation accuracy/generalization to the held-out test set.
- **Data:** Per-row measurements of speed, RPM, acceleration, throttle, engine temperature/load, heart rate, weather, visibility/precipitation, design speed, and historical accidents. Train includes `risk_level` labels; test omits them.

## Repo structure
- Root: submission outputs (e.g., `driving_risk_submission_*.csv`).
- `data/`: shared datasets (`kaggle_train.csv`, `kaggle_test.csv`, `kaggle_full_unlabeled_data.csv`).
- `team/<name>/`: personal space for each teammate with a README and `code/` folder. All scripts now live under the respective team folders.
  - Dalton’s scripts are in `team/dalton_sloan/code/`; his saved artifacts (model, stack probabilities) are in `team/dalton_sloan/artifacts/`.

## Team workflow
- Add your code/notebooks under `team/<your_name>/code/`.
- Update `team/<your_name>/README.md` with what you tried, what worked, what failed, and links to outputs.
- If you produce shared submissions, save them at repo root; keep your experiment code in your team folder.

## Team sections
- Dalton Sloan — `team/dalton_sloan/README.md`
- Alden P — `team/alden_p/README.md`
- Ashley P — `team/ashley_p/README.md`
- Ryan P — `team/ryan_p/README.md`
- Justin D — `team/justin_d/README.md`
