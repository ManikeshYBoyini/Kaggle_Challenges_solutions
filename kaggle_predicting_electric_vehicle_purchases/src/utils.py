"""Helper functions shared by all notebooks."""
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from config import (
    CAT_COLS, DATA_DIR, EXPERIMENTS_FILE, FOLDS_DIR, ID_COL, N_FOLDS,
    OOF_DIR, PREDS_DIR, SEED, SUB_DIR, TARGET,
)


def load_data():
    """Return train, test, sample_submission. Target becomes 1 (Yes) / 0 (No)."""
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    sample = pd.read_csv(DATA_DIR / "sample_submission.csv")
    train[TARGET] = (train[TARGET] == "Yes").astype(int)
    return train, test, sample


def to_category(df, cols=CAT_COLS):
    """Turn text columns into pandas 'category' type (LightGBM / XGBoost use this)."""
    df = df.copy()
    for c in cols:
        df[c] = df[c].astype("category")
    return df


def get_folds(train):
    """Return a fold number (0-4) for every train row.

    The split is made once and saved, so every model uses the same folds.
    """
    path = FOLDS_DIR / "folds.csv"
    if not path.exists():
        folds = np.zeros(len(train), dtype=int)
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        for k, (_, val_idx) in enumerate(skf.split(train, train[TARGET])):
            folds[val_idx] = k
        pd.DataFrame({ID_COL: train[ID_COL], "fold": folds}).to_csv(path, index=False)
    f = pd.read_csv(path)
    assert (f[ID_COL].values == train[ID_COL].values).all(), "folds.csv does not match train order"
    return f["fold"].values


def score(y_true, y_pred):
    """Return (AUC, logloss)."""
    return roc_auc_score(y_true, y_pred), log_loss(y_true, np.clip(y_pred, 1e-7, 1 - 1e-7))


def save_preds(name, oof, test_pred):
    """Save out-of-fold (train) and test predictions for later blending."""
    np.save(OOF_DIR / f"{name}_oof.npy", oof)
    np.save(PREDS_DIR / f"{name}_test.npy", test_pred)


def load_preds(name):
    return np.load(OOF_DIR / f"{name}_oof.npy"), np.load(PREDS_DIR / f"{name}_test.npy")


def log_experiment(model, features, auc, logloss, notes=""):
    """Add one row to experiments.csv (replaces an older row with the same model name)."""
    row = pd.DataFrame([{
        "model": model, "features": features, "cv_auc": round(auc, 6),
        "cv_logloss": round(logloss, 6), "notes": notes,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }])
    if EXPERIMENTS_FILE.exists():
        old = pd.read_csv(EXPERIMENTS_FILE)
        row = pd.concat([old[old["model"] != model], row], ignore_index=True)
    row.to_csv(EXPERIMENTS_FILE, index=False)
    return row


def make_submission(test_pred, name, sample):
    """Write a Kaggle submission file and check it is valid."""
    sub = sample[[ID_COL]].copy()
    sub[TARGET] = test_pred
    assert len(sub) == len(sample)
    assert sub[TARGET].between(0, 1).all()
    assert not sub[TARGET].isna().any()
    path = SUB_DIR / f"{name}.csv"
    sub.to_csv(path, index=False)
    print(f"Saved {path.name}  rows={len(sub)}")
    return path
