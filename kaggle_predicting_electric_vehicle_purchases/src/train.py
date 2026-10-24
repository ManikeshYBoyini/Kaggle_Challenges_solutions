"""Cross-validation training loops for each model type.

Every function:
- trains one model per fold (5 total),
- returns oof (prediction for each train row, made by the model that did NOT see it)
  and test_pred (average of the 5 models' predictions on test).
"""
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score

from config import N_FOLDS, SEED


def _report(k, y_val, p_val):
    print(f"  fold {k}: AUC = {roc_auc_score(y_val, p_val):.5f}")


def train_lgbm(X, y, X_test, folds, params=None, verbose=True):
    params = {
        "objective": "binary", "learning_rate": 0.05, "num_leaves": 63,
        "min_child_samples": 50, "feature_fraction": 0.8, "bagging_fraction": 0.8,
        "bagging_freq": 1, "lambda_l2": 1.0, "n_estimators": 5000,
        "random_state": SEED, "verbose": -1, "n_jobs": -1,
        **(params or {}),
    }
    oof, test_pred = np.zeros(len(X)), np.zeros(len(X_test))
    for k in range(N_FOLDS):
        tr, va = folds != k, folds == k
        model = lgb.LGBMClassifier(**params)
        model.fit(X[tr], y[tr], eval_set=[(X[va], y[va])], eval_metric="auc",
                  callbacks=[lgb.early_stopping(200, verbose=False)])
        oof[va] = model.predict_proba(X[va])[:, 1]
        test_pred += model.predict_proba(X_test)[:, 1] / N_FOLDS
        if verbose:
            _report(k, y[va], oof[va])
    return oof, test_pred


def train_xgb(X, y, X_test, folds, params=None, verbose=True):
    params = {
        "learning_rate": 0.05, "max_depth": 6, "min_child_weight": 5,
        "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 1.0,
        "n_estimators": 5000, "early_stopping_rounds": 200, "eval_metric": "auc",
        "tree_method": "hist", "enable_categorical": True, "max_cat_to_onehot": 1,
        "random_state": SEED, "n_jobs": -1,
        **(params or {}),
    }
    oof, test_pred = np.zeros(len(X)), np.zeros(len(X_test))
    for k in range(N_FOLDS):
        tr, va = folds != k, folds == k
        model = xgb.XGBClassifier(**params)
        model.fit(X[tr], y[tr], eval_set=[(X[va], y[va])], verbose=False)
        oof[va] = model.predict_proba(X[va])[:, 1]
        test_pred += model.predict_proba(X_test)[:, 1] / N_FOLDS
        if verbose:
            _report(k, y[va], oof[va])
    return oof, test_pred


def train_catboost(X, y, X_test, folds, cat_cols, params=None, verbose=True):
    params = {
        "iterations": 3000, "learning_rate": 0.08, "depth": 6,
        "eval_metric": "AUC", "od_type": "Iter", "od_wait": 200,
        "random_seed": SEED, "thread_count": -1, "verbose": 0,
        **(params or {}),
    }
    oof, test_pred = np.zeros(len(X)), np.zeros(len(X_test))
    for k in range(N_FOLDS):
        tr, va = folds != k, folds == k
        model = CatBoostClassifier(**params, cat_features=cat_cols)
        model.fit(X[tr], y[tr], eval_set=(X[va], y[va]), use_best_model=True)
        oof[va] = model.predict_proba(X[va])[:, 1]
        test_pred += model.predict_proba(X_test)[:, 1] / N_FOLDS
        if verbose:
            _report(k, y[va], oof[va])
    return oof, test_pred
