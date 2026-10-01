"""Model factory: LightGBM, XGBoost, RandomForest, LogisticRegression.

Each fit_X function takes (X_train, y_train) and internally splits 90/10
for early stopping. Class imbalance is handled via sample_weight.
"""
from pathlib import Path
import yaml
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.preprocessing import StandardScaler

import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier


def load_model_config():
    cfg_path = Path(__file__).parent.parent / "configs" / "models.yaml"
    return yaml.safe_load(cfg_path.read_text())


def make_balanced_weights(y):
    """Sample weights for balanced loss (replaces sklearn class_weight='balanced')."""
    return compute_sample_weight(class_weight="balanced", y=y)


def fit_lightgbm(X_train, y_train, X_val=None, y_val=None, num_classes=None, seed=42):
    import os
    raw = load_model_config()["lightgbm"]
    cfg = dict(raw)
    cfg.pop("class_weight", None)  # not native to lgb.train
    early_stop = cfg.pop("early_stopping_rounds", 30)
    n_iter = cfg.pop("num_iterations", 300)
    if os.environ.get("SMOKE_MODE") == "1":
        n_iter = 50
        early_stop = 10
    cfg["seed"] = seed
    cfg["num_class"] = num_classes if num_classes else int(np.max(y_train)) + 1

    if X_val is None:
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.1, random_state=seed, stratify=y_train
        )

    # Note: sample_weight=balanced creates extreme weights (432:1) on this 9-class
    # imbalanced problem -> unstable. Use the natural distribution.
    train_set = lgb.Dataset(X_train, label=y_train)
    val_set = lgb.Dataset(X_val, label=y_val, reference=train_set)

    model = lgb.train(
        cfg,
        train_set,
        num_boost_round=n_iter,
        valid_sets=[train_set, val_set],
        valid_names=["train", "val"],
        callbacks=[
            lgb.early_stopping(early_stop, verbose=False),
            lgb.log_evaluation(0),
        ],
    )
    return model


def predict_lightgbm(model, X):
    proba = model.predict(X)
    pred = np.argmax(proba, axis=1)
    return pred, proba


def fit_xgboost(X_train, y_train, X_val=None, y_val=None, num_classes=None, seed=42):
    import os
    raw = load_model_config()["xgboost"]
    cfg = dict(raw)
    n_est = cfg.pop("n_estimators", 300)
    early = cfg.pop("early_stopping_rounds", 25)
    if os.environ.get("SMOKE_MODE") == "1":
        n_est = 50
        early = 10
    cfg["seed"] = seed
    cfg["num_class"] = num_classes if num_classes else int(np.max(y_train)) + 1

    if X_val is None:
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.1, random_state=seed, stratify=y_train
        )

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    model = xgb.train(
        cfg, dtrain, num_boost_round=n_est,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=early,
        verbose_eval=False,
    )
    return model


def predict_xgboost(model, X):
    dtest = xgb.DMatrix(X)
    proba = model.predict(dtest)
    pred = np.argmax(proba, axis=1)
    return pred, proba


def fit_rf(X_train, y_train, seed=42, **_):
    cfg = dict(load_model_config()["random_forest"])
    cfg["random_state"] = seed
    model = RandomForestClassifier(**cfg)
    model.fit(X_train, y_train)
    return model


def predict_rf(model, X):
    pred = model.predict(X)
    proba = model.predict_proba(X)
    return pred, proba


def fit_lr(X_train, y_train, seed=42, **_):
    cfg = dict(load_model_config()["logistic_regression"])
    cfg.pop("multi_class", None)  # deprecated in sklearn 1.7
    cfg["random_state"] = seed
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    model = LogisticRegression(**cfg)
    model.fit(X_scaled, y_train)
    return (model, scaler)


def predict_lr(packed, X):
    model, scaler = packed
    X_scaled = scaler.transform(X)
    pred = model.predict(X_scaled)
    proba = model.predict_proba(X_scaled)
    return pred, proba


def fit_mlp(X_train, y_train, seed=42, **_):
    """MLP baseline: 2 hidden layers (128, 64), early stopping. Sanity DL baseline."""
    cfg = load_model_config().get("mlp", {})
    cfg.setdefault("hidden_layer_sizes", (128, 64))
    cfg.setdefault("max_iter", 100)
    cfg.setdefault("early_stopping", True)
    cfg.setdefault("validation_fraction", 0.1)
    cfg.setdefault("n_iter_no_change", 10)
    cfg["random_state"] = seed
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    model = MLPClassifier(**cfg)
    model.fit(X_scaled, y_train)
    return (model, scaler)


def predict_mlp(packed, X):
    model, scaler = packed
    X_scaled = scaler.transform(X)
    pred = model.predict(X_scaled)
    proba = model.predict_proba(X_scaled)
    return pred, proba


def fit_stability_lgbm_wrapper(X_train, y_train, X_val=None, y_val=None,
                                num_classes=None, seed=42, BS_train=None, **_):
    """Wrapper that requires BS_train kwarg for stability-aware LightGBM."""
    from src.stability_lgbm import fit_stability_lgbm
    if BS_train is None:
        raise ValueError("stability_lgbm requires BS_train")
    return fit_stability_lgbm(X_train, y_train, BS_train, X_val, y_val,
                              num_classes=num_classes, seed=seed)


def predict_stability_lgbm_wrapper(packed, X):
    from src.stability_lgbm import predict_stability_lgbm
    return predict_stability_lgbm(packed, X)


def _lazy_tabnet(*args, **kwargs):
    from src.deep_models import fit_tabnet
    return fit_tabnet(*args, **kwargs)


def _lazy_predict_tabnet(*args, **kwargs):
    from src.deep_models import predict_tabnet
    return predict_tabnet(*args, **kwargs)


def _lazy_ftt(*args, **kwargs):
    from src.deep_models import fit_fttransformer
    return fit_fttransformer(*args, **kwargs)


def _lazy_predict_ftt(*args, **kwargs):
    from src.deep_models import predict_fttransformer
    return predict_fttransformer(*args, **kwargs)


MODEL_REGISTRY = {
    "lightgbm": (fit_lightgbm, predict_lightgbm),
    "xgboost": (fit_xgboost, predict_xgboost),
    "rf": (fit_rf, predict_rf),
    "lr": (fit_lr, predict_lr),
    "mlp": (fit_mlp, predict_mlp),
    "stability_lgbm": (fit_stability_lgbm_wrapper, predict_stability_lgbm_wrapper),
    "tabnet": (_lazy_tabnet, _lazy_predict_tabnet),
    "ftt": (_lazy_ftt, _lazy_predict_ftt),
}


def fit_predict(name: str, X_train, y_train, X_test, X_val=None, y_val=None,
                num_classes=None, seed=42, BS_train=None):
    fit_fn, pred_fn = MODEL_REGISTRY[name]
    if name == "stability_lgbm":
        model = fit_fn(X_train, y_train, X_val=X_val, y_val=y_val,
                       num_classes=num_classes, seed=seed, BS_train=BS_train)
    else:
        model = fit_fn(X_train, y_train, X_val=X_val, y_val=y_val,
                       num_classes=num_classes, seed=seed)
    pred, proba = pred_fn(model, X_test)
    return model, pred, proba
