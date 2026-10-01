"""
Deep tabular baselines: TabNet (Arik & Pfister 2021) and FT-Transformer
(Gorishniy et al. 2021).

Both return softmax probabilities in the same shape as our other models, so
they integrate with the open-set detection module unchanged.
"""
from __future__ import annotations
from typing import Tuple
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def _split_for_validation(X_train, y_train, seed):
    return train_test_split(X_train, y_train, test_size=0.1, random_state=seed, stratify=y_train)


# ---------- TabNet ----------

def fit_tabnet(X_train, y_train, X_val=None, y_val=None,
               num_classes=None, seed=42, **_):
    """Train TabNet (Arik & Pfister 2021) on CPU.

    Reduced configuration for CPU budget: smaller n_d, fewer epochs, larger batch.
    """
    from pytorch_tabnet.tab_model import TabNetClassifier
    torch.set_num_threads(16)

    if X_val is None:
        X_train, X_val, y_train, y_val = _split_for_validation(X_train, y_train, seed)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train).astype(np.float32)
    X_val_s = scaler.transform(X_val).astype(np.float32)

    model = TabNetClassifier(
        n_d=32, n_a=32, n_steps=3,
        gamma=1.3, lambda_sparse=1e-3,
        optimizer_params=dict(lr=2e-2),
        verbose=0,
        device_name="cpu",
        seed=seed,
    )
    model.fit(
        X_train_s, y_train.astype(np.int64),
        eval_set=[(X_val_s, y_val.astype(np.int64))],
        eval_metric=["accuracy"],
        max_epochs=30,
        patience=8,
        batch_size=4096,
        virtual_batch_size=512,
        num_workers=0,
        drop_last=False,
    )
    return (model, scaler)


def predict_tabnet(packed, X):
    model, scaler = packed
    X_s = scaler.transform(X).astype(np.float32)
    proba = model.predict_proba(X_s)
    pred = np.argmax(proba, axis=1)
    return pred, proba


# ---------- FT-Transformer ----------

class _FTTWrapper:
    """Wraps rtdl FT-Transformer to expose .predict_proba interface."""
    def __init__(self, model, scaler, num_classes, batch_size=4096):
        self.model = model
        self.scaler = scaler
        self.num_classes = num_classes
        self.batch_size = batch_size

    def predict_proba(self, X):
        self.model.eval()
        X_s = self.scaler.transform(X).astype(np.float32)
        out = []
        with torch.no_grad():
            for i in range(0, len(X_s), self.batch_size):
                batch = torch.from_numpy(X_s[i:i + self.batch_size])
                logits = self.model(batch, None)
                proba = torch.softmax(logits, dim=1).cpu().numpy()
                out.append(proba)
        return np.concatenate(out, axis=0)


def fit_fttransformer(X_train, y_train, X_val=None, y_val=None,
                      num_classes=None, seed=42, **_):
    """Train FT-Transformer (Gorishniy et al. 2021) on CPU.

    Smallest reasonable configuration for tabular IDS, enough to be
    a credible baseline within compute budget.
    """
    from rtdl_revisiting_models import FTTransformer
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(16)

    n_classes = num_classes or int(np.max(y_train)) + 1
    n_features = X_train.shape[1]

    if X_val is None:
        X_train, X_val, y_train, y_val = _split_for_validation(X_train, y_train, seed)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train).astype(np.float32)
    X_val_s = scaler.transform(X_val).astype(np.float32)

    # Light FTT config for CPU
    model = FTTransformer(
        n_cont_features=n_features,
        cat_cardinalities=[],
        d_out=n_classes,
        n_blocks=2,
        d_block=96,
        attention_n_heads=4,
        attention_dropout=0.1,
        ffn_d_hidden_multiplier=2.0,
        ffn_dropout=0.1,
        residual_dropout=0.0,
    )

    # Standard training loop
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    criterion = torch.nn.CrossEntropyLoss()

    Xtr = torch.from_numpy(X_train_s)
    ytr = torch.from_numpy(y_train.astype(np.int64))
    Xvl = torch.from_numpy(X_val_s)
    yvl = torch.from_numpy(y_val.astype(np.int64))

    batch_size = 4096
    max_epochs = 25
    patience = 5
    best_val = float("inf")
    best_state = None
    no_improve = 0

    n = len(Xtr)
    for epoch in range(max_epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb = Xtr[idx]
            yb = ytr[idx]
            logits = model(xb, None)
            loss = criterion(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # Validation
        model.eval()
        with torch.no_grad():
            v_loss = 0.0
            for i in range(0, len(Xvl), batch_size):
                xb = Xvl[i:i + batch_size]
                yb = yvl[i:i + batch_size]
                logits = model(xb, None)
                v_loss += criterion(logits, yb).item() * len(xb)
            v_loss /= len(Xvl)
        if v_loss < best_val - 1e-4:
            best_val = v_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return _FTTWrapper(model, scaler, n_classes)


def predict_fttransformer(wrapped, X):
    proba = wrapped.predict_proba(X)
    pred = np.argmax(proba, axis=1)
    return pred, proba
