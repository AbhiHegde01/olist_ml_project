"""
Shared feature definitions.

The SAME functions are used by the training script, the notebooks and the
Streamlit predictor, so there is no train/serve mismatch.

Two prediction tasks
--------------------
* late      : will a (not yet shipped) order arrive after the estimated date?
              -> only information known at PURCHASE time is allowed.
* satisfaction : the order has been delivered - will the review be bad (1-3 stars)?
              -> purchase-time info + what happened during delivery.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

# ---- column groups --------------------------------------------------------
CATEGORICAL = ["customer_state", "seller_state", "payment_type", "category"]

PURCHASE_NUMERIC = [
    "estimated_delivery_days", "shipping_limit_days", "approval_hours",
    "n_items", "n_sellers", "total_price", "total_freight", "freight_ratio",
    "total_weight_g", "product_volume_cm3", "payment_installments",
    "distance_km", "same_state", "purchase_month", "purchase_weekday",
    "purchase_hour", "is_weekend", "seller_prior_late_rate", "seller_prior_orders",
]

DELIVERY_NUMERIC = ["actual_delivery_days", "delay_days", "is_late"]

SATISFACTION_EXTRA = ["seller_prior_bad_rate", "product_photos_qty", "product_description_length"]

# heavy right-skew -> log1p before scaling (helps LogReg / Naive Bayes)
SKEWED = {"total_price", "total_freight", "total_weight_g", "product_volume_cm3",
          "distance_km", "approval_hours", "seller_prior_orders", "shipping_limit_days",
          "product_description_length"}

TASKS = {
    "late": {
        "target": "is_late",
        "title": "Late-delivery risk",
        "positive_label": "Late",
        "numeric": PURCHASE_NUMERIC,
        "categorical": CATEGORICAL,
    },
    "satisfaction": {
        "target": "bad_review",
        "title": "Customer dissatisfaction (review 1-3 stars)",
        "positive_label": "Bad review",
        "numeric": PURCHASE_NUMERIC + DELIVERY_NUMERIC + SATISFACTION_EXTRA,
        "categorical": CATEGORICAL,
    },
}


def feature_columns(task: str) -> list[str]:
    cfg = TASKS[task]
    return cfg["numeric"] + cfg["categorical"]


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recompute features that are pure functions of other inputs.
    Called on the user's form input in the app so derived columns are consistent.
    """
    df = df.copy()
    if {"total_price", "total_freight"} <= set(df.columns):
        df["freight_ratio"] = df["total_freight"] / (df["total_price"] + df["total_freight"]).replace(0, np.nan)
    if {"customer_state", "seller_state"} <= set(df.columns):
        df["same_state"] = (df["customer_state"] == df["seller_state"]).astype(int)
    if "purchase_weekday" in df.columns:
        df["is_weekend"] = (df["purchase_weekday"] >= 5).astype(int)
    if {"actual_delivery_days", "estimated_delivery_days"} <= set(df.columns):
        df["delay_days"] = df["actual_delivery_days"] - df["estimated_delivery_days"]
        df["is_late"] = (df["delay_days"] > 0).astype(int)
    return df


def modeling_frame(master: pd.DataFrame, task: str, start: str = "2017-01-01") -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Rows usable for a task -> (X, y, purchase_timestamp)."""
    cfg = TASKS[task]
    d = master[(master["is_delivered"] == 1) & (master["order_purchase_timestamp"] >= start)]
    d = d[d[cfg["target"]].notna()]
    X = d[feature_columns(task)].copy()
    y = d[cfg["target"]].astype(int)
    return X, y, d["order_purchase_timestamp"]


def _log1p_safe(x):
    return np.log1p(np.clip(x, 0, None))


def build_preprocessor(task: str, scale: bool = True) -> ColumnTransformer:
    """
    numeric  : log1p on skewed columns, then StandardScaler (scale=False for trees)
    category : one-hot, rare levels (<1% of rows) grouped into 'infrequent'
    """
    cfg = TASKS[task]
    skewed = [c for c in cfg["numeric"] if c in SKEWED]
    plain = [c for c in cfg["numeric"] if c not in SKEWED]

    def num_pipe(log: bool):
        steps = []
        if log:
            steps.append(("log1p", FunctionTransformer(_log1p_safe, feature_names_out="one-to-one")))
        if scale:
            steps.append(("scale", StandardScaler()))
        return Pipeline(steps) if steps else "passthrough"

    return ColumnTransformer(
        [
            ("num_log", num_pipe(True), skewed),
            ("num", num_pipe(False), plain),
            ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=0.01,
                                  sparse_output=False), cfg["categorical"]),
        ],
        verbose_feature_names_out=False,
    )


def input_defaults(X: pd.DataFrame) -> dict:
    """Median / mode of every feature - used to pre-fill the predictor form."""
    out = {}
    for c in X.columns:
        if pd.api.types.is_numeric_dtype(X[c]):
            out[c] = float(X[c].median())
        else:
            out[c] = X[c].mode().iloc[0]
    return out
