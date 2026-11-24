"""
Data preparation for the Olist Brazilian E-Commerce dataset.

Loads the 9 raw Kaggle CSVs, cleans them, aggregates everything to ONE ROW PER
ORDER and writes:
    data/processed/master_orders.parquet   -> used by notebooks, training, dashboard
    data/processed/data_quality.csv        -> missing values / duplicates report
    data/processed/cleaning_log.csv        -> every cleaning step and rows affected

Run:  python src/data_prep.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

FILES = {
    "customers": "olist_customers_dataset.csv",
    "geolocation": "olist_geolocation_dataset.csv",
    "items": "olist_order_items_dataset.csv",
    "payments": "olist_order_payments_dataset.csv",
    "reviews": "olist_order_reviews_dataset.csv",
    "orders": "olist_orders_dataset.csv",
    "products": "olist_products_dataset.csv",
    "sellers": "olist_sellers_dataset.csv",
    "translation": "product_category_name_translation.csv",
}

DATE_COLS = {
    "orders": [
        "order_purchase_timestamp", "order_approved_at", "order_delivered_carrier_date",
        "order_delivered_customer_date", "order_estimated_delivery_date",
    ],
    "items": ["shipping_limit_date"],
    "reviews": ["review_creation_date", "review_answer_timestamp"],
}

# Brazil bounding box – used to drop bad geolocation points
BR_LAT = (-34.0, 5.5)
BR_LNG = (-74.0, -34.0)


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_raw(raw_dir: Path = RAW) -> dict[str, pd.DataFrame]:
    """Read every raw CSV into a dict of DataFrames (dates parsed)."""
    tables = {}
    for name, fname in FILES.items():
        df = pd.read_csv(raw_dir / fname, encoding="utf-8-sig")
        for c in DATE_COLS.get(name, []):
            df[c] = pd.to_datetime(df[c], errors="coerce")
        tables[name] = df
    return tables


def quality_report(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Rows, columns, duplicate rows and missing % per column for every table."""
    rows = []
    for name, df in tables.items():
        dup = int(df.duplicated().sum())
        for col in df.columns:
            rows.append({
                "table": name, "rows": len(df), "column": col,
                "dtype": str(df[col].dtype),
                "missing": int(df[col].isna().sum()),
                "missing_pct": round(100 * df[col].isna().mean(), 2),
                "n_unique": int(df[col].nunique()),
                "duplicate_rows_in_table": dup,
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def haversine_km(lat1, lng1, lat2, lng2):
    """Great-circle distance in km (vectorised)."""
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lng2 - lng1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def bayesian_prior_mean(df: pd.DataFrame, group: str, value: str, time_col: str,
                        prior: float, m: float = 10.0) -> pd.Series:
    """
    Leakage-safe historical average of `value` per `group`, using ONLY rows that
    happened before the current row (sorted by time_col). Smoothed towards the
    global `prior` with strength m (Bayesian average):

        smoothed = (sum_past + m * prior) / (n_past + m)
    """
    d = df[[group, value, time_col]].sort_values(time_col)
    g = d.groupby(group)[value]
    past_sum = g.cumsum() - d[value].fillna(0)
    past_n = g.cumcount()  # number of earlier rows for this group
    smoothed = (past_sum.fillna(0) + m * prior) / (past_n + m)
    return smoothed.reindex(df.index), past_n.reindex(df.index)


# --------------------------------------------------------------------------- #
# Build master table
# --------------------------------------------------------------------------- #
def build_master(t: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    log = []

    def step(desc, before, after):
        log.append({"step": desc, "rows_before": before, "rows_after": after,
                    "rows_removed": before - after})

    # ---- products: translate + impute ------------------------------------
    prod = t["products"].rename(columns={
        "product_name_lenght": "product_name_length",
        "product_description_lenght": "product_description_length",
    })
    trans = t["translation"]
    prod = prod.merge(trans, on="product_category_name", how="left")
    prod["category"] = (prod["product_category_name_english"]
                        .fillna(prod["product_category_name"]).fillna("unknown"))
    n_missing_cat = int((prod["category"] == "unknown").sum())
    log.append({"step": f"products: missing category -> 'unknown' ({n_missing_cat} products)",
                "rows_before": len(prod), "rows_after": len(prod), "rows_removed": 0})
    num_cols = ["product_name_length", "product_description_length", "product_photos_qty",
                "product_weight_g", "product_length_cm", "product_height_cm", "product_width_cm"]
    for c in num_cols:
        prod[c] = prod[c].fillna(prod.groupby("category")[c].transform("median"))
        prod[c] = prod[c].fillna(prod[c].median())
    prod["product_volume_cm3"] = prod["product_length_cm"] * prod["product_height_cm"] * prod["product_width_cm"]
    log.append({"step": "products: impute numeric attributes with category median",
                "rows_before": len(prod), "rows_after": len(prod), "rows_removed": 0})

    # ---- items: aggregate per order --------------------------------------
    items = t["items"].merge(
        prod[["product_id", "category", "product_weight_g", "product_volume_cm3",
              "product_photos_qty", "product_description_length", "product_name_length"]],
        on="product_id", how="left")
    items_agg = items.groupby("order_id").agg(
        n_items=("order_item_id", "count"),
        n_products=("product_id", "nunique"),
        n_sellers=("seller_id", "nunique"),
        total_price=("price", "sum"),
        total_freight=("freight_value", "sum"),
        total_weight_g=("product_weight_g", "sum"),
        shipping_limit_date=("shipping_limit_date", "max"),
    )
    # "main" item = most expensive item in the order -> category, seller, product attributes
    main = (items.sort_values(["order_id", "price"], ascending=[True, False])
                 .drop_duplicates("order_id")
                 .set_index("order_id")[["seller_id", "product_id", "category", "price",
                                         "product_weight_g", "product_volume_cm3",
                                         "product_photos_qty", "product_description_length",
                                         "product_name_length"]]
                 .rename(columns={"price": "main_item_price"}))
    items_agg = items_agg.join(main)

    # ---- payments: aggregate per order -----------------------------------
    pay = t["payments"]
    before = len(pay)
    pay = pay[pay["payment_type"] != "not_defined"]
    step("payments: drop payment_type == 'not_defined'", before, len(pay))
    pay_type_val = pay.groupby(["order_id", "payment_type"])["payment_value"].sum().reset_index()
    main_type = (pay_type_val.sort_values(["order_id", "payment_value"], ascending=[True, False])
                 .drop_duplicates("order_id").set_index("order_id")["payment_type"])
    pay_agg = pay.groupby("order_id").agg(
        payment_value=("payment_value", "sum"),
        payment_installments=("payment_installments", "max"),
        n_payment_methods=("payment_type", "nunique"),
        n_payments=("payment_sequential", "count"),
    )
    pay_agg["payment_type"] = main_type
    pay_agg["uses_voucher"] = pay.assign(v=pay["payment_type"].eq("voucher")).groupby("order_id")["v"].any().astype(int)
    pay_agg["payment_installments"] = pay_agg["payment_installments"].clip(lower=1)

    # ---- reviews: latest review per order --------------------------------
    rev = t["reviews"]
    before = len(rev)
    rev = (rev.sort_values("review_answer_timestamp")
              .drop_duplicates("order_id", keep="last"))
    step("reviews: keep latest review per order", before, len(rev))
    rev = rev.set_index("order_id")
    rev_agg = pd.DataFrame({
        "review_score": rev["review_score"],
        "review_has_comment": rev["review_comment_message"].notna().astype(int),
        "review_comment_message": rev["review_comment_message"],
        "review_creation_date": rev["review_creation_date"],
    })

    # ---- geolocation: zip prefix -> median lat/lng -----------------------
    geo = t["geolocation"]
    before = len(geo)
    geo = geo[geo["geolocation_lat"].between(*BR_LAT) & geo["geolocation_lng"].between(*BR_LNG)]
    step("geolocation: drop points outside Brazil", before, len(geo))
    geo = geo.groupby("geolocation_zip_code_prefix")[["geolocation_lat", "geolocation_lng"]].median()
    log.append({"step": "geolocation: aggregate to zip-prefix median lat/lng",
                "rows_before": len(geo), "rows_after": len(geo), "rows_removed": 0})

    cust = t["customers"].merge(geo, left_on="customer_zip_code_prefix",
                                right_index=True, how="left").rename(
        columns={"geolocation_lat": "customer_lat", "geolocation_lng": "customer_lng"})
    sell = t["sellers"].merge(geo, left_on="seller_zip_code_prefix",
                              right_index=True, how="left").rename(
        columns={"geolocation_lat": "seller_lat", "geolocation_lng": "seller_lng"})

    # ---- orders: join everything -----------------------------------------
    o = t["orders"].copy()
    before = len(o)
    o = o.drop_duplicates("order_id")
    step("orders: drop duplicate order_id", before, len(o))

    o = (o.merge(cust, on="customer_id", how="left")
           .merge(items_agg, left_on="order_id", right_index=True, how="left")
           .merge(pay_agg, left_on="order_id", right_index=True, how="left")
           .merge(rev_agg, left_on="order_id", right_index=True, how="left")
           .merge(sell, on="seller_id", how="left"))

    before = len(o)
    o = o[o["n_items"].notna()]
    step("orders: drop orders with no items (mostly canceled/unavailable)", before, len(o))

    # ---- time features ---------------------------------------------------
    p = o["order_purchase_timestamp"]
    o["purchase_date"] = p.dt.normalize()
    o["purchase_year_month"] = p.dt.to_period("M").astype(str)
    o["purchase_year"] = p.dt.year
    o["purchase_month"] = p.dt.month
    o["purchase_weekday"] = p.dt.day_of_week
    o["purchase_hour"] = p.dt.hour
    o["is_weekend"] = (o["purchase_weekday"] >= 5).astype(int)

    day = pd.Timedelta(days=1)
    o["approval_hours"] = (o["order_approved_at"] - p) / pd.Timedelta(hours=1)
    o["estimated_delivery_days"] = (o["order_estimated_delivery_date"] - p) / day
    o["shipping_limit_days"] = (o["shipping_limit_date"] - p) / day
    o["actual_delivery_days"] = (o["order_delivered_customer_date"] - p) / day
    o["carrier_pickup_days"] = (o["order_delivered_carrier_date"] - p) / day
    o["delay_days"] = (o["order_delivered_customer_date"] - o["order_estimated_delivery_date"]) / day
    o["is_late"] = np.where(
        o["order_delivered_customer_date"].notna(),
        (o["order_delivered_customer_date"].dt.normalize() > o["order_estimated_delivery_date"]).astype(int),
        np.nan)

    # ---- impossible dates ------------------------------------------------
    bad = (o["actual_delivery_days"] < 0) | (o["carrier_pickup_days"] < 0) | (o["approval_hours"] < 0)
    before = len(o)
    o = o[~bad.fillna(False)]
    step("orders: drop impossible timelines (delivered/shipped/approved before purchase)", before, len(o))

    # ---- money / logistics features --------------------------------------
    o["freight_ratio"] = o["total_freight"] / (o["total_price"] + o["total_freight"])
    o["order_value"] = o["total_price"] + o["total_freight"]
    o["same_state"] = (o["customer_state"] == o["seller_state"]).astype(int)
    o["distance_km"] = haversine_km(o["customer_lat"], o["customer_lng"],
                                    o["seller_lat"], o["seller_lng"])
    o["distance_km"] = o["distance_km"].fillna(
        o.groupby(["customer_state", "seller_state"])["distance_km"].transform("median"))
    o["distance_km"] = o["distance_km"].fillna(o["distance_km"].median())
    o["approval_hours"] = o["approval_hours"].fillna(o["approval_hours"].median())

    # ---- targets ---------------------------------------------------------
    o["is_delivered"] = ((o["order_status"] == "delivered") &
                         o["order_delivered_customer_date"].notna()).astype(int)
    o["bad_review"] = np.where(o["review_score"].notna(), (o["review_score"] <= 3).astype(int), np.nan)

    # ---- leakage-safe seller history (only past orders) ------------------
    delivered = o["is_delivered"] == 1
    global_late = o.loc[delivered, "is_late"].mean()
    global_bad = o.loc[delivered, "bad_review"].mean()
    o["seller_prior_late_rate"] = np.nan
    o["seller_prior_bad_rate"] = np.nan
    o["seller_prior_orders"] = 0
    d = o.loc[delivered]
    late_s, n_s = bayesian_prior_mean(d, "seller_id", "is_late", "order_purchase_timestamp", global_late)
    bad_s, _ = bayesian_prior_mean(d.assign(bad_review=d["bad_review"].fillna(global_bad)),
                                   "seller_id", "bad_review", "order_purchase_timestamp", global_bad)
    o.loc[delivered, "seller_prior_late_rate"] = late_s
    o.loc[delivered, "seller_prior_bad_rate"] = bad_s
    o.loc[delivered, "seller_prior_orders"] = n_s
    o["seller_prior_late_rate"] = o["seller_prior_late_rate"].fillna(global_late)
    o["seller_prior_bad_rate"] = o["seller_prior_bad_rate"].fillna(global_bad)
    log.append({"step": "seller history: leakage-safe Bayesian-smoothed past late/bad-review rate (m=10)",
                "rows_before": len(o), "rows_after": len(o), "rows_removed": 0})

    drop = ["customer_id", "customer_zip_code_prefix", "seller_zip_code_prefix",
            "customer_lat", "customer_lng", "seller_lat", "seller_lng", "shipping_limit_date"]
    o = o.drop(columns=drop).reset_index(drop=True)
    o["customer_city"] = o["customer_city"].str.title()
    o["seller_city"] = o["seller_city"].str.title()
    return o, pd.DataFrame(log)


def main():
    PROCESSED.mkdir(parents=True, exist_ok=True)
    print("Loading raw CSVs ...")
    tables = load_raw()
    quality_report(tables).to_csv(PROCESSED / "data_quality.csv", index=False)
    print("Building master table ...")
    master, log = build_master(tables)
    assert master["order_id"].is_unique, "master must have one row per order"
    master.to_parquet(PROCESSED / "master_orders.parquet", index=False)
    log.to_csv(PROCESSED / "cleaning_log.csv", index=False)
    print(log.to_string(index=False))
    print(f"\nmaster_orders: {master.shape[0]:,} rows x {master.shape[1]} cols")
    print(f"delivered: {int(master['is_delivered'].sum()):,}  "
          f"late rate: {master['is_late'].mean():.3f}  bad-review rate: {master['bad_review'].mean():.3f}")


if __name__ == "__main__":
    main()
