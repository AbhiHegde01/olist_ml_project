import datetime as dt

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils import BAD, BLUE, GOOD, SERIES, STATE_NAMES, load_master, load_model, page_setup, show
from src.features import add_derived  # noqa: E402  (utils puts the project root on sys.path)

page_setup("Predictor", "🔮")
st.title("🔮 Order Risk Predictor")
st.caption("Describe an order the way it looks in the data — the trained models estimate the risk. "
           "Fields you don't change use the typical (median) value from the training data.")

try:
    late_m, sat_m = load_model("late"), load_model("satisfaction")
except FileNotFoundError as e:
    st.error(f"A trained model file is missing: `{e.filename}`.\n\n"
             "Train the models first with `python src/train.py` (or `python src/train.py satisfaction` "
             "for just one), then reload this page.")
    st.stop()


@st.cache_data
def route_distance() -> pd.Series:
    d = load_master()
    return d.groupby(["customer_state", "seller_state"])["distance_km"].median()


def predict(bundle, row: dict) -> tuple[float, pd.DataFrame]:
    X = add_derived(pd.DataFrame([{**bundle["defaults"], **row}]))[bundle["features"]]
    return float(bundle["pipeline"].predict_proba(X)[0, 1]), X


def explain(bundle, X: pd.DataFrame) -> pd.Series | None:
    """Per-feature contribution in log-odds (XGBoost's built-in TreeSHAP)."""
    pipe = bundle["pipeline"]
    clf = pipe.named_steps["clf"]
    if not hasattr(clf, "get_booster"):
        return None
    import xgboost as xgb
    prep = pipe.named_steps["prep"]
    Z = prep.transform(X)
    names = prep.get_feature_names_out()
    contrib = clf.get_booster().predict(xgb.DMatrix(np.asarray(Z, dtype=float), feature_names=list(names)),
                                        pred_contribs=True)[0][:-1]
    raw = []
    for n in names:   # map one-hot columns back to the original feature
        raw.append(next((c for c in bundle["categorical"] if n.startswith(c + "_")), n))
    return pd.Series(contrib, index=raw).groupby(level=0).sum()


def gauge(p: float, thr: float, base: float, title: str):
    color = BAD if p >= thr else GOOD
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=p * 100, number={"suffix": "%", "valueformat": ".1f"},
        title={"text": title},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": color, "thickness": 0.35},
               "threshold": {"line": {"color": "gray", "width": 3}, "value": thr * 100, "thickness": 0.8},
               "steps": [{"range": [0, base * 100], "color": "rgba(128,128,128,.12)"}]}))
    fig.update_layout(height=260, margin=dict(l=20, r=20, t=60, b=10))
    st.plotly_chart(fig, width="stretch")
    st.caption(f"Grey band = average rate ({base:.1%}); grey line = decision threshold ({thr:.1%}).")


def contribution_chart(contrib: pd.Series | None, title: str):
    if contrib is None:
        return
    top = contrib.reindex(contrib.abs().sort_values(ascending=False).index).head(8).iloc[::-1]
    df = pd.DataFrame({"feature": top.index, "effect": top.values})
    df["direction"] = np.where(df["effect"] > 0, "raises risk", "lowers risk")
    fig = px.bar(df, x="effect", y="feature", orientation="h", color="direction",
                 color_discrete_map={"raises risk": BAD, "lowers risk": BLUE}, title=title,
                 labels={"effect": "Contribution (log-odds)", "feature": ""})
    show(fig, 330)


# ------------------------------------------------------------------ inputs
cats, d = late_m["categories"], late_m["defaults"]
with st.container(border=True):
    st.markdown("##### 🧾 Order details (known at checkout)")
    c = st.columns(4)
    cust = c[0].selectbox("Customer state", cats["customer_state"], index=cats["customer_state"].index("SP"),
                          format_func=lambda s: f"{s} – {STATE_NAMES.get(s, s)}")
    sell = c[1].selectbox("Seller state", cats["seller_state"], index=cats["seller_state"].index("SP"),
                          format_func=lambda s: f"{s} – {STATE_NAMES.get(s, s)}")
    cat = c[2].selectbox("Product category", cats["category"],
                         index=cats["category"].index("bed_bath_table") if "bed_bath_table" in cats["category"] else 0)
    pay = c[3].selectbox("Payment type", cats["payment_type"], index=cats["payment_type"].index("credit_card"))

    c = st.columns(4)
    price = c[0].number_input("Items total price (R$)", 1.0, 10000.0, 100.0, step=10.0)
    freight = c[1].number_input("Freight (R$)", 0.0, 500.0, 18.0, step=1.0)
    n_items = c[2].number_input("Number of items", 1, 20, 1)
    n_sellers = c[3].number_input("Number of sellers", 1, 5, 1)

    c = st.columns(4)
    weight = c[0].number_input("Total weight (g)", 10, 40000, int(d["total_weight_g"]), step=100)
    inst = c[1].slider("Installments", 1, 24, 1)
    pdate = c[2].date_input("Purchase date", dt.date(2018, 5, 15))
    hour = c[3].slider("Purchase hour", 0, 23, 14)

    c = st.columns(4)
    est = c[0].slider("Promised delivery (days)", 2, 60, int(round(d["estimated_delivery_days"])))
    default_km = float(route_distance().get((cust, sell), d["distance_km"]))
    dist = c[1].number_input("Seller → customer distance (km)", 0.0, 4000.0, float(round(default_km)), step=50.0,
                             help="Pre-filled with the median distance for this state pair.")
    s_late = c[2].slider("Seller's past late rate", 0.0, 0.6, round(d["seller_prior_late_rate"], 2), 0.01,
                         help="Share of the seller's previous orders that arrived late.")
    s_orders = c[3].number_input("Seller's past orders", 0, 2000, int(d["seller_prior_orders"]))

row = {"customer_state": cust, "seller_state": sell, "category": cat, "payment_type": pay,
       "total_price": price, "total_freight": freight, "n_items": n_items, "n_sellers": n_sellers,
       "total_weight_g": weight, "payment_installments": inst, "purchase_month": pdate.month,
       "purchase_weekday": pdate.weekday(), "purchase_hour": hour, "estimated_delivery_days": float(est),
       "distance_km": dist, "seller_prior_late_rate": s_late, "seller_prior_orders": s_orders}

tab1, tab2 = st.tabs(["🚚 Late-delivery risk (at checkout)", "⭐ Bad-review risk (after delivery)"])

with tab1:
    p, X = predict(late_m, row)
    a, b = st.columns([2, 3])
    with a:
        gauge(p, late_m["threshold"], late_m["base_rate"], "P(order arrives late)")
        if p >= late_m["threshold"]:
            st.error(f"**High risk of late delivery** ({p / late_m['base_rate']:.1f}× the average). "
                     "Suggested action: priority carrier, proactive message to the customer, or a longer promised date.")
        else:
            st.success("**Low risk** — normal fulfilment.")
    with b:
        contribution_chart(explain(late_m, X), "Why? Top factors for this order")
    grid = list(range(5, 61, 2))
    curve = [predict(late_m, {**row, "estimated_delivery_days": float(g)})[0] for g in grid]
    fig = px.line(x=grid, y=curve, markers=True, labels={"x": "Promised delivery (days)", "y": "P(late)"},
                  title="What-if: how the promised delivery window changes the risk", color_discrete_sequence=[BLUE])
    fig.add_vline(x=est, line_dash="dot", line_color="gray")
    fig.update_yaxes(tickformat=".0%")
    show(fig, 320)
    st.caption(f"Model: {late_m['model_name']}. Uses only information available at checkout (no leakage).")

with tab2:
    st.markdown("##### 📬 What happened during delivery")
    c = st.columns(3)
    actual = c[0].slider("Actual delivery time (days)", 1, 90, max(1, est - 10))
    photos = c[1].slider("Product photos", 1, 10, int(sat_m["defaults"].get("product_photos_qty", 1) or 1))
    s_bad = c[2].slider("Seller's past bad-review rate", 0.0, 1.0,
                        round(sat_m["defaults"]["seller_prior_bad_rate"], 2), 0.01)
    row2 = {**row, "actual_delivery_days": float(actual), "product_photos_qty": photos, "seller_prior_bad_rate": s_bad}
    delay = actual - est
    st.caption(f"Delivered **{abs(delay)} days {'late' if delay > 0 else 'early'}** relative to the promise.")
    p2, X2 = predict(sat_m, row2)
    a, b = st.columns([2, 3])
    with a:
        gauge(p2, sat_m["threshold"], sat_m["base_rate"], "P(review is 1–3 ★)")
        if p2 >= sat_m["threshold"]:
            st.error("**Likely dissatisfied customer.** Suggested action: customer-service contact, "
                     "voucher / discount on next order before the review is written.")
        else:
            st.success("**Likely satisfied customer** (4–5 ★).")
    with b:
        contribution_chart(explain(sat_m, X2), "Why? Top factors for this order")
    grid = list(range(1, 61, 2))
    curve = [predict(sat_m, {**row2, "actual_delivery_days": float(g)})[0] for g in grid]
    fig = px.line(x=grid, y=curve, markers=True, labels={"x": "Actual delivery time (days)", "y": "P(bad review)"},
                  title="What-if: delivery time vs probability of a bad review", color_discrete_sequence=[SERIES[1]])
    fig.add_vline(x=est, line_dash="dash", line_color=BAD, annotation_text="promised date")
    fig.update_yaxes(tickformat=".0%")
    show(fig, 320)
    st.caption(f"Model: {sat_m['model_name']}.")
