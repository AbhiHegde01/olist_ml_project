import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from utils import BLUE, SEQ, SERIES, STATE_CENTROIDS, STATE_NAMES, filters, insight, load_master, page_setup, show

page_setup("Customers & Geography", "🗺️")
df = filters(load_master())
st.title("🗺️ Customers & Geography")

dlv = df[df["is_delivered"] == 1]
st_ = dlv.groupby("customer_state").agg(orders=("order_id", "size"), revenue=("order_value", "sum"),
                                         delivery_days=("actual_delivery_days", "mean"),
                                         late_rate=("is_late", "mean"), review=("review_score", "mean"),
                                         freight=("total_freight", "mean")).reset_index()
st_["state_name"] = st_["customer_state"].map(STATE_NAMES)
st_["lat"] = st_["customer_state"].map(lambda s: STATE_CENTROIDS[s][0])
st_["lon"] = st_["customer_state"].map(lambda s: STATE_CENTROIDS[s][1])

metric = st.radio("Map metric", ["orders", "revenue", "delivery_days", "late_rate", "review", "freight"],
                  horizontal=True, format_func=lambda x: {"orders": "Orders", "revenue": "Revenue",
                  "delivery_days": "Avg delivery days", "late_rate": "Late rate", "review": "Avg review",
                  "freight": "Avg freight"}[x])
a, b = st.columns([3, 2])
with a:
    fig = px.scatter_geo(st_, lat="lat", lon="lon", size="orders", color=metric, hover_name="state_name",
                         color_continuous_scale=SEQ, size_max=45, title=f"Customer states — {metric.replace('_', ' ')}",
                         hover_data={"lat": False, "lon": False, "orders": ":,", "revenue": ":,.0f",
                                     "delivery_days": ":.1f", "late_rate": ":.1%", "review": ":.2f"})
    fig.update_geos(scope="south america", fitbounds="locations", showcountries=True, showland=True,
                    landcolor="rgba(128,128,128,.08)")
    show(fig, 480)
with b:
    t = st_.sort_values(metric, ascending=False).head(15)
    fig = px.bar(t.iloc[::-1], x=metric, y="customer_state", orientation="h", color_discrete_sequence=[BLUE],
                 title=f"Top 15 states by {metric.replace('_', ' ')}", labels={"customer_state": ""},
                 hover_name="state_name")
    show(fig, 480)

insight("São Paulo (SP) alone generates ~42% of orders and gets the fastest deliveries (~8 days). "
        "Northern states (RR, AP, AM) wait 3× longer and pay much higher freight — and give lower reviews.")

st.subheader("Customer loyalty — RFM segmentation")
ref = df["order_purchase_timestamp"].max() + pd.Timedelta(days=1)
rfm = df.groupby("customer_unique_id").agg(
    recency=("order_purchase_timestamp", lambda s: (ref - s.max()).days),
    frequency=("order_id", "nunique"), monetary=("order_value", "sum"))
rfm["R"] = pd.qcut(rfm["recency"].rank(method="first"), 4, labels=[4, 3, 2, 1]).astype(int)
rfm["M"] = pd.qcut(rfm["monetary"].rank(method="first"), 4, labels=[1, 2, 3, 4]).astype(int)
rfm["segment"] = np.select(
    [(rfm["frequency"] > 1) & (rfm["R"] >= 3), rfm["frequency"] > 1, (rfm["R"] >= 3) & (rfm["M"] >= 3),
     rfm["R"] >= 3, rfm["M"] >= 3],
    ["Loyal (repeat, recent)", "Repeat, lapsing", "Recent big spender", "Recent", "Lapsed big spender"],
    "Lapsed low value")

c = st.columns(4)
c[0].metric("Unique customers", f"{len(rfm):,}")
c[1].metric("Repeat customers", f"{(rfm['frequency'] > 1).mean():.1%}")
c[2].metric("Avg spend / customer", f"R$ {rfm['monetary'].mean():,.0f}")
c[3].metric("Median recency", f"{rfm['recency'].median():.0f} days")

a, b = st.columns(2)
with a:
    seg = rfm.groupby("segment").agg(customers=("R", "size"), avg_spend=("monetary", "mean")).reset_index()
    fig = px.bar(seg.sort_values("customers"), x="customers", y="segment", orientation="h",
                 color_discrete_sequence=[BLUE], title="Customers per RFM segment", labels={"segment": ""},
                 hover_data={"avg_spend": ":,.2f"})
    show(fig)
with b:
    f = rfm["frequency"].clip(upper=5).value_counts().sort_index().reset_index()
    f.columns = ["orders per customer", "customers"]
    f["orders per customer"] = f["orders per customer"].astype(str).replace({"5": "5+"})
    fig = px.bar(f, x="orders per customer", y="customers", log_y=True, color_discrete_sequence=[SERIES[1]],
                 title="Orders per customer (log scale)")
    show(fig)

insight("Only ~3% of customers ever place a second order. Acquisition drives growth; **retention is the "
        "untapped opportunity** — and satisfied customers (on-time delivery) are the ones who come back.")

with st.expander("State table"):
    st.dataframe(st_.drop(columns=["lat", "lon"]).sort_values("orders", ascending=False).style.format(
        {"revenue": "R$ {:,.0f}", "delivery_days": "{:.1f}", "late_rate": "{:.1%}", "review": "{:.2f}",
         "freight": "R$ {:.2f}"}), hide_index=True, width="stretch")
