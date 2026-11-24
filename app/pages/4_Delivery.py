import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils import BAD, BLUE, SEQ, SERIES, filters, insight, load_master, page_setup, show

page_setup("Delivery", "🚚")
df = filters(load_master())
dlv = df[df["is_delivered"] == 1]
st.title("🚚 Delivery Performance")

c = st.columns(5)
c[0].metric("Avg delivery time", f"{dlv['actual_delivery_days'].mean():.1f} days")
c[1].metric("Avg promised time", f"{dlv['estimated_delivery_days'].mean():.1f} days")
c[2].metric("Late deliveries", f"{dlv['is_late'].mean():.1%}")
c[3].metric("Median arrives early by", f"{-dlv['delay_days'].median():.1f} days")
c[4].metric("Same-state orders", f"{dlv['same_state'].mean():.1%}")

m = dlv.groupby("purchase_year_month").agg(late_rate=("is_late", "mean"), actual=("actual_delivery_days", "mean"),
                                           estimated=("estimated_delivery_days", "mean")).reset_index()
a, b = st.columns(2)
with a:
    fig = px.line(m, x="purchase_year_month", y=["estimated", "actual"], markers=True,
                  title="Promised vs actual delivery time (days)",
                  labels={"purchase_year_month": "", "value": "Days", "variable": ""})
    fig.update_layout(hovermode="x unified")
    show(fig)
with b:
    fig = px.bar(m, x="purchase_year_month", y="late_rate", color_discrete_sequence=[BAD],
                 title="Late-delivery rate by purchase month", labels={"purchase_year_month": "", "late_rate": "Late rate"})
    fig.update_yaxes(tickformat=".0%")
    show(fig)

a, b = st.columns(2)
with a:
    d = dlv["delay_days"].clip(-40, 30)
    fig = px.histogram(d, nbins=70, color_discrete_sequence=[BLUE], title="Delivered date − estimated date (days)",
                       labels={"value": "Days (negative = early)"})
    fig.add_vline(x=0, line_color=BAD, line_width=2, annotation_text="late →", annotation_position="top right")
    fig.update_layout(showlegend=False)
    show(fig)
with b:
    dlv2 = dlv.assign(distance_band=pd.cut(dlv["distance_km"], [-1, 100, 300, 600, 1000, 2000, 5000],
                                           labels=["<100", "100-300", "300-600", "600-1k", "1k-2k", ">2k"]))
    g = dlv2.groupby("distance_band", observed=True).agg(days=("actual_delivery_days", "mean"),
                                                         late=("is_late", "mean")).reset_index()
    fig = px.bar(g, x="distance_band", y="days", color_discrete_sequence=[BLUE], text_auto=".1f",
                 title="Avg delivery days by seller→customer distance (km)",
                 labels={"distance_band": "Distance (km)", "days": "Days"}, hover_data={"late": ":.1%"})
    show(fig)

st.subheader("Routes: seller state → customer state")
top_s = dlv["seller_state"].value_counts().index[:8]
top_c = dlv["customer_state"].value_counts().index[:12]
r = dlv[dlv["seller_state"].isin(top_s) & dlv["customer_state"].isin(top_c)]
metric = st.radio("Route metric", ["actual_delivery_days", "is_late"], horizontal=True,
                  format_func=lambda x: "Avg delivery days" if x == "actual_delivery_days" else "Late rate")
p = r.pivot_table(index="seller_state", columns="customer_state", values=metric, aggfunc="mean")
fig = px.imshow(p, color_continuous_scale=SEQ, text_auto=".2f" if metric == "is_late" else ".0f", aspect="auto",
                labels=dict(x="Customer state", y="Seller state", color=""))
show(fig, 420)

insight("Olist's promised dates are **very conservative** (orders arrive ~11 days early on average), yet ~7% still arrive late. "
        "Late rates spike after Black Friday 2017 and in Feb–Mar 2018 — the logistics network didn't scale with demand. "
        "Distance is the main structural driver: >2,000 km routes take 2–3× longer.")
