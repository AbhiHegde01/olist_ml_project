import pandas as pd
import plotly.express as px
import streamlit as st

from utils import BLUE, SEQ, WEEKDAYS, brl, filters, insight, load_master, page_setup, show

page_setup("Sales Trends", "📈")
df = filters(load_master())
st.title("📈 Sales Trends")

m = df.groupby("purchase_year_month").agg(orders=("order_id", "size"), revenue=("order_value", "sum"),
                                          aov=("order_value", "mean")).reset_index()
m["growth"] = m["orders"].pct_change() * 100

c = st.columns(4)
c[0].metric("Revenue", brl(m["revenue"].sum()))
c[1].metric("Best month (orders)", m.loc[m["orders"].idxmax(), "purchase_year_month"])
c[2].metric("Avg monthly orders", f"{m['orders'].mean():,.0f}")
c[3].metric("Avg order value", brl(df["order_value"].mean()))

a, b = st.columns(2)
with a:
    fig = px.area(m, x="purchase_year_month", y="revenue", title="Monthly revenue (BRL)",
                  labels={"purchase_year_month": "", "revenue": "Revenue"}, color_discrete_sequence=[BLUE])
    fig.update_traces(hovertemplate="%{x}<br>R$ %{y:,.0f}<extra></extra>")
    show(fig)
with b:
    fig = px.bar(m, x="purchase_year_month", y="growth", title="Month-over-month order growth (%)",
                 labels={"purchase_year_month": "", "growth": "Growth %"}, color_discrete_sequence=[BLUE])
    fig.update_traces(hovertemplate="%{x}<br>%{y:+.1f}%<extra></extra>")
    show(fig)

daily = df.groupby("purchase_date").agg(orders=("order_id", "size")).reset_index()
daily["7-day average"] = daily["orders"].rolling(7, min_periods=1).mean()
fig = px.line(daily, x="purchase_date", y=["orders", "7-day average"], title="Daily orders",
              labels={"purchase_date": "", "value": "Orders", "variable": ""})
fig.update_traces(selector=dict(name="orders"), line=dict(width=1), opacity=0.45)
fig.update_traces(selector=dict(name="7-day average"), line=dict(width=2))
top = daily.loc[daily["orders"].idxmax()]
fig.add_annotation(x=top["purchase_date"], y=top["orders"], text=f"{top['purchase_date']:%d %b %Y}: {top['orders']:,} orders",
                   showarrow=True, arrowhead=0, ay=-30)
fig.update_layout(hovermode="x unified")
show(fig)

a, b = st.columns(2)
with a:
    p = df.pivot_table(index="purchase_weekday", columns="purchase_hour", values="order_id", aggfunc="count").fillna(0)
    p.index = [WEEKDAYS[i] for i in p.index]
    fig = px.imshow(p, color_continuous_scale=SEQ, aspect="auto", title="When do customers buy? (weekday × hour)",
                    labels=dict(x="Hour of day", y="", color="Orders"))
    show(fig)
with b:
    w = df.groupby("purchase_weekday").agg(orders=("order_id", "size"), aov=("order_value", "mean")).reset_index()
    w["day"] = w["purchase_weekday"].map(dict(enumerate(WEEKDAYS)))
    fig = px.bar(w, x="day", y="orders", title="Orders by weekday", color_discrete_sequence=[BLUE],
                 labels={"day": "", "orders": "Orders"}, hover_data={"aov": ":.2f", "day": False})
    show(fig)

insight("Orders grew roughly **10× during 2017** and stabilised at ~6–7k/month in 2018. "
        "**Black Friday (24-Nov-2017)** is the single busiest day. Customers mostly shop on weekdays between 10h and 22h — "
        "marketing pushes and support staffing should follow this pattern.")

with st.expander("Monthly table"):
    st.dataframe(m.style.format({"revenue": "R$ {:,.0f}", "aov": "R$ {:,.2f}", "growth": "{:+.1f}%"}),
                 width="stretch", hide_index=True)
