import plotly.express as px
import streamlit as st

from utils import BLUE, SERIES, brl, filters, insight, load_master, page_setup, show

page_setup("Payments", "💳")
df = filters(load_master())
st.title("💳 Payments")

c = st.columns(4)
c[0].metric("Credit card share", f"{(df['payment_type'] == 'credit_card').mean():.1%}")
c[1].metric("Avg installments", f"{df['payment_installments'].mean():.1f}")
c[2].metric("Orders using vouchers", f"{df['uses_voucher'].mean():.1%}")
c[3].metric("Avg payment", brl(df["payment_value"].mean()))

a, b = st.columns(2)
with a:
    p = df.groupby("payment_type").agg(orders=("order_id", "size"), value=("payment_value", "sum")).reset_index()
    p = p.sort_values("orders", ascending=False)
    fig = px.bar(p, x="payment_type", y="orders", color_discrete_sequence=[BLUE], title="Orders by payment type",
                 labels={"payment_type": ""}, hover_data={"value": ":,.0f"}, text_auto=",")
    show(fig)
with b:
    i = df.assign(inst=df["payment_installments"].clip(upper=12)).groupby("inst").agg(
        orders=("order_id", "size"), aov=("order_value", "mean")).reset_index()
    fig = px.bar(i, x="inst", y="aov", color_discrete_sequence=[SERIES[1]], title="Average order value by installments",
                 labels={"inst": "Installments (12 = 12+)", "aov": "Avg order value (BRL)"}, hover_data={"orders": ":,"})
    show(fig)

m = df.groupby(["purchase_year_month", "payment_type"]).size().reset_index(name="orders")
fig = px.area(m, x="purchase_year_month", y="orders", color="payment_type", groupnorm="percent",
              title="Payment-type mix over time (% of orders)",
              category_orders={"payment_type": ["credit_card", "boleto", "voucher", "debit_card"]},
              labels={"purchase_year_month": "", "orders": "% of orders", "payment_type": ""})
show(fig, 400)

g = df[df["is_delivered"] == 1].groupby("payment_type").agg(review=("review_score", "mean"),
                                                          bad=("bad_review", "mean"), late=("is_late", "mean"),
                                                          approval_h=("approval_hours", "mean")).reset_index()
st.dataframe(g.style.format({"review": "{:.2f}", "bad": "{:.1%}", "late": "{:.1%}", "approval_h": "{:.1f} h"}),
             hide_index=True, width="stretch")

insight("Three in four orders are paid by credit card, usually in installments — the more expensive the order, "
        "the more installments. **Boleto** (bank slip) orders take ~1 day longer to be approved, which delays shipping.")
