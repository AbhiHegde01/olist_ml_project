import plotly.express as px
import streamlit as st

from utils import BAD, BLUE, brl, filters, insight, load_master, load_metrics, page_setup, show

page_setup("Overview", "📦")
df = filters(load_master())

st.title("📦 Olist Brazilian E-Commerce — Analytics & Prediction")
st.caption("End-to-end ML project: data cleaning → statistical analysis → feature engineering → "
           "predictive models for **late deliveries** and **customer dissatisfaction**.")

dlv = df[df["is_delivered"] == 1]
c = st.columns(6)
c[0].metric("Orders", f"{len(df):,}")
c[1].metric("Revenue", brl(df["order_value"].sum()))
c[2].metric("Avg order value", brl(df["order_value"].mean()))
c[3].metric("Avg review", f"{df['review_score'].mean():.2f} ★")
c[4].metric("Late deliveries", f"{dlv['is_late'].mean():.1%}")
c[5].metric("Unique customers", f"{df['customer_unique_id'].nunique():,}")

left, right = st.columns([3, 2])
with left:
    m = df.groupby("purchase_year_month").agg(orders=("order_id", "size")).reset_index()
    fig = px.bar(m, x="purchase_year_month", y="orders", title="Orders per month",
                 labels={"purchase_year_month": "", "orders": "Orders"}, color_discrete_sequence=[BLUE])
    fig.update_traces(hovertemplate="%{x}<br>%{y:,} orders<extra></extra>")
    show(fig)
with right:
    s = (dlv.groupby("is_late")["review_score"].mean().rename({0: "On time", 1: "Late"}).reset_index())
    s.columns = ["Delivery", "Average review"]
    fig = px.bar(s, x="Delivery", y="Average review", text_auto=".2f", color="Delivery",
                 color_discrete_map={"On time": BLUE, "Late": BAD},
                 title="Late deliveries destroy satisfaction")
    fig.update_layout(showlegend=False, yaxis_range=[0, 5])
    show(fig)

insight("**Headline finding:** a late order receives on average ~1.8 fewer stars, and the bad-review "
        "rate more than triples. Delivery reliability is the main lever for customer satisfaction — "
        "which is why we built a model that flags risky orders at checkout.")

st.subheader("How to use this dashboard")
st.markdown("""
| Page | What you'll find |
|---|---|
| **Sales Trends** | monthly revenue, growth, Black Friday, weekday × hour heatmap |
| **Customers & Geography** | state map, delivery time by state, RFM segments, repeat customers |
| **Products** | category revenue, prices, freight and bad-review rate by category |
| **Delivery** | estimated vs actual, late rate over time and by route |
| **Reviews** | score distribution and drivers of bad reviews |
| **Payments** | methods, installments, vouchers |
| **Statistics** | hypothesis tests (χ², t-test, ANOVA), correlations |
| **Model Performance** | model comparison, confusion matrix, ROC/PR, feature importance |
| **Predictor** | enter an order → late-delivery risk and bad-review probability |
""")

metrics = load_metrics()
if metrics:
    st.subheader("Model results at a glance (test period, unseen data)")
    cols = st.columns(len(metrics))
    for col, m in zip(cols, metrics.values()):
        with col:
            st.markdown(f"**{m['title']}** — {m['best_model']}")
            t = m["test"]
            a, b, cc = st.columns(3)
            a.metric("ROC-AUC", f"{t['roc_auc']:.3f}")
            b.metric("PR-AUC", f"{t['pr_auc']:.3f}", help=f"random baseline = {m['pos_rate_test']:.3f}")
            cc.metric("Recall", f"{t['recall']:.1%}")
st.caption("Use the sidebar filters — they apply to every analysis page.")
