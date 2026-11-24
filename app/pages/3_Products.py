import plotly.express as px
import streamlit as st

from utils import BAD, BLUE, SERIES, brl, filters, insight, load_master, page_setup, show

page_setup("Products", "🛍️")
df = filters(load_master())
st.title("🛍️ Products & Categories")

cat = df.groupby("category").agg(orders=("order_id", "size"), revenue=("order_value", "sum"),
                                 avg_price=("total_price", "mean"), avg_freight=("total_freight", "mean"),
                                 freight_ratio=("freight_ratio", "mean"), bad_rate=("bad_review", "mean"),
                                 review=("review_score", "mean"), late_rate=("is_late", "mean")).reset_index()
c = st.columns(4)
c[0].metric("Categories", f"{df['category'].nunique()}")
c[1].metric("Products sold", f"{df['product_id'].nunique():,}")
c[2].metric("Avg item price", brl(df["main_item_price"].mean()))
c[3].metric("Freight share of order", f"{df['freight_ratio'].mean():.1%}")

n = st.slider("Number of categories to show", 5, 30, 15)
a, b = st.columns(2)
with a:
    t = cat.nlargest(n, "revenue").iloc[::-1]
    fig = px.bar(t, x="revenue", y="category", orientation="h", color_discrete_sequence=[BLUE],
                 title="Top categories by revenue", labels={"category": "", "revenue": "Revenue (BRL)"},
                 hover_data={"orders": ":,", "avg_price": ":.2f"})
    show(fig, 520)
with b:
    t = cat[cat["orders"] >= 200].nlargest(n, "bad_rate").iloc[::-1]
    fig = px.bar(t, x="bad_rate", y="category", orientation="h", color_discrete_sequence=[BAD],
                 title="Highest bad-review rate (categories with ≥200 orders)",
                 labels={"category": "", "bad_rate": "Bad-review rate"}, hover_data={"orders": ":,"})
    fig.update_xaxes(tickformat=".0%")
    show(fig, 520)

big = cat[cat["orders"] >= 200]
fig = px.scatter(big, x="avg_price", y="review", size="orders", hover_name="category", log_x=True,
                 color_discrete_sequence=[SERIES[0]], title="Price vs average review (bubble = orders)",
                 labels={"avg_price": "Average order price (BRL, log)", "review": "Average review"})
fig.update_traces(marker=dict(line=dict(width=2, color="white"), opacity=0.8))
show(fig, 440)

a, b = st.columns(2)
with a:
    fig = px.histogram(df[df["total_price"] < df["total_price"].quantile(.98)], x="total_price", nbins=60,
                       color_discrete_sequence=[BLUE], title="Order price distribution (98% of orders)",
                       labels={"total_price": "Order item total (BRL)"})
    show(fig)
with b:
    t = df[df["total_price"] < 1000].sample(min(5000, len(df)), random_state=0)
    fig = px.scatter(t, x="total_price", y="total_freight", opacity=0.35, color_discrete_sequence=[SERIES[1]],
                     title="Price vs freight (sample of 5,000)", labels={"total_price": "Price", "total_freight": "Freight"})
    show(fig)

insight("**Health & beauty, watches/gifts and bed-bath-table** lead revenue. Office furniture and some "
        "home categories have the worst bad-review rates — bulky items with long, error-prone deliveries. "
        "For cheap items freight can exceed 30% of the order value, which hurts conversion.")

with st.expander("Category table"):
    st.dataframe(cat.sort_values("revenue", ascending=False).style.format(
        {"revenue": "R$ {:,.0f}", "avg_price": "R$ {:,.2f}", "avg_freight": "R$ {:,.2f}", "freight_ratio": "{:.1%}",
         "bad_rate": "{:.1%}", "review": "{:.2f}", "late_rate": "{:.1%}"}), hide_index=True, width="stretch")
