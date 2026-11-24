import pandas as pd
import plotly.express as px
import streamlit as st

from utils import BAD, BLUE, GOOD, SERIES, filters, insight, load_comments, load_master, page_setup, show

page_setup("Reviews", "⭐")
df = filters(load_master())
dlv = df[(df["is_delivered"] == 1) & df["review_score"].notna()]
st.title("⭐ Reviews & Customer Satisfaction")

c = st.columns(4)
c[0].metric("Average score", f"{dlv['review_score'].mean():.2f} ★")
c[1].metric("5-star share", f"{(dlv['review_score'] == 5).mean():.1%}")
c[2].metric("Bad reviews (1–3★)", f"{dlv['bad_review'].mean():.1%}")
c[3].metric("Reviews with comment", f"{dlv['review_has_comment'].mean():.1%}")

a, b = st.columns(2)
with a:
    s = dlv["review_score"].value_counts().sort_index().reset_index()
    s.columns = ["score", "orders"]
    s["score"] = s["score"].astype(int).astype(str) + "★"
    fig = px.bar(s, x="score", y="orders", color_discrete_sequence=[BLUE], title="Review score distribution")
    show(fig)
with b:
    bins = pd.cut(dlv["delay_days"], [-200, -15, -10, -5, 0, 3, 7, 14, 200],
                  labels=["≤-15", "-15..-10", "-10..-5", "-5..0", "0..3", "3..7", "7..14", ">14"])
    g = dlv.groupby(bins, observed=True)["bad_review"].mean().reset_index()
    g.columns = ["delay", "bad_rate"]
    g["late"] = ~g["delay"].astype(str).str.startswith(("≤", "-"))
    fig = px.bar(g, x="delay", y="bad_rate", color="late", color_discrete_map={False: BLUE, True: BAD},
                 title="Bad-review rate by delivery delay (days vs promised date)",
                 labels={"delay": "Delay (days)", "bad_rate": "Bad-review rate", "late": "Late"})
    fig.update_yaxes(tickformat=".0%")
    show(fig)

box = dlv[["review_score", "actual_delivery_days"]].assign(
    stars=lambda d: d["review_score"].astype(int).astype(str) + "★",
    group=lambda d: (d["review_score"] <= 3).map({True: "bad (1-3★)", False: "good (4-5★)"}))
fig = px.box(box.sort_values("stars"), x="stars", y="actual_delivery_days", color="group", points=False,
             color_discrete_map={"bad (1-3★)": BAD, "good (4-5★)": BLUE},
             title="Boxplot: delivery time by review score (box = middle 50%, line = median)",
             labels={"stars": "Review score", "actual_delivery_days": "Actual delivery (days)", "group": ""})
fig.update_yaxes(range=[0, 70])
show(fig, 400)
st.caption("Each box spans the 25th–75th percentile; whiskers reach 1.5× IQR. 1★ orders took a median "
           "~17 days vs ~9 days for 5★, and their spread is much wider.")

drivers = {
    "Delivered late": dlv.groupby("is_late")["bad_review"].mean(),
    "Multiple sellers": dlv.groupby(dlv["n_sellers"] > 1)["bad_review"].mean(),
    "More than 1 item": dlv.groupby(dlv["n_items"] > 1)["bad_review"].mean(),
    "Seller in other state": dlv.groupby(dlv["same_state"] == 0)["bad_review"].mean(),
    "Paid by boleto": dlv.groupby(dlv["payment_type"] == "boleto")["bad_review"].mean(),
}
rows = [{"factor": k, "Without": v.iloc[0], "With": v.iloc[1]} for k, v in drivers.items()]
d = pd.DataFrame(rows).melt(id_vars="factor", var_name="group", value_name="bad_rate")
fig = px.bar(d, x="bad_rate", y="factor", color="group", barmode="group", orientation="h",
             color_discrete_map={"Without": BLUE, "With": SERIES[1]},
             title="Bad-review rate with vs without each factor", labels={"factor": "", "bad_rate": "Bad-review rate"})
fig.update_xaxes(tickformat=".0%")
show(fig, 400)

m = dlv.groupby("purchase_year_month").agg(score=("review_score", "mean"), late=("is_late", "mean")).reset_index()
a, b = st.columns(2)
with a:
    fig = px.line(m, x="purchase_year_month", y="score", markers=True, color_discrete_sequence=[GOOD],
                  title="Average review by purchase month", labels={"purchase_year_month": "", "score": "Avg score"})
    show(fig)
with b:
    fig = px.scatter(m, x="late", y="score", text="purchase_year_month", color_discrete_sequence=[BLUE],
                     title="Monthly late rate vs average review", labels={"late": "Late rate", "score": "Avg score"},
                     trendline=None)
    fig.update_traces(textposition="top center", textfont_size=9, marker_size=10)
    fig.update_xaxes(tickformat=".0%")
    show(fig)

insight("Satisfaction is driven by **logistics, not product**: once an order is more than a week late, "
        "~80% of reviews are bad. Months with high late rates (Nov-17, Feb/Mar-18) have the lowest scores. "
        "Orders split across several sellers also generate more complaints (items arrive separately).")

with st.expander("Read some customer comments (Portuguese)"):
    c1, c2 = st.columns([1, 3])
    score = c1.selectbox("Score", [1, 2, 3, 4, 5], index=0)
    if c2.toggle("Load comments", help="Review text is loaded only when you ask for it, to keep the app light."):
        comments = load_comments()
        pool = dlv.loc[dlv["review_score"] == score, ["order_id", "purchase_date", "category", "is_late"]]
        pool = pool.join(comments, on="order_id", how="inner")
        st.dataframe(pool.drop(columns="order_id").sample(min(10, len(pool)), random_state=1),
                     hide_index=True, width="stretch")
