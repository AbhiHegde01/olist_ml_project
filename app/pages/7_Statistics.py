import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from scipy import stats

from utils import filters, insight, load_master, page_setup, show

page_setup("Statistics", "🧮")
df = filters(load_master())
dlv = df[(df["is_delivered"] == 1) & df["bad_review"].notna()].copy()
dlv["bad_review"] = dlv["bad_review"].astype(int)
dlv["is_late"] = dlv["is_late"].astype(int)
st.title("🧮 Statistical Analysis & Naive Bayes")

tab1, tab2, tab3 = st.tabs(["Hypothesis tests", "Correlation", "Naive Bayes — step by step"])

# --------------------------------------------------------------------------- #
with tab1:
    st.markdown("Significance level **α = 0.05**. With tens of thousands of orders, tiny effects become "
                "significant, so we also report an **effect size**.")

    def cramers_v(ct):
        chi2 = stats.chi2_contingency(ct)[0]
        return np.sqrt(chi2 / (ct.values.sum() * (min(ct.shape) - 1)))

    rows = []
    ct = pd.crosstab(dlv["is_late"], dlv["bad_review"])
    chi2, p, dof, exp = stats.chi2_contingency(ct)
    rows.append(["χ² independence", "is_late ⟂ bad_review", chi2, p, f"Cramér's V = {cramers_v(ct):.3f}"])
    ct2 = pd.crosstab(dlv["payment_type"], dlv["bad_review"])
    c2, p2, *_ = stats.chi2_contingency(ct2)
    rows.append(["χ² independence", "payment_type ⟂ bad_review", c2, p2, f"Cramér's V = {cramers_v(ct2):.3f}"])
    g = dlv.loc[dlv["bad_review"] == 0, "actual_delivery_days"]
    b = dlv.loc[dlv["bad_review"] == 1, "actual_delivery_days"]
    t, pt = stats.ttest_ind(b, g, equal_var=False)
    d = (b.mean() - g.mean()) / np.sqrt((b.var() + g.var()) / 2)
    rows.append(["Welch t-test", "delivery days: bad vs good reviews", t, pt,
                 f"Cohen's d = {d:.2f} ({b.mean():.1f} vs {g.mean():.1f} days)"])
    u, pu = stats.mannwhitneyu(b, g)
    rows.append(["Mann-Whitney U", "delivery days: bad vs good reviews", u, pu, "non-parametric"])
    top = dlv["customer_state"].value_counts().index[:10]
    groups = [dlv.loc[dlv["customer_state"] == s, "review_score"] for s in top]
    if len(groups) > 1:
        f, pf = stats.f_oneway(*groups)
        rows.append(["One-way ANOVA", "review score across top-10 states", f, pf, ""])
        h, ph = stats.kruskal(*groups)
        rows.append(["Kruskal-Wallis", "review score across top-10 states", h, ph, ""])
    r, pr = stats.pearsonr(dlv["distance_km"], dlv["total_freight"])
    rows.append(["Pearson r", "distance ↔ freight", r, pr, f"r = {r:.3f}"])
    rho, prho = stats.spearmanr(dlv["distance_km"], dlv["actual_delivery_days"])
    rows.append(["Spearman ρ", "distance ↔ delivery days", rho, prho, f"ρ = {rho:.3f}"])
    res = pd.DataFrame(rows, columns=["Test", "H₀ / variables", "Statistic", "p-value", "Effect size"])
    res["Decision"] = np.where(res["p-value"] < 0.05, "Reject H₀", "Fail to reject H₀")
    st.dataframe(res.style.format({"Statistic": "{:,.2f}", "p-value": "{:.2e}"}), hide_index=True, width="stretch")

    a, b_ = st.columns(2)
    with a:
        st.markdown("**Contingency table: late × bad review (observed)**")
        st.dataframe(ct.rename(index={0: "on time", 1: "late"}, columns={0: "good", 1: "bad"}), width="stretch")
    with b_:
        st.markdown("**Expected counts if independent**")
        st.dataframe(pd.DataFrame(exp, index=["on time", "late"], columns=["good", "bad"]).round(0), width="stretch")
    st.latex(r"\chi^2=\sum_{i,j}\frac{(O_{ij}-E_{ij})^2}{E_{ij}},\qquad E_{ij}=\frac{R_i\,C_j}{N},\qquad V=\sqrt{\frac{\chi^2}{N(\min(r,c)-1)}}")
    insight("Every test rejects H₀: late deliveries and long delivery times are strongly associated with bad reviews "
            "(Cramér's V ≈ 0.3, Cohen's d ≈ 0.5+) — the statistical basis for our satisfaction model.")

# --------------------------------------------------------------------------- #
with tab2:
    cols = ["order_value", "total_freight", "freight_ratio", "n_items", "n_sellers", "payment_installments",
            "distance_km", "estimated_delivery_days", "actual_delivery_days", "delay_days", "is_late",
            "seller_prior_late_rate", "review_score"]
    method = st.radio("Method", ["spearman", "pearson"], horizontal=True)
    cm = dlv[cols].corr(method=method)
    fig = px.imshow(cm, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto")
    show(fig, 600)
    st.caption("Diverging scale: red = positive, blue = negative, white = no correlation.")

# --------------------------------------------------------------------------- #
with tab3:
    st.markdown("Choose an order's characteristics. The posterior is computed **by hand** with Bayes' theorem, "
                "the naive independence assumption and Laplace smoothing (α = 1).")
    st.latex(r"P(C\mid x)=\frac{P(C)\prod_j P(x_j\mid C)}{\sum_{c}P(c)\prod_j P(x_j\mid c)},\qquad "
             r"P(x_j=v\mid C)=\frac{N_{C,j,v}+\alpha}{N_C+\alpha K_j}")
    feats = ["is_late", "payment_type", "same_state", "n_sellers_gt1"]
    data = dlv.assign(n_sellers_gt1=(dlv["n_sellers"] > 1).astype(int))
    c1, c2, c3, c4 = st.columns(4)
    q = {"is_late": c1.selectbox("Delivered late?", [0, 1], index=1, format_func=lambda x: "Yes" if x else "No"),
         "payment_type": c2.selectbox("Payment type", sorted(data["payment_type"].dropna().unique())),
         "same_state": c3.selectbox("Seller in same state?", [0, 1], format_func=lambda x: "Yes" if x else "No"),
         "n_sellers_gt1": c4.selectbox("Multiple sellers?", [0, 1], format_func=lambda x: "Yes" if x else "No")}
    alpha, N = 1.0, len(data)
    rows, unnorm = [], {}
    for c in [0, 1]:
        sub = data[data["bad_review"] == c]
        prior = len(sub) / N
        prod = prior
        for f in feats:
            K = data[f].nunique()
            cnt = int((sub[f] == q[f]).sum())
            p = (cnt + alpha) / (len(sub) + alpha * K)
            prod *= p
            rows.append({"class": "bad (1)" if c else "good (0)", "feature": f, "value": str(q[f]),
                         "count": cnt, "N_C": len(sub), "K": K, "P(x|C)": p})
        unnorm[c] = (prior, prod)
    st.dataframe(pd.DataFrame(rows).style.format({"P(x|C)": "{:.4f}"}), hide_index=True, width="stretch")
    ev = unnorm[0][1] + unnorm[1][1]
    a, b_, c_ = st.columns(3)
    a.metric("Prior P(bad)", f"{unnorm[1][0]:.3f}")
    b_.metric("Evidence P(x)", f"{ev:.5f}")
    c_.metric("Posterior P(bad | x)", f"{unnorm[1][1] / ev:.1%}", delta=f"{unnorm[1][1] / ev - unnorm[1][0]:+.1%} vs prior",
              delta_color="inverse")
    st.markdown(f"P(good)·ΠP(x|good) = `{unnorm[0][1]:.6f}`  P(bad)·ΠP(x|bad) = `{unnorm[1][1]:.6f}` → "
                f"predicted class: **{'BAD review' if unnorm[1][1] > unnorm[0][1] else 'good review'}**")
    st.caption("Notebook 03 shows the same computation and verifies it equals scikit-learn's CategoricalNB.")
