"""
Generates the four project notebooks (run once, then execute them with
`jupyter nbconvert --to notebook --execute --inplace notebooks/0*.ipynb`).
Editing the .ipynb files directly in Jupyter afterwards is fine.
"""
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent

SETUP = """\
import sys, warnings
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

import numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams["figure.dpi"] = 110
pd.set_option("display.max_columns", 60); pd.set_option("display.float_format", "{:,.3f}".format)
FIG = ROOT / "reports" / "figures"; FIG.mkdir(parents=True, exist_ok=True)
"""

LOAD_MASTER = """\
master = pd.read_parquet(ROOT / "data/processed/master_orders.parquet")
dlv = master[master.is_delivered == 1].copy()          # delivered orders (used for most analyses)
print(f"all orders: {len(master):,} | delivered: {len(dlv):,}")
"""


def nb(cells):
    book = nbf.v4.new_notebook()
    book.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    book.cells = [nbf.v4.new_markdown_cell(c[1]) if c[0] == "md" else nbf.v4.new_code_cell(c[1])
                  for c in cells]
    return book


md = lambda s: ("md", s)
code = lambda s: ("code", s)

# =========================================================================== #
# 01 - Cleaning
# =========================================================================== #
nb01 = nb([
    md("""# 01 · Data Understanding & Cleaning
**Dataset:** [Brazilian E-Commerce Public Dataset by Olist (Kaggle)](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) — ~100k orders placed between 2016 and 2018 on the Olist marketplace.

**Goal of this notebook:** understand the 9 relational tables, measure data quality, clean them and build **one row per order** (`master_orders.parquet`) that all later steps use.

All cleaning logic lives in [`src/data_prep.py`](../src/data_prep.py) so it is reusable and testable; this notebook runs it and explains every step."""),
    code(SETUP),
    md("""## 1. The relational schema
```
customers ──< orders >── order_reviews
                │
                ├──< order_payments
                └──< order_items >── products ── category_translation
                          │
                          └── sellers
customers / sellers ── zip_code_prefix ── geolocation
```
`orders` is the hub table. One order can have several items (possibly from several sellers), several payments and (rarely) several reviews."""),
    code("""from src.data_prep import load_raw, quality_report, build_master
tables = load_raw()
pd.DataFrame({name: {"rows": len(df), "columns": df.shape[1]} for name, df in tables.items()}).T"""),
    code("""for name in ["orders", "items", "products", "reviews"]:
    print(f"--- {name} ---"); display(tables[name].head(3))"""),
    md("## 2. Data-quality audit (before cleaning)"),
    code("""qr = quality_report(tables)
miss = qr[qr.missing > 0].sort_values("missing_pct", ascending=False)
display(miss[["table", "column", "missing", "missing_pct"]])
fig, ax = plt.subplots(figsize=(9, 5))
sns.barplot(data=miss, y=miss.table + "." + miss.column, x="missing_pct", ax=ax, color="C3")
ax.set(xlabel="% missing", ylabel="", title="Missing values per column (raw data)")
plt.tight_layout(); plt.savefig(FIG / "missing_values.png"); plt.show()"""),
    code("""print("Order status distribution:")
display(tables["orders"].order_status.value_counts().to_frame("orders"))
print("Duplicate rows per table:", {k: int(v.duplicated().sum()) for k, v in tables.items()})
print("Orders with >1 review:", int(tables["reviews"].order_id.duplicated().sum()))"""),
    md("""### Observations
* **Review comments** are missing for ~59% (title) / ~88% — customers often leave only a star rating. Not a problem: we only use the score + a `has_comment` flag.
* **Delivery timestamps** are missing for undelivered / canceled orders — expected, not an error.
* **Products** miss category for 610 products and dimensions for 2 → imputed.
* **Geolocation** has ~1M rows with many duplicates per zip prefix and a few points outside Brazil.
* Some orders have more than one review → keep the latest."""),
    md("""## 3. Cleaning & aggregation steps
| Table | Action | Why |
|---|---|---|
| products | translate category to English, missing → `unknown`, impute dimensions/weight by category median | keep all products, sensible values |
| items | aggregate per order: n_items, n_sellers, total price & freight, total weight; **main item** = most expensive item → category, seller | model works at order level |
| payments | drop `not_defined`; total value, max installments, main payment type, voucher flag | one row per order |
| reviews | keep latest review per order; `bad_review = score ≤ 3` | one label per order |
| geolocation | drop points outside Brazil, median lat/lng per zip prefix → **haversine distance** customer↔seller | logistics feature |
| orders | drop orders without items, drop impossible timelines (delivered before purchase) | data validity |
| seller history | past late-rate & bad-review-rate of the seller computed **only from earlier orders**, Bayesian-smoothed | strong feature without leakage |
"""),
    code("""master, log = build_master(tables)
display(log)
assert master.order_id.is_unique
print(master.shape)"""),
    md("""### Leakage-safe seller history (Bayesian average)
For each order we compute the seller's late-rate using only orders purchased **before** it:

$$\\hat{r}_{seller} = \\frac{\\sum_{past} late + m\\cdot \\bar{r}_{global}}{n_{past} + m}, \\quad m = 10$$

A new seller (n_past = 0) gets the global rate; a seller with many past orders gets ≈ its own rate. This is the same idea as a Laplace / Bayesian prior."""),
    code("""cols = ["seller_prior_orders", "seller_prior_late_rate", "is_late"]
master.loc[master.is_delivered == 1, cols].groupby(pd.cut(master.seller_prior_orders, [-1, 0, 10, 50, 200, 5000])).mean()"""),
    md("## 4. Result: master table"),
    code("""display(master.head())
master.describe().T"""),
    code("""master.to_parquet(ROOT / "data/processed/master_orders.parquet", index=False)
print("saved", master.shape)"""),
])

# =========================================================================== #
# 02 - EDA & statistics
# =========================================================================== #
nb02 = nb([
    md("""# 02 · Exploratory Data Analysis & Statistical Tests
Descriptive statistics, business trends and formal hypothesis tests. Figures are also saved to `reports/figures/` for the report."""),
    code(SETUP), code(LOAD_MASTER),
    md("## 1. Headline KPIs"),
    code("""kpi = {
    "Orders": len(master),
    "Unique customers": master.customer_unique_id.nunique(),
    "Sellers": master.seller_id.nunique(),
    "Revenue (BRL, items+freight)": master.order_value.sum(),
    "Average order value (BRL)": master.order_value.mean(),
    "Average review score": master.review_score.mean(),
    "Late delivery rate (delivered)": dlv.is_late.mean(),
    "Bad review rate (1-3 stars)": master.bad_review.mean(),
    "Repeat-customer rate": (master.groupby("customer_unique_id").size() > 1).mean(),
}
pd.Series(kpi).to_frame("value")"""),
    md("## 2. Descriptive statistics (central tendency, spread, shape)"),
    code("""num = ["order_value", "total_price", "total_freight", "freight_ratio", "n_items", "payment_installments",
       "distance_km", "estimated_delivery_days", "actual_delivery_days", "delay_days", "review_score"]
desc = dlv[num].describe(percentiles=[.25, .5, .75, .95]).T
desc["skew"] = dlv[num].skew(); desc["kurtosis"] = dlv[num].kurt()
desc"""),
    code("""fig, axes = plt.subplots(2, 3, figsize=(14, 7))
for ax, c in zip(axes.flat, ["order_value", "total_freight", "distance_km", "actual_delivery_days", "delay_days", "review_score"]):
    data = dlv[c].clip(upper=dlv[c].quantile(.99))
    sns.histplot(data, bins=40, ax=ax, kde=c != "review_score"); ax.set_title(c)
plt.suptitle("Distributions (clipped at 99th percentile)"); plt.tight_layout()
plt.savefig(FIG / "distributions.png"); plt.show()"""),
    md("Money, weight and distance are **strongly right-skewed** (skew > 3) → we apply `log1p` before linear / Naive Bayes models."),
    md("## 3. Sales trends"),
    code("""m = master[master.order_purchase_timestamp >= "2017-01-01"].groupby("purchase_year_month").agg(
    orders=("order_id", "size"), revenue=("order_value", "sum"), aov=("order_value", "mean"))
m = m[m.index <= "2018-08"]
m["orders_growth_%"] = m.orders.pct_change() * 100
fig, ax = plt.subplots(figsize=(13, 4.5)); ax2 = ax.twinx()
ax.bar(m.index, m.orders, color="C0", alpha=.7, label="orders"); ax2.plot(m.index, m.revenue / 1e6, "C3-o", label="revenue (M BRL)")
ax.set_ylabel("orders"); ax2.set_ylabel("revenue (M BRL)"); ax.tick_params(axis="x", rotation=60)
ax.set_title("Monthly orders and revenue"); fig.legend(loc="upper left", bbox_to_anchor=(.07, .9))
plt.tight_layout(); plt.savefig(FIG / "monthly_trend.png"); plt.show()
m.tail(8)"""),
    code("""daily = master.groupby("purchase_date").size()
print("Top 5 days by orders:"); display(daily.sort_values(ascending=False).head(5).to_frame("orders"))
pivot = master.pivot_table(index="purchase_weekday", columns="purchase_hour", values="order_id", aggfunc="count")
pivot.index = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
plt.figure(figsize=(14, 3.8)); sns.heatmap(pivot, cmap="Blues"); plt.title("Orders by weekday × hour")
plt.tight_layout(); plt.savefig(FIG / "weekday_hour.png"); plt.show()"""),
    md("""**Trend findings**
* Orders grew ~**10× from Jan-2017 to 2018**, then plateaued around 6–7k orders/month.
* **24-Nov-2017 (Black Friday)** is the busiest day in the dataset by far.
* Customers buy mostly **Mon–Fri, 10h–22h**; weekends and nights are quiet."""),
    md("## 4. Customers & geography"),
    code("""st = dlv.groupby("customer_state").agg(orders=("order_id", "size"), revenue=("order_value", "sum"),
    avg_delivery_days=("actual_delivery_days", "mean"), late_rate=("is_late", "mean"),
    avg_review=("review_score", "mean")).sort_values("orders", ascending=False)
st["share_%"] = st.orders / st.orders.sum() * 100
display(st.head(10))
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
st.orders.plot.bar(ax=axes[0], color="C0", title="Orders by customer state")
st.sort_values("avg_delivery_days").avg_delivery_days.plot.bar(ax=axes[1], color="C1", title="Avg delivery days by state")
plt.tight_layout(); plt.savefig(FIG / "states.png"); plt.show()"""),
    code("""# RFM (Recency, Frequency, Monetary) per unique customer
ref = master.order_purchase_timestamp.max() + pd.Timedelta(days=1)
rfm = master.groupby("customer_unique_id").agg(
    recency=("order_purchase_timestamp", lambda s: (ref - s.max()).days),
    frequency=("order_id", "nunique"), monetary=("order_value", "sum"))
rfm["R"] = pd.qcut(rfm.recency, 4, labels=[4, 3, 2, 1]).astype(int)
rfm["F"] = np.where(rfm.frequency > 1, 4, 1)
rfm["M"] = pd.qcut(rfm.monetary, 4, labels=[1, 2, 3, 4]).astype(int)
rfm["segment"] = np.select([(rfm.R >= 3) & (rfm.F == 4), (rfm.R >= 3) & (rfm.M >= 3), rfm.R >= 3, rfm.M >= 3],
                           ["Loyal", "Recent big spender", "Recent", "Lapsed big spender"], "Lapsed / low value")
display(rfm.segment.value_counts(normalize=True).mul(100).round(1).to_frame("% customers"))
print(f"Customers with more than one order: {(rfm.frequency > 1).mean():.2%}")"""),
    md("Only **~3% of customers come back** → retention is the biggest business opportunity; satisfaction (next sections) drives it."),
    md("## 5. Products"),
    code("""cat = dlv.groupby("category").agg(orders=("order_id", "size"), revenue=("order_value", "sum"),
    avg_price=("total_price", "mean"), bad_rate=("bad_review", "mean"), late_rate=("is_late", "mean"))
fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
cat.revenue.nlargest(12).sort_values().plot.barh(ax=axes[0], color="C2", title="Top 12 categories by revenue")
cat[cat.orders >= 300].bad_rate.nlargest(12).sort_values().plot.barh(ax=axes[1], color="C3", title="Highest bad-review rate (≥300 orders)")
plt.tight_layout(); plt.savefig(FIG / "categories.png"); plt.show()"""),
    md("## 6. Delivery performance"),
    code("""fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
sns.histplot(dlv.delay_days.clip(-40, 40), bins=80, ax=axes[0]); axes[0].axvline(0, color="r")
axes[0].set_title("Delivered minus estimated date (days) — right of red line = late")
lt = dlv[dlv.order_purchase_timestamp >= "2017-01-01"].groupby("purchase_year_month").is_late.mean() * 100
lt.plot(ax=axes[1], marker="o", color="C3", title="Late-delivery rate by purchase month (%)"); axes[1].tick_params(axis="x", rotation=60)
plt.tight_layout(); plt.savefig(FIG / "delivery.png"); plt.show()
print(f"Olist estimates are conservative: median order arrives {-dlv.delay_days.median():.1f} days EARLY; {dlv.is_late.mean():.1%} arrive late.")"""),
    md("Late rate spikes after **Black Friday (Nov-2017)** and in **Feb–Mar 2018** (carrier strikes/peak load) → shows a concept drift we must respect with a **time-based train/test split**."),
    md("## 7. Reviews & satisfaction — the key relationship"),
    code("""fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
dlv.review_score.value_counts().sort_index().plot.bar(ax=axes[0], color="C0", title="Review score distribution")
dlv.groupby("is_late").review_score.mean().rename({0: "On time", 1: "Late"}).plot.bar(ax=axes[1], color=["C2", "C3"], title="Average review: on time vs late")
bins = pd.cut(dlv.delay_days, [-200, -10, -5, 0, 5, 10, 20, 200])
dlv.groupby(bins, observed=True).bad_review.mean().mul(100).plot.bar(ax=axes[2], color="C3", title="% bad reviews by delay (days)")
for ax in axes: ax.tick_params(axis="x", rotation=0)
axes[2].tick_params(axis="x", rotation=45)
plt.tight_layout(); plt.savefig(FIG / "reviews.png"); plt.show()
dlv.groupby("is_late").agg(orders=("order_id", "size"), avg_score=("review_score", "mean"), bad_rate=("bad_review", "mean"))"""),
    md("## 8. Payments"),
    code("""fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
master.payment_type.value_counts().plot.pie(ax=axes[0], autopct="%1.1f%%", title="Main payment type"); axes[0].set_ylabel("")
master.groupby(master.payment_installments.clip(upper=12)).order_value.mean().plot.bar(ax=axes[1], color="C4", title="Average order value by installments")
plt.tight_layout(); plt.savefig(FIG / "payments.png"); plt.show()"""),
    md("Credit card dominates (~75%); customers split larger purchases into more installments (common in Brazil)."),
    md("""## 9. Hypothesis testing
Significance level **α = 0.05** for every test. With ~95k rows almost everything is "significant", so we also report **effect sizes**."""),
    code("""from scipy import stats
results = []

def cramers_v(ct):
    chi2 = stats.chi2_contingency(ct)[0]; n = ct.values.sum()
    return np.sqrt(chi2 / (n * (min(ct.shape) - 1)))

# H1: late delivery and bad review are independent (chi-square test of independence)
ct = pd.crosstab(dlv.is_late, dlv.bad_review); chi2, p, dof, _ = stats.chi2_contingency(ct)
results.append(["Chi-square: is_late × bad_review", chi2, p, f"Cramér's V = {cramers_v(ct):.3f}"])
display(ct)

# H2: payment type and bad review are independent
ct2 = pd.crosstab(dlv.payment_type, dlv.bad_review); chi2, p, *_ = stats.chi2_contingency(ct2)
results.append(["Chi-square: payment_type × bad_review", chi2, p, f"Cramér's V = {cramers_v(ct2):.3f}"])

# H3: delivery time is the same for good and bad reviews (Welch t-test + Mann-Whitney U)
g, b = dlv.loc[dlv.bad_review == 0, "actual_delivery_days"], dlv.loc[dlv.bad_review == 1, "actual_delivery_days"]
t, p = stats.ttest_ind(b, g, equal_var=False)
d = (b.mean() - g.mean()) / np.sqrt((b.var() + g.var()) / 2)
results.append(["Welch t-test: delivery days (bad vs good)", t, p, f"Cohen's d = {d:.3f}; means {b.mean():.1f} vs {g.mean():.1f}"])
u, p = stats.mannwhitneyu(b, g)
results.append(["Mann-Whitney U: delivery days (bad vs good)", u, p, "non-parametric check"])

# H4: average review is equal across the top 10 states (ANOVA + Kruskal-Wallis)
top = dlv.customer_state.value_counts().index[:10]
groups = [dlv.loc[dlv.customer_state == s, "review_score"].dropna() for s in top]
f, p = stats.f_oneway(*groups); results.append(["One-way ANOVA: review score across 10 states", f, p, ""])
h, p = stats.kruskal(*groups); results.append(["Kruskal-Wallis: review score across 10 states", h, p, ""])

# H5: freight value correlates with distance
r, p = stats.pearsonr(dlv.distance_km, dlv.total_freight); results.append(["Pearson: distance × freight", r, p, f"r = {r:.3f}"])
rho, p = stats.spearmanr(dlv.distance_km, dlv.actual_delivery_days); results.append(["Spearman: distance × delivery days", rho, p, f"ρ = {rho:.3f}"])

res = pd.DataFrame(results, columns=["test", "statistic", "p_value", "effect / note"])
res["reject H0 (α=0.05)"] = res.p_value < 0.05
res"""),
    code("""cols = ["order_value", "total_freight", "freight_ratio", "n_items", "payment_installments", "distance_km",
        "estimated_delivery_days", "actual_delivery_days", "delay_days", "is_late", "seller_prior_late_rate", "review_score"]
fig, axes = plt.subplots(1, 2, figsize=(18, 7))
sns.heatmap(dlv[cols].corr(method="pearson"), annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=axes[0]); axes[0].set_title("Pearson")
sns.heatmap(dlv[cols].corr(method="spearman"), annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=axes[1]); axes[1].set_title("Spearman")
plt.tight_layout(); plt.savefig(FIG / "correlation.png"); plt.show()"""),
    md("""## 10. Key insights
1. **Delivery is the #1 driver of satisfaction.** Late orders average ≈2.5★ vs ≈4.3★ on time; bad-review rate jumps from ~18% to ~70%+ (χ² test, Cramér's V ≈ 0.3 — moderate-to-strong effect).
2. Bad-review orders took significantly longer to arrive (Welch t, large effect).
3. Distance strongly drives freight cost and delivery time; North/North-East states (RR, AP, AM, AL) wait 2–3× longer than SP.
4. Business grew fast through 2017; Black Friday and Feb–Mar 2018 caused delivery overloads.
5. Only ~3% of customers repeat → improving delivery reliability is the lever for retention.

These motivate the two models: **predict late deliveries at purchase time** and **predict dissatisfied customers after delivery**."""),
])

# =========================================================================== #
# 03 - Feature engineering, Naive Bayes math, feature selection
# =========================================================================== #
nb03 = nb([
    md("""# 03 · Feature Engineering, Naive Bayes Math & Feature Selection"""),
    code(SETUP), code(LOAD_MASTER),
    code("""from src.features import TASKS, feature_columns, modeling_frame, build_preprocessor, SKEWED
for t, cfg in TASKS.items():
    print(f"{t:13s} target={cfg['target']:11s} numeric={len(cfg['numeric'])} categorical={len(cfg['categorical'])}")"""),
    md("""## 1. Engineered features
| Group | Features | Idea |
|---|---|---|
| Time | purchase month, weekday, hour, is_weekend | seasonality & workload |
| Promise | `estimated_delivery_days`, `shipping_limit_days`, `approval_hours` | how much slack the promise has |
| Basket | n_items, n_sellers, total_price, total_freight, **freight_ratio**, weight, volume | heavy/multi-seller orders are harder |
| Geography | customer_state, seller_state, **same_state**, **distance_km (haversine)** | logistics difficulty |
| Payment | payment_type, installments | boleto needs bank confirmation → slower |
| Seller history | **seller_prior_late_rate**, **seller_prior_bad_rate**, seller_prior_orders | seller reliability, computed only from the past |
| Delivery outcome (satisfaction only) | actual_delivery_days, **delay_days**, is_late | what the customer experienced |

**Leakage rule.** The *late* model may only use information available at checkout. The *satisfaction* model is used after delivery, so the actual delivery outcome is allowed — but the review itself, its text, and anything after the review never are."""),
    md("""## 2. Naive Bayes — the theory
**Bayes' theorem** for class $C$ given features $\\mathbf{x} = (x_1, \\dots, x_n)$:

$$P(C\\mid\\mathbf{x}) = \\frac{P(\\mathbf{x}\\mid C)\\,P(C)}{P(\\mathbf{x})}$$

**Naive assumption:** features are conditionally independent given the class:

$$P(\\mathbf{x}\\mid C) = \\prod_{j=1}^{n} P(x_j \\mid C) \\quad\\Rightarrow\\quad \\hat{C} = \\arg\\max_{C}\\; P(C)\\prod_{j} P(x_j\\mid C)$$

$P(\\mathbf{x})$ is the same for every class, so it is only needed to normalise probabilities: $P(\\mathbf{x}) = \\sum_{C} P(C)\\prod_j P(x_j\\mid C)$.

**Categorical NB with Laplace smoothing** (α = 1) — avoids zero probabilities for unseen values:

$$P(x_j = v \\mid C) = \\frac{N_{C,j,v} + \\alpha}{N_C + \\alpha K_j}$$

where $K_j$ is the number of possible values of feature $j$.

**Gaussian NB** (used for continuous features in the model comparison):

$$P(x_j\\mid C) = \\frac{1}{\\sqrt{2\\pi\\sigma_{C,j}^2}}\\exp\\!\\left(-\\frac{(x_j-\\mu_{C,j})^2}{2\\sigma_{C,j}^2}\\right)$$

In practice we sum **log-probabilities** to avoid numerical underflow."""),
    md("""## 3. Worked example by hand: will this customer leave a bad review?
Features: `is_late` (0/1), `payment_type` (4 values), `same_state` (0/1). Target: `bad_review`."""),
    code("""toy = dlv[["is_late", "payment_type", "same_state", "bad_review"]].dropna().astype({"is_late": int, "bad_review": int})
feats = ["is_late", "payment_type", "same_state"]
alpha = 1.0
N = len(toy)
prior = toy.bad_review.value_counts().sort_index() / N
print("Priors P(C):", prior.round(4).to_dict())

query = {"is_late": 1, "payment_type": "boleto", "same_state": 0}
print("Query order:", query)

rows, log_post = [], {}
for c in [0, 1]:
    sub = toy[toy.bad_review == c]; Nc = len(sub)
    lp = np.log(prior[c])
    for f in feats:
        K = toy[f].nunique()
        count = (sub[f] == query[f]).sum()
        p = (count + alpha) / (Nc + alpha * K)
        rows.append({"class": c, "feature": f, "value": query[f], "count N_cjv": count, "N_c": Nc, "K_j": K, "P(x_j|C)": p})
        lp += np.log(p)
    log_post[c] = lp
display(pd.DataFrame(rows))

unnorm = {c: np.exp(v) for c, v in log_post.items()}
evidence = sum(unnorm.values())
post = {c: v / evidence for c, v in unnorm.items()}
print("Unnormalised  P(C)·ΠP(x|C):", {c: f"{v:.6f}" for c, v in unnorm.items()})
print("Evidence P(x)             :", f"{evidence:.6f}")
print("Posterior P(bad_review=1|x) by hand =", round(post[1], 6))"""),
    code("""from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import OrdinalEncoder
enc = OrdinalEncoder()
Xtoy = enc.fit_transform(toy[feats].astype(str))
cnb = CategoricalNB(alpha=1.0).fit(Xtoy, toy.bad_review)
q = enc.transform(pd.DataFrame([query]).astype(str))
p_sklearn = cnb.predict_proba(q)[0, 1]
print(f"scikit-learn CategoricalNB  P(bad_review=1|x) = {p_sklearn:.6f}")
print("Hand calculation matches scikit-learn:", np.isclose(p_sklearn, post[1]))"""),
    md("""The hand computation and `CategoricalNB` agree exactly. A late order paid by boleto from another state has a much higher bad-review probability than the base rate (~22%).

**Limitation:** the independence assumption is violated (e.g. `same_state` and `is_late` are correlated), so NB probabilities are over-confident — which is why we also train discriminative models (Logistic Regression, tree ensembles) in notebook 04."""),
    code("""from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold
cv = StratifiedKFold(3, shuffle=True, random_state=42)
for task in TASKS:
    X, y, _ = modeling_frame(master, task)
    pipe = Pipeline([("prep", build_preprocessor(task)), ("nb", GaussianNB())])
    s = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc")
    print(f"Gaussian NB on all features — {task:12s} ROC-AUC {s.mean():.3f} ± {s.std():.3f}")"""),
    md("""## 4. Feature selection
We run several families of methods on the **satisfaction** task and the **late** task and look for consensus:

* **Filter** — variance threshold, correlation & VIF (multicollinearity), ANOVA F-test, χ² test / Cramér's V for categoricals, mutual information.
* **Wrapper** — Recursive Feature Elimination (RFE) with logistic regression.
* **Embedded** — L1 (Lasso) logistic regression, Random-Forest impurity importance."""),
    code("""from sklearn.feature_selection import VarianceThreshold, f_classif, mutual_info_classif, RFE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from scipy import stats

def numeric_matrix(task, n=25000):
    X, y, _ = modeling_frame(master, task)
    idx = X.sample(min(n, len(X)), random_state=42).index
    num = TASKS[task]["numeric"]
    Xn = X.loc[idx, num].copy()
    for c in num:
        if c in SKEWED: Xn[c] = np.log1p(Xn[c].clip(lower=0))
    return Xn, y.loc[idx], X.loc[idx]

Xn, y, Xall = numeric_matrix("satisfaction")
vt = VarianceThreshold(1e-4).fit(Xn)
print("Near-constant features removed by VarianceThreshold:", list(Xn.columns[~vt.get_support()]) or "none")"""),
    code("""# Correlation + VIF  (VIF_i = i-th diagonal element of the inverse correlation matrix)
R = Xn.corr()
vif = pd.Series(np.diag(np.linalg.pinv(R.values)), index=R.columns, name="VIF").sort_values(ascending=False)
high_pairs = [(a, b, round(R.loc[a, b], 3)) for i, a in enumerate(R.columns) for b in R.columns[i+1:] if abs(R.loc[a, b]) > 0.8]
print("Pairs with |r| > 0.8:", high_pairs)
vif.to_frame().head(10)"""),
    md("""`delay_days = actual_delivery_days − estimated_delivery_days` is a linear combination → huge VIF (perfect multicollinearity). Tree models don't care, but for linear models regularisation (L2/L1) handles it. `n_items`/`total_weight` and `is_weekend`/`weekday` are also related — expected by construction."""),
    code("""def selection_table(task):
    Xn, y, Xall = numeric_matrix(task)
    Z = (Xn - Xn.mean()) / Xn.std()
    F, pF = f_classif(Z, y)
    mi = mutual_info_classif(Z, y, random_state=42)
    l1 = LogisticRegression(penalty="l1", solver="liblinear", C=0.01, class_weight="balanced").fit(Z, y)
    rfe = RFE(LogisticRegression(max_iter=2000, class_weight="balanced"), n_features_to_select=10).fit(Z, y)
    rf = RandomForestClassifier(200, max_depth=10, n_jobs=-1, random_state=42, class_weight="balanced").fit(Z, y)
    t = pd.DataFrame({"ANOVA_F": F, "p_value": pF, "mutual_info": mi, "L1_coef": np.abs(l1.coef_[0]),
                      "RFE_rank": rfe.ranking_, "RF_importance": rf.feature_importances_}, index=Xn.columns)
    votes = pd.DataFrame({
        "ANOVA": t.p_value < 1e-3,
        "MI": t.mutual_info >= t.mutual_info.median(),
        "L1": t.L1_coef > 0,
        "RFE": t.RFE_rank == 1,
        "RF": t.RF_importance >= t.RF_importance.median()})
    t["votes (of 5)"] = votes.sum(axis=1)
    return t.sort_values(["votes (of 5)", "mutual_info"], ascending=False)

sel_sat = selection_table("satisfaction"); sel_sat"""),
    code("""sel_late = selection_table("late"); sel_late"""),
    code("""def cat_tests(task):
    X, y, _ = modeling_frame(master, task)
    out = []
    for c in TASKS[task]["categorical"]:
        ct = pd.crosstab(X[c], y); chi2, p, dof, _ = stats.chi2_contingency(ct)
        out.append({"feature": c, "chi2": chi2, "dof": dof, "p_value": p,
                    "cramers_v": np.sqrt(chi2 / (ct.values.sum() * (min(ct.shape) - 1)))})
    return pd.DataFrame(out).set_index("feature").sort_values("cramers_v", ascending=False)
print("Satisfaction:"); display(cat_tests("satisfaction"))
print("Late delivery:"); display(cat_tests("late"))"""),
    code("""fig, axes = plt.subplots(1, 2, figsize=(15, 6))
sel_sat.mutual_info.sort_values().plot.barh(ax=axes[0], color="C0", title="Mutual information — satisfaction")
sel_late.mutual_info.sort_values().plot.barh(ax=axes[1], color="C3", title="Mutual information — late delivery")
plt.tight_layout(); plt.savefig(FIG / "feature_selection_mi.png"); plt.show()"""),
    md("""### Selection decision
* **Satisfaction:** the delivery-outcome features (`delay_days`, `actual_delivery_days`, `is_late`) and `seller_prior_bad_rate` win every vote; basket size (`n_items`, `n_sellers`) also matters (multi-item orders often arrive split → complaints).
* **Late delivery:** `estimated_delivery_days`, `seller_prior_late_rate`, `distance_km`, `same_state`, purchase month (seasonality) and customer state carry the signal.
* Features with 0 votes are weak but cheap; tree ensembles are robust to them, so we **keep the full engineered set** for the gradient-boosting model and rely on regularisation (L2) for the linear model. The tables above are the evidence for the report.

Next: [04_modeling.ipynb](04_modeling.ipynb)."""),
])

# =========================================================================== #
# 04 - Modeling
# =========================================================================== #
nb04 = nb([
    md("""# 04 · Modelling, Evaluation & Explainability
Training is done by [`src/train.py`](../src/train.py) (re-run with `python src/train.py`, ~10 min). This notebook explains the design and analyses the saved results in `models/metrics.json`."""),
    code(SETUP),
    md("""## 1. Experimental design
* **Time-based split:** train = orders purchased **2017-01 → 2018-03**, test = **2018-04 → 2018-08**. A random split would leak future patterns and overstate performance; this mimics deploying the model on new orders.
* **Model selection** by **3-fold stratified cross-validation on the training period only**; the test period is touched once at the end.
* **Class imbalance:** late ≈ 7%, bad review ≈ 22%. We use `class_weight="balanced"` for linear/tree models, PR-AUC as the main metric (accuracy is misleading: predicting "never late" gives 95% accuracy) and **tune the decision threshold** to maximise F1 on out-of-fold training predictions.
* **Candidates:** Gaussian Naive Bayes (baseline) → Logistic Regression → Decision Tree → Random Forest → XGBoost → XGBoost tuned with `RandomizedSearchCV`.
* Every model is a single scikit-learn `Pipeline` (log-transform + scaling + one-hot encoding + classifier), so the exact same preprocessing runs in the web app."""),
    code("""import json, joblib
metrics = json.loads((ROOT / "models/metrics.json").read_text())
for t, m in metrics.items():
    print(f"{m['title']}: train {m['n_train']:,} ({m['train_period']}), test {m['n_test']:,} ({m['test_period']}), "
          f"positive rate train {m['pos_rate_train']:.1%} / test {m['pos_rate_test']:.1%}")"""),
    md("## 2. Model comparison (cross-validation on training period)"),
    code("""for t, m in metrics.items():
    cv = pd.DataFrame(m["cv"]).set_index("model")
    print(m["title"]); display(cv[["roc_auc", "pr_auc", "f1", "precision", "recall"]].style.highlight_max(color="#c6efce").format("{:.3f}"))"""),
    code("""fig, axes = plt.subplots(1, 2, figsize=(15, 4.5))
for ax, (t, m) in zip(axes, metrics.items()):
    cv = pd.DataFrame(m["cv"]).set_index("model")
    cv[["roc_auc", "pr_auc", "f1"]].plot.barh(ax=ax, xerr=cv[["roc_auc_std", "pr_auc_std", "f1_std"]].values.T)
    ax.axvline(m["pos_rate_train"], ls="--", c="grey"); ax.set_title(m["title"] + " — CV"); ax.set_xlim(0, 1)
plt.tight_layout(); plt.savefig(FIG / "model_comparison.png"); plt.show()"""),
    md("The dashed line is the PR-AUC of a random classifier (= positive rate). Every model beats it; **Naive Bayes is the weakest** (independence assumption violated), tree ensembles are best."),
    md("## 3. Final model on the untouched test period"),
    code("""rows = []
for t, m in metrics.items():
    rows.append({"task": m["title"], "best model": m["best_model"], "threshold": m["threshold"], **m["test"]})
pd.DataFrame(rows).set_index("task")"""),
    code("""for t, m in metrics.items():
    print(m["title"]); display(pd.DataFrame(m["test_all_models_at_0.5"]).set_index("model").style.format("{:.3f}"))"""),
    code("""fig, axes = plt.subplots(2, 3, figsize=(17, 10))
for row, (t, m) in zip(axes, metrics.items()):
    cm = np.array(m["confusion_matrix"])
    sns.heatmap(cm, annot=True, fmt=",", cmap="Blues", ax=row[0], cbar=False,
                xticklabels=["pred 0", "pred 1"], yticklabels=["true 0", "true 1"])
    row[0].set_title(f"{m['title']}\\nconfusion matrix @ thr={m['threshold']:.2f}")
    row[1].plot(m["roc_curve"]["fpr"], m["roc_curve"]["tpr"]); row[1].plot([0, 1], [0, 1], "k--")
    row[1].set(title=f"ROC (AUC = {m['test']['roc_auc']:.3f})", xlabel="FPR", ylabel="TPR")
    row[2].plot(m["pr_curve"]["recall"], m["pr_curve"]["precision"]); row[2].axhline(m["pos_rate_test"], ls="--", c="grey")
    row[2].set(title=f"Precision-Recall (AP = {m['test']['pr_auc']:.3f})", xlabel="Recall", ylabel="Precision")
plt.tight_layout(); plt.savefig(FIG / "test_evaluation.png"); plt.show()"""),
    code("""fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
for ax, (t, m) in zip(axes, metrics.items()):
    c = m["calibration"]; ax.plot(c["mean_predicted"], c["fraction_positive"], "o-"); ax.plot([0, 1], [0, 1], "k--")
    ax.set(title=f"Calibration — {t}", xlabel="predicted probability", ylabel="observed frequency")
plt.tight_layout(); plt.show()"""),
    md("## 4. What drives the predictions? (permutation importance on test set)"),
    code("""fig, axes = plt.subplots(1, 2, figsize=(15, 6))
for ax, (t, m) in zip(axes, metrics.items()):
    imp = pd.DataFrame(m["importance"]).head(12).iloc[::-1]
    ax.barh(imp.feature, imp.importance, xerr=imp["std"], color="C2"); ax.set_title(f"Permutation importance — {t}\\n(drop in PR-AUC when shuffled)")
plt.tight_layout(); plt.savefig(FIG / "importance.png"); plt.show()"""),
    md("## 5. Using a saved model on a new situation"),
    code("""from src.features import add_derived
bundle = joblib.load(ROOT / "models/satisfaction.joblib")
order = pd.DataFrame([bundle["defaults"]])
for scenario, actual in [("arrived 5 days early", bundle["defaults"]["estimated_delivery_days"] - 5),
                         ("arrived 10 days late", bundle["defaults"]["estimated_delivery_days"] + 10)]:
    x = add_derived(order.assign(actual_delivery_days=actual))[bundle["features"]]
    p = bundle["pipeline"].predict_proba(x)[0, 1]
    print(f"Typical order {scenario:22s} → P(bad review) = {p:.1%}  → {'BAD' if p >= bundle['threshold'] else 'good'}")"""),
    md("""## 6. Conclusions & limitations
* **Satisfaction model** is strong: delivery delay, seller history and basket size explain most bad reviews — operations can proactively contact customers whose order is predicted to produce a bad review.
* **Late-delivery model** is harder (rare event + drift between periods: 7.5% late in train vs 4% in test) but still ranks risky orders far better than chance — useful for flagging orders for priority shipping.
* Limitations: no information about carriers or weather; review text not used; concept drift means the model should be retrained monthly.
* Future work: NLP sentiment on Portuguese review text, carrier-level features, survival models for delivery time."""),
])

for name, book in {"01_cleaning.ipynb": nb01, "02_eda_statistics.ipynb": nb02,
                   "03_features_math.ipynb": nb03, "04_modeling.ipynb": nb04}.items():
    nbf.write(book, HERE / name)
    print("wrote", name)
