"""
Builds the 4 compact figures used in the written report  ->  reports/figures/report_*.png

Run:  python reports/make_report_figures.py      (needs data/processed + models/metrics.json)
"""
import json
import os
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

BLUE, ORANGE, AQUA, RED, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#e34948", "#8a8a85"
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
                     "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})

cols = ["order_id", "order_purchase_timestamp", "purchase_year_month", "customer_state", "category",
        "total_price", "total_freight", "estimated_delivery_days", "actual_delivery_days", "delay_days",
        "is_late", "review_score", "bad_review", "is_delivered", "distance_km", "n_items",
        "seller_prior_late_rate", "freight_ratio"]
df = pd.read_parquet(ROOT / "data/processed/master_orders.parquet", columns=cols)
df = df[(df.order_purchase_timestamp >= "2017-01-01") & (df.order_purchase_timestamp < "2018-09-01")]
dlv = df[(df.is_delivered == 1) & df.review_score.notna()]

# ---------------------------------------------------------------- Fig 1: the dataset
s = dlv.sample(6, random_state=7)
show = pd.DataFrame({
    "order_id": s.order_id.str[:8] + "…",
    "purchased": s.order_purchase_timestamp.dt.strftime("%Y-%m-%d"),
    "state": s.customer_state, "category": s.category.str.slice(0, 18),
    "price (R$)": s.total_price.round(2), "freight (R$)": s.total_freight.round(2),
    "promised (d)": s.estimated_delivery_days.round(1), "actual (d)": s.actual_delivery_days.round(1),
    "is_late": s.is_late.astype(int), "review ★": s.review_score.astype(int),
    "bad_review": s.bad_review.astype(int)})
fig, ax = plt.subplots(figsize=(10, 1.9))
ax.axis("off")
t = ax.table(cellText=show.values, colLabels=show.columns, loc="center", cellLoc="center")
t.auto_set_font_size(False); t.set_fontsize(7.5); t.scale(1, 1.35)
for (r, c), cell in t.get_celld().items():
    cell.set_edgecolor("#d6d5d0")
    if r == 0:
        cell.set_facecolor("#e8f0fb"); cell.set_text_props(weight="bold")
    if r > 0 and show.columns[c] in ("is_late", "bad_review"):
        cell.set_text_props(weight="bold", color=RED if show.iloc[r - 1, c] == 1 else "#333")
ax.set_title("Sample of the cleaned order-level table (98,500 orders × 60 columns)", pad=4)
plt.tight_layout(); plt.savefig(OUT / "report_1_dataset.png", bbox_inches="tight"); plt.close()

# ---------------------------------------------------------------- Fig 2: EDA (6 panels incl. boxplot)
fig, axes = plt.subplots(2, 3, figsize=(13, 6.6))
m = df.groupby("purchase_year_month").size()
ax = axes[0, 0]
ax.bar(range(len(m)), m.values, color=BLUE, width=0.75)
ax.set_xticks(range(0, len(m), 4)); ax.set_xticklabels(m.index[::4], fontsize=7)
ax.set_title("(a) Orders per month"); ax.set_ylabel("orders")
bf = list(m.index).index("2017-11")
ax.annotate("Black Friday", (bf, m.iloc[bf]), xytext=(bf - 9, m.iloc[bf] - 600), fontsize=7,
            arrowprops=dict(arrowstyle="-", color=GREY))

ax = axes[0, 1]
sc = dlv.review_score.value_counts(normalize=True).sort_index() * 100
ax.bar([f"{int(i)}★" for i in sc.index], sc.values, color=[RED, RED, RED, BLUE, BLUE], width=0.6)
for i, v in enumerate(sc.values):
    ax.text(i, v + 1, f"{v:.0f}%", ha="center", fontsize=8)
ax.set_title("(b) Review scores (red = bad class)"); ax.set_ylabel("% of orders")

ax = axes[0, 2]
groups = [dlv.loc[dlv.review_score == k, "actual_delivery_days"].values for k in [1, 2, 3, 4, 5]]
bp = ax.boxplot(groups, tick_labels=["1★", "2★", "3★", "4★", "5★"], showfliers=False, patch_artist=True,
                widths=0.55, medianprops=dict(color="black", linewidth=1.5))
for patch, col in zip(bp["boxes"], [RED, RED, RED, BLUE, BLUE]):
    patch.set_facecolor(col); patch.set_alpha(0.55)
ax.set_title("(c) Boxplot: delivery days by score"); ax.set_ylabel("actual delivery (days)")

ax = axes[1, 0]
bins = pd.cut(dlv.delay_days, [-200, -10, -5, 0, 3, 7, 14, 200],
              labels=["≤-10", "-10..-5", "-5..0", "0..3", "3..7", "7..14", ">14"])
g = dlv.groupby(bins, observed=True).bad_review.mean() * 100
ax.bar(g.index.astype(str), g.values, color=[BLUE] * 3 + [RED] * 4, width=0.7)
for i, v in enumerate(g.values):
    ax.text(i, v + 1.5, f"{v:.0f}%", ha="center", fontsize=7)
ax.tick_params(axis="x", labelsize=7)
ax.set_title("(d) Bad-review rate by days late"); ax.set_ylabel("% bad reviews")
ax.set_xlabel("days late (negative = early)")

ax = axes[1, 1]
d = dlv.delay_days.clip(-40, 30)
ax.hist(d[d <= 0], bins=np.arange(-40, 1, 1), color=BLUE)
ax.hist(d[d > 0], bins=np.arange(0, 31, 1), color=RED)
ax.axvline(0, color="black", lw=1)
ax.set_title(f"(e) Histogram: delivered − promised ({dlv.is_late.mean():.1%} late)")
ax.set_xlabel("days (negative = early)"); ax.set_ylabel("orders")

ax = axes[1, 2]
vars_ = {"delay_days": "delay", "actual_delivery_days": "delivery days", "is_late": "is late",
         "distance_km": "distance", "total_freight": "freight", "n_items": "items", "review_score": "review score"}
c = dlv[list(vars_)].corr(method="spearman").rename(index=vars_, columns=vars_)
im = ax.imshow(c.values, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(c))); ax.set_xticklabels(c.columns, rotation=45, ha="right", fontsize=7)
ax.set_yticks(range(len(c))); ax.set_yticklabels(c.index, fontsize=7)
for i in range(len(c)):
    for j in range(len(c)):
        ax.text(j, i, f"{c.values[i, j]:.2f}", ha="center", va="center", fontsize=6,
                color="white" if abs(c.values[i, j]) > 0.5 else "black")
ax.set_title("(f) Spearman correlation"); ax.spines[:].set_visible(False)
plt.tight_layout(); plt.savefig(OUT / "report_2_eda.png", bbox_inches="tight"); plt.close()

# ---------------------------------------------------------------- Fig 3: evaluation (one row)
met = json.loads((ROOT / "models/metrics.json").read_text())
fig, axes = plt.subplots(1, 4, figsize=(15, 3.7), gridspec_kw={"width_ratios": [1.35, 1, 0.9, 0.9]})
ax = axes[0]
names = [r["model"].replace(" (Gaussian)", "").replace("Logistic Regression", "Logistic Reg.")
         for r in met["satisfaction"]["cv"]]
y = np.arange(len(names)); h = 0.38
for k, (task, col) in enumerate([("satisfaction", ORANGE), ("late", BLUE)]):
    vals = [r["pr_auc"] for r in met[task]["cv"]]
    off = h / 2 if k == 0 else -h / 2
    ax.barh(y + off, vals, height=h, color=col, label="Bad review" if task == "satisfaction" else "Late delivery")
    for yy, v in zip(y + off, vals):
        ax.text(v + 0.005, yy, f"{v:.2f}", va="center", fontsize=6.5)
ax.set_yticks(y); ax.set_yticklabels(names, fontsize=8); ax.set_xlim(0, 0.85)
ax.set_title("(a) CV PR-AUC by model"); ax.legend(fontsize=6.5, loc="center right")

ax = axes[1]
for task, col, lab in [("satisfaction", ORANGE, "Bad review"), ("late", BLUE, "Late")]:
    r = met[task]["roc_curve"]
    ax.plot(r["fpr"], r["tpr"], color=col, lw=2, label=f"{lab} (AUC {met[task]['test']['roc_auc']:.2f})")
ax.plot([0, 1], [0, 1], "--", color=GREY, lw=1, label="random (0.50)")
ax.set_xlabel("false positive rate"); ax.set_ylabel("true positive rate")
ax.set_title("(b) ROC curves (test)"); ax.legend(fontsize=6.5, loc="lower right")

for ax, task, title in [(axes[2], "satisfaction", "(c) Bad review"), (axes[3], "late", "(d) Late delivery")]:
    cm = np.array(met[task]["confusion_matrix"])
    ax.imshow(cm, cmap="Blues")
    lab = ["good", "bad"] if task == "satisfaction" else ["on time", "late"]
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", fontsize=10, weight="bold",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["pred " + l for l in lab], fontsize=8)
    ax.set_yticks([0, 1]); ax.set_yticklabels(["true " + l for l in lab], fontsize=8)
    ax.set_title(f"{title}: confusion matrix"); ax.spines[:].set_visible(False)
plt.tight_layout(); plt.savefig(OUT / "report_3_evaluation.png", bbox_inches="tight"); plt.close()
print("saved:", sorted(p.name for p in OUT.glob("report_*.png")))
