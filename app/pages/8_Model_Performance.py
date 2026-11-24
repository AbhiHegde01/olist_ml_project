import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils import BLUE, SEQ, SERIES, load_metrics, page_setup, show

page_setup("Model Performance", "🤖")
st.title("🤖 Model Performance")
metrics = load_metrics()
if not metrics:
    st.error("models/metrics.json not found — run `python src/train.py`.")
    st.stop()

task = st.radio("Model", list(metrics), horizontal=True, format_func=lambda t: metrics[t]["title"])
m = metrics[task]

st.markdown(f"""
**Target:** `{m['target']}` · **Best model:** {m['best_model']} · **Decision threshold:** {m['threshold']:.3f}
**Train:** {m['n_train']:,} orders ({m['train_period']}, {m['pos_rate_train']:.1%} positive) ·
**Test:** {m['n_test']:,} orders ({m['test_period']}, {m['pos_rate_test']:.1%} positive) — time-based split, test never used for selection.
""")

t = m["test"]
c = st.columns(6)
for col, (k, label) in zip(c, [("roc_auc", "ROC-AUC"), ("pr_auc", "PR-AUC"), ("f1", "F1"),
                               ("precision", "Precision"), ("recall", "Recall"), ("accuracy", "Accuracy")]):
    col.metric(label, f"{t[k]:.3f}")
st.caption(f"PR-AUC of a random model = positive rate = {m['pos_rate_test']:.3f}. Accuracy is shown for completeness "
           "but is misleading on imbalanced data.")

st.subheader("1 · Model comparison (3-fold CV on training period)")
cv = pd.DataFrame(m["cv"])
a, b = st.columns([3, 2])
with a:
    long = cv.melt(id_vars="model", value_vars=["roc_auc", "pr_auc", "f1"], var_name="metric", value_name="score")
    fig = px.bar(long, x="score", y="model", color="metric", barmode="group", orientation="h",
                 labels={"model": ""}, category_orders={"metric": ["roc_auc", "pr_auc", "f1"]})
    fig.add_vline(x=m["pos_rate_train"], line_dash="dash", line_color="gray",
                  annotation_text="random PR-AUC", annotation_position="bottom right")
    show(fig, 420)
with b:
    st.dataframe(cv.set_index("model")[["roc_auc", "pr_auc", "f1", "precision", "recall"]]
                 .style.format("{:.3f}").highlight_max(color="rgba(27,175,122,.25)"), width="stretch")
    st.caption("Naive Bayes is the baseline; the best model is chosen by CV PR-AUC.")
    with st.expander("Tuned XGBoost hyper-parameters"):
        st.json(m["best_params"])

st.subheader("2 · Evaluation on the unseen test period")
a, b, c_ = st.columns(3)
with a:
    cm = np.array(m["confusion_matrix"])
    lbl = ["Negative", m["positive_label"]]
    fig = px.imshow(cm, text_auto=",", color_continuous_scale=SEQ, x=["Pred " + l for l in lbl],
                    y=["True " + l for l in lbl], title="Confusion matrix")
    fig.update_coloraxes(showscale=False)
    show(fig, 360)
with b:
    fig = go.Figure(go.Scatter(x=m["roc_curve"]["fpr"], y=m["roc_curve"]["tpr"], line=dict(color=BLUE, width=2),
                               name="model", hovertemplate="FPR %{x:.2f}<br>TPR %{y:.2f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], line=dict(dash="dash", color="gray", width=1), name="random"))
    fig.update_layout(title=f"ROC curve (AUC {t['roc_auc']:.3f})", xaxis_title="False positive rate",
                      yaxis_title="True positive rate", showlegend=False)
    show(fig, 360)
with c_:
    fig = go.Figure(go.Scatter(x=m["pr_curve"]["recall"], y=m["pr_curve"]["precision"], line=dict(color=SERIES[1], width=2),
                               hovertemplate="Recall %{x:.2f}<br>Precision %{y:.2f}<extra></extra>"))
    fig.add_hline(y=m["pos_rate_test"], line_dash="dash", line_color="gray")
    fig.update_layout(title=f"Precision–Recall (AP {t['pr_auc']:.3f})", xaxis_title="Recall", yaxis_title="Precision",
                      showlegend=False)
    show(fig, 360)

tn, fp, fn, tp = cm.ravel()
st.markdown(f"Out of **{tp + fn:,}** actual *{m['positive_label'].lower()}* orders in the test period the model catches "
            f"**{tp:,} ({tp / (tp + fn):.0%})**; when it raises an alert it is right **{tp / max(tp + fp, 1):.0%}** of the time "
            f"(vs {m['pos_rate_test']:.0%} by chance → **{tp / max(tp + fp, 1) / m['pos_rate_test']:.1f}× lift**).")

a, b = st.columns(2)
with a:
    imp = pd.DataFrame(m["importance"]).head(15).iloc[::-1]
    fig = px.bar(imp, x="importance", y="feature", orientation="h", error_x="std", color_discrete_sequence=[BLUE],
                 title="Permutation importance (drop in PR-AUC)", labels={"feature": ""})
    show(fig, 460)
with b:
    cal = m["calibration"]
    fig = go.Figure(go.Scatter(x=cal["mean_predicted"], y=cal["fraction_positive"], mode="lines+markers",
                               line=dict(color=BLUE, width=2), marker=dict(size=8), name="model"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], line=dict(dash="dash", color="gray", width=1), name="perfect"))
    fig.update_layout(title="Calibration (reliability) curve", xaxis_title="Predicted probability",
                      yaxis_title="Observed frequency", showlegend=False)
    show(fig, 460)

with st.expander("All models on the test period (threshold 0.5)"):
    st.dataframe(pd.DataFrame(m["test_all_models_at_0.5"]).set_index("model").style.format("{:.3f}"), width="stretch")
with st.expander("Methodology"):
    st.markdown("""
* **Pipeline:** `log1p` on skewed numeric features → `StandardScaler` (linear/NB models) → one-hot encoding with rare levels grouped → classifier. One scikit-learn `Pipeline`, the same object serves predictions in this app.
* **Imbalance:** `class_weight='balanced'` for LogReg/trees; evaluation by PR-AUC; decision threshold chosen to maximise F1 on out-of-fold training predictions.
* **Leakage control:** time-based split; seller history features computed only from earlier orders; the late-delivery model sees only checkout-time information.
* **Drift:** the late rate in the test period (2018-04→08) is much lower than in training (Feb–Mar 2018 crisis), which makes the late task harder — a realistic deployment condition.
""")
