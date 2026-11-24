# Olist Brazilian E-Commerce — Analytics Dashboard & Predictive Models

End-to-end machine-learning project on the [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) (~100k orders, 2016–2018).

**Live app:** https://olist-ml-project.streamlit.app

| | |
|---|---|
| **Business question 1** | At checkout, which orders are at risk of arriving **late**? |
| **Business question 2** | After delivery, which customers are likely to leave a **bad review (1–3★)**? |
| **Deliverables** | cleaned dataset · statistical analysis · feature engineering & selection · Naive Bayes math · 6 compared models per task · interactive dashboard · hosted predictor |

## Pipeline

```
data/raw/*.csv (9 tables)
      │  src/data_prep.py      clean · merge · aggregate to 1 row per order · leakage-safe seller history
      ▼
data/processed/master_orders.parquet
      │  src/features.py       shared feature definitions + sklearn preprocessing pipeline
      │  src/train.py          CV model comparison · XGBoost tuning · threshold tuning · test evaluation
      ▼
models/late.joblib · models/satisfaction.joblib · models/metrics.json
      │
      ▼
app/ (Streamlit)  Overview · Sales · Customers & Geo · Products · Delivery · Reviews · Payments ·
                  Statistics (hypothesis tests + interactive Naive Bayes) · Model Performance · Predictor
```

## Repository layout

| Path | Content |
|---|---|
| `notebooks/01_cleaning.ipynb` | schema, data-quality audit, cleaning steps & log |
| `notebooks/02_eda_statistics.ipynb` | descriptive stats, trends, geography, RFM, hypothesis tests (χ², Welch t, Mann-Whitney, ANOVA, Kruskal-Wallis, Pearson/Spearman) |
| `notebooks/03_features_math.ipynb` | feature engineering, **Naive Bayes derivation + hand-computed example verified against scikit-learn**, feature selection (variance, correlation, VIF, ANOVA F, χ², mutual information, RFE, L1, RF importance) |
| `notebooks/04_modeling.ipynb` | model comparison, test evaluation, ROC/PR, calibration, permutation importance, conclusions |
| `src/` | reusable Python modules (cleaning, features, training) |
| `app/` | Streamlit dashboard (`app/Home.py` is the entry point) |
| `models/` | trained pipelines and metrics |
| `reports/figures/` | figures exported by the notebooks |
| `tests/test_app_pages.py` | smoke test that runs every dashboard page |

## Methodology highlights

* **One row per order**: items, payments, reviews and geolocation are aggregated; the "main item" (most expensive) defines category and seller.
* **Engineered features**: haversine seller→customer distance, freight ratio, promised delivery window, purchase time, basket size, and **seller history** (past late-rate & bad-review rate computed only from earlier orders, Bayesian-smoothed: `(Σpast + m·global)/(n_past + m)`).
* **No leakage**: the late-delivery model only uses information available at checkout; the train/test split is **by time** (train 2017-01→2018-03, test 2018-04→2018-08).
* **Imbalanced targets** (7% late, 22% bad reviews): class weights, PR-AUC as the main metric, decision threshold tuned for F1 on out-of-fold predictions.
* **Models**: Gaussian Naive Bayes (baseline) → Logistic Regression → Decision Tree → Random Forest → XGBoost → tuned XGBoost (RandomizedSearchCV). Best model selected by cross-validated PR-AUC, never by the test set.
* **Explainability**: permutation importance (global) and XGBoost TreeSHAP contributions per prediction (in the app).

## Results

See `models/metrics.json`, the **Model Performance** page, or notebook 04. <!-- RESULTS_TABLE -->

## Run locally

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows  (source .venv/bin/activate on macOS/Linux)
pip install -r requirements-dev.txt
# put the 9 Kaggle CSVs in data/raw/
python src/data_prep.py           # -> data/processed/master_orders.parquet
python src/train.py               # -> models/*.joblib, models/metrics.json  (~15-20 min)
streamlit run app/Home.py
```

## Deploy (Streamlit Community Cloud)

1. Push this repository to GitHub (raw CSVs are git-ignored; the processed parquet and models are committed — the app only needs those).
2. Go to <https://share.streamlit.io> → **Create app** → choose the repo, branch `main`, main file **`app/Home.py`**.
3. In *Advanced settings* choose Python **3.13** (same as training, so the pinned scikit-learn / XGBoost versions load the saved models).
4. Deploy, then open every page and try the Predictor.

## Dataset

Olist, *Brazilian E-Commerce Public Dataset by Olist*, Kaggle, CC BY-NC-SA 4.0.
