"""Shared helpers for the Streamlit dashboard: data loading, filters, chart styling."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ---- validated categorical palette (fixed order) + roles --------------------
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
BLUE, ORANGE, AQUA = SERIES[0], SERIES[1], SERIES[2]
GOOD, BAD = "#1baf7a", "#e34948"
SEQ = ["#dbe9fa", "#a9cbf2", "#6fa6e6", "#2a78d6", "#1a56a3", "#0f3a72"]   # one hue, light -> dark

pio.templates["olist"] = go.layout.Template(layout=dict(
    colorway=SERIES,
    font=dict(family="Inter, Segoe UI, sans-serif", size=13),
    margin=dict(l=10, r=10, t=50, b=10),
    hoverlabel=dict(font_size=12),
    xaxis=dict(showgrid=False, linecolor="rgba(128,128,128,.4)"),
    yaxis=dict(gridcolor="rgba(128,128,128,.15)", zeroline=False),
    legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0, title=None),
    bargap=0.25,
))
pio.templates.default = "plotly+olist"

STATE_NAMES = {
    "AC": "Acre", "AL": "Alagoas", "AP": "Amapá", "AM": "Amazonas", "BA": "Bahia", "CE": "Ceará",
    "DF": "Distrito Federal", "ES": "Espírito Santo", "GO": "Goiás", "MA": "Maranhão",
    "MT": "Mato Grosso", "MS": "Mato Grosso do Sul", "MG": "Minas Gerais", "PA": "Pará",
    "PB": "Paraíba", "PR": "Paraná", "PE": "Pernambuco", "PI": "Piauí", "RJ": "Rio de Janeiro",
    "RN": "Rio Grande do Norte", "RS": "Rio Grande do Sul", "RO": "Rondônia", "RR": "Roraima",
    "SC": "Santa Catarina", "SP": "São Paulo", "SE": "Sergipe", "TO": "Tocantins",
}
# approximate state centroids (lat, lng) - used for the map and the predictor's distance default
STATE_CENTROIDS = {
    "AC": (-9.0, -70.5), "AL": (-9.6, -36.6), "AP": (1.4, -51.8), "AM": (-4.2, -64.6), "BA": (-12.5, -41.7),
    "CE": (-5.2, -39.5), "DF": (-15.8, -47.9), "ES": (-19.6, -40.7), "GO": (-15.9, -49.8), "MA": (-5.4, -45.4),
    "MT": (-12.9, -55.9), "MS": (-20.5, -54.8), "MG": (-18.5, -44.6), "PA": (-3.8, -52.5), "PB": (-7.1, -36.8),
    "PR": (-24.6, -51.6), "PE": (-8.4, -37.9), "PI": (-7.4, -42.7), "RJ": (-22.3, -42.6), "RN": (-5.8, -36.6),
    "RS": (-29.8, -53.2), "RO": (-10.9, -62.8), "RR": (2.1, -61.4), "SC": (-27.3, -50.5), "SP": (-22.2, -48.7),
    "SE": (-10.6, -37.4), "TO": (-10.2, -48.3),
}

WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def page_setup(title: str, icon: str = "📦"):
    st.set_page_config(page_title=f"{title} · Olist Analytics", page_icon=icon, layout="wide")
    st.markdown("""<style>
      [data-testid="stMetricValue"] {font-size: 1.6rem;}
      .block-container {padding-top: 2rem;}
    </style>""", unsafe_allow_html=True)


MASTER = ROOT / "data" / "processed" / "master_orders.parquet"
# Columns the dashboard never uses (the long review text is loaded separately, only on demand).
_SKIP = {"order_status", "order_approved_at", "order_delivered_carrier_date", "order_delivered_customer_date",
         "order_estimated_delivery_date", "customer_city", "seller_city", "n_products", "seller_id",
         "product_weight_g", "product_name_length", "n_payment_methods", "n_payments",
         "review_creation_date", "purchase_year", "carrier_pickup_days", "review_comment_message"}


@st.cache_resource(show_spinner="Loading data ...")
def load_master() -> pd.DataFrame:
    """
    Loaded ONCE per server and shared by every page and user (cache_resource = no copies).
    Pages must treat it as read-only: filter / .assign() / .copy() - never assign columns in place.
    """
    import pyarrow.parquet as pq
    cols = [c for c in pq.read_schema(MASTER).names if c not in _SKIP]
    df = pd.read_parquet(MASTER, columns=cols)
    ts = df["order_purchase_timestamp"]
    df = df[(ts >= "2017-01-01") & (ts < "2018-09-01")].reset_index(drop=True)
    floats = df.select_dtypes("float64").columns
    df[floats] = df[floats].astype("float32")          # halves numeric memory, precision is plenty
    return df


@st.cache_resource(show_spinner="Loading review comments ...")
def load_comments() -> pd.DataFrame:
    df = pd.read_parquet(MASTER, columns=["order_id", "review_comment_message"])
    return df.dropna().set_index("order_id")


@st.cache_data
def load_metrics() -> dict:
    path = ROOT / "models" / "metrics.json"
    return json.loads(path.read_text()) if path.exists() else {}


@st.cache_resource(show_spinner="Loading model ...")
def load_model(task: str) -> dict:
    return joblib.load(ROOT / "models" / f"{task}.joblib")


def filters(df: pd.DataFrame) -> pd.DataFrame:
    """Global sidebar filters (persist across pages through session_state keys)."""
    st.sidebar.header("Filters")
    dmin, dmax = df["purchase_date"].min().date(), df["purchase_date"].max().date()
    rng = st.sidebar.date_input("Purchase date", value=st.session_state.get("f_dates", (dmin, dmax)),
                                min_value=dmin, max_value=dmax)
    if isinstance(rng, (list, tuple)) and len(rng) == 2:
        st.session_state["f_dates"] = tuple(rng)
    start, end = st.session_state.get("f_dates", (dmin, dmax))
    states = st.sidebar.multiselect("Customer state", sorted(df["customer_state"].dropna().unique()),
                                    key="f_states", placeholder="All states")
    cats = st.sidebar.multiselect("Product category", sorted(df["category"].dropna().unique()),
                                  key="f_cats", placeholder="All categories")
    out = df[(df["purchase_date"].dt.date >= start) & (df["purchase_date"].dt.date <= end)]
    if states:
        out = out[out["customer_state"].isin(states)]
    if cats:
        out = out[out["category"].isin(cats)]
    st.sidebar.caption(f"{len(out):,} of {len(df):,} orders selected")
    st.sidebar.divider()
    st.sidebar.caption("Data: Olist Brazilian E-Commerce (Kaggle), Jan-2017 → Aug-2018")
    if out.empty:
        st.warning("No orders match the current filters.")
        st.stop()
    return out


def brl(x: float) -> str:
    if abs(x) >= 1e6:
        return f"R$ {x/1e6:,.2f} M"
    if abs(x) >= 1e3:
        return f"R$ {x/1e3:,.1f} k"
    return f"R$ {x:,.2f}"


def insight(text: str):
    st.info(text, icon="💡")


def show(fig, height: int = 380):
    fig.update_layout(height=height)
    st.plotly_chart(fig, width="stretch")
