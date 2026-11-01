"""ShelfSense: 7-day sales forecasting for a CPG business.

Run locally:      streamlit run app.py
Run on Databricks: deploy as a Databricks App (see app.yaml and README.md).
"""
import io
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))
from shelfsense import __version__  # noqa: E402
from shelfsense.data import REQUIRED_COLUMNS, generate_sales, kpis, prepare, validate_sales  # noqa: E402
from shelfsense.model import MAX_WAPE, forecast_next_days, train_and_evaluate  # noqa: E402

TEAL, PURPLE, INK, AMBER = "#3FB8AE", "#6B2C91", "#2A2E3A", "#C8962E"
CAT_COLORS = {"Beverages": TEAL, "Snacks": PURPLE, "Dairy": AMBER, "Personal Care": "#8A8F9C"}

st.set_page_config(page_title="ShelfSense", page_icon="📈", layout="wide")


# ---------- state ----------
@st.cache_data(show_spinner=False)
def _synthetic(days: int, seed: int) -> pd.DataFrame:
    return prepare(generate_sales(days=days, seed=seed))


@st.cache_resource(show_spinner="Training model…")
def _train(df_hash: int, _df: pd.DataFrame):
    return train_and_evaluate(_df)


if "sales" not in st.session_state:
    st.session_state.sales = _synthetic(730, 42)
    st.session_state.source = "Synthetic (seed 42, 730 days)"

df: pd.DataFrame = st.session_state.sales
result = _train(int(pd.util.hash_pandas_object(df).sum()), df)

# ---------- sidebar ----------
st.sidebar.title("ShelfSense")
st.sidebar.caption(f"v{__version__} · CPG sales forecasting")
page = st.sidebar.radio(
    "Go to",
    ["Sales dashboard", "Data management", "Sales analytics", "Forecast", "Model insights", "Actual vs predicted"],
)
st.sidebar.divider()
st.sidebar.caption(f"Data: {st.session_state.source}")
st.sidebar.caption(f"{df['date'].min():%d %b %Y} → {df['date'].max():%d %b %Y} · {df['sku'].nunique()} SKUs")
gate = "PASS" if result.passes_gate else "FAIL"
st.sidebar.caption(f"Model WAPE {result.metrics['WAPE %']:.1f}% · gate {gate}")


def inr(x: float) -> str:
    if x >= 1e7:
        return f"₹{x/1e7:.2f} Cr"
    if x >= 1e5:
        return f"₹{x/1e5:.1f} L"
    return f"₹{x:,.0f}"


def style(fig, h=380):
    fig.update_layout(height=h, margin=dict(l=10, r=10, t=40, b=10), plot_bgcolor="white",
                      font=dict(color=INK), legend=dict(orientation="h", y=-0.15))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="#EEEEEE")
    return fig


# ---------- pages ----------
if page == "Sales dashboard":
    st.title("Sales dashboard")
    k = kpis(df)
    c = st.columns(4)
    c[0].metric("Revenue, last 28 days", inr(k["revenue"]), f"{k['revenue_change_pct']:+.1f}%")
    c[1].metric("Units, last 28 days", f"{k['units']:,.0f}", f"{k['units_change_pct']:+.1f}%")
    c[2].metric("Average selling price", f"₹{k['avg_price']:.1f}")
    c[3].metric("Days on promotion", f"{k['promo_share_pct']:.0f}%")
    st.caption(f"Top product by revenue in the last 28 days: **{k['top_product']}**")

    daily = df.groupby("date", as_index=False)["revenue"].sum()
    daily["7-day average"] = daily["revenue"].rolling(7).mean()
    fig = go.Figure()
    fig.add_scatter(x=daily["date"], y=daily["revenue"], name="Daily revenue", line=dict(color="#C9CCD3", width=1))
    fig.add_scatter(x=daily["date"], y=daily["7-day average"], name="7-day average", line=dict(color=TEAL, width=2.5))
    st.plotly_chart(style(fig.update_layout(title="Daily revenue (₹)")), use_container_width=True)

    monthly = df.assign(month=df["date"].dt.to_period("M").dt.to_timestamp()).groupby(["month", "category"], as_index=False)["revenue"].sum()
    st.plotly_chart(style(px.bar(monthly, x="month", y="revenue", color="category", color_discrete_map=CAT_COLORS,
                                 title="Monthly revenue by category (₹)")), use_container_width=True)

elif page == "Data management":
    st.title("Sales data management")
    st.write("Upload your own daily sales file or generate synthetic data. Every upload is validated before the model sees it.")
    tab1, tab2, tab3 = st.tabs(["Upload CSV", "Generate synthetic data", "Current data"])
    with tab1:
        st.caption("Required columns: " + ", ".join(f"`{c}`" for c in REQUIRED_COLUMNS))
        up = st.file_uploader("Daily sales CSV", type="csv")
        if up is not None:
            raw = pd.read_csv(up)
            problems = validate_sales(raw)
            if problems:
                st.error("Data validation failed. The current data was kept.")
                for p in problems:
                    st.write(f"- {p}")
            else:
                st.session_state.sales = prepare(raw)
                st.session_state.source = f"Uploaded: {up.name}"
                st.success(f"Loaded {len(raw):,} rows. The model will retrain on the new data.")
                st.rerun()
    with tab2:
        days = st.slider("Days of history", 120, 1095, 730, step=30)
        seed = st.number_input("Random seed", 0, 9999, 42)
        if st.button("Generate", type="primary"):
            st.session_state.sales = _synthetic(days, int(seed))
            st.session_state.source = f"Synthetic (seed {seed}, {days} days)"
            st.rerun()
    with tab3:
        st.dataframe(df.tail(200), use_container_width=True, hide_index=True)
        buf = io.StringIO()
        df.to_csv(buf, index=False)
        st.download_button("Download current data as CSV", buf.getvalue(), "shelfsense_sales.csv", "text/csv")

elif page == "Sales analytics":
    st.title("Sales analytics")
    end = df["date"].max()
    window = st.select_slider("Period", options=[28, 90, 180, 365], value=90, format_func=lambda d: f"Last {d} days")
    cur = df[df["date"] > end - pd.Timedelta(days=window)]
    c1, c2 = st.columns(2)
    by_cat = cur.groupby("category", as_index=False)["revenue"].sum().sort_values("revenue")
    c1.plotly_chart(style(px.bar(by_cat, x="revenue", y="category", orientation="h", color="category",
                                 color_discrete_map=CAT_COLORS, title="Revenue by category (₹)"), 320).update_layout(showlegend=False),
                    use_container_width=True)
    by_prod = cur.groupby(["product_name", "category"], as_index=False)["revenue"].sum().sort_values("revenue")
    c2.plotly_chart(style(px.bar(by_prod, x="revenue", y="product_name", orientation="h", color="category",
                                 color_discrete_map=CAT_COLORS, title="Revenue by product (₹)"), 320).update_layout(showlegend=False),
                    use_container_width=True)
    dow = cur.assign(day=cur["date"].dt.day_name()).groupby(["day", "category"], as_index=False)["units_sold"].mean()
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    st.plotly_chart(style(px.line(dow, x="day", y="units_sold", color="category", color_discrete_map=CAT_COLORS,
                                  category_orders={"day": order}, markers=True, title="Average units per SKU by weekday")),
                    use_container_width=True)
    promo = cur.groupby(["category", "promo"], as_index=False)["units_sold"].mean()
    promo["promo"] = promo["promo"].map({0: "No promo", 1: "Promo"})
    st.plotly_chart(style(px.bar(promo, x="category", y="units_sold", color="promo", barmode="group",
                                 color_discrete_map={"No promo": "#C9CCD3", "Promo": PURPLE},
                                 title="Promotion lift: average daily units per SKU"), 320), use_container_width=True)

elif page == "Forecast":
    st.title("Forecast: next 7 days")
    if not result.passes_gate:
        st.warning(f"This model did not pass the release gate (must beat the naive baseline and stay under {MAX_WAPE:.0f}% WAPE). Treat the numbers below as indicative only.")
    skus = sorted(df["sku"].unique())
    names = df.drop_duplicates("sku").set_index("sku")["product_name"].to_dict()
    with st.expander("Plan a promotion (what-if)"):
        promo_skus = st.multiselect("SKUs on promotion next week (10% off)", skus, format_func=lambda s: f"{s} · {names[s]}")
    fc = forecast_next_days(result, df, promo_plan={s: 1 for s in promo_skus})
    c = st.columns(3)
    c[0].metric("Forecast units, next 7 days", f"{fc['forecast_units'].sum():,.0f}")
    c[1].metric("Forecast revenue, next 7 days", inr(fc["forecast_revenue"].sum()))
    c[2].metric("SKUs on promotion", len(promo_skus))

    sku = st.selectbox("Product", skus, format_func=lambda s: f"{s} · {names[s]}")
    hist = df[df["sku"] == sku].tail(35)
    f = fc[fc["sku"] == sku]
    fig = go.Figure()
    fig.add_scatter(x=hist["date"], y=hist["units_sold"], name="Actual", line=dict(color=INK, width=2))
    fig.add_scatter(x=list(f["date"]) + list(f["date"][::-1]), y=list(f["upper"]) + list(f["lower"][::-1]),
                    fill="toself", fillcolor="rgba(63,184,174,0.18)", line=dict(width=0), mode="lines", name="80% range")
    fig.add_scatter(x=f["date"], y=f["forecast_units"], name="Forecast", line=dict(color=TEAL, width=3, dash="dot"))
    st.plotly_chart(style(fig.update_layout(title=f"{names[sku]}: last 5 weeks and next 7 days (units)")), use_container_width=True)

    table = fc.pivot_table(index=["sku", "product_name"], columns=fc["date"].dt.strftime("%a %d %b"),
                           values="forecast_units", sort=False).reset_index()
    st.subheader("Units by SKU and day")
    st.dataframe(table, use_container_width=True, hide_index=True)
    st.download_button("Download forecast CSV", fc.to_csv(index=False), "shelfsense_forecast.csv", "text/csv")

elif page == "Model insights":
    st.title("Model insights")
    m, b = result.metrics, result.baseline_metrics
    st.write("The model is scored on the last 28 days, which it never saw during training. "
             "It is compared with a naive baseline that repeats the same weekday from last week.")
    c = st.columns(4)
    c[0].metric("WAPE", f"{m['WAPE %']:.1f}%", f"{m['WAPE %'] - b['WAPE %']:+.1f} pts vs baseline", delta_color="inverse")
    c[1].metric("MAPE", f"{m['MAPE %']:.1f}%", f"{m['MAPE %'] - b['MAPE %']:+.1f} pts vs baseline", delta_color="inverse")
    c[2].metric("RMSE (units)", f"{m['RMSE units']:.0f}", f"{m['RMSE units'] - b['RMSE units']:+.0f} vs baseline", delta_color="inverse")
    c[3].metric("Bias", f"{m['Bias %']:+.1f}%", help="Positive = over-forecast (excess stock). Negative = under-forecast (stock-outs).")
    if result.passes_gate:
        st.success(f"Release gate passed: beats the baseline and WAPE is under {MAX_WAPE:.0f}%.")
    else:
        st.error(f"Release gate failed: the model must beat the baseline and keep WAPE under {MAX_WAPE:.0f}%.")

    c1, c2 = st.columns(2)
    comp = pd.DataFrame({"Model": ["ShelfSense (gradient boosting)", "Naive: same day last week"],
                         "WAPE %": [m["WAPE %"], b["WAPE %"]]})
    c1.plotly_chart(style(px.bar(comp, x="Model", y="WAPE %", color="Model", color_discrete_sequence=[TEAL, "#C9CCD3"],
                                 title="Error on holdout (lower is better)"), 340).update_layout(showlegend=False),
                    use_container_width=True)
    imp = result.feature_importance.head(10).sort_values("importance")
    c2.plotly_chart(style(px.bar(imp, x="importance", y="feature", orientation="h", color_discrete_sequence=[PURPLE],
                                 title="What drives the forecast (permutation importance)"), 340), use_container_width=True)
    h = result.holdout
    per = h.groupby("product_name").apply(lambda g: (g["actual"] - g["predicted"]).abs().sum() / g["actual"].sum() * 100,
                                          include_groups=False).rename("WAPE %").reset_index().sort_values("WAPE %")
    st.subheader("Error by product")
    st.dataframe(per.style.format({"WAPE %": "{:.1f}"}), use_container_width=True, hide_index=True)
    with st.expander("Training parameters"):
        st.json(result.params)

elif page == "Actual vs predicted":
    st.title("Actual vs predicted")
    st.write("Holdout period: the last 28 days. The model made these predictions without seeing the actual sales.")
    h = result.holdout
    level = st.radio("Level", ["Total", "Category", "Product"], horizontal=True)
    if level == "Total":
        sel = h
        label = "All products"
    elif level == "Category":
        cat = st.selectbox("Category", sorted(h["category"].unique()))
        sel, label = h[h["category"] == cat], cat
    else:
        prod = st.selectbox("Product", sorted(h["product_name"].unique()))
        sel, label = h[h["product_name"] == prod], prod
    agg = sel.groupby("date", as_index=False)[["actual", "predicted", "baseline"]].sum()
    fig = go.Figure()
    fig.add_scatter(x=agg["date"], y=agg["actual"], name="Actual", line=dict(color=INK, width=2.5))
    fig.add_scatter(x=agg["date"], y=agg["predicted"], name="Model", line=dict(color=TEAL, width=2.5))
    fig.add_scatter(x=agg["date"], y=agg["baseline"], name="Naive baseline", line=dict(color="#B5B8C0", width=1.5, dash="dash"))
    st.plotly_chart(style(fig.update_layout(title=f"{label}: daily units, actual vs predicted"), 420), use_container_width=True)
    c1, c2 = st.columns(2)
    sc = px.scatter(sel, x="actual", y="predicted", color="category", color_discrete_map=CAT_COLORS, opacity=0.6,
                    title="Each dot is one SKU-day")
    mx = float(max(sel["actual"].max(), sel["predicted"].max()))
    sc.add_scatter(x=[0, mx], y=[0, mx], mode="lines", line=dict(color="#B5B8C0", dash="dot"), name="Perfect forecast")
    c1.plotly_chart(style(sc, 360), use_container_width=True)
    err = sel.assign(error=sel["predicted"] - sel["actual"])
    c2.plotly_chart(style(px.histogram(err, x="error", nbins=40, color_discrete_sequence=[PURPLE],
                                       title="Forecast error distribution (units)"), 360), use_container_width=True)
