import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from datetime import datetime

# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    layout="wide",
    page_title="Rakshits-Insights-Terminal"
)

# Initialize Session State
if "theme" not in st.session_state:
    st.session_state.theme = "Black"

# Dynamic styling variables
BG = "#0E1117" if st.session_state.theme == "Black" else "#FFFFFF"
TEXT = "#FFFFFF" if st.session_state.theme == "Black" else "#111111"
PANEL = "#161B22" if st.session_state.theme == "Black" else "#F8F9FA"
BORDER = "#30363D" if st.session_state.theme == "Black" else "#E0E0E0"
LABEL_COLOR = "#FFFFFF" if st.session_state.theme == "Black" else "#111111"

# ============================================================
# 2. GLOBAL CSS INJECTION (FORCE WHITE METRIC HEADERS)
# ============================================================
st.markdown(
    f"""
    <style>
    /* Prevent top padding overlap with Streamlit toolbar */
    header[data-testid="stHeader"] {{
        background-color: transparent !important;
        z-index: 1;
    }}
    .block-container {{
        padding-top: 3rem !important;
        padding-bottom: 2rem !important;
    }}
    .stApp {{
        background-color: {BG} !important;
        color: {TEXT} !important;
    }}

    /* Header styling */
    .header-title {{
        font-size: 24px;
        font-weight: 800;
        color: {TEXT} !important;
        display: inline-block;
    }}
    .header-subtitle {{
        font-size: 13px;
        font-weight: 400;
        color: #A3B1C2 !important;
        margin-left: 6px;
    }}

    /* Container Box Styling */
    div[data-testid="metric-container"] {{
        background-color: {PANEL} !important;
        border: 1px solid {BORDER} !important;
        border-radius: 8px;
        padding: 12px 16px !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }}

    /* FORCE METRIC CARD HEADER LABELS TO SOLID WHITE */
    div[data-testid="stMetricLabel"],
    div[data-testid="stMetricLabel"] *,
    div[data-testid="stMetricLabel"] p,
    div[data-testid="stMetricLabel"] label,
    div[data-testid="stMetricLabel"] span {{
        color: #FFFFFF !important;
        font-size: 14px !important;
        font-weight: 700 !important;
        opacity: 1 !important;
    }}

    /* FORCE METRIC CARD VALUES TO SOLID WHITE */
    div[data-testid="stMetricValue"],
    div[data-testid="stMetricValue"] *,
    div[data-testid="stMetricValue"] div {{
        color: #FFFFFF !important;
        font-size: 22px !important;
        font-weight: 800 !important;
        opacity: 1 !important;
    }}

    /* Dropdown & Input styling */
    div[data-baseweb="select"] > div {{
        background-color: {PANEL} !important;
        color: {TEXT} !important;
        border-color: {BORDER} !important;
    }}
    label[data-testid="stWidgetLabel"] {{
        color: {TEXT} !important;
        font-weight: 600 !important;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# ============================================================
# 3. HEADER ROW
# ============================================================
head_col1, head_col2 = st.columns([8, 2])

with head_col1:
    st.markdown(
        f"""
        <div>
            <span class="header-title">Rakshits-Insights-Terminal</span>
            <span class="header-subtitle">(CAN SLIM Quantitative Intelligence)</span>
        </div>
        """,
        unsafe_allow_html=True
    )

with head_col2:
    is_dark = st.checkbox("Dark Theme", value=(st.session_state.theme == "Black"))
    st.session_state.theme = "Black" if is_dark else "White"

st.markdown(f"<hr style='margin: 10px 0 20px 0; border-color: {BORDER};'>", unsafe_allow_html=True)

# ============================================================
# 4. CONFIGURATION & DATA ENGINE
# ============================================================
DYNAMIC_UNIVERSE = [
    "SYRMA.NS", "BSE.NS", "LMW.NS", "PVRINOX.NS", "METROPOLIS.NS",
    "ECLERX.NS", "HAL.NS", "BEL.NS", "VBL.NS", "DIXON.NS",
    "ZOMATO.NS", "CDSL.NS", "KALYANKJIL.NS", "SUZLON.NS", "MCX.NS"
]

timeframe = st.selectbox("Select ML Candle Interval:", ["5M", "15M", "30M", "1H", "1D"], index=1)

TIMEFRAME_CONFIG = {
    "5M": {"interval": "5m", "period": "60d", "bars": 100},
    "15M": {"interval": "15m", "period": "60d", "bars": 100},
    "30M": {"interval": "30m", "period": "60d", "bars": 100},
    "1H": {"interval": "1h", "period": "730d", "bars": 100},
    "1D": {"interval": "1d", "period": "2y", "bars": 100}
}
selected_config = TIMEFRAME_CONFIG[timeframe]

@st.cache_data(ttl=300)
def execute_quant_pipeline(tickers, interval, period):
    raw_metrics = []
    for t in tickers:
        try:
            stock = yf.Ticker(t)
            hist = stock.history(period=period, interval=interval, auto_adjust=False)
            if hist.empty or len(hist) < 50:
                continue
            
            hist = hist.dropna(subset=["Open", "High", "Low", "Close", "Volume"])
            cp = hist["Close"].astype(float)
            current_price = float(cp.iloc[-1])

            # Technical momentum
            lookback_1 = min(63, len(cp) - 1)
            lookback_2 = min(126, len(cp) - 1)
            q1_perf = (current_price - cp.iloc[-lookback_1]) / cp.iloc[-lookback_1]
            q2_perf = (cp.iloc[-lookback_1] - cp.iloc[-lookback_2]) / cp.iloc[-lookback_2]
            weighted_momentum = q1_perf * 0.60 + q2_perf * 0.40

            # Accumulation / Distribution
            delta_price = cp.diff()
            vol = hist["Volume"].astype(float)
            volume_window = min(30, len(hist))
            green_vol = np.where(delta_price > 0, vol, 0)[-volume_window:].sum()
            red_vol = np.where(delta_price < 0, vol, 0)[-volume_window:].sum()
            vol_velocity = (green_vol - red_vol) / (green_vol + red_vol + 1e-6)

            # Fundamentals safely fetched
            info = stock.info if hasattr(stock, 'info') else {}
            eps_g = info.get("earningsGrowth", 0)
            eps_raw = eps_g if eps_g is not None else q1_perf * 0.5
            pe_ratio = info.get("trailingPE", "N/A")
            market_cap = info.get("marketCap", 0)
            roe = info.get("returnOnEquity", 0)

            pivot_window = min(60, len(hist) - 5)
            pivot_price = hist["High"].iloc[-pivot_window:-5].max()
            pct_from_pivot = ((current_price - pivot_price) / pivot_price) * 100

            # Random Forest Classifier
            df_features = hist.copy()
            df_features["Returns"] = df_features["Close"].pct_change()
            df_features["MA10"] = df_features["Close"].rolling(10).mean()
            df_features["MA30"] = df_features["Close"].rolling(30).mean()
            df_features["Vol_MA10"] = df_features["Volume"].rolling(10).mean()
            df_features["Target"] = np.where(df_features["Close"].shift(-5) > df_features["Close"], 1, 0)
            df_features.dropna(inplace=True)

            feature_cols = ["Close", "Volume", "Returns", "MA10", "MA30", "Vol_MA10"]
            X_ml = df_features[feature_cols].values
            y_ml = df_features["Target"].values

            clf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
            clf.fit(X_ml[:-5], y_ml[:-5])
            probabilities = clf.predict_proba(np.array([df_features[feature_cols].iloc[-1]]))
            prob_higher = probabilities[0][1] * 100 if probabilities.shape[1] == 2 else 50.0

            raw_metrics.append({
                "ticker": t, "hist": hist, "current_price": current_price, "pivot_price": pivot_price,
                "pct_from_pivot": pct_from_pivot, "raw_momentum": weighted_momentum, "raw_vol_velocity": vol_velocity,
                "raw_eps": eps_raw, "ml_prob": prob_higher, "pe_ratio": pe_ratio,
                "market_cap": f"₹{market_cap/1e9:.1f}B" if market_cap else "N/A",
                "roe": f"{roe*100:.1f}%" if roe else "N/A"
            })
        except Exception:
            continue

    if not raw_metrics:
        return pd.DataFrame(), {}

    df = pd.DataFrame(raw_metrics)
    df["Price Strength (RS)"] = (df["raw_momentum"].rank(pct=True) * 98 + 1).astype(int)
    df["EPS Rating"] = (df["raw_eps"].rank(pct=True) * 98 + 1).astype(int)
    df["Master Score"] = ((df["Price Strength (RS)"] * 0.5 + df["EPS Rating"] * 0.5)).astype(int)
    df["Acc/Dis Grade"] = df["raw_vol_velocity"].apply(lambda v: "A" if v > 0.12 else ("B" if v > -0.02 else "C"))

    records_dictionary = {}
    final_grid = []
    for _, row in df.iterrows():
        t = row["ticker"]
        rec = {
            "Ticker": t, "Price": f"₹{row['current_price']:.2f}", "Master Score": f"{row['Master Score']}/99",
            "EPS Rating": f"{row['EPS Rating']}/99", "Price Strength (RS)": f"{row['Price Strength (RS)']}/99",
            "Acc/Dis Grade": row["Acc/Dis Grade"], "Pivot Delta": f"{row['pct_from_pivot']:.1f}%",
            "ML Direction Prob": f"{row['ml_prob']:.1f}%", "P/E": str(row["pe_ratio"]), "Mkt Cap": row["market_cap"],
            "ROE": row["roe"], "raw_pivot": row["pivot_price"], "raw_hist": row["hist"],
            "raw_master": row["Master Score"], "raw_eps": row["EPS Rating"], "raw_rs": row["Price Strength (RS)"],
            "raw_pivot_delta": row["pct_from_pivot"], "raw_ml_prob": row["ml_prob"]
        }
        final_grid.append(rec)
        records_dictionary[t] = rec

    df_sorted = pd.DataFrame(final_grid).sort_values(by="raw_ml_prob", ascending=False)
    return df_sorted, records_dictionary

# Execute Pipeline
with st.spinner("Fetching market feed & computing metrics..."):
    df_ranking, master_records = execute_quant_pipeline(DYNAMIC_UNIVERSE, selected_config["interval"], selected_config["period"])

# Display Ranking
st.subheader(f"📊 Top Ranked Candidates — {timeframe}")
if not df_ranking.empty:
    st.dataframe(
        df_ranking[["Ticker", "Price", "ML Direction Prob", "Master Score", "EPS Rating", "Price Strength (RS)", "Acc/Dis Grade", "P/E", "Mkt Cap", "ROE"]],
        use_container_width=True, hide_index=True
    )

# Search Input
st.markdown("---")
search_query = st.text_input("Search Ticker Symbol (e.g. HAL.NS, SYRMA.NS):", "").strip().upper()

active_selection = None
if search_query:
    if search_query in master_records:
        active_selection = search_query
    else:
        with st.spinner(f"Evaluating {search_query}..."):
            _, extra_rec = execute_quant_pipeline([search_query], selected_config["interval"], selected_config["period"])
            if search_query in extra_rec:
                master_records.update(extra_rec)
                active_selection = search_query
            else:
                st.error("Data fetch failed. Verify ticker format on Yahoo Finance (e.g. TATAMOTORS.NS).")
else:
    if master_records:
        active_selection = list(master_records.keys())[0]

# ============================================================
# 5. METRICS & PLOTLY CHARTING
# ============================================================
if active_selection and active_selection in master_records:
    s = master_records[active_selection]
    df_chart = s["raw_hist"].copy()
    df_chart.index = pd.to_datetime(df_chart.index)
    if df_chart.index.tz is not None:
        df_chart.index = df_chart.index.tz_localize(None)
    df_chart = df_chart.sort_index()

    last_timestamp = df_chart.index[-1].strftime('%Y-%m-%d %H:%M:%S')

    st.markdown(f"### Live Metrics — **{active_selection}**")
    st.caption(f"⏱️ **Last Feed Timestamp:** `{last_timestamp}` (NSE Feed)")

    # Scorecard
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Master Score", s["Master Score"])
    c2.metric("EPS Rating", s["EPS Rating"])
    c3.metric("Price Strength (RS)", s["Price Strength (RS)"])
    c4.metric("P/E Ratio", s["P/E"])
    c5.metric("Market Cap", s["Mkt Cap"])
    c6.metric("ROE", s["ROE"])

    # Regression Projection
    pattern_length = 10
    future_steps = 5
    close_values = df_chart["Close"].astype(float).values

    X_pattern, y_pattern = [], []
    for i in range(pattern_length, len(close_values) - future_steps):
        pattern = close_values[i - pattern_length:i]
        base_price = pattern[0]
        if base_price == 0: continue
        X_pattern.append((pattern / base_price) - 1)
        y_pattern.append((close_values[i + future_steps] / close_values[i]) - 1)

    predicted_return = 0.0
    if len(X_pattern) >= 20:
        pattern_model = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)
        pattern_model.fit(np.array(X_pattern), np.array(y_pattern))
        current_pattern = close_values[-pattern_length:]
        if current_pattern[0] != 0:
            current_normalized = (current_pattern / current_pattern[0]) - 1
            predicted_return = float(pattern_model.predict(current_normalized.reshape(1, -1))[0])

    current_price = float(df_chart["Close"].iloc[-1])
    vis_return = predicted_return if abs(predicted_return) > 0.002 else (0.005 if predicted_return >= 0 else -0.005)
    future_prices = [current_price * (1 + vis_return * (i / future_steps)) for i in range(1, future_steps + 1)]

    timeframe_offsets = {"5M": pd.Timedelta(minutes=5), "15M": pd.Timedelta(minutes=15), "30M": pd.Timedelta(minutes=30), "1H": pd.Timedelta(hours=1), "1D": pd.Timedelta(days=1)}
    step_delta = timeframe_offsets.get(timeframe, pd.Timedelta(days=1))

    future_dates = [df_chart.index[-1] + (i * step_delta) for i in range(1, future_steps + 1)]
    projection_dates = [df_chart.index[-1]] + future_dates
    projection_prices = [current_price] + future_prices

    chart_data = df_chart.tail(selected_config["bars"])

    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=chart_data.index, open=chart_data["Open"], high=chart_data["High"],
        low=chart_data["Low"], close=chart_data["Close"], name=f"{timeframe} Candles"
    ))

    fig.add_trace(go.Scatter(
        x=chart_data.index, y=[s["raw_pivot"]] * len(chart_data),
        mode="lines", name="Pivot Level", line=dict(color="orange", width=2, dash="dot")
    ))

    fig.add_trace(go.Scatter(
        x=projection_dates, y=projection_prices, mode="lines+markers",
        name=f"ML Projected Horizon ({predicted_return*100:+.2f}%)",
        line=dict(color="#00FFFF", width=3), marker=dict(size=8, color="#00FFFF")
    ))

    fig.update_layout(
        title=f"{active_selection} — Chart & Projection",
        yaxis_title="Price (INR)",
        xaxis_rangeslider_visible=False, height=520, margin=dict(l=15, r=15, t=40, b=15),
        template="plotly_dark" if st.session_state.theme == "Black" else "plotly_white"
    )

    st.plotly_chart(fig, use_container_width=True)
