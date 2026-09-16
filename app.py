import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from datetime import datetime

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    layout="wide",
    page_title="Rakshits-Insights-Terminal"
)

# ============================================================
# BRANDING HEADER & ROTATING 3D RUBIK'S CUBE
# ============================================================
if "theme" not in st.session_state:
    st.session_state.theme = "Black"

header_col, toggle_col = st.columns([9, 1])

with header_col:
    st.markdown(
        """
        <style>
        @keyframes rotateCube {
            0% { transform: rotateX(-22deg) rotateY(0deg); }
            100% { transform: rotateX(-22deg) rotateY(360deg); }
        }
        .rubiks-container {
            width: 32px;
            height: 32px;
            perspective: 200px;
            display: inline-block;
            vertical-align: middle;
            margin-right: 12px;
        }
        .rubiks-cube {
            width: 100%;
            height: 100%;
            position: relative;
            transform-style: preserve-3d;
            animation: rotateCube 6s infinite linear;
        }
        .cube-face {
            position: absolute;
            width: 32px;
            height: 32px;
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            grid-template-rows: repeat(3, 1fr);
            gap: 1px;
            background-color: #000;
            border: 1px solid #000;
            box-sizing: border-box;
        }
        .cube-face div { border-radius: 1px; }
        .front  { transform: translateZ(16px); }
        .back   { transform: rotateY(180deg) translateZ(16px); }
        .right  { transform: rotateY(90deg) translateZ(16px); }
        .left   { transform: rotateY(-90deg) translateZ(16px); }
        .top    { transform: rotateX(90deg) translateZ(16px); }
        .bottom { transform: rotateX(-90deg) translateZ(16px); }

        .c-red    { background-color: #E63946; }
        .c-blue   { background-color: #1D3557; }
        .c-white  { background-color: #F1FAEE; }
        .c-yellow { background-color: #FFB703; }
        .c-green  { background-color: #2A9D8F; }
        .c-orange { background-color: #FB8500; }
        
        .title-text {
            font-size: 26px;
            font-weight: 800;
            letter-spacing: 0.5px;
            display: inline-block;
            vertical-align: middle;
        }
        </style>

        <div style="padding: 10px 0px;">
            <div class="rubiks-container">
                <div class="rubiks-cube">
                    <div class="cube-face front"><div class="c-red"></div><div class="c-red"></div><div class="c-red"></div><div class="c-red"></div><div class="c-red"></div><div class="c-red"></div><div class="c-red"></div><div class="c-red"></div><div class="c-red"></div></div>
                    <div class="cube-face back"><div class="c-orange"></div><div class="c-orange"></div><div class="c-orange"></div><div class="c-orange"></div><div class="c-orange"></div><div class="c-orange"></div><div class="c-orange"></div><div class="c-orange"></div><div class="c-orange"></div></div>
                    <div class="cube-face right"><div class="c-blue"></div><div class="c-blue"></div><div class="c-blue"></div><div class="c-blue"></div><div class="c-blue"></div><div class="c-blue"></div><div class="c-blue"></div><div class="c-blue"></div><div class="c-blue"></div></div>
                    <div class="cube-face left"><div class="c-green"></div><div class="c-green"></div><div class="c-green"></div><div class="c-green"></div><div class="c-green"></div><div class="c-green"></div><div class="c-green"></div><div class="c-green"></div><div class="c-green"></div></div>
                    <div class="cube-face top"><div class="c-white"></div><div class="c-white"></div><div class="c-white"></div><div class="c-white"></div><div class="c-white"></div><div class="c-white"></div><div class="c-white"></div><div class="c-white"></div><div class="c-white"></div></div>
                    <div class="cube-face bottom"><div class="c-yellow"></div><div class="c-yellow"></div><div class="c-yellow"></div><div class="c-yellow"></div><div class="c-yellow"></div><div class="c-yellow"></div><div class="c-yellow"></div><div class="c-yellow"></div><div class="c-yellow"></div></div>
                </div>
            </div>
            <span class="title-text">
                Rakshits-Insights-Terminal 
                <span style="font-size:13px; font-weight:400; opacity:0.65;">(CAN SLIM Quantitative Intelligence)</span>
            </span>
        </div>
        """,
        unsafe_allow_html=True
    )

with toggle_col:
    theme = st.toggle("⚫", value=(st.session_state.theme == "Black"))
    st.session_state.theme = "Black" if theme else "White"

BG = "#0E1117" if st.session_state.theme == "Black" else "#FFFFFF"
TEXT = "#FFFFFF" if st.session_state.theme == "Black" else "#111111"
PANEL = "#161B22" if st.session_state.theme == "Black" else "#F5F5F5"
BORDER = "#30363D" if st.session_state.theme == "Black" else "#D0D0D0"

st.markdown(
    f"""
    <style>
    .stApp {{ background-color:{BG}; color:{TEXT}; }}
    .block-container {{ padding-top:1rem; }}
    h1,h2,h3,h4,h5,h6,p,label {{ color:{TEXT} !important; }}
    div[data-testid="metric-container"] {{ background:{PANEL}; border:1px solid {BORDER}; border-radius:8px; padding:10px; }}
    input, textarea {{ background-color:{PANEL} !important; color:{TEXT} !important; }}
    div[data-baseweb="select"] > div {{ background-color:{PANEL}; color:{TEXT}; }}
    </style>
    """,
    unsafe_allow_html=True
)

# ============================================================
# MULTI-CAP DISCOVERY UNIVERSE
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

# ============================================================
# QUANTITATIVE PIPELINE
# ============================================================
@st.cache_data(ttl=300)
def execute_quant_pipeline(tickers, interval, period):
    raw_metrics = []
    for t in tickers:
        try:
            stock = yf.Ticker(t)
            hist = stock.history(period=period, interval=interval, auto_adjust=False)
            if hist.empty or len(hist) < 100:
                continue
            
            hist = hist.dropna(subset=["Open", "High", "Low", "Close", "Volume"])
            cp = hist["Close"].astype(float)
            current_price = float(cp.iloc[-1])

            # CAN SLIM Technical Metrics
            lookback_1 = min(63, len(cp) - 1)
            lookback_2 = min(126, len(cp) - 1)
            q1_perf = (current_price - cp.iloc[-lookback_1]) / cp.iloc[-lookback_1]
            q2_perf = (cp.iloc[-lookback_1] - cp.iloc[-lookback_2]) / cp.iloc[-lookback_2]
            weighted_momentum = q1_perf * 0.60 + q2_perf * 0.40

            delta_price = cp.diff()
            vol = hist["Volume"].astype(float)
            volume_window = min(30, len(hist))
            green_vol = np.where(delta_price > 0, vol, 0)[-volume_window:].sum()
            red_vol = np.where(delta_price < 0, vol, 0)[-volume_window:].sum()
            vol_velocity = (green_vol - red_vol) / (green_vol + red_vol + 1e-6)

            info = stock.info
            eps_g = info.get("earningsGrowth", 0)
            eps_raw = eps_g if eps_g is not None else q1_perf * 0.5
            pe_ratio = info.get("trailingPE", "N/A")
            market_cap = info.get("marketCap", 0)
            roe = info.get("returnOnEquity", 0)

            high_52w = cp.max()
            pivot_window = min(60, len(hist) - 5)
            pivot_price = hist["High"].iloc[-pivot_window:-5].max()
            pct_from_pivot = ((current_price - pivot_price) / pivot_price) * 100

            # ML Classifier
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

with st.spinner(f"Fetching live feed & running ML pipeline for {timeframe}..."):
    df_ranking, master_records = execute_quant_pipeline(DYNAMIC_UNIVERSE, selected_config["interval"], selected_config["period"])

st.subheader(f"📊 Top Ranked Growth Candidates — {timeframe}")
if not df_ranking.empty:
    st.dataframe(
        df_ranking[["Ticker", "Price", "ML Direction Prob", "Master Score", "EPS Rating", "Price Strength (RS)", "Acc/Dis Grade", "P/E", "Mkt Cap", "ROE"]],
        use_container_width=True, hide_index=True
    )

st.markdown("---")
search_query = st.text_input("Target Ticker Search (e.g. HAL.NS, SUZLON.NS):", "").strip().upper()

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
                st.error("Data fetch failed. Ensure correct Yahoo Finance ticker format (e.g. TATAMOTORS.NS).")
else:
    if list(master_records.keys()):
        active_selection = list(master_records.keys())[0]

# ============================================================
# ANALYSIS & PREDICTION
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
    st.caption(f"⏱️ **Latest Live Feed Timestamp:** `{last_timestamp}` (NSE Exchange Feed)")

    # Scorecard
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Master Score", s["Master Score"])
    c2.metric("EPS Rating", s["EPS Rating"])
    c3.metric("Price Strength (RS)", s["Price Strength (RS)"])
    c4.metric("P/E Ratio", s["P/E"])
    c5.metric("Market Cap", s["Mkt Cap"])
    c6.metric("ROE", s["ROE"])

    # Pattern Machine Learning Model
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
        pattern_model = RandomForestRegressor(n_estimators=150, max_depth=8, random_state=42)
        pattern_model.fit(np.array(X_pattern), np.array(y_pattern))
        current_pattern = close_values[-pattern_length:]
        if current_pattern[0] != 0:
            current_normalized = (current_pattern / current_pattern[0]) - 1
            predicted_return = float(pattern_model.predict(current_normalized.reshape(1, -1))[0])

    current_price = float(df_chart["Close"].iloc[-1])
    
    # Scale movement visually if near zero to ensure projection visibility
    vis_return = predicted_return if abs(predicted_return) > 0.002 else (0.005 if predicted_return >= 0 else -0.005)
    future_prices = [current_price * (1 + vis_return * (i / future_steps)) for i in range(1, future_steps + 1)]

    timeframe_offsets = {"5M": pd.Timedelta(minutes=5), "15M": pd.Timedelta(minutes=15), "30M": pd.Timedelta(minutes=30), "1H": pd.Timedelta(hours=1), "1D": pd.Timedelta(days=1)}
    step_delta = timeframe_offsets.get(timeframe, pd.Timedelta(days=1))
    
    future_dates = [df_chart.index[-1] + (i * step_delta) for i in range(1, future_steps + 1)]
    projection_dates = [df_chart.index[-1]] + future_dates
    projection_prices = [current_price] + future_prices

    chart_data = df_chart.tail(selected_config["bars"])

    fig = go.Figure()

    # Candles
    fig.add_trace(go.Candlestick(
        x=chart_data.index, open=chart_data["Open"], high=chart_data["High"],
        low=chart_data["Low"], close=chart_data["Close"], name=f"{timeframe} Candles"
    ))

    # Breakout Pivot
    fig.add_trace(go.Scatter(
        x=chart_data.index, y=[s["raw_pivot"]] * len(chart_data),
        mode="lines", name="Pivot Level", line=dict(color="orange", width=2, dash="dot")
    ))

    # Projection Line
    fig.add_trace(go.Scatter(
        x=projection_dates, y=projection_prices, mode="lines+markers",
        name=f"ML Projected Path ({predicted_return*100:+.2f}%)",
        line=dict(color="#00FFFF", width=4), marker=dict(size=10, color="#00FFFF", symbol="diamond")
    ))

    # Current Price Marker
    fig.add_trace(go.Scatter(
        x=[df_chart.index[-1]], y=[current_price], mode="markers",
        name="Latest Price", marker=dict(size=12, color="yellow", symbol="circle")
    ))

    fig.update_layout(
        title=f"{active_selection} — {timeframe} Intraday & Projected Horizon | Last Tick: {last_timestamp}",
        yaxis_title="Price (INR)", xaxis_title=f"Time ({timeframe} Intervals)",
        xaxis_rangeslider_visible=False, height=550, margin=dict(l=15, r=15, t=50, b=15),
        hovermode="x unified", template="plotly_dark" if st.session_state.theme == "Black" else "plotly_white"
    )

    if timeframe in ["5M", "15M", "30M", "1H"]:
        fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"]), dict(bounds=[15.5, 9.25], pattern="hour")])
    elif timeframe == "1D":
        fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])

    st.plotly_chart(fig, use_container_width=True)

    st.info(
        f"💡 **Analyst & ML Summary for {active_selection}:** "
        f"Latest Close: ₹{current_price:.2f} | ML Expected Path Target: ₹{future_prices[-1]:.2f} "
        f"({predicted_return * 100:+.2f}%) over the next 5 {timeframe} bars. "
        f"Risk Control Stop Floor: ₹{s['raw_pivot'] * 0.93:.2f} (-7%)."
    )
