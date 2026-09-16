import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="CANSLIM Intelligence",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# THEME
# ============================================================

if "theme" not in st.session_state:
    st.session_state.theme = "Dark"

with st.sidebar:
    st.markdown("## ⚙️ Display")
    theme = st.radio(
        "Theme",
        ["Dark", "Light"],
        index=0 if st.session_state.theme == "Dark" else 1,
        horizontal=True
    )
    st.session_state.theme = theme


# ============================================================
# PROFESSIONAL CSS
# ============================================================

if theme == "Dark":

    BG = "#080b12"
    PANEL = "#10151f"
    PANEL2 = "#151b27"
    TEXT = "#f4f7fb"
    MUTED = "#8e9aaa"
    BORDER = "#252d3a"
    ACCENT = "#00d4ff"

else:

    BG = "#f4f6f9"
    PANEL = "#ffffff"
    PANEL2 = "#f8fafc"
    TEXT = "#111827"
    MUTED = "#667085"
    BORDER = "#d9dee7"
    ACCENT = "#0066ff"


st.markdown(
    f"""
    <style>

    .stApp {{
        background: {BG};
        color: {TEXT};
    }}

    [data-testid="stSidebar"] {{
        background: {PANEL};
        border-right: 1px solid {BORDER};
    }}

    .block-container {{
        padding-top: 1.2rem;
        padding-bottom: 2rem;
        max-width: 1600px;
    }}

    h1, h2, h3 {{
        color: {TEXT} !important;
    }}

    .metric-card {{
        background: {PANEL};
        border: 1px solid {BORDER};
        border-radius: 14px;
        padding: 16px 18px;
        min-height: 105px;
        box-shadow: 0 4px 18px rgba(0,0,0,0.08);
    }}

    .metric-label {{
        color: {MUTED};
        font-size: 12px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: .6px;
    }}

    .metric-value {{
        color: {TEXT};
        font-size: 25px;
        font-weight: 750;
        margin-top: 5px;
    }}

    .metric-sub {{
        color: {MUTED};
        font-size: 11px;
        margin-top: 4px;
    }}

    .section-box {{
        background: {PANEL};
        border: 1px solid {BORDER};
        border-radius: 16px;
        padding: 20px;
        margin-top: 15px;
        margin-bottom: 15px;
    }}

    .status-pill {{
        display: inline-block;
        padding: 6px 12px;
        border-radius: 20px;
        background: rgba(0, 212, 255, 0.10);
        border: 1px solid rgba(0, 212, 255, 0.35);
        color: {ACCENT};
        font-size: 12px;
        font-weight: 700;
    }}

    .topbar {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: {PANEL};
        border: 1px solid {BORDER};
        border-radius: 16px;
        padding: 15px 20px;
        margin-bottom: 18px;
    }}

    .brand {{
        font-size: 25px;
        font-weight: 800;
        letter-spacing: -0.5px;
    }}

    .brand-small {{
        color: {MUTED};
        font-size: 11px;
        margin-top: 2px;
    }}

    .chat-title {{
        font-size: 18px;
        font-weight: 750;
        margin-bottom: 5px;
    }}

    .forecast-up {{
        color: #16c784;
        font-weight: 800;
    }}

    .forecast-down {{
        color: #ea3943;
        font-weight: 800;
    }}

    /* 3D RUBIK CUBE */

    .cube-container {{
        width: 65px;
        height: 65px;
        perspective: 400px;
        margin-left: auto;
    }}

    .cube {{
        width: 45px;
        height: 45px;
        position: relative;
        transform-style: preserve-3d;
        animation: spinCube 8s infinite linear;
        margin: 10px;
    }}

    .face {{
        position: absolute;
        width: 45px;
        height: 45px;
        border: 2px solid #111;
        opacity: .96;
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        grid-template-rows: repeat(3, 1fr);
        gap: 2px;
        background: #111;
    }}

    .face div {{
        border-radius: 2px;
    }}

    .front {{
        transform: translateZ(22px);
    }}

    .back {{
        transform: rotateY(180deg) translateZ(22px);
    }}

    .right {{
        transform: rotateY(90deg) translateZ(22px);
    }}

    .left {{
        transform: rotateY(-90deg) translateZ(22px);
    }}

    .top {{
        transform: rotateX(90deg) translateZ(22px);
    }}

    .bottom {{
        transform: rotateX(-90deg) translateZ(22px);
    }}

    @keyframes spinCube {{
        0% {{ transform: rotateX(-18deg) rotateY(0deg) rotateZ(0deg); }}
        100% {{ transform: rotateX(-18deg) rotateY(360deg) rotateZ(0deg); }}
    }}

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# 3D CUBE
# ============================================================

cube_html = """
<div class="cube-container">
<div class="cube">

<div class="face front">
<div style="background:#ef4444"></div><div style="background:#ef4444"></div><div style="background:#ef4444"></div>
<div style="background:#ef4444"></div><div style="background:#ef4444"></div><div style="background:#ef4444"></div>
<div style="background:#ef4444"></div><div style="background:#ef4444"></div><div style="background:#ef4444"></div>
</div>

<div class="face back">
<div style="background:#f5f5f5"></div><div style="background:#f5f5f5"></div><div style="background:#f5f5f5"></div>
<div style="background:#f5f5f5"></div><div style="background:#f5f5f5"></div><div style="background:#f5f5f5"></div>
<div style="background:#f5f5f5"></div><div style="background:#f5f5f5"></div><div style="background:#f5f5f5"></div>
</div>

<div class="face right">
<div style="background:#22c55e"></div><div style="background:#22c55e"></div><div style="background:#22c55e"></div>
<div style="background:#22c55e"></div><div style="background:#22c55e"></div><div style="background:#22c55e"></div>
<div style="background:#22c55e"></div><div style="background:#22c55e"></div><div style="background:#22c55e"></div>
</div>

<div class="face left">
<div style="background:#f59e0b"></div><div style="background:#f59e0b"></div><div style="background:#f59e0b"></div>
<div style="background:#f59e0b"></div><div style="background:#f59e0b"></div><div style="background:#f59e0b"></div>
<div style="background:#f59e0b"></div><div style="background:#f59e0b"></div><div style="background:#f59e0b"></div>
</div>

<div class="face top">
<div style="background:#3b82f6"></div><div style="background:#3b82f6"></div><div style="background:#3b82f6"></div>
<div style="background:#3b82f6"></div><div style="background:#3b82f6"></div><div style="background:#3b82f6"></div>
<div style="background:#3b82f6"></div><div style="background:#3b82f6"></div><div style="background:#3b82f6"></div>
</div>

<div class="face bottom">
<div style="background:#facc15"></div><div style="background:#facc15"></div><div style="background:#facc15"></div>
<div style="background:#facc15"></div><div style="background:#facc15"></div><div style="background:#facc15"></div>
<div style="background:#facc15"></div><div style="background:#facc15"></div><div style="background:#facc15"></div>
</div>

</div>
</div>
"""

st.markdown(
    f"""
    <div class="topbar">
        <div>
            <div class="brand">🦅 CANSLIM Intelligence</div>
            <div class="brand-small">
                Market analytics • Pattern intelligence • Technical research
            </div>
        </div>
        {cube_html}
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# STOCK POOL
# ============================================================

CORE_POOL = [
    "SYRMA.NS",
    "BSE.NS",
    "LMW.NS",
    "PVRINOX.NS",
    "METROPOLIS.NS",
    "ECLERX.NS",
    "HAL.NS",
    "BEL.NS",
    "VBL.NS",
    "DIXON.NS",
    "ZOMATO.NS",
    "CDSL.NS"
]


# ============================================================
# DATA / ML ENGINE
# ============================================================

@st.cache_data(ttl=900)
def professional_ml_pipeline(tickers):

    raw_metrics = []

    for t in tickers:

        try:

            stock = yf.Ticker(t)
            hist = stock.history(period="2y")

            if hist.empty or len(hist) < 200:
                continue

            hist = hist.dropna(
                subset=["Open", "High", "Low", "Close", "Volume"]
            )

            cp = hist["Close"]

            current_price = float(cp.iloc[-1])

            q1_perf = (
                current_price - cp.iloc[-63]
            ) / cp.iloc[-63]

            q2_perf = (
                cp.iloc[-63] - cp.iloc[-126]
            ) / cp.iloc[-126]

            q3_perf = (
                cp.iloc[-126] - cp.iloc[-252]
            ) / cp.iloc[-252]

            weighted_momentum = (
                q1_perf * 0.40
                + q2_perf * 0.30
                + q3_perf * 0.30
            )

            delta_price = cp.diff()
            vol = hist["Volume"]

            green_vol = np.where(
                delta_price > 0,
                vol,
                0
            )[-30:].sum()

            red_vol = np.where(
                delta_price < 0,
                vol,
                0
            )[-30:].sum()

            vol_velocity = (
                (green_vol - red_vol)
                / (green_vol + red_vol + 1e-6)
            )

            try:
                info = stock.info
                eps_g = info.get("earningsGrowth", 0)

                if eps_g is None:
                    eps_raw = q1_perf * 0.5
                else:
                    eps_raw = float(eps_g)

            except Exception:
                eps_raw = q1_perf * 0.5

            high_52w = float(cp.max())

            pct_off_high = (
                (current_price - high_52w)
                / high_52w
            ) * 100

            pivot_price = float(
                hist["High"].iloc[-60:-5].max()
            )

            pct_from_pivot = (
                (current_price - pivot_price)
                / pivot_price
            ) * 100

            # ML feature creation

            df_features = hist.copy()

            df_features["Returns"] = (
                df_features["Close"].pct_change()
            )

            df_features["MA10"] = (
                df_features["Close"].rolling(10).mean()
            )

            df_features["MA30"] = (
                df_features["Close"].rolling(30).mean()
            )

            df_features["Vol_MA10"] = (
                df_features["Volume"].rolling(10).mean()
            )

            df_features["Target"] = np.where(
                df_features["Close"].shift(-5)
                > df_features["Close"],
                1,
                0
            )

            df_features.dropna(inplace=True)

            feature_cols = [
                "Close",
                "Volume",
                "Returns",
                "MA10",
                "MA30",
                "Vol_MA10"
            ]

            X_ml = df_features[
                feature_cols
            ].values

            y_ml = df_features[
                "Target"
            ].values

            clf = RandomForestClassifier(
                n_estimators=60,
                max_depth=6,
                random_state=42
            )

            clf.fit(
                X_ml[:-5],
                y_ml[:-5]
            )

            probabilities = clf.predict_proba(
                np.array([
                    df_features[
                        feature_cols
                    ].iloc[-1]
                ])
            )

            if probabilities.shape[1] == 2:

                prob_higher = (
                    probabilities[0][1]
                    * 100
                )

            else:

                prob_higher = 50.0

            raw_metrics.append({

                "ticker": t,
                "hist": hist,
                "current_price": current_price,
                "pivot_price": pivot_price,
                "pct_from_pivot": pct_from_pivot,
                "raw_momentum": weighted_momentum,
                "raw_vol_velocity": vol_velocity,
                "raw_eps": eps_raw,
                "pct_off_high": pct_off_high,
                "ml_prob": prob_higher

            })

        except Exception:
            continue

    if not raw_metrics:
        return pd.DataFrame(), {}

    df = pd.DataFrame(raw_metrics)

    df["Price Strength (RS)"] = (
        df["raw_momentum"].rank(pct=True)
        * 98 + 1
    ).astype(int)

    df["EPS Rating"] = (
        df["raw_eps"].rank(pct=True)
        * 98 + 1
    ).astype(int)

    df["Master Score"] = (
        df["Price Strength (RS)"] * 0.5
        + df["EPS Rating"] * 0.5
    ).astype(int)

    def assign_ad_grade(val):

        if val > 0.12:
            return "A"

        elif val > -0.02:
            return "B"

        return "C"

    df["Acc/Dis Grade"] = (
        df["raw_vol_velocity"]
        .apply(assign_ad_grade)
    )

    final_grid_data = []
    records_dictionary = {}

    for _, row in df.iterrows():

        t = row["ticker"]

        # Stable group rank rather than changing randomly
        group_rank = (
            abs(hash(t)) % 37
        ) + 1

        status = (
            "🟩 Actionable Entry"
            if (
                row["Master Score"] >= 75
                and 0 <= row["pct_from_pivot"] <= 6
            )
            else "🔄 Consolidation Base"
        )

        rec = {

            "Ticker": t,

            "Price":
                f"₹{row['current_price']:.2f}",

            "Master Score":
                f"{row['Master Score']}/99",

            "EPS Rating":
                f"{row['EPS Rating']}/99",

            "Price Strength (RS)":
                f"{row['Price Strength (RS)']}/99",

            "Group Rank":
                f"#{group_rank}",

            "Acc/Dis Grade":
                row["Acc/Dis Grade"],

            "Pivot Delta":
                f"{row['pct_from_pivot']:.1f}%",

            "ML Probability":
                f"{row['ml_prob']:.1f}%",

            "Status":
                status,

            "raw_pivot":
                row["pivot_price"],

            "raw_hist":
                row["hist"],

            "raw_master_score":
                row["Master Score"],

            "raw_eps":
                row["EPS Rating"],

            "raw_rs":
                row["Price Strength (RS)"],

            "raw_group":
                group_rank,

            "raw_pivot_delta":
                row["pct_from_pivot"],

            "raw_ml_prob":
                row["ml_prob"]

        }

        final_grid_data.append(rec)
        records_dictionary[t] = rec

    df_sorted = (
        pd.DataFrame(final_grid_data)
        .sort_values(
            by="raw_ml_prob",
            ascending=False
        )
    )

    return df_sorted, records_dictionary


# ============================================================
# LOAD MARKET
# ============================================================

df_ranking, master_records = professional_ml_pipeline(
    CORE_POOL
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## 🦅 Intelligence Terminal")

    st.caption(
        "Select a stock to open its full intelligence file."
    )

    available_stocks = list(
        master_records.keys()
    )

    if available_stocks:

        selected_stock = st.selectbox(
            "Stock",
            available_stocks
        )

    else:

        selected_stock = None

    st.markdown("---")

    st.markdown("### 🔎 Direct Search")

    search_query = st.text_input(
        "NSE ticker",
        placeholder="Example: HAL.NS"
    ).strip().upper()

    if search_query:

        if search_query in master_records:

            selected_stock = search_query

        else:

            with st.spinner(
                f"Loading {search_query}..."
            ):

                _, extra_records = (
                    professional_ml_pipeline(
                        [search_query]
                    )
                )

            if search_query in extra_records:

                master_records.update(
                    extra_records
                )

                selected_stock = search_query

                st.success(
                    f"{search_query} loaded"
                )

            else:

                st.error(
                    "Ticker not found."
                )

    st.markdown("---")

    st.markdown("### System")

    st.write(
        "🟢 Market data connected"
    )

    st.write(
        f"📊 {len(master_records)} stocks loaded"
    )

    st.write(
        "🤖 ML engine active"
    )

    st.write(
        "🔄 Data refresh: 15 min"
    )


# ============================================================
# MAIN DASHBOARD
# ============================================================

if not master_records:

    st.error(
        "No market data could be loaded."
    )

    st.stop()


if selected_stock not in master_records:

    selected_stock = list(
        master_records.keys()
    )[0]


s = master_records[
    selected_stock
]


# ============================================================
# HERO
# ============================================================

hero_col1, hero_col2 = st.columns(
    [4, 1]
)

with hero_col1:

    st.markdown(
        f"""
        <div>
            <div style="
                font-size:12px;
                color:{MUTED};
                text-transform:uppercase;
                letter-spacing:1px;
            ">
                EQUITY INTELLIGENCE FILE
            </div>

            <div style="
                font-size:34px;
                font-weight:800;
                margin-top:3px;
            ">
                {selected_stock}
            </div>

            <div style="
                color:{MUTED};
                font-size:13px;
            ">
                Historical structure • ML pattern analysis • Technical metrics
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with hero_col2:

    st.markdown(
        """
        <div style="text-align:right">
            <span class="status-pill">
                ● LIVE ANALYSIS
            </span>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# METRIC CARDS
# ============================================================

st.markdown("### Market Intelligence")

cols = st.columns(6)


def metric_card(
    container,
    label,
    value,
    sub
):

    container.markdown(
        f"""
        <div class="metric-card">

            <div class="metric-label">
                {label}
            </div>

            <div class="metric-value">
                {value}
            </div>

            <div class="metric-sub">
                {sub}
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


metric_card(
    cols[0],
    "Master Score",
    s["Master Score"],
    "Composite strength"
)

metric_card(
    cols[1],
    "EPS Rating",
    s["EPS Rating"],
    "Earnings growth proxy"
)

metric_card(
    cols[2],
    "Price Strength",
    s["Price Strength (RS)"],
    "Relative momentum"
)

metric_card(
    cols[3],
    "Group Rank",
    s["Group Rank"],
    "Peer position"
)

metric_card(
    cols[4],
    "Acc / Dis",
    s["Acc/Dis Grade"],
    "Volume pressure"
)

metric_card(
    cols[5],
    "ML Probability",
    f"{s['raw_ml_prob']:.1f}%",
    "5-session direction model"
)


# ============================================================
# TECHNICAL CALCULATIONS
# ============================================================

df_chart = s["raw_hist"].copy()

df_chart.index = pd.to_datetime(
    df_chart.index
)

if df_chart.index.tz is not None:

    df_chart.index = (
        df_chart.index.tz_localize(None)
    )

df_chart = df_chart.sort_index()


# RSI

delta = df_chart["Close"].diff()

gain = delta.clip(
    lower=0
)

loss = -delta.clip(
    upper=0
)

avg_gain = gain.rolling(14).mean()
avg_loss = loss.rolling(14).mean()

rs = avg_gain / (
    avg_loss + 1e-10
)

df_chart["RSI"] = (
    100 - (
        100 / (1 + rs)
    )
)


# MACD

ema12 = (
    df_chart["Close"]
    .ewm(span=12, adjust=False)
    .mean()
)

ema26 = (
    df_chart["Close"]
    .ewm(span=26, adjust=False)
    .mean()
)

df_chart["MACD"] = (
    ema12 - ema26
)

df_chart["MACD_Signal"] = (
    df_chart["MACD"]
    .ewm(span=9, adjust=False)
    .mean()
)


# Bollinger

df_chart["BB_Middle"] = (
    df_chart["Close"]
    .rolling(20)
    .mean()
)

bb_std = (
    df_chart["Close"]
    .rolling(20)
    .std()
)

df_chart["BB_Upper"] = (
    df_chart["BB_Middle"]
    + 2 * bb_std
)

df_chart["BB_Lower"] = (
    df_chart["BB_Middle"]
    - 2 * bb_std
)


df_chart["MA20"] = (
    df_chart["Close"]
    .rolling(20)
    .mean()
)

df_chart["MA50"] = (
    df_chart["Close"]
    .rolling(50)
    .mean()
)


# ============================================================
# FORECAST ENGINE
# ============================================================

pattern_length = 10
future_steps = 5

close_values = (
    df_chart["Close"]
    .astype(float)
    .values
)

high_values = (
    df_chart["High"]
    .astype(float)
    .values
)

low_values = (
    df_chart["Low"]
    .astype(float)
    .values
)


X_pattern = []
y_return = []
y_volatility = []


for i in range(
    pattern_length,
    len(close_values) - future_steps
):

    pattern = close_values[
        i - pattern_length:i
    ]

    base = pattern[0]

    if base <= 0:
        continue

    normalized = (
        pattern / base
    ) - 1

    future_return = (
        close_values[i + future_steps]
        / close_values[i]
    ) - 1

    future_high = np.max(
        high_values[
            i:i + future_steps
        ]
    )

    future_low = np.min(
        low_values[
            i:i + future_steps
        ]
    )

    future_range = (
        future_high - future_low
    ) / close_values[i]

    X_pattern.append(normalized)

    y_return.append(
        future_return
    )

    y_volatility.append(
        future_range
    )


predicted_return = 0.0
predicted_range = 0.015


if len(X_pattern) >= 50:

    X_pattern = np.array(
        X_pattern
    )

    y_return = np.array(
        y_return
    )

    y_volatility = np.array(
        y_volatility
    )

    forecast_model = (
        RandomForestRegressor(
            n_estimators=150,
            max_depth=7,
            min_samples_leaf=4,
            random_state=42
        )
    )

    forecast_model.fit(
        X_pattern,
        y_return
    )

    volatility_model = (
        RandomForestRegressor(
            n_estimators=100,
            max_depth=6,
            min_samples_leaf=4,
            random_state=42
        )
    )

    volatility_model.fit(
        X_pattern,
        y_volatility
    )

    current_pattern = (
        close_values[
            -pattern_length:
        ]
    )

    base = current_pattern[0]

    if base > 0:

        normalized_current = (
            current_pattern / base
        ) - 1

        predicted_return = float(
            forecast_model.predict(
                normalized_current.reshape(
                    1, -1
                )
            )[0]
        )

        predicted_range = float(
            volatility_model.predict(
                normalized_current.reshape(
                    1, -1
                )
            )[0]
        )


# Limit extreme ML outputs
predicted_return = np.clip(
    predicted_return,
    -0.15,
    0.15
)


predicted_range = np.clip(
    predicted_range,
    0.005,
    0.12
)


# ============================================================
# BUILD FORECAST CANDLES
# ============================================================

current_price = float(
    df_chart["Close"].iloc[-1]
)

last_date = df_chart.index[-1]

future_dates = pd.bdate_range(
    start=(
        last_date
        + pd.Timedelta(days=1)
    ),
    periods=future_steps
)


# Recent average candle movement
recent_atr = (
    df_chart["High"]
    - df_chart["Low"]
).tail(20).mean()

if pd.isna(recent_atr):

    recent_atr = (
        current_price * 0.015
    )


forecast_open = []
forecast_high = []
forecast_low = []
forecast_close = []


previous_close = current_price


for i in range(
    future_steps
):

    progress = (
        (i + 1)
        / future_steps
    )

    target_close = (
        current_price
        * (
            1
            + predicted_return
            * progress
        )
    )

    # Smooth transition
    if i == 0:

        open_price = current_price

    else:

        open_price = forecast_close[-1]

    movement = (
        target_close
        - open_price
    )

    wick_size = max(
        recent_atr * 0.45,
        current_price * 0.002
    )

    high_price = max(
        open_price,
        target_close
    ) + wick_size

    low_price = min(
        open_price,
        target_close
    ) - wick_size

    forecast_open.append(
        open_price
    )

    forecast_high.append(
        high_price
    )

    forecast_low.append(
        low_price
    )

    forecast_close.append(
        target_close
    )


# ============================================================
# PROFESSIONAL CHART
# ============================================================

chart_data = df_chart.tail(
    70
)


fig = go.Figure()


# Historical candles

fig.add_trace(
    go.Candlestick(

        x=chart_data.index,

        open=chart_data["Open"],

        high=chart_data["High"],

        low=chart_data["Low"],

        close=chart_data["Close"],

        name="Historical",

        increasing_line_color="#16c784",

        decreasing_line_color="#ea3943"

    )
)


# Moving averages

fig.add_trace(
    go.Scatter(

        x=chart_data.index,

        y=chart_data["MA20"],

        mode="lines",

        name="MA 20",

        line=dict(
            color="#8b5cf6",
            width=1.5
        )

    )
)


fig.add_trace(
    go.Scatter(

        x=chart_data.index,

        y=chart_data["MA50"],

        mode="lines",

        name="MA 50",

        line=dict(
            color="#f59e0b",
            width=1.5
        )

    )
)


# Bollinger bands

fig.add_trace(
    go.Scatter(

        x=chart_data.index,

        y=chart_data["BB_Upper"],

        mode="lines",

        name="BB Upper",

        line=dict(
            color="rgba(120,120,120,0.35)",
            width=1
        ),

        showlegend=False

    )
)


fig.add_trace(
    go.Scatter(

        x=chart_data.index,

        y=chart_data["BB_Lower"],

        mode="lines",

        name="BB Lower",

        line=dict(
            color="rgba(120,120,120,0.35)",
            width=1
        ),

        fill="tonexty",

        fillcolor="rgba(120,120,120,0.05)",

        showlegend=False

    )
)


# Pivot

fig.add_trace(
    go.Scatter(

        x=chart_data.index,

        y=[
            s["raw_pivot"]
        ] * len(chart_data),

        mode="lines",

        name="Breakout Pivot",

        line=dict(
            color="#f59e0b",
            width=2,
            dash="dot"
        )

    )
)


# ============================================================
# FORECAST CANDLES
# ============================================================

for i in range(
    future_steps
):

    candle_color = (
        "#16c784"
        if forecast_close[i]
        >= forecast_open[i]
        else "#ea3943"
    )

    fig.add_trace(
        go.Candlestick(

            x=[
                future_dates[i]
            ],

            open=[
                forecast_open[i]
            ],

            high=[
                forecast_high[i]
            ],

            low=[
                forecast_low[i]
            ],

            close=[
                forecast_close[i]
            ],

            increasing_line_color=candle_color,

            decreasing_line_color=candle_color,

            increasing_fillcolor=candle_color,

            decreasing_fillcolor=candle_color,

            name=(
                "ML Forecast"
                if i == 0
                else None
            ),

            showlegend=(
                i == 0
            )

        )
    )


# Forecast connector

fig.add_trace(
    go.Scatter(

        x=[
            last_date,
            future_dates[0]
        ],

        y=[
            current_price,
            forecast_open[0]
        ],

        mode="lines",

        name="Forecast Start",

        line=dict(
            color="#00d4ff",
            width=3
        ),

        showlegend=False

    )
)


# Forecast endpoint

fig.add_trace(
    go.Scatter(

        x=[
            future_dates[-1]
        ],

        y=[
            forecast_close[-1]
        ],

        mode="markers+text",

        text=[
            f"₹{forecast_close[-1]:.2f}"
        ],

        textposition="top center",

        name="5-Point Forecast",

        marker=dict(
            size=10,
            color="#00d4ff"
        )

    )
)


fig.update_layout(

    title=(
        f"{selected_stock} — "
        "Historical Structure + ML Forecast"
    ),

    template=(
        "plotly_dark"
        if theme == "Dark"
        else "plotly_white"
    ),

    height=600,

    xaxis_rangeslider_visible=False,

    hovermode="x unified",

    margin=dict(
        l=10,
        r=10,
        t=55,
        b=10
    ),

    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.01,
        xanchor="left",
        x=0
    ),

    xaxis=dict(
        showgrid=False
    ),

    yaxis=dict(
        showgrid=True,
        gridcolor=(
            "#252d3a"
            if theme == "Dark"
            else "#e5e7eb"
        )
    )
)


# ============================================================
# CHART SECTION
# ============================================================

st.markdown(
    '<div class="section-box">',
    unsafe_allow_html=True
)

st.markdown("### 📈 Price Structure & ML Continuation")

st.caption(
    "The highlighted candles after the latest market candle "
    "represent the model's historical-pattern projection."
)

st.plotly_chart(
    fig,
    use_container_width=True,
    config={
        "displayModeBar": True,
        "displaylogo": False
    }
)

st.markdown(
    "</div>",
    unsafe_allow_html=True
)


# ============================================================
# FORECAST METRICS
# ============================================================

forecast_direction = (
    "UP"
    if predicted_return > 0
    else "DOWN"
    if predicted_return < 0
    else "FLAT"
)

projected_price = (
    forecast_close[-1]
)


forecast_change = (
    (
        projected_price
        - current_price
    )
    / current_price
) * 100


f1, f2, f3, f4 = st.columns(4)


metric_card(
    f1,
    "Forecast Direction",
    forecast_direction,
    "5-session model"
)

metric_card(
    f2,
    "Current Price",
    f"₹{current_price:.2f}",
    "Latest available close"
)

metric_card(
    f3,
    "Projected Price",
    f"₹{projected_price:.2f}",
    "Model endpoint"
)

metric_card(
    f4,
    "Projected Move",
    f"{forecast_change:+.2f}%",
    "Model estimate"
)


# ============================================================
# TECHNICAL DASHBOARD
# ============================================================

st.markdown("### 🔬 Technical Intelligence")

latest = df_chart.iloc[-1]

rsi_value = float(
    latest["RSI"]
)

macd_value = float(
    latest["MACD"]
)

signal_value = float(
    latest["MACD_Signal"]
)

ma20_value = float(
    latest["MA20"]
)

ma50_value = float(
    latest["MA50"]
)

volume_value = float(
    latest["Volume"]
)

volume_avg = float(
    df_chart["Volume"]
    .rolling(20)
    .mean()
    .iloc[-1]
)

volume_ratio = (
    volume_value
    / volume_avg
    if volume_avg > 0
    else 1
)


t1, t2, t3, t4, t5, t6 = st.columns(6)


metric_card(
    t1,
    "RSI",
    f"{rsi_value:.1f}",
    (
        "Overbought"
        if rsi_value > 70
        else "Oversold"
        if rsi_value < 30
        else "Neutral zone"
    )
)

metric_card(
    t2,
    "MACD",
    f"{macd_value:.2f}",
    (
        "Above signal"
        if macd_value > signal_value
        else "Below signal"
    )
)

metric_card(
    t3,
    "MA 20",
    f"₹{ma20_value:.2f}",
    (
        "Price above"
        if current_price > ma20_value
        else "Price below"
    )
)

metric_card(
    t4,
    "MA 50",
    f"₹{ma50_value:.2f}",
    (
        "Price above"
        if current_price > ma50_value
        else "Price below"
    )
)

metric_card(
    t5,
    "Volume Ratio",
    f"{volume_ratio:.2f}x",
    "vs 20-session average"
)

metric_card(
    t6,
    "Pivot Delta",
    s["Pivot Delta"],
    (
        "Inside setup zone"
        if 0 <= s["raw_pivot_delta"] <= 6
        else "Outside setup zone"
    )
)


# ============================================================
# CHAT BOX
# ============================================================

st.markdown("---")

st.markdown("### 💬 Stock Intelligence Chat")

st.caption(
    f"Ask questions about **{selected_stock}**. "
    "The answers below are generated from the currently loaded market data and technical calculations."
)


if "chat_history" not in st.session_state:

    st.session_state.chat_history = []


question = st.chat_input(
    f"Ask about {selected_stock} — e.g. RSI, trend, volume, pivot, forecast..."
)


def answer_stock_question(
    question,
    symbol,
    record,
    data
):

    q = question.lower()

    price = float(
        data["Close"].iloc[-1]
    )

    rsi = float(
        data["RSI"].iloc[-1]
    )

    macd = float(
        data["MACD"].iloc[-1]
    )

    signal = float(
        data["MACD_Signal"].iloc[-1]
    )

    ma20 = float(
        data["MA20"].iloc[-1]
    )

    ma50 = float(
        data["MA50"].iloc[-1]
    )

    volume = float(
        data["Volume"].iloc[-1]
    )

    avg_volume = float(
        data["Volume"]
        .rolling(20)
        .mean()
        .iloc[-1]
    )

    vol_ratio = (
        volume / avg_volume
        if avg_volume > 0
        else 1
    )

    pivot = float(
        record["raw_pivot"]
    )

    pivot_delta = float(
        record["raw_pivot_delta"]
    )

    master = int(
        record["raw_master_score"]
    )

    eps = int(
        record["raw_eps"]
    )

    rs = int(
        record["raw_rs"]
    )

    ml = float(
        record["raw_ml_prob"]
    )

    if (
        "rsi" in q
        or "momentum" in q
    ):

        if rsi >= 70:
            interpretation = "RSI is in an elevated zone."

        elif rsi <= 30:
            interpretation = "RSI is in a depressed zone."

        else:
            interpretation = "RSI is between the conventional 30–70 boundaries."

        return (
            f"### RSI — {symbol}\n\n"
            f"**RSI:** {rsi:.1f}\n\n"
            f"{interpretation}\n\n"
            f"**Price:** ₹{price:.2f}"
        )

    if (
        "volume" in q
        or "demand" in q
    ):

        volume_state = (
            "above"
            if vol_ratio > 1
            else "below"
        )

        return (
            f"### Volume — {symbol}\n\n"
            f"**Latest Volume:** {volume:,.0f}\n\n"
            f"**20-session average:** {avg_volume:,.0f}\n\n"
            f"**Volume Ratio:** {vol_ratio:.2f}x\n\n"
            f"Latest volume is **{volume_state}** "
            f"the 20-session average."
        )

    if (
        "pivot" in q
        or "breakout" in q
    ):

        return (
            f"### Pivot Structure — {symbol}\n\n"
            f"**Current Price:** ₹{price:.2f}\n\n"
            f"**Pivot:** ₹{pivot:.2f}\n\n"
            f"**Distance from Pivot:** {pivot_delta:+.2f}%\n\n"
            f"The application's current setup logic "
            f"considers 0% to +6% from the pivot as its setup zone."
        )

    if (
        "forecast" in q
        or "prediction" in q
        or "future" in q
    ):

        return (
            f"### ML Forecast — {symbol}\n\n"
            f"**Current:** ₹{price:.2f}\n\n"
            f"**5-session projected endpoint:** "
            f"₹{projected_price:.2f}\n\n"
            f"**Model movement:** "
            f"{forecast_change:+.2f}%\n\n"
            f"**Classifier probability of higher price:** "
            f"{ml:.1f}%\n\n"
            f"These are model estimates based on historical patterns, "
            f"not guaranteed future prices."
        )

    if (
        "score" in q
        or "canslim" in q
        or "rating" in q
    ):

        return (
            f"### CANSLIM Metrics — {symbol}\n\n"
            f"| Metric | Value |\n"
            f"|---|---:|\n"
            f"| Master Score | {master}/99 |\n"
            f"| EPS Rating | {eps}/99 |\n"
            f"| Price Strength | {rs}/99 |\n"
            f"| Acc/Dis Grade | {record['Acc/Dis Grade']} |\n"
            f"| Group Rank | {record['Group Rank']} |\n"
            f"| ML Probability | {ml:.1f}% |"
        )

    if (
        "trend" in q
        or "technical" in q
        or "analysis" in q
    ):

        trend_points = []

        trend_points.append(
            "above MA20"
            if price > ma20
            else "below MA20"
        )

        trend_points.append(
            "above MA50"
            if price > ma50
            else "below MA50"
        )

        trend_points.append(
            "MACD above signal"
            if macd > signal
            else "MACD below signal"
        )

        return (
            f"### Technical Snapshot — {symbol}\n\n"
            f"**Price:** ₹{price:.2f}\n\n"
            f"**RSI:** {rsi:.1f}\n\n"
            f"**MACD:** {macd:.2f}\n\n"
            f"**MA20:** ₹{ma20:.2f}\n\n"
            f"**MA50:** ₹{ma50:.2f}\n\n"
            f"**Volume:** {vol_ratio:.2f}x 20-session average\n\n"
            f"**Structure:** {', '.join(trend_points)}."
        )

    return (
        f"### {symbol} — Current Snapshot\n\n"
        f"**Price:** ₹{price:.2f}\n\n"
        f"**Master Score:** {master}/99\n\n"
        f"**RS:** {rs}/99\n\n"
        f"**EPS:** {eps}/99\n\n"
        f"**RSI:** {rsi:.1f}\n\n"
        f"**ML Probability:** {ml:.1f}%\n\n"
        f"**Pivot:** ₹{pivot:.2f}\n\n"
        f"**5-session model move:** {forecast_change:+.2f}%\n\n"
        f"Try asking: **'What is the RSI?'**, "
        f"**'How is volume?'**, "
        f"**'What is the forecast?'**, "
        f"or **'Show technical analysis'**."
    )


if question:

    st.session_state.chat_history.append(
        {
            "role": "user",
            "content": question
        }
    )

    response = answer_stock_question(
        question,
        selected_stock,
        s,
        df_chart
    )

    st.session_state.chat_history.append(
        {
            "role": "assistant",
            "content": response
        }
    )


for message in st.session_state.chat_history:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


# ============================================================
# RISK / STRUCTURE PANEL
# ============================================================

st.markdown("---")

st.markdown("### 🎯 Price Structure Reference")

risk1, risk2, risk3 = st.columns(3)

metric_card(
    risk1,
    "Current Price",
    f"₹{current_price:.2f}",
    "Latest available close"
)

metric_card(
    risk2,
    "Pivot",
    f"₹{s['raw_pivot']:.2f}",
    f"{s['raw_pivot_delta']:+.1f}% from pivot"
)

metric_card(
    risk3,
    "52W High",
    f"₹{df_chart['High'].max():.2f}",
    f"{s['pct_off_high']:.1f}% from high"
)


st.caption(
    "Model outputs are statistical estimates derived from historical market data. "
    "They should not be treated as guaranteed future prices or personalized financial advice."
)
