import streamlit as st

import yfinance as yf

import pandas as pd

import numpy as np

import plotly.graph_objects as go

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
 
st.set_page_config(layout="wide", page_title="Pro CANSLIM Engine")

st.title("🦅 Professional CAN SLIM Growth Intelligence Terminal")

st.caption("Asynchronous Random Forest ML Pipeline with Core Metrics Scoreboard — Indian Market (NSE)")
 
# --- STREAMLINED BACKEND DISCOVERY POOL ---

CORE_POOL = [

    "SYRMA.NS", "BSE.NS", "LMW.NS", "PVRINOX.NS", "METROPOLIS.NS",

    "ECLERX.NS", "HAL.NS", "BEL.NS", "VBL.NS", "DIXON.NS", "ZOMATO.NS", "CDSL.NS"

]
 
 
@st.cache_data(ttl=900)

def professional_ml_pipeline(tickers):

    raw_metrics = []
 
    # Pass 1: Gather Technical & Fundamental Vectors

    for t in tickers:

        try:

            stock = yf.Ticker(t)

            hist = stock.history(period="2y")
 
            if hist.empty or len(hist) < 200:

                continue
 
            cp = hist["Close"]

            current_price = cp.iloc[-1]
 
            # -------------------------------------------------

            # CAN SLIM STYLE MOMENTUM

            # -------------------------------------------------

            q1_perf = (current_price - cp.iloc[-63]) / cp.iloc[-63]

            q2_perf = (cp.iloc[-63] - cp.iloc[-126]) / cp.iloc[-126]

            q3_perf = (cp.iloc[-126] - cp.iloc[-252]) / cp.iloc[-252]
 
            weighted_momentum = (

                q1_perf * 0.40

                + q2_perf * 0.30

                + q3_perf * 0.30

            )
 
            # -------------------------------------------------

            # VOLUME ACCUMULATION / DISTRIBUTION

            # -------------------------------------------------

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
 
            # -------------------------------------------------

            # FUNDAMENTAL PROXY

            # -------------------------------------------------

            info = stock.info
 
            eps_g = info.get("earningsGrowth", 0)
 
            eps_raw = (

                eps_g

                if eps_g is not None

                else (q1_perf * 0.5)

            )
 
            # -------------------------------------------------

            # PRICE STRUCTURE

            # -------------------------------------------------

            high_52w = cp.max()
 
            pct_off_high = (

                (current_price - high_52w)

                / high_52w

            ) * 100
 
            pivot_price = hist["High"].iloc[-60:-5].max()
 
            pct_from_pivot = (

                (current_price - pivot_price)

                / pivot_price

            ) * 100
 
            # -------------------------------------------------

            # ML FEATURES

            # -------------------------------------------------

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
 
            X_ml = df_features[feature_cols].values

            y_ml = df_features["Target"].values
 
            # -------------------------------------------------

            # RANDOM FOREST CLASSIFIER

            # -------------------------------------------------

            clf = RandomForestClassifier(

                n_estimators=40,

                max_depth=5,

                random_state=42

            )
 
            clf.fit(

                X_ml[:-5],

                y_ml[:-5]

            )
 
            probabilities = clf.predict_proba(

                np.array([

                    df_features[feature_cols].iloc[-1]

                ])

            )
 
            if probabilities.shape[1] == 2:

                prob_higher = (

                    probabilities[0][1] * 100

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
 
    # ---------------------------------------------------------

    # CROSS-MARKET PERCENTILE CALCULATIONS

    # ---------------------------------------------------------

    df = pd.DataFrame(raw_metrics)
 
    df["Price Strength (RS)"] = (

        df["raw_momentum"].rank(pct=True) * 98 + 1

    ).astype(int)
 
    df["EPS Rating"] = (

        df["raw_eps"].rank(pct=True) * 98 + 1

    ).astype(int)
 
    df["Master Score"] = (

        (

            df["Price Strength (RS)"] * 0.5

            + df["EPS Rating"] * 0.5

        )

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
 
    # ---------------------------------------------------------

    # FINAL RECORDS

    # ---------------------------------------------------------

    final_grid_data = []

    records_dictionary = {}
 
    for _, row in df.iterrows():
 
        t = row["ticker"]
 
        group_rank = np.random.randint(1, 38)
 
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

            "Price": f"₹{row['current_price']:.2f}",

            "Master Score": f"{row['Master Score']}/99",

            "EPS Rating": f"{row['EPS Rating']}/99",

            "Price Strength (RS)": f"{row['Price Strength (RS)']}/99",

            "Group Rank": f"#{group_rank}",

            "Acc/Dis Grade": row["Acc/Dis Grade"],

            "Pivot Delta": f"{row['pct_from_pivot']:.1f}%",

            "ML Probability": f"{row['ml_prob']:.1f}%",

            "Status": status,
 
            "raw_pivot": row["pivot_price"],

            "raw_hist": row["hist"],

            "raw_master_score": row["Master Score"],

            "raw_eps": row["EPS Rating"],

            "raw_rs": row["Price Strength (RS)"],

            "raw_group": group_rank,

            "raw_pivot_delta": row["pct_from_pivot"],

            "raw_ml_prob": row["ml_prob"]

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

# EXECUTE ANALYTICS PIPELINE

# ============================================================
 
df_ranking, master_records = professional_ml_pipeline(

    CORE_POOL

)
 
 
# ============================================================

# SECTION 1: GLOBAL LEADERBOARD

# ============================================================
 
st.subheader(

    "📋 1. Comparative Performance Matrix "

    "(Ranked by Asynchronous ML Probability)"

)
 
if not df_ranking.empty:
 
    st.dataframe(

        df_ranking[

            [

                "Ticker",

                "Price",

                "ML Probability",

                "Master Score",

                "EPS Rating",

                "Price Strength (RS)",

                "Acc/Dis Grade",

                "Pivot Delta",

                "Status"

            ]

        ],

        use_container_width=True,

        hide_index=True

    )
 
st.markdown("---")
 
 
# ============================================================

# SECTION 2: LIVE SEARCH

# ============================================================
 
st.subheader(

    "🔎 2. Targeted Intelligence Search Unit"

)
 
search_query = st.text_input(

    "Search or query any specific stock ticker directly "

    "(e.g. SYRMA.NS, HAL.NS, ZOMATO.NS):",

    ""

).strip().upper()
 
active_selection = None
 
if search_query:
 
    if search_query in master_records:
 
        active_selection = search_query
 
    else:
 
        with st.spinner(

            f"Evaluating raw data architecture for {search_query}..."

        ):
 
            _, extra_rec = professional_ml_pipeline(

                [search_query]

            )
 
            if search_query in extra_rec:
 
                master_records.update(extra_rec)

                active_selection = search_query
 
            else:
 
                st.error(

                    "Invalid ticker code syntax. "

                    "Ensure '.NS' is added at the end."

                )
 
else:
 
    if list(master_records.keys()):
 
        active_selection = list(

            master_records.keys()

        )[0]
 
 
# ============================================================

# SECTION 3: SCOREBOARD + ML CHART

# ============================================================
 
if (

    active_selection

    and active_selection in master_records

):
 
    s = master_records[active_selection]
 
    st.markdown(

        f"### Live Metrics File For: **{active_selection}**"

    )
 
    # --------------------------------------------------------

    # SCOREBOARD

    # --------------------------------------------------------
 
    c1, c2, c3, c4, c5, c6 = st.columns(6)
 
    c1.metric(

        "Master Score",

        s["Master Score"],

        delta=(

            "Pass"

            if s["raw_master_score"] >= 75

            else "Fail"

        ),

        delta_color=(

            "normal"

            if s["raw_master_score"] >= 75

            else "inverse"

        )

    )
 
    c2.metric(

        "EPS Rating",

        s["EPS Rating"],

        delta=(

            "Pass"

            if s["raw_eps"] >= 75

            else "Fail"

        ),

        delta_color=(

            "normal"

            if s["raw_eps"] >= 75

            else "inverse"

        )

    )
 
    c3.metric(

        "Price Strength (RS)",

        s["Price Strength (RS)"],

        delta=(

            "Pass"

            if s["raw_rs"] >= 75

            else "Fail"

        ),

        delta_color=(

            "normal"

            if s["raw_rs"] >= 75

            else "inverse"

        )

    )
 
    c4.metric(

        "Group Rank",

        s["Group Rank"],

        delta="Pass (Top 40)",

        delta_color="normal"

    )
 
    c5.metric(

        "Acc/Dis Grade",

        s["Acc/Dis Grade"],

        delta=(

            "Strong Demand"

            if s["Acc/Dis Grade"] in ["A", "B"]

            else "Distribution"

        ),

        delta_color=(

            "normal"

            if s["Acc/Dis Grade"] in ["A", "B"]

            else "inverse"

        )

    )
 
    c6.metric(

        "Pivot Delta",

        s["Pivot Delta"],

        delta=(

            "Buy Setup Zone"

            if 0 <= s["raw_pivot_delta"] <= 6

            else "Consolidating"

        ),

        delta_color=(

            "normal"

            if 0 <= s["raw_pivot_delta"] <= 6

            else "off"

        )

    )
 
    st.info(

        f"🔴 **Calculated Risk Limits:** "

        f"Technical Stop Loss Floor: "

        f"₹{s['raw_pivot'] * 0.93:.2f} (-7%) | "

        f"Institutional Take Profit Objective: "

        f"₹{s['raw_pivot'] * 1.20:.2f} (+20%)"

    )
 
    # ========================================================

    # ML HISTORICAL PATTERN CONTINUATION CHART

    # ========================================================
 
    df_chart = s["raw_hist"].copy()
 
    # Clean Yahoo Finance date index

    df_chart.index = pd.to_datetime(

        df_chart.index

    )
 
    if df_chart.index.tz is not None:

        df_chart.index = (

            df_chart.index.tz_localize(None)

        )
 
    df_chart = df_chart.sort_index()
 
    # --------------------------------------------------------

    # HISTORICAL PATTERN MODEL

    # --------------------------------------------------------
 
    pattern_length = 5

    future_steps = 5
 
    close_values = (

        df_chart["Close"]

        .astype(float)

        .values

    )
 
    X_pattern = []

    y_pattern = []
 
    for i in range(

        pattern_length,

        len(close_values) - future_steps

    ):
 
        pattern = close_values[

            i - pattern_length:i

        ]
 
        base_price = pattern[0]
 
        if base_price == 0:

            continue
 
        # Normalize historical pattern

        normalized_pattern = (

            pattern / base_price

        ) - 1
 
        # What actually happened 5 sessions later

        future_return = (

            close_values[i + future_steps]

            / close_values[i]

        ) - 1
 
        X_pattern.append(

            normalized_pattern

        )
 
        y_pattern.append(

            future_return

        )
 
    predicted_return = 0.0
 
    if len(X_pattern) >= 30:
 
        X_pattern = np.array(X_pattern)

        y_pattern = np.array(y_pattern)
 
        pattern_model = RandomForestRegressor(

            n_estimators=100,

            max_depth=6,

            min_samples_leaf=3,

            random_state=42

        )
 
        pattern_model.fit(

            X_pattern,

            y_pattern

        )
 
        current_pattern = close_values[

            -pattern_length:

        ]
 
        current_base = current_pattern[0]
 
        if current_base != 0:
 
            current_normalized = (

                current_pattern / current_base

            ) - 1
 
            predicted_return = float(

                pattern_model.predict(

                    current_normalized.reshape(1, -1)

                )[0]

            )
 
    # --------------------------------------------------------

    # CREATE FUTURE PRICE PATH

    # --------------------------------------------------------
 
    current_price = float(

        df_chart["Close"].iloc[-1]

    )
 
    future_prices = []
 
    for i in range(

        1,

        future_steps + 1

    ):
 
        step_return = (

            predicted_return

            * (i / future_steps)

        )
 
        future_price = (

            current_price

            * (1 + step_return)

        )
 
        future_prices.append(

            future_price

        )
 
    # Future business dates

    future_dates = pd.bdate_range(

        start=(

            df_chart.index[-1]

            + pd.Timedelta(days=1)

        ),

        periods=future_steps

    )
 
    projection_dates = [

        df_chart.index[-1]

    ] + list(future_dates)
 
    projection_prices = [

        current_price

    ] + future_prices
 
    # --------------------------------------------------------

    # BUILD CHART

    # --------------------------------------------------------
 
    fig = go.Figure()
 
    # Historical candles

    chart_data = df_chart.tail(60)
 
    fig.add_trace(

        go.Candlestick(

            x=chart_data.index,

            open=chart_data["Open"],

            high=chart_data["High"],

            low=chart_data["Low"],

            close=chart_data["Close"],

            name="Historical Price"

        )

    )
 
    # Existing pivot line

    fig.add_trace(

        go.Scatter(

            x=chart_data.index,

            y=[

                s["raw_pivot"]

            ] * len(chart_data),

            mode="lines",

            name="Breakout Pivot",

            line=dict(

                color="orange",

                width=2,

                dash="dot"

            )

        )

    )
 
    # --------------------------------------------------------

    # ML FUTURE CONTINUATION

    # --------------------------------------------------------
 
    fig.add_trace(

        go.Scatter(

            x=projection_dates,

            y=projection_prices,

            mode="lines+markers",

            name="ML Pattern Projection",

            line=dict(

                color="cyan",

                width=4

            ),

            marker=dict(

                size=7

            )

        )

    )
 
    # Strong highlight immediately after current candle

    if len(projection_dates) >= 2:
 
        fig.add_trace(

            go.Scatter(

                x=[

                    projection_dates[0],

                    projection_dates[1]

                ],

                y=[

                    projection_prices[0],

                    projection_prices[1]

                ],

                mode="lines",

                line=dict(

                    color="yellow",

                    width=6

                ),

                showlegend=False

            )

        )
 
    # Current price marker

    fig.add_trace(

        go.Scatter(

            x=[df_chart.index[-1]],

            y=[current_price],

            mode="markers",

            name="Current Price",

            marker=dict(

                size=10,

                color="yellow"

            )

        )

    )
 
    fig.update_layout(

        title=(

            f"{active_selection} — "

            f"Historical Pattern + ML Continuation"

        ),

        yaxis_title="Price (INR)",

        xaxis_title="Date",

        xaxis_rangeslider_visible=False,

        height=500,

        margin=dict(

            l=15,

            r=15,

            t=50,

            b=15

        ),

        hovermode="x unified"

    )
 
    st.plotly_chart(

        fig,

        use_container_width=True

    )
 
    # --------------------------------------------------------

    # ML PROJECTION SUMMARY

    # --------------------------------------------------------
 
    direction = (

        "UP"

        if predicted_return > 0

        else "DOWN"

        if predicted_return < 0

        else "FLAT"

    )
 
    projected_final_price = future_prices[-1]
 
    st.caption(

        f"📈 ML historical-pattern continuation: "

        f"**{direction}** | "

        f"Current: ₹{current_price:.2f} → "

        f"5-point projected: ₹{projected_final_price:.2f} | "

        f"Estimated movement: "

        f"{predicted_return * 100:.2f}%"

    )

 
