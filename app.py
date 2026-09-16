import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

st.set_page_config(layout="wide", page_title="Pro CANSLIM Engine")
st.title("🦅 Professional CAN SLIM Growth Intelligence Terminal")
st.caption("Advanced Multi-Factor Percentile Ranking Pipeline — National Stock Exchange (NSE)")

# --- PRO BACKEND DISCOVERY POOL (Core Growth Leaders) ---
PRO_POOL = [
    "PVRINOX.NS", "LMW.NS", "ECLERX.NS", "METROPOLIS.NS", 
    "HAL.NS", "BEL.NS", "ZOMATO.NS", "BSE.NS", "CDSL.NS", 
    "TRENT.NS", "TATAMOTORS.NS", "SYRMA.NS", "DIXON.NS", "VBL.NS"
]

@st.cache_data(ttl=900)
def professional_grading_pipeline(tickers):
    raw_metrics = []
    
    # PHASE 1: Fetch Raw Data Matrices
    for t in tickers:
        try:
            stock = yf.Ticker(t)
            hist = stock.history(period="2y")
            if hist.empty or len(hist) < 252: continue
            
            cp = hist['Close']
            
            # 1. Precise Time-Weighted Momentum Calculation (CAN SLIM Standard)
            # Heavy 40% weight on the immediate 3 months, 30% on prior two quarters
            q1_perf = (cp.iloc[-1] - cp.iloc[-63]) / cp.iloc[-63]
            q2_perf = (cp.iloc[-63] - cp.iloc[-126]) / cp.iloc[-126]
            q3_perf = (cp.iloc[-126] - cp.iloc[-252]) / cp.iloc[-252]
            weighted_momentum = (q1_perf * 0.40) + (q2_perf * 0.30) + (q3_perf * 0.30)
            
            # 2. Institutional Volume Velocity (Accumulation/Distribution Intensity)
            # Measures if volume is expanding heavier on green days vs red days over 30 days
            delta_price = cp.diff()
            vol = hist['Volume']
            green_vol = np.where(delta_price > 0, vol, 0)[-30:].sum()
            red_vol = np.where(delta_price < 0, vol, 0)[-30:].sum()
            vol_velocity = (green_vol - red_vol) / (green_vol + red_vol + 1e-6)
            
            # 3. Fundamental Earnings Momentum Metric
            info = stock.info
            eps_growth = info.get('earningsGrowth', 0)
            eps_raw = eps_growth if eps_growth is not None else 0.15 # Baseline standard
            
            # 4. Nearness to 52-Week High
            high_52w = cp.iloc[-252:].max()
            current_price = cp.iloc[-1]
            pct_off_high = ((current_price - high_52w) / high_52w) * 100
            
            # 5. Breakout Pivot Detection (Last 60 days consolidation high ceiling)
            pivot_price = hist['High'].iloc[-60:-5].max()
            pct_from_pivot = ((current_price - pivot_price) / pivot_price) * 100
            
            raw_metrics.append({
                "ticker": t, "hist": hist, "current_price": current_price, "pivot_price": pivot_price,
                "pct_from_pivot": pct_from_pivot, "raw_momentum": weighted_momentum, 
                "raw_vol_velocity": vol_velocity, "raw_eps": eps_raw, "pct_off_high": pct_off_high
            })
        except Exception as e:
            continue
            
    if not raw_metrics: return pd.DataFrame(), {}
    
    # PHASE 2: Competitive Cross-Market Percentile Ranking (The 1-99 Pro Scale)
    df = pd.DataFrame(raw_metrics)
    
    # Mathematical Percentile distribution across the entire running universe
    df['Price Strength (RS)'] = (df['raw_momentum'].rank(pct=True) * 98 + 1).astype(int)
    df['EPS Rating'] = (df['raw_eps'].rank(pct=True) * 98 + 1).astype(int)
    
    # Map Accumulation intensity to clear institutional alphabet letters
    def assign_ad_grade(val):
        if val > 0.15: return 'A-' or 'A'
        elif val > 0.0: return 'B'
        elif val > -0.15: return 'C'
        return 'D'
    df['Acc/Dis Grade'] = df['raw_vol_velocity'].apply(assign_ad_grade)
    
    # Calculate Unified Composite Master Rating
    df['Composite Rating'] = ((df['Price Strength (RS)'] * 0.5) + (df['EPS Rating'] * 0.5)).astype(int)
    
    # Generate final clean records for rendering
    final_analysis = []
    records_dictionary = {}
    
    for _, row in df.iterrows():
        t = row['ticker']
        
        # Determine actionable status tags
        if row['Composite Rating'] >= 80 and 0 <= row['pct_from_pivot'] <= 5.5:
            status = "🟩 Actionable Breakout"
        elif row['pct_off_high'] >= -15:
            status = "🔄 Building Valid Base"
        else:
            status = "⚠️ Lagging / Avoid"
            
        rec = {
            "Ticker": t,
            "Current Price": f"₹{row['current_price']:.2f}",
            "Composite Rating": f"{row['Composite Rating']}/99",
            "EPS Rating": f"{row['EPS Rating']}/99",
            "Price Strength (RS)": f"{row['Price Strength (RS)']}/99",
            "Acc/Dis Grade": row['Acc/Dis Grade'],
            "Pivot Delta": f"{row['pct_from_pivot']:.1f}%",
            "Off 52W High": f"{row['pct_off_high']:.1f}%",
            "Status": status,
            "raw_pivot": row['pivot_price'], "raw_hist": row['hist']
        }
        final_analysis.append(rec)
        records_dictionary[t] = rec
        
    # Sort entire table grid by highest Composite Strength overall
    df_sorted = pd.DataFrame(final_analysis).sort_values(by="Composite Rating", ascending=False)
    return df_sorted, records_dictionary

# Execute Pipeline
df_ranking, master_records = professional_grading_pipeline(PRO_POOL)

# --- WEB TERMINAL DASHBOARD LAYOUT ---
st.subheader("📋 1. Comparative Performance Matrix & Percentile Standings")
if not df_ranking.empty:
    st.dataframe(df_ranking[["Ticker", "Current Price", "Composite Rating", "EPS Rating", "Price Strength (RS)", "Acc/Dis Grade", "Pivot Delta", "Off 52W High", "Status"]], use_container_width=True, hide_index=True)

st.markdown("---")

# --- LIVE ISOLATION & PRO SEARCH OPPORTUNITY ---
st.subheader("🔎 2. Targeted Intelligence Search Unit")
search_query = st.text_input("Search or inject any outside NSE ticker to calculate its ratings instantly (e.g. PVRINOX.NS, LMW.NS, ECLERX.NS):", "").strip().upper()

active_selection = None
if search_query:
    if search_query in master_records:
        active_selection = search_query
    else:
        with st.spinner(f"Running deep mathematical grading profile for {search_query}..."):
            _, extra_rec = professional_grading_pipeline([search_query])
            if search_query in extra_rec:
                master_records.update(extra_rec)
                active_selection = search_query
            else:
                st.error("Ticker unrecognized. Verify code and verify '.NS' suffix is used.")
else:
    if list(master_records.keys()):
        active_selection = list(master_records.keys())[0]

# Render Analytical Breakdown & Predictive AI Candlestick Continuation
if active_selection and active_selection in master_records:
    s = master_records[active_selection]
    
    st.markdown(f"### Core Focus File: **{active_selection}**")
    st.info(f"🛡️ **Algorithmic Execution Constraints:** Initial Stop Loss Floor: ₹{s['raw_pivot']*0.93:.2f} (-7%) | Institutional Take Profit Objective: ₹{s['raw_pivot']*1.20:.2f} (+20%)")

    # ML Continuation Model Architecture
    df_chart = s['raw_hist'].copy()
    df_chart['Day_Index'] = np.arange(len(df_chart))
    
    # Isolate last 20 active candles to compute immediate velocity vectors
    X_train = df_chart[['Day_Index']].values[-20:]
    y_train = df_chart['Close'].values[-20:]
    ml_model = LinearRegression().fit(X_train, y_train)
    
    future_x = np.array([[len(df_chart) + i] for i in range(1, 6)])
    future_y = ml_model.predict(future_x)
    future_timeline = pd.date_range(start=df_chart.index[-1] + pd.Timedelta(days=1), periods=5)

    # Rendering Interactive Visual Framework
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df_chart.index[-60:], open=df_chart['Open'].iloc[-60:], high=df_chart['High'].iloc[-60:], low=df_chart['Low'].iloc[-60:], close=df_chart['Close'].iloc[-60:], name="Price Candles"))
    fig.add_trace(go.Scatter(x=df_chart.index[-60:], y=[s['raw_pivot']]*60, mode='lines', name='Resistance Pivot Ceiling', line=dict(color='orange', width=2, dash='dot')))
    fig.add_trace(go.Scatter(x=future_timeline, y=future_y, mode='lines+markers', name='5-Day Predictive AI Continuation Vector', line=dict(color='cyan', width=3)))
    
    fig.update_layout(yaxis_title="Price (INR)", xaxis_rangeslider_visible=False, height=450, margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(fig, use_container_width=True)
