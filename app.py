import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(layout="wide", page_title="AI Velocity Scanner")
st.title("🦅 Predictive Machine Learning & Velocity Scanner")
st.caption("Asynchronous Random Forest Classification Model Pipeline — Indian Market Universe")

CORE_POOL = [
    "SYRMA.NS", "BSE.NS", "LMW.NS", "PVRINOX.NS", "METROPOLIS.NS", 
    "ECLERX.NS", "HAL.NS", "BEL.NS", "VBL.NS", "DIXON.NS", "ZOMATO.NS", "CDSL.NS"
]

@st.cache_data(ttl=900)
def process_machine_learning_pipeline(tickers):
    analyzed_pool = []
    
    for t in tickers:
        try:
            stock = yf.Ticker(t)
            hist = stock.history(period="2y") # Collect broad depth history for ML training arrays
            if hist.empty or len(hist) < 200: continue
            
            cp = hist['Close']
            current_price = cp.iloc[-1]
            
            # --- FEATURE ENGINEERING CELL FOR THE ML ENGINE ---
            df_features = hist.copy()
            df_features['Returns'] = df_features['Close'].pct_change()
            df_features['MA10'] = df_features['Close'].rolling(10).mean()
            df_features['MA30'] = df_features['Close'].rolling(30).mean()
            df_features['Vol_MA10'] = df_features['Volume'].rolling(10).mean()
            
            # Target Vector generation: Will price be higher in 5 trading sessions?
            df_features['Target'] = np.where(df_features['Close'].shift(-5) > df_features['Close'], 1, 0)
            df_features.dropna(inplace=True)
            
            # Splitting Features matrix
            feature_cols = ['Close', 'Volume', 'Returns', 'MA10', 'MA30', 'Vol_MA10']
            X = df_features[feature_cols].values
            y = df_features['Target'].values
            
            # Train the Multi-Decision Classifier Model
            # Trains dynamically on history up to the immediate candle
            clf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
            clf.fit(X[:-5], y[:-5])
            
            # Predict directional hit score for current live parameters
            current_vector = np.array([df_features[feature_cols].iloc[-1]])
            prob_higher = clf.predict_proba(current_vector)[0][1] * 100
            
            # Velocity Benchmarks
            return_3m = ((current_price - cp.iloc[-63]) / cp.iloc[-63]) * 100
            return_1y = ((current_price - cp.iloc[-252]) / cp.iloc[-252]) * 100
            
            recent_pivot = hist['High'].iloc[-40:-3].max()
            pivot_delta = ((current_price - recent_pivot) / recent_pivot) * 100
            high_52w = cp.max()
            pct_off_high = ((current_price - high_52w) / high_52w) * 100
            
            analyzed_pool.append({
                "Ticker": t, "Price (₹)": f"{current_price:.2f}",
                "3-Month Return": return_3m, "1-Year Return": return_1y,
                "ML Win Probability": prob_higher, "Delta From Pivot": pivot_delta,
                "Distance to 52W High": f"{pct_off_high:.1f}%",
                "raw_pivot": recent_pivot, "raw_hist": hist, "raw_current": current_price
            })
        except Exception as e:
            continue
            
    if not analyzed_pool: return pd.DataFrame(), {}
    
    df_out = pd.DataFrame(analyzed_pool).sort_values(by="ML Win Probability", ascending=False)
    df_out['3-Month Return'] = df_out['3-Month Return'].apply(lambda x: f"{x:+.1f}%")
    df_out['1-Year Return'] = df_out['1-Year Return'].apply(lambda x: f"{x:+.1f}%")
    df_out['Delta From Pivot'] = df_out['Delta From Pivot'].apply(lambda x: f"{x:+.1f}%")
    df_out['ML Win Probability'] = df_out['ML Win Probability'].apply(lambda x: f"{x:.1f}%")
    
    records_map = {r['Ticker']: r for r in analyzed_pool}
    return df_out, records_map

df_grid, master_map = process_machine_learning_pipeline(CORE_POOL)

# --- USER DISPLAY INTERFACE ---
st.subheader("📋 1. Core Predictive Matrix (Ranked by Asynchronous ML Probability Score)")
if not df_grid.empty:
    st.dataframe(df_grid[["Ticker", "Price (₹)", "ML Win Probability", "3-Month Return", "1-Year Return", "Delta From Pivot", "Distance to 52W High"]], use_container_width=True, hide_index=True)

st.markdown("---")
st.subheader("🔎 2. Targeted Technical Target Matrix")

search_box = st.text_input("Filter or run any specific stock code:", "").strip().upper()
active_ticker = search_box if (search_box in master_map) else (list(master_map.keys()) if master_map else None)

if active_ticker and active_ticker in master_map:
    s = master_map[active_ticker]
    st.markdown(f"### Performance Focus File: **{active_ticker}**")
    st.info(f"🔴 **Calculated Risk Limits:** Technical Stop Loss Floor: ₹{s['raw_pivot']*0.93:.2f} | Execution Target Range: ₹{s['raw_pivot']*1.20:.2f} to ₹{s['raw_pivot']*1.25:.2f}")
    
    df_c = s['raw_hist'].copy()
    
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df_c.index[-60:], open=df_c['Open'].iloc[-60:], high=df_c['High'].iloc[-60:], low=df_c['Low'].iloc[-60:], close=df_c['Close'].iloc[-60:], name="Candles"))
    fig.add_trace(go.Scatter(x=df_c.index[-60:], y=[s['raw_pivot']]*60, mode='lines', name='Chart Pivot Ceiling', line=dict(color='orange', width=2, dash='dot')))
    
    fig.update_layout(yaxis_title="Price (INR)", xaxis_rangeslider_visible=False, height=450, margin=dict(l=15, r=15, t=15, b=15))
    st.plotly_chart(fig, use_container_width=True)
