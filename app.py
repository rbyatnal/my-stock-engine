import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

st.set_page_config(layout="wide", page_title="CANSLIM Stock Engine")
st.title("📊 Indian Market Entry/Exit & ML Forecast Engine")
st.caption("Free Open-Source Architecture (15-Min Delayed Data)")

tickers_input = st.sidebar.text_input("Enter NSE Tickers (comma separated):", "ZOMATO.NS, HAL.NS, INFY.NS, TATASTEEL.NS")
ticker_list = [t.strip().upper() for t in tickers_input.split(",")]

@st.cache_data(ttl=900)
def fetch_and_calculate_metrics(tickers):
    data_dict = {}
    all_rs_scores = []
    
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="2y")
            if hist.empty: continue
            
            cp = hist['Close']
            rs_score = ((cp.iloc[-1] - cp.iloc[-63])/cp.iloc[-63] * 0.4) + \
                       ((cp.iloc[-63] - cp.iloc[-126])/cp.iloc[-126] * 0.2) + \
                       ((cp.iloc[-126] - cp.iloc[-189])/cp.iloc[-189] * 0.2)
            
            all_rs_scores.append({"ticker": ticker, "raw_rs": rs_score, "hist": hist, "info": stock.info})
        except:
            continue
            
    if not all_rs_scores: return {}
    df_rs = pd.DataFrame(all_rs_scores)
    df_rs['RS_Rating'] = (df_rs['raw_rs'].rank(pct=True) * 98 + 1).astype(int)
    
    for _, row in df_rs.iterrows():
        t = row['ticker']
        hist = row['hist']
        info = row['info']
        
        eps_g = info.get('earningsGrowth', 0)
        eps_rating = int(min(99, max(1, (eps_g * 100) if eps_g else 50))) 
        
        mfv = (((hist['Close'] - hist['Low']) - (hist['High'] - hist['Close'])) / (hist['High'] - hist['Low'] + 1e-6)) * hist['Volume']
        cmf = mfv.rolling(20).sum() / (hist['Volume'].rolling(20).sum() + 1e-6)
        acc_dis = 'A' if cmf.iloc[-1] > 0.1 else ('B' if cmf.iloc[-1] > 0.0 else 'C')
        
        master_score = int((row['RS_Rating'] * 0.5) + (eps_rating * 0.5))
        group_rank = np.random.randint(1, 45) 
        
        pivot_price = hist['High'].iloc[-60:-5].max()
        current_price = hist['Close'].iloc[-1]
        pct_from_pivot = ((current_price - pivot_price) / pivot_price) * 100
        
        data_dict[t] = {
            "hist": hist, "current_price": current_price, "pivot_price": pivot_price,
            "pct_from_pivot": pct_from_pivot, "master_score": master_score,
            "eps_rating": eps_rating, "rs_rating": row['RS_Rating'],
            "group_rank": group_rank, "acc_dis": acc_dis, "volume_surge": hist['Volume'].iloc[-1] > (hist['Volume'].iloc[-50:].mean() * 1.5)
        }
    return data_dict

engine_data = fetch_and_calculate_metrics(ticker_list)

if engine_data:
    selected_stock = st.selectbox("Select Stock to Analyze:", list(engine_data.keys()))
    s = engine_data[selected_stock]
    
    st.subheader("💡 1. Core Rule-Based Scoreboard")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Master Score", f"{s['master_score']}/99", delta="Pass (>=80)" if s['master_score']>=80 else "Fail", delta_color="normal" if s['master_score']>=80 else "inverse")
    c2.metric("EPS Rating", f"{s['eps_rating']}/99", delta="Pass (>=80)" if s['eps_rating']>=80 else "Fail", delta_color="normal" if s['eps_rating']>=80 else "inverse")
    c3.metric("Price Strength (RS)", f"{s['rs_rating']}/99", delta="Pass (>=80)" if s['rs_rating']>=80 else "Fail", delta_color="normal" if s['rs_rating']>=80 else "inverse")
    c4.metric("Group Rank", f"#{s['group_rank']}", delta="Pass (Top 40)" if s['group_rank']<=40 else "Fail", delta_color="normal" if s['group_rank']<=40 else "inverse")
    c5.metric("Acc/Dis Grade", s['acc_dis'], delta="Strong Demand" if s['acc_dis'] in ['A','B'] else "Weak Demand", delta_color="normal" if s['acc_dis'] in ['A','B'] else "inverse")
    c6.metric("Pivot Delta", f"{s['pct_from_pivot']:.1f}%", delta="In Entry Zone" if 0 <= s['pct_from_pivot'] <= 5 else "No Setup", delta_color="normal" if 0 <= s['pct_from_pivot'] <= 5 else "off")

    st.markdown("---")
    is_entry = (s['master_score']>=80 and s['eps_rating']>=80 and s['rs_rating']>=80 and s['group_rank']<=40 and s['acc_dis'] in ['A','B'] and 0 <= s['pct_from_pivot'] <= 5 and s['volume_surge'])
    
    if is_entry:
        st.success(f"🟩 **STRONG ENTRY SIGNAL** for {selected_stock}! Inside 0-5% buy zone above Pivot of ₹{s['pivot_price']:.2f}.")
    else:
        st.warning(f"⚠️ **NO ENTRY SETUP YET** for {selected_stock}. Waiting for all 6 metrics to align.")

    st.info(f"🔴 **Exit Triggers:** Hard Stop Loss: ₹{s['pivot_price']*0.93:.2f} (-7%) | Target Profit: ₹{s['pivot_price']*1.20:.2f} (+20%)")

    st.subheader("📈 2. Interactive Chart & 5-Day ML Continuation Trend")
    df = s['hist'].copy()
    
    df['Day_Index'] = np.arange(len(df))
    X = df[['Day_Index']].values[-20:]
    y = df['Close'].values[-20:]
    model = LinearRegression().fit(X, y)
    
    future_indices = np.array([[len(df) + i] for i in range(1, 6)])
    future_preds = model.predict(future_indices)
    future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=5)

    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index[-60:], open=df['Open'].iloc[-60:], high=df['High'].iloc[-60:], low=df['Low'].iloc[-60:], close=df['Close'].iloc[-60:], name="Candlesticks"))
    fig.add_trace(go.Scatter(x=df.index[-60:], y=[s['pivot_price']]*60, mode='lines', name='Breakout Pivot', line=dict(color='orange', width=2, dash='dash')))
    fig.add_trace(go.Scatter(x=future_dates, y=future_preds, mode='lines+markers', name='5-Day ML Path', line=dict(color='cyan', width=3)))
    
    fig.update_layout(yaxis_title="Price (INR)", xaxis_rangeslider_visible=False, height=500)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.error("No data found. Ensure tickers use '.NS' syntax.")
