import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from datetime import datetime

# ============================================================
# PAGE CONFIG (MUST BE THE VERY FIRST STREAMLIT CALL)
# ============================================================
st.set_page_config(
    layout="wide",
    page_title="Rakshits-Insights-Terminal"
)

# ============================================================
# SESSION STATE INITIALIZATION
# ============================================================
if "theme" not in st.session_state:
    st.session_state.theme = "Black"

# ============================================================
# DYNAMIC STYLING & CONTRAST CSS
# ============================================================
BG = "#0E1117" if st.session_state.theme == "Black" else "#FFFFFF"
TEXT = "#FFFFFF" if st.session_state.theme == "Black" else "#111111"
PANEL = "#161B22" if st.session_state.theme == "Black" else "#F5F5F5"
BORDER = "#30363D" if st.session_state.theme == "Black" else "#D0D0D0"
LABEL_COLOR = "#A3B1C2" if st.session_state.theme == "Black" else "#555555"

st.markdown(
    f"""
    <style>
    /* Unclip main viewport and hide default Streamlit header bar */
    header[data-testid="stHeader"] {{
        background-color: transparent !important;
        z-index: 1;
    }}
    .block-container {{
        padding-top: 3.5rem !important;
        padding-bottom: 2rem !important;
    }}
    .stApp {{
        background-color: {BG};
        color: {TEXT};
    }}

    /* Custom Header Styling */
    .header-container {{
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 10px;
    }}
    .title-text {{
        font-size: 24px;
        font-weight: 800;
        color: {TEXT} !important;
    }}
    .sub-title-text {{
        font-size: 13px;
        font-weight: 400;
        color: {LABEL_COLOR} !important;
    }}

    /* Explicit styling for Streamlit Metrics */
    div[data-testid="metric-container"] {{
        background-color: {PANEL} !important;
        border: 1px solid {BORDER} !important;
        border-radius: 8px;
        padding: 12px !important;
    }}
    div[data-testid="stMetricLabel"] > label, div[data-testid="stMetricLabel"] {{
        color: {LABEL_COLOR} !important;
        font-size: 14px !important;
        font-weight: 600 !important;
    }}
    div[data-testid="stMetricValue"] {{
        color: {TEXT} !important;
        font-size: 22px !important;
        font-weight: 700 !important;
    }}

    /* Selectbox & Input Contrast Fixes */
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
# HEADER ROW
# ============================================================
header_col, toggle_col = st.columns([9, 1])

with header_col:
    st.markdown(
        """
        <div class="header-container">
            <span class="title-text">
                Rakshits-Insights-Terminal 
                <span class="sub-title-text">(CAN SLIM Quantitative Intelligence)</span>
            </span>
        </div>
        """,
        unsafe_allow_html=True
    )

with toggle_col:
    theme = st.toggle("⚫", value=(st.session_state.theme == "Black"))
    st.session_state.theme = "Black" if theme else "White"

st.markdown("<hr style='margin: 5px 0 20px 0; border-color: #30363D;'>", unsafe_allow_html=True)
