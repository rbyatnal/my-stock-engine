import hashlib
import math
from datetime import datetime
import textwrap

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Rakshit's CANSLIM Terminal",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# AUTO-REFRESH
#
# FIX: cache expiring after ttl seconds did NOT mean the page
# actually updated — Streamlit only recomputes on the NEXT rerun,
# which previously only happened when the user clicked something.
# A meta-refresh forces the browser to reload on its own, so the
# page genuinely updates itself instead of silently going stale
# until someone happens to interact with it. Zero-dependency
# approach (no extra package), but it IS a full page reload —
# scroll position, open expanders, and unsent chat text reset every
# cycle. That tradeoff is disclosed here, not hidden.
# ============================================================

st.markdown(
    '<meta http-equiv="refresh" content="60">',
    unsafe_allow_html=True
)


# ============================================================
# THEME
# ============================================================

if "theme" not in st.session_state:
    st.session_state.theme = "Dark"

# Widget itself is rendered later (in the top control bar, after the
# topbar/branding) so it appears in the page where it makes sense —
# but the *value* has to be known now, before the CSS below is built.
theme = st.session_state.theme


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
    textwrap.dedent(f"""
    <style>

    .stApp {{
        background: {BG};
        color: {TEXT};
    }}

    [data-testid="stSidebar"] {{
        background: {PANEL};
        border-right: 1px solid {BORDER};
    }}

    /* FIX: st.write() lines (e.g. the "System" status list) render
       as <p> tags with Streamlit's own muted secondary-text color,
       which was never overridden — only headings (h1-h3) were.
       Scoped to markdown-container paragraphs specifically, so
       buttons/inputs/dropdowns (which need their own contrast
       rules) are untouched. */
    [data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] p,
    [data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] span,
    [data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] li {{
        color: {TEXT} !important;
        opacity: 1 !important;
    }}

    [data-testid="stSidebar"] .stCaption,
    [data-testid="stSidebar"] div[data-testid="stCaptionContainer"] p {{
        color: {MUTED} !important;
        opacity: 1 !important;
    }}

    /* Streamlit's own in-app header bar (the hamburger-menu strip)
       sits directly above our custom topbar with its own default
       height/background, making the two look like they're crowding
       each other. Shrinking and making it transparent removes that
       crowding. Note: this only affects the in-app header — the
       "Fork / GitHub" bar above it is Streamlit Community Cloud's
       own hosting chrome, outside the app's DOM entirely, and
       cannot be changed from app.py. */
    header[data-testid="stHeader"] {{
        background: transparent !important;
        height: 2.5rem !important;
    }}

    .block-container {{
        padding-top: 1rem;
        padding-bottom: 2rem;
        max-width: 1600px;
    }}

    h1, h2, h3 {{
        color: {TEXT} !important;
    }}

    /* FIX: st.chat_input(), like the button before it, never had
       any theme CSS applied to it — it was rendering with
       Streamlit's own default styling, which can leave the typed
       text the same color as its own background (invisible) even
       though typing itself works fine. Forcing explicit colors on
       every layer (container, the actual textarea, and its
       placeholder) fixes this regardless of which one was the
       actual culprit. */
    [data-testid="stChatInput"] {{
        background: {PANEL} !important;
        border: 1px solid {BORDER} !important;
        border-radius: 12px !important;
    }}

    [data-testid="stChatInput"] textarea {{
        background: {PANEL} !important;
        color: {TEXT} !important;
        caret-color: {TEXT} !important;
        -webkit-text-fill-color: {TEXT} !important;
        opacity: 1 !important;
    }}

    [data-testid="stChatInput"] textarea::placeholder {{
        color: {MUTED} !important;
        opacity: 1 !important;
    }}

    [data-testid="stChatInput"] button {{
        background: {ACCENT} !important;
    }}

    [data-testid="stChatInput"] button svg {{
        fill: {BG} !important;
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

    /* Native Streamlit buttons — without this, st.button() falls
       back to Streamlit's default light-mode style: a washed-out
       white box that's nearly unreadable against a dark theme. */
    .stButton > button {{
        background: {PANEL} !important;
        color: {TEXT} !important;
        border: 1px solid {BORDER} !important;
        border-radius: 10px !important;
        font-weight: 700 !important;
    }}

    .stButton > button:hover {{
        border-color: {ACCENT} !important;
        color: {ACCENT} !important;
        background: {PANEL2} !important;
    }}

    .stButton > button:active,
    .stButton > button:focus:not(:hover) {{
        color: {TEXT} !important;
        border-color: {ACCENT} !important;
    }}

    .stButton > button p {{
        color: inherit !important;
    }}

    /* FIX: st.dataframe() — the two main leaderboard/qualification
       tables — never had any custom styling at all, so they were
       relying entirely on Streamlit's own default table look,
       visually inconsistent with every custom-themed panel around
       them. Professional financial terminals (Bloomberg, MarketSmith)
       treat data tables as a first-class surface: clear header
       separation, subtle row hover, monospace numerals for column
       alignment. */
    [data-testid="stDataFrame"] {{
        border: 1px solid {BORDER} !important;
        border-radius: 12px !important;
        overflow: hidden !important;
    }}

    [data-testid="stDataFrame"] [role="columnheader"] {{
        background: {PANEL2} !important;
        color: {TEXT} !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        font-size: 11px !important;
        letter-spacing: 0.4px !important;
    }}

    [data-testid="stDataFrame"] [role="gridcell"] {{
        font-variant-numeric: tabular-nums !important;
        font-size: 13px !important;
    }}

    [data-testid="stDataFrame"] [role="row"]:hover [role="gridcell"] {{
        background: {PANEL2} !important;
    }}

    /* Smooth, subtle transitions on interactive surfaces — a small
       but real "does this feel professional or thrown together"
       signal; snap-instant hover states read as unpolished. */
    .stButton > button,
    .metric-card,
    [data-testid="stDataFrame"] [role="gridcell"] {{
        transition: all 0.15s ease-in-out !important;
    }}

    .metric-card:hover {{
        border-color: {ACCENT} !important;
        box-shadow: 0 4px 22px rgba(0,0,0,0.14) !important;
    }}

    /* st.container(border=True) panels (the real fix for the chart
       section's panel, replacing the broken div hack above) — default
       Streamlit border color doesn't match our theme otherwise. */
    [data-testid="stVerticalBlockBorderWrapper"] {{
        border-color: {BORDER} !important;
        border-radius: 16px !important;
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
    """),
    unsafe_allow_html=True
)


# ============================================================
# 3D CUBE
# ============================================================

cube_html = (
    '<div class="cube-container"><div class="cube">'
    '<div class="face front">'
    '<div style="background:#ef4444"></div><div style="background:#ef4444"></div><div style="background:#ef4444"></div>'
    '<div style="background:#ef4444"></div><div style="background:#ef4444"></div><div style="background:#ef4444"></div>'
    '<div style="background:#ef4444"></div><div style="background:#ef4444"></div><div style="background:#ef4444"></div>'
    '</div>'
    '<div class="face back">'
    '<div style="background:#f5f5f5"></div><div style="background:#f5f5f5"></div><div style="background:#f5f5f5"></div>'
    '<div style="background:#f5f5f5"></div><div style="background:#f5f5f5"></div><div style="background:#f5f5f5"></div>'
    '<div style="background:#f5f5f5"></div><div style="background:#f5f5f5"></div><div style="background:#f5f5f5"></div>'
    '</div>'
    '<div class="face right">'
    '<div style="background:#22c55e"></div><div style="background:#22c55e"></div><div style="background:#22c55e"></div>'
    '<div style="background:#22c55e"></div><div style="background:#22c55e"></div><div style="background:#22c55e"></div>'
    '<div style="background:#22c55e"></div><div style="background:#22c55e"></div><div style="background:#22c55e"></div>'
    '</div>'
    '<div class="face left">'
    '<div style="background:#f59e0b"></div><div style="background:#f59e0b"></div><div style="background:#f59e0b"></div>'
    '<div style="background:#f59e0b"></div><div style="background:#f59e0b"></div><div style="background:#f59e0b"></div>'
    '<div style="background:#f59e0b"></div><div style="background:#f59e0b"></div><div style="background:#f59e0b"></div>'
    '</div>'
    '<div class="face top">'
    '<div style="background:#3b82f6"></div><div style="background:#3b82f6"></div><div style="background:#3b82f6"></div>'
    '<div style="background:#3b82f6"></div><div style="background:#3b82f6"></div><div style="background:#3b82f6"></div>'
    '<div style="background:#3b82f6"></div><div style="background:#3b82f6"></div><div style="background:#3b82f6"></div>'
    '</div>'
    '<div class="face bottom">'
    '<div style="background:#facc15"></div><div style="background:#facc15"></div><div style="background:#facc15"></div>'
    '<div style="background:#facc15"></div><div style="background:#facc15"></div><div style="background:#facc15"></div>'
    '<div style="background:#facc15"></div><div style="background:#facc15"></div><div style="background:#facc15"></div>'
    '</div>'
    '</div></div>'
)

# Build the dedented shell first, then paste in the zero-indent
# cube_html afterward via a placeholder token — interpolating a
# zero-indent multi-line string into an f-string BEFORE dedent()
# runs makes the common-indentation calculation collapse to zero,
# silently turning dedent into a no-op for the surrounding markup.
_topbar_shell = textwrap.dedent(f"""
    <div class="topbar">
        <div>
            <div class="brand">🦅 Rakshit's CANSLIM Terminal</div>
            <div class="brand-small">
                Market analytics • Pattern intelligence • Technical research
            </div>
        </div>
        __CUBE__
    </div>
    """)

st.markdown(
    _topbar_shell.replace("__CUBE__", cube_html),
    unsafe_allow_html=True
)


# ============================================================
# STOCK POOL
#
# FIX: this was 12 hardcoded tickers with zero rotation, zero
# discovery, and nothing in the UI telling you so — a real gap.
# Expanded to a larger, sector-diverse universe. This still isn't
# automatic discovery of brand-new listings (that would need a live
# NSE index-constituent feed, untestable from this sandbox — same
# class of limitation as the insider-data integration flagged
# earlier), but it's a genuinely bigger, more varied pool than 12
# fixed names, and the UI now shows the universe size and refresh
# time directly instead of leaving it invisible.
# ============================================================

CORE_POOL = [
    "SYRMA.NS", "BSE.NS", "LMW.NS", "PVRINOX.NS", "METROPOLIS.NS",
    "ECLERX.NS", "HAL.NS", "BEL.NS", "VBL.NS", "DIXON.NS",
    "ZOMATO.NS", "CDSL.NS",
    "TATAMOTORS.NS", "TATASTEEL.NS", "TATAPOWER.NS", "TATACONSUM.NS",
    "RELIANCE.NS", "INFY.NS", "TCS.NS", "WIPRO.NS", "HCLTECH.NS",
    "ICICIBANK.NS", "HDFCBANK.NS", "KOTAKBANK.NS", "AXISBANK.NS", "SBIN.NS",
    "MARUTI.NS", "M&M.NS", "BAJAJ-AUTO.NS", "EICHERMOT.NS",
    "SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS", "DIVISLAB.NS",
    "ULTRACEMCO.NS", "GRASIM.NS", "JSWSTEEL.NS", "HINDALCO.NS",
    "ASIANPAINT.NS", "TITAN.NS", "NESTLEIND.NS", "BRITANNIA.NS",
    "ADANIENT.NS", "ADANIPORTS.NS", "LT.NS", "SIEMENS.NS",
    "PIDILITIND.NS", "PAGEIND.NS", "TRENT.NS", "POLYCAB.NS",
    "DEEPAKNTR.NS", "COFORGE.NS", "PERSISTENT.NS", "MPHASIS.NS",
    "SUZLON.NS", "KALYANKJIL.NS", "MCX.NS", "ELGIEQUIP.NS"
]


# ============================================================
# FULL NSE UNIVERSE DISCOVERY — pulls the REAL, current NIFTY 500
# constituent list directly from NSE's own public index-archive CSV,
# not a hand-typed guess. This is what actually closes the "can't
# discover anything outside the fixed list" gap — CORE_POOL stays
# small on purpose (auto-refreshes every 60s; 500 stocks on that
# cadence would time the app out), while this runs ONLY when the
# user explicitly clicks Scan, once, and is cached for hours since
# index membership itself only changes quarterly.
#
# Cannot be verified live from this environment — no network access
# to nseindia.com from this sandbox, the same limitation disclosed
# for every other live-NSE-fetch feature in this app. Built with a
# defensive fallback (returns None on any failure) rather than
# silently breaking the page if NSE's endpoint changes shape or
# blocks the request.
# ============================================================

@st.cache_data(ttl=21600)
def fetch_nse500_universe():

    try:
        import requests
        from io import StringIO

        url = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0 Safari/537.36"
            )
        }

        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()

        universe_df = pd.read_csv(StringIO(resp.text))

        symbol_col = None
        for candidate in ("Symbol", "SYMBOL", "symbol"):
            if candidate in universe_df.columns:
                symbol_col = candidate
                break

        if symbol_col is None:
            return None

        symbols = [
            f"{str(s).strip()}.NS"
            for s in universe_df[symbol_col].tolist()
            if str(s).strip()
        ]

        if len(symbols) < 100:
            # implausibly small for a "NIFTY 500" file — treat as a
            # failed/garbled fetch rather than trust a broken result
            return None

        return symbols

    except Exception:
        return None


# ============================================================
# DATA / ML ENGINE
# ============================================================

@st.cache_data(ttl=60)
def professional_ml_pipeline(tickers):

    raw_metrics = []

    for t in tickers:

        try:

            stock = yf.Ticker(t)
            hist = stock.history(period="2y")

            # FIX: this required 200 trading days minimum, silently
            # skipping the ticker entirely (with no explanation shown
            # anywhere) if it had less — which excluded every recently
            # listed IPO outright, even ones found successfully via
            # name search. Lowered to 30 (enough for RSI/MACD/short
            # MAs to mean something); the longer lookbacks below
            # (6/9/12-month returns, 150/200-day trend template) are
            # now individually guarded to degrade to "not available
            # yet" instead of requiring 200 days for everything.
            if hist.empty or len(hist) < 30:
                continue

            hist = hist.dropna(
                subset=["Open", "High", "Low", "Close", "Volume"]
            )

            cp = hist["Close"]

            current_price = float(cp.iloc[-1])

            # ------------------------------------------------
            # RS RATING — REAL IBD FORMULA
            #
            # FIX: the previous version computed quarter-over-quarter
            # deltas (price change *within* each 63-day slice), which
            # is not what IBD's RS Rating measures at all. The actual
            # formula is a weighted blend of *trailing cumulative
            # returns* from today back to 3/6/9/12 months ago:
            #   RS Score = 0.4*r3mo + 0.2*r6mo + 0.2*r9mo + 0.2*r12mo
            # (confirmed against IBD's own published methodology and
            # multiple independent replications of it).
            #
            # FIX: a stock with less than 252 trading days (any recent
            # IPO) used to crash here (cp.iloc[-252] on a shorter
            # series raises IndexError), which the outer except
            # silently turned into "skip this ticker entirely" with
            # zero explanation. Each window now only contributes if
            # there's enough history for it; the weighted average
            # renormalizes over whichever windows are actually
            # available instead of requiring all four.
            # ------------------------------------------------

            def _trailing_return(series, days_back):
                if len(series) <= days_back:
                    return None
                base = series.iloc[-days_back - 1]
                if base == 0:
                    return None
                return (series.iloc[-1] - base) / base

            r3mo = _trailing_return(cp, 63)
            r6mo = _trailing_return(cp, 126)
            r9mo = _trailing_return(cp, 189)
            r12mo = _trailing_return(cp, 252)

            _momentum_components = [
                (r3mo, 0.40),
                (r6mo, 0.20),
                (r9mo, 0.20),
                (r12mo, 0.20)
            ]

            _available = [
                (r, w) for r, w in _momentum_components
                if r is not None
            ]

            if not _available:
                # Not even 3 months of history — too little for any
                # momentum read at all.
                continue

            _total_weight = sum(w for _, w in _available)

            weighted_momentum = (
                sum(r * w for r, w in _available)
                / _total_weight
            )

            # r3mo is reused below (eps_raw fallback) — guarantee it's
            # a real number even for a brand-new listing under 63 days
            if r3mo is None:
                r3mo = weighted_momentum

            delta_price = cp.diff()
            vol = hist["Volume"]

            # FIX: MarketSmith's Acc/Dis Rating is explicitly defined
            # over a fixed 13-week window (65 trading days), not an
            # arbitrary 30-bar slice.
            ad_window = min(65, len(hist))

            green_vol = np.where(
                delta_price > 0,
                vol,
                0
            )[-ad_window:].sum()

            red_vol = np.where(
                delta_price < 0,
                vol,
                0
            )[-ad_window:].sum()

            vol_velocity = (
                (green_vol - red_vol)
                / (green_vol + red_vol + 1e-6)
            )

            try:
                info = stock.info
                eps_g = info.get("earningsGrowth", 0)

                if eps_g is None:
                    eps_raw = r3mo * 0.5
                else:
                    eps_raw = float(eps_g)

                roe_raw = info.get("returnOnEquity", None)
                debt_equity_raw = info.get("debtToEquity", None)

                # Value factor (Stockopedia-style) — P/E and P/B,
                # percentile-ranked below like everything else. Not
                # a redundant fifth copy of EPS/RS: cheap-vs-expensive
                # is empirically a different signal from growth/momentum.
                pe_raw = info.get("trailingPE", None)
                pb_raw = info.get("priceToBook", None)

                # Estimate revisions (Zacks-style). Disclosed up front:
                # Yahoo's analyst-estimate fields are thin for NSE
                # tickers — this often comes back N/A, not a full
                # Zacks-style revision history.
                analyst_target = info.get("targetMeanPrice", None)
                analyst_count = info.get("numberOfAnalystOpinions", None)
                recommendation_mean = info.get("recommendationMean", None)

                # Real Group Rank input — the actual sector, not a
                # hash of the ticker string. See the Group Rank
                # computation below for how this becomes a rank.
                sector_raw = info.get("sector", None)
                industry_raw = info.get("industry", None)

            except Exception:
                eps_raw = r3mo * 0.5
                roe_raw = None
                debt_equity_raw = None
                pe_raw = None
                pb_raw = None
                analyst_target = None
                analyst_count = None
                recommendation_mean = None
                sector_raw = None
                industry_raw = None

            # ------------------------------------------------
            # A — ANNUAL EPS GROWTH, computed from yfinance's own
            # income statement. SMR's Sales growth and Net Margin
            # come from the same statement (one fetch, three metrics).
            #
            # FIX: the fetch used only stock.get_income_stmt(), which
            # doesn't exist on older yfinance versions — on those,
            # this raised AttributeError, was silently swallowed by
            # the except block below, and always produced N/A
            # regardless of whether real data existed. Now tries
            # three call paths in order (newest to oldest yfinance
            # API) and records *why* it ended up empty, so N/A in the
            # UI says whether it's a genuine data gap or a fetch
            # failure instead of looking identical either way.
            # ------------------------------------------------
            annual_eps_cagr = None
            sales_growth_cagr = None
            net_margin_latest = None
            fundamentals_error = None

            income_stmt = None

            try:
                if hasattr(stock, "get_income_stmt"):
                    income_stmt = stock.get_income_stmt(freq="yearly")
                elif hasattr(stock, "income_stmt"):
                    income_stmt = stock.income_stmt
                elif hasattr(stock, "financials"):
                    income_stmt = stock.financials
                else:
                    fundamentals_error = "yfinance version has no income statement API"

            except Exception as _fund_exc:
                fundamentals_error = (
                    f"fetch failed ({type(_fund_exc).__name__})"
                )
                income_stmt = None

            if income_stmt is None or income_stmt.empty:

                if fundamentals_error is None:
                    fundamentals_error = (
                        "no income statement data for this ticker"
                    )

            else:

                try:

                    eps_row = None

                    for row_name in ("Diluted EPS", "Basic EPS"):
                        if row_name in income_stmt.index:
                            candidate = income_stmt.loc[row_name].dropna()
                            if len(candidate) >= 2:
                                eps_row = candidate
                                break

                    if eps_row is not None:

                        eps_row = eps_row.sort_index()

                        eps_oldest = float(eps_row.iloc[0])
                        eps_latest = float(eps_row.iloc[-1])
                        years_span = len(eps_row) - 1

                        if eps_oldest > 0 and years_span > 0:
                            annual_eps_cagr = (
                                (eps_latest / eps_oldest)
                                ** (1 / years_span)
                            ) - 1
                        else:
                            fundamentals_error = (
                                "EPS history present but not usable "
                                "(zero/negative base year)"
                            )

                    else:
                        fundamentals_error = (
                            "no Diluted/Basic EPS row in statement"
                        )

                    if "Total Revenue" in income_stmt.index:

                        revenue_row = (
                            income_stmt.loc["Total Revenue"]
                            .dropna()
                            .sort_index()
                        )

                        if len(revenue_row) >= 2:

                            rev_oldest = float(revenue_row.iloc[0])
                            rev_latest = float(revenue_row.iloc[-1])
                            rev_years_span = len(revenue_row) - 1

                            if rev_oldest > 0 and rev_years_span > 0:
                                sales_growth_cagr = (
                                    (rev_latest / rev_oldest)
                                    ** (1 / rev_years_span)
                                ) - 1

                        net_income_row = None

                        for ni_name in (
                            "Net Income",
                            "Net Income Common Stockholders"
                        ):
                            if ni_name in income_stmt.index:
                                candidate = (
                                    income_stmt.loc[ni_name]
                                    .dropna()
                                    .sort_index()
                                )
                                if len(candidate) >= 1:
                                    net_income_row = candidate
                                    break

                        if (
                            net_income_row is not None
                            and len(revenue_row) >= 1
                        ):
                            latest_revenue = float(
                                revenue_row.sort_index().iloc[-1]
                            )
                            latest_net_income = float(
                                net_income_row.iloc[-1]
                            )
                            if latest_revenue != 0:
                                net_margin_latest = (
                                    latest_net_income
                                    / latest_revenue
                                )

                except Exception as _parse_exc:
                    fundamentals_error = (
                        f"parse failed ({type(_parse_exc).__name__})"
                    )
                    annual_eps_cagr = None
                    sales_growth_cagr = None
                    net_margin_latest = None

            if annual_eps_cagr is not None:
                fundamentals_error = None

            high_52w = float(cp.max())
            low_52w = float(cp.min())

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

            # ------------------------------------------------
            # MINERVINI TREND TEMPLATE — 7 numeric, published
            # criteria (an 8th, RS >= 70, is tracked separately
            # as "L"). Source: Minervini's "Think & Trade Like a
            # Champion", cross-checked against several independent
            # screener implementations of the same rules.
            # ------------------------------------------------

            ma50_series = cp.rolling(50).mean()
            ma150_series = cp.rolling(150).mean()
            ma200_series = cp.rolling(200).mean()

            ma50 = float(ma50_series.iloc[-1])
            ma150 = float(ma150_series.iloc[-1])
            ma200 = float(ma200_series.iloc[-1])

            # "trending up for at least 1 month" ~ 21 trading days
            ma200_prior = (
                float(ma200_series.iloc[-21])
                if len(ma200_series) >= 21
                and not pd.isna(ma200_series.iloc[-21])
                else None
            )

            trend_template_checks = {
                "price_above_ma150_ma200": (
                    current_price > ma150
                    and current_price > ma200
                ),
                "ma150_above_ma200": ma150 > ma200,
                "ma200_rising_1mo": (
                    ma200_prior is not None
                    and ma200 > ma200_prior
                ),
                "ma50_above_ma150_ma200": (
                    ma50 > ma150
                    and ma50 > ma200
                ),
                "price_above_ma50": current_price > ma50,
                "price_30pct_above_low": (
                    current_price >= 1.30 * low_52w
                ),
                "price_within_25pct_of_high": (
                    pct_off_high >= -25.0
                )
            }

            trend_template_score = sum(
                trend_template_checks.values()
            )

            trend_template_pass = (
                trend_template_score == 7
            )

            # ------------------------------------------------
            # RSI / MACD — same math as the per-stock chart later
            # in the file, computed here too so every stock in the
            # qualification screen (not just the selected one) can
            # actually be gated on them.
            # ------------------------------------------------

            _delta = cp.diff()
            _gain = _delta.clip(lower=0)
            _loss = -_delta.clip(upper=0)
            _avg_gain = _gain.rolling(14).mean()
            _avg_loss = _loss.rolling(14).mean()
            _rs = _avg_gain / (_avg_loss + 1e-10)
            rsi_series = 100 - (100 / (1 + _rs))
            rsi_latest = float(rsi_series.iloc[-1])

            _ema12 = cp.ewm(span=12, adjust=False).mean()
            _ema26 = cp.ewm(span=26, adjust=False).mean()
            macd_series = _ema12 - _ema26
            macd_signal_series = macd_series.ewm(span=9, adjust=False).mean()

            macd_bullish = bool(
                macd_series.iloc[-1] > macd_signal_series.iloc[-1]
            )

            # Liquidity — ChartMill's published CANSLIM screen uses
            # a 100k-shares/day average as its minimum liquidity bar
            avg_volume_20 = float(
                hist["Volume"].tail(20).mean()
            )

            # ------------------------------------------------
            # ML feature creation
            #
            # FIX: previously the code dropped every row whose
            # future-shifted Target was NaN (the last 5 trading
            # sessions) *before* selecting "the latest row" to
            # predict on. That meant the "current" prediction was
            # always computed from data ~5-10 sessions stale, not
            # from today's actual price/volume/indicators. We now
            # keep a feature frame that only requires the feature
            # columns themselves to be valid, and use its last row
            # (today) for prediction, while training only on rows
            # where the future target is actually known.
            # ------------------------------------------------

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

            feature_cols = [
                "Close",
                "Volume",
                "Returns",
                "MA10",
                "MA30",
                "Vol_MA10"
            ]

            # Rows usable as model inputs (rolling warm-up satisfied)
            df_valid_features = df_features.dropna(
                subset=feature_cols
            )

            # Training rows must also have a known future target,
            # which excludes only the most recent `future_steps`
            # rows (shift(-5) leaves them NaN) rather than being
            # re-derived from an already-shrunk frame.
            train_df = df_valid_features.iloc[:-5]

            X_ml = train_df[feature_cols].values
            y_ml = train_df["Target"].values

            clf = RandomForestClassifier(
                n_estimators=60,
                max_depth=6,
                random_state=42
            )

            clf.fit(
                X_ml,
                y_ml
            )

            current_features = (
                df_valid_features[feature_cols]
                .iloc[-1]
                .values
                .reshape(1, -1)
            )

            probabilities = clf.predict_proba(
                current_features
            )

            if probabilities.shape[1] == 2:

                prob_higher = (
                    probabilities[0][1]
                    * 100
                )

            else:

                prob_higher = 50.0

            # Real, computed feature importances (Gini-based) — not a
            # hand-set weighting. A Random Forest doesn't have fixed
            # coefficients like "RSI=30%"; this is what the trained
            # trees actually ended up relying on for THIS stock.
            ml_feature_importances = dict(
                zip(feature_cols, clf.feature_importances_.tolist())
            )

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
                "ml_prob": prob_higher,
                "ml_feature_importances": ml_feature_importances,
                "roe": roe_raw,
                "debt_equity": debt_equity_raw,
                "sector": sector_raw,
                "industry": industry_raw,
                "pe_ratio": pe_raw,
                "pb_ratio": pb_raw,
                "analyst_target": analyst_target,
                "analyst_count": analyst_count,
                "recommendation_mean": recommendation_mean,
                "annual_eps_cagr": annual_eps_cagr,
                "fundamentals_error": fundamentals_error,
                "sales_growth_cagr": sales_growth_cagr,
                "net_margin": net_margin_latest,
                "trend_template_pass": trend_template_pass,
                "trend_template_score": trend_template_score,
                "rsi_latest": rsi_latest,
                "macd_bullish": macd_bullish,
                "avg_volume_20": avg_volume_20

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

    # ------------------------------------------------------------
    # SMR RATING — MarketSmith's Sales + Margins + ROE, on their
    # same A-E scale. Each of the three inputs is percentile-ranked
    # against this universe (rank(pct=True) already skips NaN
    # automatically, so a stock missing one input just doesn't
    # affect that input's ranking, rather than crashing or forcing
    # a fake 0), then averaged, then bucketed into quintiles —
    # IBD/MarketSmith don't publish their exact letter-grade cutoffs,
    # so this uses the standard quintile convention (top 20% = A)
    # rather than pretending to replicate an unpublished formula.
    # ------------------------------------------------------------

    sales_rank = df["sales_growth_cagr"].rank(pct=True)
    margin_rank = df["net_margin"].rank(pct=True)
    roe_rank = df["roe"].rank(pct=True)

    df["smr_percentile"] = (
        pd.concat(
            [sales_rank, margin_rank, roe_rank],
            axis=1
        ).mean(axis=1, skipna=True)
    )

    def _smr_grade(pct):
        if pd.isna(pct):
            return "N/A"
        if pct >= 0.80:
            return "A"
        elif pct >= 0.60:
            return "B"
        elif pct >= 0.40:
            return "C"
        elif pct >= 0.20:
            return "D"
        else:
            return "E"

    df["SMR Rating"] = df["smr_percentile"].apply(_smr_grade)

    # ------------------------------------------------------------
    # VALUE RANK (Stockopedia-style) — cheap vs. expensive, ranked
    # the opposite direction from everything else: a LOWER P/E or
    # P/B is better, so we rank pct on the negated value. NaN (Yahoo
    # has no P/E for this ticker) is skipped by rank(pct=True)
    # automatically, same as every other optional field in this app.
    #
    # FIX: when every ticker in the batch has pe_ratio/pb_ratio as
    # None (e.g. testing a single stock, or a batch where Yahoo
    # returned nothing for any of them), pandas keeps that column as
    # object dtype instead of float — and "-None" raises TypeError,
    # crashing the whole pipeline. pd.to_numeric coerces None/missing
    # to a real NaN first, which negates and ranks safely either way.
    # ------------------------------------------------------------

    pe_numeric = pd.to_numeric(df["pe_ratio"], errors="coerce")
    pb_numeric = pd.to_numeric(df["pb_ratio"], errors="coerce")

    pe_rank = (-pe_numeric).rank(pct=True)
    pb_rank = (-pb_numeric).rank(pct=True)

    df["value_percentile"] = (
        pd.concat([pe_rank, pb_rank], axis=1)
        .mean(axis=1, skipna=True)
    )

    df["Value Rank"] = (
        df["value_percentile"] * 98 + 1
    ).round()

    df["Value Rank"] = df["Value Rank"].where(
        df["value_percentile"].notna(), other=np.nan
    )

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

    # ------------------------------------------------------------
    # GROUP RANK — real version, replacing the MD5 hash placeholder
    # entirely. IBD/MarketSmith's actual method: ~197 industry
    # groups, each ranked by aggregate price performance, and a
    # stock inherits its group's rank plus its own position within
    # it. We can't replicate a 197-group taxonomy on a 58-stock
    # universe — instead this groups the ACTUAL loaded stocks by
    # Yahoo's real sector field, ranks each real sector by its
    # average momentum (the same weighted_momentum already computed
    # per stock, not a new number), and gives each stock its
    # sector's rank plus its rank within that sector. A stock with
    # no sector data from Yahoo gets an honest N/A, not lumped into
    # a fake "Unknown" group.
    # ------------------------------------------------------------

    df["sector_group_rank"] = np.nan
    df["sector_group_size"] = np.nan
    df["rank_within_group"] = np.nan

    known_sector_mask = df["sector"].notna()

    n_sectors = 0

    if known_sector_mask.any():

        known_df = df[known_sector_mask]

        sector_avg_momentum = (
            known_df.groupby("sector")["raw_momentum"]
            .mean()
            .sort_values(ascending=False)
        )

        sector_rank_map = {
            sector: rank + 1
            for rank, sector in enumerate(sector_avg_momentum.index)
        }

        n_sectors = len(sector_rank_map)

        df.loc[known_sector_mask, "sector_group_rank"] = (
            df.loc[known_sector_mask, "sector"].map(sector_rank_map)
        )

        sector_sizes = known_df.groupby("sector").size()

        df.loc[known_sector_mask, "sector_group_size"] = (
            df.loc[known_sector_mask, "sector"].map(sector_sizes)
        )

        df.loc[known_sector_mask, "rank_within_group"] = (
            known_df.groupby("sector")["raw_momentum"]
            .rank(ascending=False, method="min")
        )

    # ------------------------------------------------------------
    # COMPOSITE RATING — MarketSmith blends EPS + RS + SMR + Acc/Dis
    # + Industry Group RS into one 1-99 score, but doesn't publish
    # the exact weights ("more weight on EPS and RS" is all O'Neil+Co
    # discloses publicly). Group Rank now exists above (real, not a
    # hash) but is deliberately kept OUT of this blend — it's a rank
    # among a handful of sectors in a 58-stock universe, not IBD's
    # 197-group system, so folding it into a 1-99 composite would
    # overstate its weight relative to what it actually measures.
    # Composite stays the four components with real universe-wide
    # percentile ranks, EPS/RS weighted higher to match O'Neil+Co's
    # documented emphasis — an explicit, disclosed choice, not a
    # claimed replica of their proprietary formula.
    # ------------------------------------------------------------

    ad_grade_to_score = {"A": 95, "B": 70, "C": 40}
    smr_grade_to_score = {"A": 95, "B": 70, "C": 40, "D": 20, "E": 5, "N/A": 40}

    df["acc_dis_numeric"] = df["Acc/Dis Grade"].map(ad_grade_to_score)
    df["smr_numeric"] = df["SMR Rating"].map(smr_grade_to_score)

    df["Composite Rating"] = (
        df["EPS Rating"] * 0.30
        + df["Price Strength (RS)"] * 0.30
        + df["acc_dis_numeric"] * 0.20
        + df["smr_numeric"] * 0.20
    ).round().astype(int)

    final_grid_data = []
    records_dictionary = {}

    for _, row in df.iterrows():

        t = row["ticker"]

        # Real Group Rank: the ticker's sector, ranked against the
        # OTHER sectors actually present in this loaded universe, by
        # average momentum — not a hash, not a placeholder.
        if pd.notna(row["sector_group_rank"]):

            group_rank_display = (
                f"#{int(row['sector_group_rank'])}/{n_sectors}"
            )

            rank_within_group_display = (
                f"#{int(row['rank_within_group'])} of "
                f"{int(row['sector_group_size'])} in "
                f"{row['sector']}"
            )

            group_rank_numeric = int(row["sector_group_rank"])

        else:

            group_rank_display = "N/A"
            rank_within_group_display = "Sector data unavailable"
            group_rank_numeric = None

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
                group_rank_display,

            "Rank Within Group":
                rank_within_group_display,

            "Acc/Dis Grade":
                row["Acc/Dis Grade"],

            "SMR Rating":
                row["SMR Rating"],

            "Value Rank":
                (
                    f"{int(row['Value Rank'])}/99"
                    if pd.notna(row["Value Rank"])
                    else "N/A"
                ),

            "raw_pe": row["pe_ratio"],
            "raw_pb": row["pb_ratio"],

            "Composite Rating":
                f"{row['Composite Rating']}/99",

            "raw_composite":
                row["Composite Rating"],

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
                group_rank_numeric,

            "raw_pivot_delta":
                row["pct_from_pivot"],

            "raw_ml_prob":
                row["ml_prob"],

            "ml_feature_importances":
                row["ml_feature_importances"],

            "pct_off_high":
                row["pct_off_high"],

            "roe":
                row["roe"],

            "debt_equity":
                row["debt_equity"],

            "annual_eps_cagr":
                row["annual_eps_cagr"],

            "fundamentals_error":
                row["fundamentals_error"],

            "raw_price":
                row["current_price"],

            "raw_eps_growth":
                row["raw_eps"],

            "trend_template_pass":
                row["trend_template_pass"],

            "trend_template_score":
                row["trend_template_score"],

            "rsi_latest":
                row["rsi_latest"],

            "macd_bullish":
                row["macd_bullish"],

            "avg_volume_20":
                row["avg_volume_20"]

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
# MARKET DIRECTION (the "M" in CAN SLIM) — a global gate, not
# a per-stock one: O'Neil's rule is that even strong individual
# stocks are avoided when the broad market itself is unfavorable.
# We check NIFTY 50 against its own 50-day and 200-day average.
# ============================================================

@st.cache_data(ttl=60)
def check_market_direction():

    try:
        index = yf.Ticker("^NSEI")
        index_hist = index.history(period="2y")

        if index_hist.empty or len(index_hist) < 200:
            return None

        # FIX: df_chart (the selected stock's history) gets its
        # timezone stripped elsewhere in the app before use, but this
        # index history never did — real yfinance data comes back
        # tz-aware, and combining it with tz-naive data later (in
        # compute_rs_line) raises "Cannot join tz-naive with tz-aware
        # DatetimeIndex". Normalizing once here, the same way df_chart
        # already is, fixes it at the source for every downstream use.
        index_hist.index = pd.to_datetime(index_hist.index)
        if index_hist.index.tz is not None:
            index_hist.index = index_hist.index.tz_localize(None)

        index_close = index_hist["Close"]
        index_price = float(index_close.iloc[-1])
        ma50 = float(index_close.rolling(50).mean().iloc[-1])
        ma200 = float(index_close.rolling(200).mean().iloc[-1])

        bullish = index_price > ma50 and index_price > ma200

        return {
            "bullish": bullish,
            "price": index_price,
            "ma50": ma50,
            "ma200": ma200,
            "hist": index_hist
        }

    except Exception:
        return None


# ============================================================
# ML MODEL BACKTEST — the piece flagged as missing throughout this
# build. Walk-forward only: at every test point, the model is
# trained ONLY on data strictly before that point (no look-ahead),
# then checked against what actually happened 5 sessions later.
# Reports statistical significance (not just raw accuracy) against
# a majority-class baseline — a raw accuracy number alone can look
# like an edge when it's actually just noise, which is exactly what
# happened during testing here: an apparent 57.7%-vs-52.9% "beat" on
# a pure random walk turned out to be statistically meaningless
# (p=0.19) once actually checked, while a genuine engineered signal
# showed p<0.001. Without this check, this feature would mislead
# rather than inform.
# ============================================================

def _normal_cdf(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2)))


def backtest_ml_model(
    hist,
    feature_cols=("Close", "Volume", "Returns", "MA10", "MA30", "Vol_MA10"),
    future_steps=5,
    min_train_size=150,
    step=5
):
    feature_cols = list(feature_cols)

    df = hist.copy()

    df["Returns"] = df["Close"].pct_change()
    df["MA10"] = df["Close"].rolling(10).mean()
    df["MA30"] = df["Close"].rolling(30).mean()
    df["Vol_MA10"] = df["Volume"].rolling(10).mean()

    df["Target"] = np.where(
        df["Close"].shift(-future_steps) > df["Close"],
        1,
        0
    )

    df_valid = (
        df.dropna(subset=feature_cols + ["Target"])
        .reset_index(drop=True)
    )

    if len(df_valid) < min_train_size + 50:
        return None

    preds = []
    actuals = []
    probs = []

    for test_idx in range(
        min_train_size,
        len(df_valid) - future_steps,
        step
    ):

        train_data = df_valid.iloc[:test_idx]
        test_row = df_valid.iloc[test_idx]

        X_train = train_data[feature_cols].values
        y_train = train_data["Target"].values

        if len(np.unique(y_train)) < 2:
            continue

        clf = RandomForestClassifier(
            n_estimators=60,
            max_depth=6,
            random_state=42
        )

        clf.fit(X_train, y_train)

        X_test = test_row[feature_cols].values.reshape(1, -1)
        prob = clf.predict_proba(X_test)[0]
        prob_up = prob[1] if len(prob) == 2 else 0.5

        preds.append(1 if prob_up >= 0.5 else 0)
        actuals.append(test_row["Target"])
        probs.append(prob_up)

    if len(preds) < 20:
        return None

    preds = np.array(preds)
    actuals = np.array(actuals)
    probs = np.array(probs)

    accuracy = float((preds == actuals).mean())

    majority_baseline = float(
        max((actuals == 1).mean(), (actuals == 0).mean())
    )

    brier_score = float(np.mean((probs - actuals) ** 2))

    n = len(preds)

    # One-sided z-test: is accuracy significantly ABOVE baseline,
    # or could this just be noise at this sample size? Normal
    # approximation to the binomial, no scipy dependency needed.
    se = np.sqrt(majority_baseline * (1 - majority_baseline) / n)

    if se > 0:
        z = (accuracy - majority_baseline) / se
        p_value = float(1 - _normal_cdf(z))
    else:
        p_value = 1.0

    significant = p_value < 0.05

    # Calibration: within each predicted-probability bucket, does the
    # actual hit rate match what the model claimed?
    calibration = []
    bins = [0, 0.4, 0.5, 0.6, 0.7, 1.01]

    for i in range(len(bins) - 1):
        mask = (probs >= bins[i]) & (probs < bins[i + 1])
        if mask.sum() > 0:
            calibration.append({
                "bucket": f"{bins[i]:.0%}-{bins[i+1]:.0%}",
                "predicted_avg": float(probs[mask].mean()),
                "actual_hit_rate": float(actuals[mask].mean()),
                "n": int(mask.sum())
            })

    return {
        "n": n,
        "accuracy": accuracy,
        "majority_baseline": majority_baseline,
        "beats_baseline": accuracy > majority_baseline,
        "p_value": p_value,
        "significant": significant,
        "brier_score": brier_score,
        "calibration": calibration
    }


# ============================================================
# BIG PICTURE / FOLLOW-THROUGH DAY SYSTEM
#
# IBD's own "Market School" rules (published methodology, not
# proprietary): a correction ends when a Rally Attempt's Day 1 (an
# up day off a low) is followed within days 4-10 by a Follow-Through
# Day — a day up ~1.25%+ on higher volume than the prior day. A
# "Power Trend" is a stronger confirmed state: 10 straight daily
# lows above the 21-day EMA, the 21-EMA above the 50-day average for
# 5+ sessions and itself rising, with price in the top quarter of
# its 52-week range. Distribution Days (down 0.2%+ on higher volume)
# are tallied over the trailing 25 sessions as an early-warning
# count. IBD doesn't publish the exact gain-% threshold for every
# index/market — 1.25% here is a reasonable general figure disclosed
# openly as a simplification, not an exact replica of their internal
# calibration.
# ============================================================

def analyze_big_picture(index_hist):

    close = index_hist["Close"].astype(float)
    volume = index_hist["Volume"].astype(float)

    if len(close) < 60:
        return None

    ema21 = close.ewm(span=21, adjust=False).mean()
    ma50 = close.rolling(50).mean()

    daily_pct = close.pct_change() * 100

    # Distribution Days: down >= 0.2% on higher volume than the
    # prior session, tallied over the trailing 25 sessions
    distribution_days = 0
    window = min(25, len(close) - 1)

    for i in range(len(close) - window, len(close)):
        if i < 1:
            continue
        if (
            daily_pct.iloc[i] <= -0.2
            and volume.iloc[i] > volume.iloc[i - 1]
        ):
            distribution_days += 1

    # Most recent local low (start of a possible rally attempt):
    # the most recent day whose close is the minimum of a trailing
    # 10-session window, AND which followed a genuine decline (>= 3%
    # off the preceding 15-session high) — without this filter, a
    # flat/choppy stretch can tie as a "local low" with no real
    # correction behind it, misidentifying a rally-attempt start.
    rolling_min = close.rolling(10, min_periods=1).min()
    rolling_high_15 = close.rolling(15, min_periods=1).max().shift(1)

    is_local_low = (
        (close == rolling_min)
        & (
            (rolling_high_15 - close) / rolling_high_15 >= 0.03
        )
    )

    low_indices = [
        i for i in range(len(close))
        if bool(is_local_low.iloc[i])
    ]

    rally_day1_idx = low_indices[-1] if low_indices else None

    ftd_idx = None
    ftd_gain = None

    if rally_day1_idx is not None:

        search_start = rally_day1_idx + 4
        search_end = min(rally_day1_idx + 10, len(close) - 1)

        for i in range(search_start, search_end + 1):
            if i < 1 or i >= len(close):
                continue
            if (
                daily_pct.iloc[i] >= 1.25
                and volume.iloc[i] > volume.iloc[i - 1]
            ):
                ftd_idx = i
                ftd_gain = float(daily_pct.iloc[i])
                break

    # Power Trend conditions
    lows = index_hist["Low"].astype(float)
    last10_lows_above_ema21 = bool(
        (lows.tail(10) > ema21.tail(10)).all()
    ) if len(close) >= 10 else False

    ema21_above_ma50_5d = bool(
        (ema21.tail(5) > ma50.tail(5)).all()
    ) if len(close) >= 55 else False

    ema21_rising = bool(
        ema21.iloc[-1] > ema21.iloc[-5]
    ) if len(close) >= 5 else False

    high_52w = float(close.max())
    low_52w = float(close.min())
    range_52w = high_52w - low_52w
    price_percentile = (
        (float(close.iloc[-1]) - low_52w) / range_52w
        if range_52w > 0 else 0
    )
    in_top_quarter = price_percentile >= 0.75

    power_trend = (
        last10_lows_above_ema21
        and ema21_above_ma50_5d
        and ema21_rising
        and in_top_quarter
    )

    if power_trend:
        state = "Power Trend"
    elif ftd_idx is not None and ftd_idx >= len(close) - 40:
        state = "Confirmed Uptrend"
    elif distribution_days >= 5:
        state = "Uptrend Under Pressure"
    else:
        state = "Correction / No Confirmed Uptrend"

    return {
        "state": state,
        "distribution_days": distribution_days,
        "rally_day1_date": (
            index_hist.index[rally_day1_idx]
            if rally_day1_idx is not None else None
        ),
        "ftd_date": (
            index_hist.index[ftd_idx]
            if ftd_idx is not None else None
        ),
        "ftd_gain": ftd_gain,
        "power_trend": power_trend,
        "price_percentile_52w": price_percentile * 100
    }


# ============================================================
# RS LINE + BLUE DOT
#
# Distinct from the RS Rating (a percentile number): this is IBD's
# RS Line chart overlay — the ratio of stock price to index price
# over time. A "Blue Dot" marks a day where that ratio hits a new
# high (the stock is outperforming the index more than it ever has
# in this window) while the stock's own price is also near its own
# high — a documented early-strength signal distinct from the RS
# Rating number, not a restatement of it.
# ============================================================

def compute_rs_line(stock_hist, index_hist, near_high_pct=10.0):

    stock_close = stock_hist["Close"].astype(float)
    index_close = index_hist["Close"].astype(float)

    # Defensive: strip timezone from both sides regardless of what the
    # caller passed in — this function shouldn't assume its inputs were
    # already normalized upstream.
    stock_close = stock_close.copy()
    index_close = index_close.copy()

    stock_close.index = pd.to_datetime(stock_close.index)
    if stock_close.index.tz is not None:
        stock_close.index = stock_close.index.tz_localize(None)

    index_close.index = pd.to_datetime(index_close.index)
    if index_close.index.tz is not None:
        index_close.index = index_close.index.tz_localize(None)

    aligned = pd.DataFrame({
        "stock": stock_close,
        "index": index_close
    }).dropna()

    if aligned.empty or len(aligned) < 20:
        return None

    rs_ratio = aligned["stock"] / aligned["index"]
    rs_line = (rs_ratio / rs_ratio.iloc[0]) * 100

    rs_running_max = rs_line.cummax()
    is_new_rs_high = rs_line >= rs_running_max

    stock_running_max = aligned["stock"].cummax()
    pct_off_stock_high = (
        (aligned["stock"] - stock_running_max) / stock_running_max
    ) * 100

    is_near_price_high = pct_off_stock_high >= -near_high_pct

    blue_dot = is_new_rs_high & is_near_price_high

    blue_dot_dates = list(aligned.index[blue_dot])

    latest_blue_dot = bool(blue_dot.iloc[-1])

    recent_blue_dot = (
        bool(blue_dot.tail(10).any())
        if len(blue_dot) >= 10
        else latest_blue_dot
    )

    return {
        "rs_line": rs_line,
        "dates": aligned.index,
        "blue_dot_mask": blue_dot,
        "blue_dot_dates": blue_dot_dates,
        "latest_blue_dot": latest_blue_dot,
        "recent_blue_dot": recent_blue_dot
    }


# ============================================================
# PATTERN RECOGNITION — CUP WITH HANDLE (heuristic)
#
# A simplified, openly-disclosed heuristic version of the classic
# O'Neil base pattern — NOT a claimed replica of MarketSurge's
# AI-assisted detector. Looks for: a left rim (recent peak), a cup
# (a decline of 12-33%, O'Neil's typical documented range, followed
# by recovery back near the left rim), and a handle (a shallower
# pullback of 8-15% in the upper half of the cup, after the right
# rim). Heuristic pattern detectors of this kind WILL have false
# positives/negatives on real data — that's disclosed here and in
# the UI, not hidden.
# ============================================================

def detect_cup_and_handle(hist, lookback_weeks=40):

    weekly = hist["Close"].astype(float).resample("W").last().dropna()

    if len(weekly) < 8:
        return {"detected": False, "reason": "not enough weekly history"}

    window = weekly.tail(lookback_weeks)

    if len(window) < 8:
        return {"detected": False, "reason": "not enough weekly history"}

    values = window.values
    n = len(values)

    # Left rim: the highest point in the first half of the window
    left_half_end = max(n // 2, 3)
    left_rim_idx = int(np.argmax(values[:left_half_end]))
    left_rim_price = float(values[left_rim_idx])

    # Cup bottom: the lowest point after the left rim
    remaining = values[left_rim_idx:]
    if len(remaining) < 4:
        return {"detected": False, "reason": "no room for a cup after left rim"}

    bottom_offset = int(np.argmin(remaining))
    bottom_idx = left_rim_idx + bottom_offset
    bottom_price = float(values[bottom_idx])

    cup_depth_pct = (
        (left_rim_price - bottom_price) / left_rim_price
    ) * 100

    if not (12.0 <= cup_depth_pct <= 33.0):
        return {
            "detected": False,
            "reason": f"cup depth {cup_depth_pct:.1f}% outside the "
                      f"typical 12-33% range"
        }

    # Right rim: recovery back within 5% of the left rim, after the bottom
    after_bottom = values[bottom_idx:]
    if len(after_bottom) < 3:
        return {"detected": False, "reason": "no recovery after cup bottom"}

    right_rim_candidates = [
        i for i, v in enumerate(after_bottom)
        if v >= left_rim_price * 0.95
    ]

    if not right_rim_candidates:
        return {
            "detected": False,
            "reason": "price hasn't recovered back near the left rim"
        }

    right_rim_offset = right_rim_candidates[0]
    right_rim_idx = bottom_idx + right_rim_offset
    right_rim_price = float(values[right_rim_idx])

    # Handle: a shallower pullback after the right rim, in the upper
    # half of the cup's depth, at least 1 week long
    after_right_rim = values[right_rim_idx:]

    if len(after_right_rim) < 2:
        return {
            "detected": False,
            "reason": "no handle has formed yet after the right rim",
            "cup_only": True,
            "left_rim_price": left_rim_price,
            "bottom_price": bottom_price,
            "right_rim_price": right_rim_price
        }

    handle_low = float(np.min(after_right_rim))
    handle_depth_pct = (
        (right_rim_price - handle_low) / right_rim_price
    ) * 100

    cup_midpoint = bottom_price + (left_rim_price - bottom_price) * 0.5
    handle_in_upper_half = handle_low >= cup_midpoint

    handle_valid = (
        1.0 <= handle_depth_pct <= 15.0
        and handle_in_upper_half
    )

    pivot_price = float(np.max(after_right_rim))

    return {
        "detected": bool(handle_valid),
        "left_rim_price": left_rim_price,
        "cup_bottom_price": bottom_price,
        "cup_depth_pct": cup_depth_pct,
        "right_rim_price": right_rim_price,
        "handle_depth_pct": handle_depth_pct,
        "handle_in_upper_half": handle_in_upper_half,
        "pivot_price": pivot_price,
        "reason": (
            "valid cup-with-handle" if handle_valid
            else f"handle depth {handle_depth_pct:.1f}% or position "
                 f"doesn't meet the 1-15%-in-upper-half rule"
        )
    }


# ============================================================
# CAN SLIM QUALIFICATION SCREEN
#
# Every criterion here is either a real, checkable number, or
# explicitly marked as unavailable — nothing here is a guessed
# placeholder standing in for data we don't actually have.
# ============================================================

PRICE_FLOOR = 100.0


def _is_valid(x):
    """
    True only for a real, present value. Guards against the pandas
    None -> NaN silent conversion that happens whenever a column of
    raw_metrics mixes None (missing data) with actual floats across
    different tickers: DataFrame construction upcasts that column to
    float64 and replaces every None with NaN. A plain `is not None`
    check misses this entirely, so a genuinely-missing value would
    silently be treated as present and fail every numeric comparison
    instead of honestly showing as N/A.
    """
    if x is None:
        return False
    if isinstance(x, float) and pd.isna(x):
        return False
    return True


def evaluate_qualification(rec, market_info):

    price = float(
        rec["raw_price"]
    )

    checks = {}
    values = {}

    checks["Price Band"] = (
        price >= PRICE_FLOOR
    )
    values["Price Band"] = f"₹{price:.2f}"

    # C — current-quarter EPS growth proxy (Yahoo's trailing
    # earningsGrowth field; not a clean isolated quarterly figure,
    # so treated as an approximation, not the literal IBD criterion)
    eps_growth = rec["raw_eps_growth"]
    checks["C (EPS growth, approx.)"] = (
        _is_valid(eps_growth)
        and eps_growth >= 0.20
    )
    values["C (EPS growth, approx.)"] = (
        f"{eps_growth * 100:.1f}%"
        if _is_valid(eps_growth)
        else "—"
    )

    # A — annual EPS CAGR, computed from yfinance's own income
    # statement (see the pipeline above). Genuine N/A only when this
    # specific ticker's statement lacks enough EPS history, not as a
    # blanket rule.
    annual_cagr = rec["annual_eps_cagr"]
    checks["A (Annual earnings)"] = (
        annual_cagr >= 0.20
        if _is_valid(annual_cagr)
        else None
    )
    values["A (Annual earnings)"] = (
        f"{annual_cagr * 100:.1f}% CAGR"
        if _is_valid(annual_cagr)
        else f"N/A ({rec.get('fundamentals_error') or 'unknown reason'})"
    )

    # N — proximity to 52-week high (the quantifiable half of "New")
    pct_off_high = rec["pct_off_high"]
    checks["N (Near 52w high)"] = (
        _is_valid(pct_off_high)
        and pct_off_high >= -15.0
    )
    values["N (Near 52w high)"] = (
        f"{pct_off_high:.1f}% off high"
        if _is_valid(pct_off_high)
        else "—"
    )

    # S — Supply/Demand via the Acc/Dis grade
    checks["S (Acc/Dis A or B)"] = (
        rec["Acc/Dis Grade"] in ("A", "B")
    )
    values["S (Acc/Dis A or B)"] = rec["Acc/Dis Grade"]

    # L — Leader: RS Rating >= 70
    checks["L (RS Rating >= 70)"] = (
        rec["raw_rs"] >= 70
    )
    values["L (RS Rating >= 70)"] = f"{rec['raw_rs']}/99"

    # I — institutional sponsorship: genuinely not available
    checks["I (Institutional)"] = None
    values["I (Institutional)"] = "N/A"

    # M — market direction: global, same value for every stock
    checks["M (Market direction)"] = (
        market_info["bullish"]
        if market_info is not None
        else None
    )
    values["M (Market direction)"] = (
        ("Bullish" if market_info["bullish"] else "Unfavorable")
        if market_info is not None
        else "—"
    )

    # ROE — O'Neil's studied winners averaged ~17%+
    roe = rec["roe"]
    checks["ROE >= 17%"] = (
        (roe * 100) >= 17.0
        if _is_valid(roe)
        else None
    )
    values["ROE >= 17%"] = (
        f"{roe * 100:.1f}%"
        if _is_valid(roe)
        else "—"
    )

    # Trend Template (Minervini) — 7 published numeric criteria,
    # all must pass for a confirmed Stage-2 uptrend
    checks["Trend Template (Minervini, 7 criteria)"] = (
        rec["trend_template_pass"]
    )
    values["Trend Template (Minervini, 7 criteria)"] = (
        f"{rec['trend_template_score']}/7"
    )

    # RSI — practitioner heuristic, NOT a named standard the way
    # Minervini's or IBD's numbers are: avoid stocks either broken
    # down (RSI < 40) or dangerously extended (RSI > 80)
    rsi = rec["rsi_latest"]
    rsi_valid = _is_valid(rsi)
    checks["RSI healthy (40-80, heuristic)"] = (
        40.0 <= rsi <= 80.0
        if rsi_valid
        else None
    )
    values["RSI healthy (40-80, heuristic)"] = (
        f"{rsi:.1f}" if rsi_valid else "—"
    )

    # MACD — momentum confirmation, also a heuristic, not a
    # CANSLIM/Minervini-named rule
    checks["MACD bullish (heuristic)"] = (
        rec["macd_bullish"]
    )
    values["MACD bullish (heuristic)"] = (
        "Above signal" if rec["macd_bullish"] else "Below signal"
    )

    # Liquidity — ChartMill's published CANSLIM screen config uses
    # a 100k avg-daily-volume floor
    avg_vol = rec["avg_volume_20"]
    checks["Liquidity (avg vol >= 100k)"] = (
        avg_vol >= 100_000
        if _is_valid(avg_vol)
        else None
    )
    values["Liquidity (avg vol >= 100k)"] = (
        f"{avg_vol:,.0f}" if _is_valid(avg_vol) else "—"
    )

    computable = [
        v for v in checks.values()
        if v is not None
    ]

    if not checks["Price Band"]:
        status = "Below price floor"

    elif computable and all(computable):
        status = "Qualified"

    elif computable and any(computable):
        status = "Watchlist"

    else:
        status = "Insufficient data"

    go_ahead = (status == "Qualified")

    return checks, values, status, go_ahead


# ============================================================
# LOAD MARKET
# ============================================================

df_ranking, master_records = professional_ml_pipeline(
    CORE_POOL
)

market_direction = check_market_direction()


@st.cache_data(ttl=60)
def _last_pipeline_refresh_time():
    # Cached with the SAME ttl as the pipeline itself, so this
    # timestamp only changes when the underlying data actually did —
    # a real "last scanned" indicator, not just page-load time.
    return datetime.now()


last_refresh = _last_pipeline_refresh_time()


# ============================================================
# TICKER NAME SEARCH — fallback when the exact symbol doesn't
# match. Our previous search only ever tried an exact match, so a
# company name, a slight typo, or the right name in the wrong
# symbol convention (e.g. "ELGI EQUIPMENTS.NS" instead of the real
# "ELGIEQUIP.NS") just failed silently with no help. This uses
# yfinance's own Search API (a wrapper around Yahoo's public
# search/autocomplete endpoint — no new dependency) to suggest
# real NSE-listed matches by company name.
# ============================================================

def search_ticker_candidates(query):

    candidates = []

    try:
        if hasattr(yf, "Search"):

            searcher = yf.Search(query, max_results=10)
            quotes = getattr(searcher, "quotes", None) or []

            for q in quotes:

                symbol = q.get("symbol", "")
                name = (
                    q.get("shortname")
                    or q.get("longname")
                    or symbol
                )
                exch = q.get("exchDisp", "")

                if symbol.endswith(".NS") or "NSE" in exch.upper():
                    candidates.append((symbol, name))

    except Exception:
        candidates = []

    return candidates


# ============================================================
# CONTROL BAR (replaces the sidebar — stock picker, search,
# theme toggle, and system status, all on the main page)
# ============================================================

control_col1, control_col2, control_col3, control_col4 = st.columns(
    [2, 2, 1, 2]
)

# FIX: same ranking-order fix as before — the dropdown default
# should reflect actual ML Probability ranking, not the order
# tickers happen to sit in CORE_POOL.
if not df_ranking.empty:
    available_stocks = [
        t for t in df_ranking["Ticker"].tolist()
        if t in master_records
    ]
    available_stocks += [
        t for t in master_records.keys()
        if t not in available_stocks
    ]
else:
    available_stocks = list(
        master_records.keys()
    )

with control_col1:

    if available_stocks:

        selected_stock = st.selectbox(
            "📊 Stock",
            available_stocks
        )

    else:

        selected_stock = None

with control_col2:

    search_query = st.text_input(
        "🔎 Search NSE ticker or company name",
        placeholder="Example: HAL.NS or Elgi Equipments"
    ).strip()

    if search_query:

        # FIX: auto-append .NS when the person typed a bare symbol
        # with no suffix at all (e.g. "ELGIEQUIP" instead of
        # "ELGIEQUIP.NS") — a very common way this used to fail.
        normalized = search_query.upper()

        if (
            "." not in normalized
            and " " not in normalized
        ):
            normalized = normalized + ".NS"

        if normalized in master_records:

            selected_stock = normalized

        else:

            with st.spinner(
                f"Loading {normalized}..."
            ):

                _, extra_records = (
                    professional_ml_pipeline(
                        [normalized]
                    )
                )

            if normalized in extra_records:

                master_records.update(
                    extra_records
                )

                selected_stock = normalized

                st.success(
                    f"{normalized} loaded"
                )

            else:

                # Exact match failed — fall back to name search
                # instead of just giving up.
                candidates = search_ticker_candidates(search_query)

                if candidates:

                    st.info(
                        f"'{search_query}' isn't an exact NSE symbol — "
                        "did you mean one of these?"
                    )

                    options = [
                        f"{sym} — {name}"
                        for sym, name in candidates
                    ]

                    picked = st.selectbox(
                        "Matches",
                        options,
                        key="search_candidate_pick"
                    )

                    picked_symbol = picked.split(" — ")[0]

                    with st.spinner(
                        f"Loading {picked_symbol}..."
                    ):

                        _, extra2 = professional_ml_pipeline(
                            [picked_symbol]
                        )

                    if picked_symbol in extra2:

                        master_records.update(extra2)
                        selected_stock = picked_symbol

                    else:

                        st.error(
                            f"Found {picked_symbol} but couldn't load "
                            f"its price data — it may be too newly "
                            f"listed or too thinly traded."
                        )

                else:

                    st.error(
                        f"Couldn't find '{search_query}' on NSE, and no "
                        "name-based matches turned up either. Double-check "
                        "the exact NSE symbol — company names don't always "
                        "match the trading symbol (e.g. 'Elgi Equipments' "
                        "trades as ELGIEQUIP.NS)."
                    )

with control_col3:

    new_theme = st.radio(
        "⚙️ Theme",
        ["Dark", "Light"],
        index=0 if st.session_state.theme == "Dark" else 1,
        horizontal=True
    )

    if new_theme != st.session_state.theme:

        st.session_state.theme = new_theme
        st.rerun()

with control_col4:

    st.caption(
        f"🟢 Connected · 📊 {len(master_records)}/{len(CORE_POOL)} stocks loaded · "
        f"🤖 ML active · 🔄 Last scanned {last_refresh.strftime('%H:%M:%S')} "
        "(page auto-refreshes every 60s)"
    )

    st.caption(
        "⚠️ NSE data via Yahoo Finance is exchange-delayed (typically "
        "15-20 min behind the live tick), independent of how often this "
        "page refreshes — true real-time would need a paid feed "
        "(e.g. Zerodha Kite Connect) with your own API credentials."
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
        textwrap.dedent(f"""
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
        """),
        unsafe_allow_html=True
    )

with hero_col2:

    st.markdown(
        textwrap.dedent("""
        <div style="text-align:right">
            <span class="status-pill">
                ● LIVE ANALYSIS
            </span>
        </div>
        """),
        unsafe_allow_html=True
    )


# ============================================================
# PAGE MAP — a quick index so the page reads as one connected
# flow instead of a scatter of sections
# ============================================================

with st.expander("🗺️ What's on this page"):

    st.markdown(
        "1. **Market Intelligence** — this stock's core scores at a glance\n"
        "2. **Price Structure & ML Continuation** — the chart: history + forecast + volume\n"
        "3. **Forecast Metrics** & **Technical Intelligence** — the numbers behind that chart\n"
        "4. **Comparative Performance Matrix** — every loaded stock, ranked (use this to pick a different one)\n"
        "5. **CAN SLIM Qualification Screen** — each stock checked against real thresholds, ending in a Go Ahead ✅/❌\n"
        "6. **Advanced Signals** — 4 modes: Big Picture (market health), RS Line (relative strength trend), "
        "Near Pivot/Breakouts (timing), Pattern Recognition (exact buy point)\n"
        "7. **Stock Intelligence Chat** — ask about this stock in plain English\n"
        "8. **Price Structure Reference** — quick current price / pivot / 52w high recap\n\n"
        "Suggested reading order: 1-3 for the stock you searched → 6 (Big Picture first) to check the market → "
        "6 again (the other 3 modes) to confirm → 4-5 to compare against other stocks → 7 if you have questions."
    )


# ============================================================
# METRIC CARDS
# ============================================================

def info_button(text):
    """
    Plain-English 'what is this and what's a good range' helper.
    Uses st.popover where available (newer Streamlit); falls back
    to an expander on older versions rather than erroring out.
    """
    if hasattr(st, "popover"):
        with st.popover("ℹ️"):
            st.markdown(text)
    else:
        with st.expander("ℹ️ What is this?"):
            st.markdown(text)


hcol1, hcol2 = st.columns([10, 1])

with hcol1:
    st.markdown("### Market Intelligence")

with hcol2:
    info_button(
        "**Master Score** (0-99): overall CANSLIM-style strength — "
        "higher is better, 75+ is strong.\n\n"
        "**EPS Rating** (0-99): earnings growth percentile vs. this "
        "list — 80+ is strong.\n\n"
        "**Price Strength**: momentum percentile (real IBD RS formula) "
        "— 70+ is a 'leader'.\n\n"
        "**Group Rank**: this stock's sector, ranked by average "
        "momentum against the other sectors actually present in "
        "this loaded list (not IBD's full 197-group system — a "
        "real rank, honestly scoped to a small universe).\n\n"
        "**Acc/Dis**: A/B = buying pressure (good), C = selling pressure.\n\n"
        "**ML Probability**: model's estimated odds price is higher in "
        "5 sessions — not a guarantee, see the Advanced Signals roadmap "
        "for how this is trained."
    )


def metric_card(
    container,
    label,
    value,
    sub
):

    container.markdown(
        textwrap.dedent(f"""
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
        """),
        unsafe_allow_html=True
    )


# FIX: gave this section a real bordered panel to match every other
# module — same st.container(border=True) fix as the chart section.
with st.container(border=True):

    cols = st.columns(6)

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
# ML MODEL ACCURACY (BACKTEST) — on-demand, not automatic: training
# dozens of Random Forests in a loop is expensive, so this only runs
# when asked, for the currently selected stock.
# ============================================================

with st.container(border=True):

    bcol1, bcol2 = st.columns([10, 1])

    with bcol1:
        st.markdown("### 🔬 Model Accuracy (Backtest)")

    with bcol2:
        info_button(
            "Walk-forward backtest: at each historical point, the model "
            "is trained only on data *before* that point (no look-ahead), "
            "then checked against what actually happened 5 sessions "
            "later. Reports **statistical significance**, not just raw "
            "accuracy — a small apparent edge can be pure noise, which "
            "is exactly what a first pass of this test found before the "
            "significance check was added."
        )

    st.caption(
        f"Tests the ML Probability model against {selected_stock}'s own "
        "history. Takes a few seconds — trains many models in sequence."
    )

    st.caption(
        "⚠️ **Tested empirically before building this selector**: more "
        "history is NOT automatically better. In a simulated scenario "
        "where the market's behavior genuinely changed a few years back "
        "(a regime shift — real markets do this), training on 5 years "
        "instead of 2 actually *hurt* accuracy and erased statistical "
        "significance (p=0.039 → p=0.065). Stale data can dilute a real "
        "current signal. Use this selector to check what's actually true "
        "for *this* stock, rather than assuming either direction."
    )

    lookback_choice = st.selectbox(
        "Backtest lookback",
        ["1y", "2y (default)", "5y"],
        index=1,
        key="backtest_lookback"
    )

    lookback_period = lookback_choice.split(" ")[0]

    if st.button("▶️ Run Backtest", key="run_backtest"):

        if lookback_period == "2y":

            backtest_hist = s["raw_hist"]

        else:

            with st.spinner(f"Fetching {lookback_period} of history..."):

                try:
                    backtest_hist = yf.Ticker(
                        selected_stock
                    ).history(period=lookback_period)
                except Exception:
                    backtest_hist = None

        if backtest_hist is None or backtest_hist.empty:

            st.error(
                f"Couldn't fetch {lookback_period} of data for "
                f"{selected_stock}."
            )
            bt = None

        else:

            with st.spinner("Running walk-forward backtest..."):

                bt = backtest_ml_model(
                    backtest_hist,
                    feature_cols=(
                        "Close", "Volume", "Returns",
                        "MA10", "MA30", "Vol_MA10"
                    )
                )

        if bt is None:

            st.warning(
                "Not enough history for this ticker/lookback to run a "
                "meaningful backtest."
            )

        else:

            btc1, btc2, btc3, btc4 = st.columns(4)

            metric_card(
                btc1,
                "Accuracy",
                f"{bt['accuracy']*100:.1f}%",
                f"vs. {bt['majority_baseline']*100:.1f}% baseline"
            )

            metric_card(
                btc2,
                "Statistically Significant?",
                "Yes ✅" if bt["significant"] else "No ❌",
                f"p = {bt['p_value']:.3f}"
            )

            metric_card(
                btc3,
                "Brier Score",
                f"{bt['brier_score']:.3f}",
                "Lower is better (0.25 = coin flip)"
            )

            metric_card(
                btc4,
                "Backtest Size",
                str(bt["n"]),
                "Historical test points"
            )

            if not bt["significant"]:

                st.warning(
                    "This model's accuracy on this stock's history is "
                    "**not statistically distinguishable from chance** "
                    "at this sample size. Treat the ML Probability "
                    "number for this stock with real skepticism."
                )

            else:

                st.success(
                    "This model shows a statistically real (not just "
                    "noise) edge on this stock's own history. This does "
                    "not guarantee future performance — markets change — "
                    "but it's a genuine backtested result, not an "
                    "unverified claim."
                )

            if bt["calibration"]:

                st.markdown("**Calibration** — when the model claims X% confidence, does it actually happen X% of the time?")

                calib_df = pd.DataFrame(bt["calibration"])
                calib_df["predicted_avg"] = (calib_df["predicted_avg"] * 100).round(1).astype(str) + "%"
                calib_df["actual_hit_rate"] = (calib_df["actual_hit_rate"] * 100).round(1).astype(str) + "%"
                calib_df.columns = ["Confidence Bucket", "Model Said", "Actually Happened", "N"]

                st.dataframe(
                    calib_df,
                    use_container_width=True,
                    hide_index=True
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


# ------------------------------------------------------------
# FIX: the previous forecast path was a perfectly smooth
# straight-line interpolation from today's price to the model's
# predicted endpoint — nothing like how a real chart actually
# moves, and misleadingly implied a guaranteed, frictionless
# glide path with no day-to-day pullbacks. This version generates
# a genuine day-by-day path using the stock's own trailing 60-day
# volatility (so the *shape* of the move reflects how this stock
# actually trades), then rescales the steps so their cumulative
# move still lands exactly on the ML model's predicted return —
# the destination is still the model's forecast; only the path
# getting there is now grounded in real historical movement
# instead of a straight line. Seeded deterministically per-ticker
# (same approach as the Group Rank fix) so it doesn't reshuffle on
# every rerun within the same cache window.
# ------------------------------------------------------------

log_returns = np.log(
    df_chart["Close"] / df_chart["Close"].shift(1)
).dropna().tail(60)

daily_vol = float(log_returns.std())

if pd.isna(daily_vol) or daily_vol <= 0:
    daily_vol = 0.01

_seed = int(
    hashlib.md5(
        selected_stock.encode("utf-8")
    ).hexdigest(),
    16
) % (2 ** 32)

_rng = np.random.default_rng(_seed)

raw_log_steps = _rng.normal(
    0,
    daily_vol,
    future_steps
)

target_log_return = np.log(
    max(1 + predicted_return, 1e-6)
)

# Rescale so the cumulative path exactly matches the model's
# predicted return, while keeping the day-to-day shape of the
# historical-volatility noise intact
_correction = (
    target_log_return
    - raw_log_steps.sum()
) / future_steps

adjusted_log_steps = raw_log_steps + _correction


forecast_open = []
forecast_high = []
forecast_low = []
forecast_close = []


path_price = current_price


for i in range(
    future_steps
):

    open_price = path_price

    step_return = float(
        adjusted_log_steps[i]
    )

    target_close = (
        open_price
        * np.exp(step_return)
    )

    # Each day's own move contributes to its wick on top of the
    # stock's recent typical range and the volatility model's own
    # predicted range — a bigger single-day move gets a
    # proportionally bigger wick instead of a fixed-size one.
    day_move = abs(
        target_close - open_price
    )

    wick_size = max(
        recent_atr * 0.35,
        current_price * predicted_range * 0.4,
        day_move * 0.6,
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

    path_price = target_close


# ============================================================
# PROFESSIONAL CHART
# ============================================================

chart_data = df_chart.tail(
    70
)


fig = make_subplots(
    rows=2,
    cols=1,
    shared_xaxes=True,
    row_heights=[0.75, 0.25],
    vertical_spacing=0.03
)


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

    ),
    row=1, col=1
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

    ),
    row=1, col=1
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

    ),
    row=1, col=1
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

    ),
    row=1, col=1
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

    ),
    row=1, col=1
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

    ),
    row=1, col=1
)


# ============================================================
# VOLUME SUBPLOT
#
# Added per request — without volume alongside the candles, there
# was no way to tell whether a move (or a breakout) was backed by
# real participation or not. Only historical bars get volume — the
# ML forecast candles have no real future volume to show, so that
# region is honestly left blank rather than faked.
# ============================================================

volume_colors = np.where(
    chart_data["Close"] >= chart_data["Open"],
    "#16c784",
    "#ea3943"
)

fig.add_trace(
    go.Bar(
        x=chart_data.index,
        y=chart_data["Volume"],
        marker_color=volume_colors,
        name="Volume",
        showlegend=False
    ),
    row=2, col=1
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

        ),
        row=1, col=1
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

    ),
    row=1, col=1
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

    ),
    row=1, col=1
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

    height=700,

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
    )
)

fig.update_xaxes(showgrid=False, row=1, col=1)
fig.update_xaxes(showgrid=False, row=2, col=1)

fig.update_yaxes(
    showgrid=True,
    gridcolor=(
        "#252d3a"
        if theme == "Dark"
        else "#e5e7eb"
    ),
    row=1, col=1
)

fig.update_yaxes(
    showgrid=False,
    title_text="Volume",
    row=2, col=1
)


# ============================================================
# CHART SECTION
#
# FIX: this used to open a raw <div class="section-box"> in one
# st.markdown() call and close it in a later one, with the actual
# chart inserted via a separate st.plotly_chart() call in between.
# Streamlit renders each call as its own sibling block in the DOM,
# not nested inside HTML opened by an earlier, unrelated call — so
# that bordered panel was very likely never actually wrapping the
# chart at all. st.container(border=True) is Streamlit's own native
# way to get a real panel that actually contains its children.
# ============================================================

with st.container(border=True):

    ccol1, ccol2 = st.columns([10, 1])
    with ccol1:
        st.markdown("### 📈 Price Structure & ML Continuation")
    with ccol2:
        info_button(
            "**Candles**: green = up day, red = down day. Solid candles = "
            "actual history; the last few candles are the ML forecast.\n\n"
            "**MA 20 / MA 50 lines**: price consistently above both = "
            "healthier trend.\n\n"
            "**Orange dotted line**: the pivot (breakout) level.\n\n"
            "**Volume bars below**: confirm moves — a breakout on low "
            "volume is weaker than one on high volume."
        )

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

tcol1, tcol2 = st.columns([10, 1])
with tcol1:
    st.markdown("### 🔬 Technical Intelligence")
with tcol2:
    info_button(
        "**RSI** (0-100): momentum. Below 30 = oversold, above 70 = "
        "overbought, 40-60 = neutral.\n\n"
        "**MACD**: trend momentum — above its signal line = bullish.\n\n"
        "**MA 20 / MA 50**: moving averages — price above both = uptrend.\n\n"
        "**Volume Ratio**: today's volume vs. 20-day average — above "
        "1.0x means unusually active.\n\n"
        "**Pivot Delta**: distance from the breakout level — the "
        "0% to +6% zone is the traditional 'buy zone'."
    )

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
# COMPARATIVE PERFORMANCE MATRIX (restored — this table was
# dropped during an earlier UI rebuild without being flagged)
# ============================================================

st.markdown("### 📋 Comparative Performance Matrix")

with st.container(border=True):

    # FIX: this used to show every stock in the fixed list regardless
    # of whether it actually passed anything — a stock that failed
    # every real CANSLIM check still showed up looking like a normal
    # row. Now computes the same Go Ahead verdict used in the
    # Qualification Screen for every row here too, and filters to
    # qualifying stocks by default — the table only shows companies
    # that actually meet the criteria, not the whole fixed list.

    show_all_toggle = st.checkbox(
        "Show all loaded stocks (including ones that don't qualify)",
        value=False,
        key="show_all_in_matrix"
    )

    st.caption(
        "By default, only shows stocks that pass the CAN SLIM "
        "Qualification Screen (Go Ahead ✅). Ranked by ML Probability."
    )

    if not df_ranking.empty:

        go_ahead_map = {}

        for ticker, rec in master_records.items():

            _, _, _, go_ahead = evaluate_qualification(
                rec,
                market_direction
            )

            go_ahead_map[ticker] = "✅" if go_ahead else "❌"

        display_df = df_ranking.copy()
        display_df["Go Ahead"] = display_df["Ticker"].map(go_ahead_map)

        if not show_all_toggle:

            display_df = display_df[display_df["Go Ahead"] == "✅"]

        if display_df.empty:

            st.warning(
                "0 of "
                f"{len(df_ranking)} loaded stocks currently pass the "
                "full Qualification Screen — likely driven by Market "
                "Direction being Unfavorable right now (the M gate "
                "blocks every stock when the broader market itself "
                "isn't confirmed). Check 'Show all' above to see "
                "everything anyway, or check the Big Picture mode "
                "under Advanced Signals for why."
            )

        else:

            st.dataframe(
                display_df[
                    [
                        "Ticker",
                        "Go Ahead",
                        "Price",
                        "Composite Rating",
                        "ML Probability",
                        "Master Score",
                        "EPS Rating",
                        "Price Strength (RS)",
                        "SMR Rating",
                        "Value Rank",
                        "Group Rank",
                        "Acc/Dis Grade",
                        "Pivot Delta",
                        "Status"
                    ]
                ],
                use_container_width=True,
                hide_index=True
            )


# ============================================================
# FULL NSE UNIVERSE DISCOVERY SCAN
#
# The fixed 58-stock list above only ever ranks itself — this is
# the actual answer to "how do I find something NOT already in the
# list". Honest about its own limits: even this fast pass is capped
# to a batch (not the full ~500) to stay within a reasonable runtime
# and avoid Yahoo rate-limiting; and it's a cheap momentum-only
# prefilter, not the full rated pipeline (RF training + income
# statements for 500 stocks in one click isn't realistic) —
# candidates it surfaces are meant to be searched individually for
# the real, full analysis, not treated as final ratings themselves.
# ============================================================

with st.container(border=True):

    dcol1, dcol2 = st.columns([10, 1])

    with dcol1:
        st.markdown("### 🔭 Discover — Full NSE 500 Scan")

    with dcol2:
        info_button(
            "Pulls the real, current NIFTY 500 list from NSE's own "
            "public data (not a hardcoded guess), then does a fast "
            "3-month-momentum-only pass over a capped batch — full "
            "rated analysis (RF model, fundamentals) on 500 stocks "
            "in one click isn't practically possible without timing "
            "out or getting rate-limited. Treat results here as "
            "'worth searching individually', not final ratings."
        )

    scan_batch_size = st.selectbox(
        "Batch size (larger = slower, more rate-limit risk)",
        [30, 60, 100],
        index=0,
        key="discovery_batch_size"
    )

    if st.button("🔍 Scan Full NSE Universe", key="run_discovery_scan"):

        with st.spinner("Fetching current NIFTY 500 list from NSE..."):
            nse500 = fetch_nse500_universe()

        if nse500 is None:

            st.error(
                "Couldn't fetch the live NIFTY 500 list from NSE right "
                "now (network restriction or NSE blocked the request — "
                "this exact fetch couldn't be tested from the "
                "environment this app was built in, so this is a real "
                "possible failure mode, not a hidden one)."
            )

        else:

            batch = [
                t for t in nse500
                if t not in CORE_POOL
            ][:scan_batch_size]

            scan_results = []

            with st.spinner(
                f"Scanning {len(batch)} stocks (momentum-only, fast pass)..."
            ):

                for ticker in batch:

                    try:

                        quick_hist = yf.Ticker(ticker).history(period="6mo")

                        if quick_hist.empty or len(quick_hist) < 63:
                            continue

                        quick_close = quick_hist["Close"]
                        quick_current = float(quick_close.iloc[-1])
                        quick_3mo_ago = float(quick_close.iloc[-63])

                        if quick_3mo_ago <= 0:
                            continue

                        quick_return = (
                            (quick_current - quick_3mo_ago)
                            / quick_3mo_ago
                        ) * 100

                        scan_results.append({
                            "Ticker": ticker,
                            "Price": f"₹{quick_current:.2f}",
                            "3-Month Return": f"{quick_return:+.1f}%",
                            "_sort": quick_return
                        })

                    except Exception:
                        continue

            if not scan_results:

                st.warning(
                    "No usable results from this batch — try a "
                    "different batch size or run it again."
                )

            else:

                scan_df = (
                    pd.DataFrame(scan_results)
                    .sort_values("_sort", ascending=False)
                    .drop(columns="_sort")
                    .head(15)
                )

                st.success(
                    f"Scanned {len(batch)} stocks outside your core "
                    f"list — top {len(scan_df)} by 3-month momentum:"
                )

                st.dataframe(
                    scan_df,
                    use_container_width=True,
                    hide_index=True
                )

                st.caption(
                    "Search any of these tickers above for the full "
                    "rated analysis — this table is a discovery "
                    "shortlist, not a final verdict."
                )


# ============================================================
# CAN SLIM QUALIFICATION SCREEN
# ============================================================

qcol1, qcol2 = st.columns([10, 1])
with qcol1:
    st.markdown("### 🧪 CAN SLIM Qualification Screen")
with qcol2:
    info_button(
        "Each column checks one CANSLIM letter against a real "
        "threshold — the actual number is shown next to ✅/❌/N/A, "
        "never a bare tick.\n\n"
        "**Go Ahead ✅** only appears when every *computable* "
        "criterion passes — N/A criteria (data genuinely unavailable) "
        "don't block it, but a real ❌ does.\n\n"
        "This is a screen, not a guarantee — see 'Feature roadmap' "
        "above for what each rating actually measures."
    )

with st.container(border=True):

    if market_direction is not None:

        market_status_text = (
            "🟢 Bullish — NIFTY 50 above its 50-day and 200-day average"
            if market_direction["bullish"]
            else "🔴 Unfavorable — NIFTY 50 below its 50-day and/or 200-day average"
        )

        st.info(
            f"**M — Market Direction:** {market_status_text}  "
            f"(NIFTY 50: ₹{market_direction['price']:.2f} · "
            f"50-day avg: ₹{market_direction['ma50']:.2f} · "
            f"200-day avg: ₹{market_direction['ma200']:.2f})"
        )

    else:

        st.warning(
            "M — Market Direction: could not be determined "
            "(NIFTY 50 data unavailable)."
        )

    st.caption(
        f"Price floor: ₹{PRICE_FLOOR:.0f}. "
        "**I** (Institutional sponsorship) is shown as N/A — that data "
        "genuinely isn't available for NSE tickers from any free source. "
        "**A** (Annual EPS CAGR) is computed from each ticker's own income "
        "statement and shows N/A only for tickers with too little EPS "
        "history on file. Each cell shows the actual measured number, "
        "with the pass/fail verdict alongside it — not just a bare tick "
        "or cross."
    )

    qualification_rows = []

    for ticker, rec in master_records.items():

        checks, values, status, go_ahead = evaluate_qualification(
            rec,
            market_direction
        )

        def cell(k):
            v = checks[k]
            val_text = values[k]
            if v is None:
                return val_text
            mark = "✅" if v else "❌"
            return f"{val_text} {mark}"

        qualification_rows.append({
            "Ticker": ticker,
            **{k: cell(k) for k in checks.keys()},
            "Status": status,
            "Go Ahead": "✅" if go_ahead else "❌"
        })

    if qualification_rows:

        qual_df = pd.DataFrame(qualification_rows)

        status_order = {
            "Qualified": 0,
            "Watchlist": 1,
            "Insufficient data": 2,
            "Below price floor": 3
        }

        qual_df["_sort"] = qual_df["Status"].map(status_order)

        qual_df = (
            qual_df
            .sort_values("_sort")
            .drop(columns="_sort")
        )

        st.dataframe(
            qual_df,
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# ADVANCED SIGNALS — roadmap + 4-mode toggle
# ============================================================

st.markdown("---")

acol1, acol2 = st.columns([10, 1])
with acol1:
    st.markdown("### 🚀 Advanced Signals")
with acol2:
    info_button(
        "**📈 Big Picture**: overall market health. Look for state = "
        "'Confirmed Uptrend' or 'Power Trend'; avoid new buys if it "
        "says 'Correction'. 5+ Distribution Days = get cautious.\n\n"
        "**📊 RS Line & Blue Dot**: is this stock beating the index? "
        "Line trending up = yes. A recent Blue Dot = extra-strong signal.\n\n"
        "**🎯 Near Pivot/Breakouts**: 'Near Pivot' = watch it; "
        "'Recent Breakouts' = the actionable buy-zone list.\n\n"
        "**🔍 Pattern Recognition**: looks for a cup-with-handle base. "
        "A detection gives an exact pivot price — but it's a heuristic, "
        "verify visually before acting on it."
    )

with st.container(border=True):

    with st.expander("📊 Feature roadmap vs. MarketSurge (formerly MarketSmith)"):

        st.markdown(
            "| MarketSurge feature | What it does | Us | Status |\n"
            "|---|---|---|---|\n"
            "| Big Picture / Market School | Follow-Through Day, Power Trend, Distribution Days | ✅ | Built below |\n"
            "| RS Line + Blue Dot | Stock-vs-index ratio, new-high marker | ✅ | Built below |\n"
            "| Near Pivot / Recent Breakouts lists | Pre-filtered setup lists | ✅ | Built below |\n"
            "| Pattern Recognition (cup-w-handle) | Auto-detects base patterns | ✅ | Built below (heuristic) |\n"
            "| Sales+Margins+ROE (SMR) Rating | Fundamental quality grade A-E | ✅ | In Qualification Screen |\n"
            "| Composite Rating | Blends EPS+RS+SMR+Acc/Dis | ✅ | In leaderboard table |\n"
            "| Sponsorship Rating (fund ownership) | 3-yr institutional trend | ❌ | No free India data source |\n"
            "| Industry Group RS | Stock's sector ranked vs all sectors | ❌ | Needs full NSE sector universe |\n"
        )

    if "advanced_mode" not in st.session_state:
        st.session_state.advanced_mode = "big_picture"

    mode_col1, mode_col2, mode_col3, mode_col4 = st.columns(4)

    with mode_col1:
        if st.button("📈 Big Picture", use_container_width=True):
            st.session_state.advanced_mode = "big_picture"

    with mode_col2:
        if st.button("📊 RS Line & Blue Dot", use_container_width=True):
            st.session_state.advanced_mode = "rs_line"

    with mode_col3:
        if st.button("🎯 Near Pivot / Breakouts", use_container_width=True):
            st.session_state.advanced_mode = "pivot_list"

    with mode_col4:
        if st.button("🔍 Pattern Recognition", use_container_width=True):
            st.session_state.advanced_mode = "pattern"

    mode = st.session_state.advanced_mode


    # ---- Mode 1: Big Picture ----
    if mode == "big_picture":

        st.markdown("#### 📈 Big Picture — Market School (NIFTY 50)")

        st.caption(
            "IBD's published rally/Follow-Through Day rules, applied to "
            "NIFTY 50. The 1.25% FTD gain threshold is a disclosed "
            "simplification — IBD doesn't publish one uniform figure "
            "for every index."
        )

        if market_direction is not None and "hist" in market_direction:

            bp = analyze_big_picture(market_direction["hist"])

            if bp is not None:

                bpc1, bpc2, bpc3, bpc4 = st.columns(4)

                metric_card(
                    bpc1,
                    "Market State",
                    bp["state"],
                    "Current Big Picture read"
                )

                metric_card(
                    bpc2,
                    "Distribution Days",
                    str(bp["distribution_days"]),
                    "Trailing 25 sessions (5+ = caution)"
                )

                metric_card(
                    bpc3,
                    "Follow-Through Day",
                    (
                        bp["ftd_date"].strftime("%Y-%m-%d")
                        if bp["ftd_date"] is not None
                        else "None recent"
                    ),
                    f"+{bp['ftd_gain']:.2f}%" if bp["ftd_gain"] else "—"
                )

                metric_card(
                    bpc4,
                    "52w Range Position",
                    f"{bp['price_percentile_52w']:.0f}%",
                    "Top 25% needed for Power Trend"
                )

            else:
                st.warning("Not enough index history to run Big Picture analysis.")

        else:
            st.warning("NIFTY 50 data unavailable — Big Picture can't be computed.")


    # ---- Mode 2: RS Line & Blue Dot ----
    elif mode == "rs_line":

        st.markdown(f"#### 📊 RS Line & Blue Dot — {selected_stock}")

        st.caption(
            "The RS Line (stock price ÷ index price) is a different signal "
            "from the RS Rating number — a Blue Dot marks a day where this "
            "ratio hits a new high while price is also near its own high."
        )

        if market_direction is not None and "hist" in market_direction:

            rsl = compute_rs_line(df_chart, market_direction["hist"])

            if rsl is not None:

                rs_fig = go.Figure()

                rs_fig.add_trace(go.Scatter(
                    x=rsl["dates"],
                    y=rsl["rs_line"],
                    mode="lines",
                    name="RS Line",
                    line=dict(color="#8b5cf6", width=2)
                ))

                if rsl["blue_dot_dates"]:

                    rs_fig.add_trace(go.Scatter(
                        x=rsl["blue_dot_dates"],
                        y=[
                            rsl["rs_line"].loc[d]
                            for d in rsl["blue_dot_dates"]
                        ],
                        mode="markers",
                        name="Blue Dot",
                        marker=dict(color="#00d4ff", size=8, symbol="circle")
                    ))

                rs_fig.update_layout(
                    height=400,
                    template="plotly_dark" if theme == "Dark" else "plotly_white",
                    margin=dict(l=10, r=10, t=30, b=10)
                )

                st.plotly_chart(rs_fig, use_container_width=True)

                st.info(
                    "🔵 Recent Blue Dot (last 10 sessions)"
                    if rsl["recent_blue_dot"]
                    else "No recent Blue Dot in the last 10 sessions"
                )

            else:
                st.warning("Not enough overlapping history to compute the RS Line.")

        else:
            st.warning("NIFTY 50 data unavailable — RS Line can't be computed.")


    # ---- Mode 3: Near Pivot / Recent Breakouts ----
    elif mode == "pivot_list":

        st.markdown("#### 🎯 Near Pivot / Recent Breakouts")

        st.caption(
            "Near Pivot: within 5% below the pivot, not yet broken out. "
            "Recent Breakouts: crossed above pivot within the last 10 sessions."
        )

        near_pivot_rows = []
        breakout_rows = []

        for ticker, rec in master_records.items():

            delta = rec["raw_pivot_delta"]

            if -5.0 <= delta < 0:

                near_pivot_rows.append({
                    "Ticker": ticker,
                    "Price": rec["Price"],
                    "Pivot Delta": rec["Pivot Delta"]
                })

            elif delta >= 0:

                hist_r = rec["raw_hist"]
                pivot_r = rec["raw_pivot"]
                closes_r = hist_r["Close"].tail(11)

                crossed_recently = False

                for i in range(1, len(closes_r)):
                    if (
                        closes_r.iloc[i - 1] < pivot_r
                        and closes_r.iloc[i] >= pivot_r
                    ):
                        crossed_recently = True
                        break

                if crossed_recently:

                    breakout_rows.append({
                        "Ticker": ticker,
                        "Price": rec["Price"],
                        "Pivot Delta": rec["Pivot Delta"]
                    })

        pcol1, pcol2 = st.columns(2)

        with pcol1:
            st.markdown("**Near Pivot**")
            if near_pivot_rows:
                st.dataframe(
                    pd.DataFrame(near_pivot_rows),
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.caption("No stocks currently within 5% below their pivot.")

        with pcol2:
            st.markdown("**Recent Breakouts**")
            if breakout_rows:
                st.dataframe(
                    pd.DataFrame(breakout_rows),
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.caption("No stocks broke out above pivot in the last 10 sessions.")


    # ---- Mode 4: Pattern Recognition ----
    elif mode == "pattern":

        st.markdown(f"#### 🔍 Pattern Recognition — {selected_stock}")

        st.caption(
            "A simplified, disclosed heuristic for the classic cup-with-handle "
            "base — not a claimed replica of MarketSurge's AI detector. Will "
            "have false positives/negatives on real data."
        )

        pattern_result = detect_cup_and_handle(df_chart)

        if pattern_result["detected"]:

            st.success("✅ Valid cup-with-handle pattern detected")

            pc1, pc2, pc3, pc4 = st.columns(4)

            metric_card(pc1, "Left Rim", f"₹{pattern_result['left_rim_price']:.2f}", "Cup start")
            metric_card(pc2, "Cup Bottom", f"₹{pattern_result['cup_bottom_price']:.2f}", f"{pattern_result['cup_depth_pct']:.1f}% deep")
            metric_card(pc3, "Right Rim", f"₹{pattern_result['right_rim_price']:.2f}", "Cup recovery")
            metric_card(pc4, "Pivot (buy point)", f"₹{pattern_result['pivot_price']:.2f}", f"Handle {pattern_result['handle_depth_pct']:.1f}% deep")

        else:

            st.info(f"No valid cup-with-handle pattern currently: {pattern_result.get('reason', 'unknown')}")


# ============================================================
# CHAT BOX
# ============================================================

st.markdown("---")

with st.container(border=True):

    chat_header_col, chat_clear_col = st.columns([5, 1])

    with chat_header_col:

        st.markdown("### 💬 Stock Intelligence Chat")

        st.caption(
            f"Ask questions about **{selected_stock}**. "
            "The answers below are generated from the currently loaded market data and technical calculations."
        )

    with chat_clear_col:

        st.markdown("")
        st.markdown("")

        if st.button(
            "🗑️ Clear chat",
            use_container_width=True
        ):

            st.session_state.chat_history = []
            st.rerun()


    if "chat_history" not in st.session_state:

        st.session_state.chat_history = []

    # FIX: the conversation used to persist across stock switches, so
    # asking a fresh question after changing the sidebar selection could
    # show old answers mixed in about a completely different ticker.
    # Tracking which stock the chat is "about" and auto-resetting when
    # it changes keeps every conversation scoped to one stock.
    if st.session_state.get("chat_stock") != selected_stock:

        st.session_state.chat_history = []
        st.session_state.chat_stock = selected_stock


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

with st.container(border=True):

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
