"""
app.py  ─  v5.0  專業交易平台（UI 層）
─────────────────────────────────────────────────────────────────
重構要點（v5.0）：
  ✓ 移除左側 Sidebar → 頂部橫向控制列（st.columns）
  ✓ TradingView Lightweight Charts 取代 Plotly K 線
  ✓ 股票代號自動對應中文名稱（components.stock_names）
  ✓ 推薦清單新增「推薦理由」技術指標欄
  ✓ 數字字體改用 JetBrains Mono（Google Fonts CDN）
  ✓ 台股紅漲綠跌 Morandi 低飽和配色 + 米白 #FDFDFD 底色
  ✓ 思考鏈（ThinkingLog）小標籤 + 打字機滾動效果
  ✓ @st.fragment 隔離 K 線 → 不中斷推演進程

架構：
  app.py         → Streamlit UI（本檔）
  engine.py      → 業務邏輯（Pipeline / 決策引擎）
  auth.py        → 用戶認證
  components/
    stock_names.py       → 台股中文名稱對照
    tradingview_chart.py → TradingView LW Charts 嵌入
"""

# ── 標準函式庫 ────────────────────────────────────────────────
import datetime

# ── 第三方套件 ────────────────────────────────────────────────
import yfinance as yf
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── 本地模組 ──────────────────────────────────────────────────
import auth
import engine
from engine import (
    ThinkingLogger,
    generate_asset_driven_pool,
    compute_portfolio_summary,
    build_holdings_rows,
    generate_daily_guide,
    run_full_pipeline,
    TW_STOCK_UNIVERSE,
)
from data_fetcher import DataFetcher
from components.stock_names import get_name, display as stock_display, display_list
from components.tradingview_chart import tradingview_chart


# ═══════════════════════════════════════════════════════════════
#  頁面設定（第一個 st 指令）
# ═══════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="台股量化交易平台",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",   # 側邊欄預設收起
)

st.markdown("""
<style>
/* ══════════════════════════════════════════════════════════════
   Professional Trading Platform ─ Design System v5.0
   Palette:
     BG        #FDFDFD  |  Surface  #FFFFFF
     Accent    #607D8B  |  Sage     #7A9E87
     Amber     #B8966A  |  Text     #383838
     Muted     #7A7A7A  |  Faint    #AEAEAE
     Border    #E8E3DB  |  Rise     #B85450  Fall #5A8A7A
   Fonts:
     UI        Noto Sans TC / Inter
     Heading   Noto Serif TC
     Numbers   JetBrains Mono (monospace)
══════════════════════════════════════════════════════════════ */

/* ─ Google Fonts ─────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Noto+Serif+TC:wght@300;400;600&family=Noto+Sans+TC:wght@300;400;500&family=Inter:wght@300;400;500&family=JetBrains+Mono:wght@400;500&display=swap');

/* ─ Root ─────────────────────────────────────────────────── */
:root {
    --bg      : #FDFDFD;
    --surface : #FFFFFF;
    --accent  : #607D8B;
    --sage    : #7A9E87;
    --amber   : #B8966A;
    --text    : #383838;
    --muted   : #7A7A7A;
    --faint   : #AEAEAE;
    --border  : #E8E3DB;
    --shadow  : rgba(56,40,20,.06);
    --rise    : #B85450;
    --fall    : #5A8A7A;
    --mono    : 'JetBrains Mono', 'Fira Code', 'Courier New', monospace;
}

/* ─ Global ───────────────────────────────────────────────── */
.stApp {
    background-color: var(--bg) !important;
    font-family: 'Noto Sans TC', 'Inter', sans-serif;
    color: var(--text);
}

/* ─ Hide default sidebar toggle when collapsed ───────────── */
[data-testid="collapsedControl"] { display: none !important; }

/* ─ Typography ───────────────────────────────────────────── */
h1, h2, h3 {
    font-family: 'Noto Serif TC', Georgia, serif !important;
    color: var(--text) !important;
    font-weight: 600 !important;
    letter-spacing: -0.01em;
}
h4, h5, h6, p {
    font-family: 'Noto Sans TC', 'Inter', sans-serif !important;
    color: var(--text) !important;
}

/* ─ Monospace numbers (apply via .mono class in HTML) ────── */
.mono {
    font-family: var(--mono) !important;
    font-feature-settings: 'tnum' 1;
    letter-spacing: -0.01em;
}

/* ─ Buttons ──────────────────────────────────────────────── */
[data-testid="stButton"] > button {
    background: transparent !important;
    border: 1.5px solid var(--border) !important;
    border-radius: 24px !important;
    color: var(--muted) !important;
    font-family: 'Noto Sans TC', 'Inter', sans-serif !important;
    font-size: .83rem !important;
    letter-spacing: .04em;
    padding: 7px 18px !important;
    transition: all .2s ease !important;
    box-shadow: none !important;
}
[data-testid="stButton"] > button:hover {
    border-color: var(--accent) !important;
    color: var(--accent) !important;
    background: rgba(96,125,139,.06) !important;
}
[data-testid="stButton"] > button[kind="primary"] {
    background: var(--accent) !important;
    border-color: var(--accent) !important;
    color: #fff !important;
    font-weight: 500 !important;
}
[data-testid="stButton"] > button[kind="primary"]:hover {
    background: #506879 !important;
    border-color: #506879 !important;
}

/* ─ Metric Cards ─────────────────────────────────────────── */
div[data-testid="metric-container"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 14px !important;
    padding: 18px 20px !important;
    box-shadow: 0 2px 14px var(--shadow) !important;
    transition: box-shadow .2s ease !important;
}
div[data-testid="metric-container"]:hover {
    box-shadow: 0 6px 24px rgba(56,40,20,.10) !important;
}
[data-testid="metric-label"] > div {
    color: var(--muted) !important;
    font-size: .72rem !important;
    letter-spacing: .07em;
    text-transform: uppercase;
    font-family: 'Noto Sans TC', 'Inter', sans-serif !important;
}
[data-testid="metric-value"] > div {
    color: var(--text) !important;
    font-family: var(--mono) !important;
    font-size: 1.45rem !important;
    font-weight: 500 !important;
}
[data-testid="metric-delta"] {
    font-size: .75rem !important;
    font-family: var(--mono) !important;
}

/* ─ Tabs ─────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid var(--border) !important;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    border-radius: 8px 8px 0 0 !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    color: var(--muted) !important;
    font-family: 'Noto Sans TC', 'Inter', sans-serif !important;
    font-size: .84rem !important;
    padding: 8px 16px !important;
    transition: all .18s ease !important;
}
.stTabs [aria-selected="true"] {
    color: var(--accent) !important;
    border-bottom: 2px solid var(--accent) !important;
    background: rgba(96,125,139,.05) !important;
}

/* ─ Expanders ────────────────────────────────────────────── */
[data-testid="stExpander"] {
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    background: var(--surface) !important;
    box-shadow: 0 1px 8px var(--shadow) !important;
    margin: 6px 0 !important;
    overflow: hidden;
}
[data-testid="stExpander"] summary {
    color: var(--text) !important;
    font-family: 'Noto Sans TC', 'Inter', sans-serif !important;
    font-size: .85rem !important;
    padding: 10px 16px !important;
}

/* ─ Inputs ───────────────────────────────────────────────── */
.stTextInput > div > div > input,
.stNumberInput > div > div > input,
.stTextArea > div > div > textarea {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
    font-family: 'Noto Sans TC', 'Inter', sans-serif !important;
    font-size: .87rem !important;
}
.stTextInput > div > div > input:focus,
.stNumberInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(96,125,139,.13) !important;
}

/* ─ NumberInput mono font ────────────────────────────────── */
.stNumberInput > div > div > input {
    font-family: var(--mono) !important;
}

/* ─ Select / Radio ───────────────────────────────────────── */
[data-testid="stSelectbox"] > div > div {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
}
.stRadio label { color: var(--text) !important; font-size: .84rem !important; }

/* ─ Toggle ───────────────────────────────────────────────── */
.stToggle label { color: var(--text) !important; }

/* ─ Slider ───────────────────────────────────────────────── */
.stSlider [data-testid="stTickBar"] { color: var(--faint) !important; }

/* ─ Alerts ───────────────────────────────────────────────── */
[data-testid="stAlert"] {
    border-radius: 10px !important;
    border: 1px solid var(--border) !important;
}

/* ─ Divider ──────────────────────────────────────────────── */
hr { border: none !important; border-top: 1px solid var(--border) !important; margin: 18px 0 !important; }

/* ─ DataFrames ───────────────────────────────────────────── */
[data-testid="stDataFrame"] {
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    overflow: hidden !important;
}

/* ─ Progress Bar ─────────────────────────────────────────── */
[data-testid="stProgress"] > div { background: var(--border) !important; border-radius: 8px !important; }
[data-testid="stProgress"] > div > div { background: var(--accent) !important; border-radius: 8px !important; }

/* ═══════════════════════════════════════════════════════════
   Custom Components
═══════════════════════════════════════════════════════════ */

/* ─ KPI Cards ────────────────────────────────────────────── */
.kpi-wrap {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 20px 22px 14px;
    text-align: center;
    position: relative;
    overflow: hidden;
    box-shadow: 0 2px 16px var(--shadow);
    transition: box-shadow .22s ease, transform .22s ease;
}
.kpi-wrap:hover {
    box-shadow: 0 8px 28px rgba(56,40,20,.10);
    transform: translateY(-2px);
}
.kpi-wrap::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, var(--accent) 0%, var(--sage) 100%);
    border-radius: 16px 16px 0 0;
}
.kpi-lbl {
    font-size: .68rem;
    font-family: 'Noto Sans TC', 'Inter', sans-serif;
    color: var(--muted);
    letter-spacing: .12em;
    text-transform: uppercase;
    margin-bottom: 10px;
}
.kpi-val {
    font-size: 1.50rem;
    font-family: var(--mono);
    font-weight: 500;
    color: var(--text);
    line-height: 1.2;
    letter-spacing: -0.02em;
}
.kpi-dlt {
    font-size: .73rem;
    font-family: var(--mono);
    margin-top: 8px;
    opacity: .85;
}

/* ─ Section Headers ──────────────────────────────────────── */
.section-bar {
    font-size: .66rem;
    font-family: 'Noto Sans TC', 'Inter', sans-serif;
    font-weight: 600;
    color: var(--faint);
    letter-spacing: .22em;
    text-transform: uppercase;
    display: flex;
    align-items: center;
    gap: 12px;
    margin: 26px 0 14px;
}
.section-bar::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
}

/* ─ Action Cards ─────────────────────────────────────────── */
.action-card {
    border-radius: 0 12px 12px 0;
    padding: 14px 20px;
    margin: 8px 0;
    border-left-width: 4px;
    border-left-style: solid;
}
.action-title {
    font-size: 1.02rem;
    font-weight: 700;
    color: #383838;
    margin-bottom: 6px;
}
.action-body {
    color: #5A5A5A;
    font-size: .88rem;
    line-height: 1.65;
}
.action-reason {
    margin-top: 8px;
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
}

/* ─ Thinking Log Tags ────────────────────────────────────── */
.think-wrap {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 14px 18px;
    min-height: 56px;
    max-height: 180px;
    overflow-y: auto;
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    align-content: flex-start;
}
.think-tag {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    font-size: .74rem;
    font-family: var(--mono);
    padding: 3px 10px;
    border-radius: 20px;
    border: 1px solid;
    animation: tagIn .25s ease both;
    white-space: nowrap;
}
@keyframes tagIn {
    from { opacity: 0; transform: scale(.85) translateY(4px); }
    to   { opacity: 1; transform: scale(1) translateY(0); }
}
.think-tag.data   { background: rgba(96,125,139,.08);  border-color: rgba(96,125,139,.30);  color: #506879; }
.think-tag.ga     { background: rgba(184,150,106,.08); border-color: rgba(184,150,106,.30); color: #8A6A30; }
.think-tag.mc     { background: rgba(122,158,135,.08); border-color: rgba(122,158,135,.30); color: #3A7A5A; }
.think-tag.hold   { background: rgba(122,110,158,.08); border-color: rgba(122,110,158,.30); color: #5A4A8E; }
.think-tag.info   { background: rgba(158,158,158,.08); border-color: rgba(158,158,158,.30); color: #5A5A5A; }
.think-cursor {
    display: inline-block;
    width: 6px; height: 13px;
    background: var(--accent);
    border-radius: 2px;
    animation: blink .8s step-end infinite;
    vertical-align: middle;
    margin-left: 2px;
}
@keyframes blink { 50% { opacity: 0; } }

/* ─ Indicator Pill (recommendation reason) ───────────────── */
.ind-pill {
    display: inline-flex;
    align-items: center;
    gap: 3px;
    font-size: .70rem;
    font-family: 'Noto Sans TC', 'Inter', sans-serif;
    font-weight: 500;
    padding: 2px 9px;
    border-radius: 12px;
    background: rgba(96,125,139,.10);
    color: var(--accent);
    border: 1px solid rgba(96,125,139,.20);
}
.ind-pill.bull { background: rgba(184,84,80,.08);  color: var(--rise); border-color: rgba(184,84,80,.20); }
.ind-pill.bear { background: rgba(90,138,122,.08); color: var(--fall); border-color: rgba(90,138,122,.20); }

/* ─ Top Control Bar ──────────────────────────────────────── */
.ctrl-bar {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 16px 20px;
    margin-bottom: 16px;
    box-shadow: 0 2px 12px var(--shadow);
}

/* ─ Mobile responsive columns ────────────────────────────── */
@media (max-width: 768px) {
    .kpi-val { font-size: 1.20rem; }
    .kpi-wrap { padding: 14px 14px 10px; }
    [data-testid="column"] { min-width: 100% !important; }
}

/* ─ Caption / Code ───────────────────────────────────────── */
[data-testid="stCaptionContainer"] p, .stCaption {
    color: var(--muted) !important;
    font-size: .76rem !important;
}
code {
    background: rgba(96,125,139,.10) !important;
    color: var(--accent) !important;
    border-radius: 6px !important;
    padding: 2px 7px !important;
    font-size: .80rem !important;
    font-family: var(--mono) !important;
}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
#  快取資料函數（5 分鐘 TTL）
# ═══════════════════════════════════════════════════════════════

@st.cache_data(ttl=300, show_spinner=False)
def fetch_realtime_quotes(codes: tuple[str, ...]) -> dict:
    """批次取得即時報價（5 分鐘快取）。"""
    fetcher = DataFetcher()
    result: dict[str, dict] = {}
    for code in codes:
        try:
            df = fetcher.fetch(code, period="5d", interval="1d")
            if df is None or df.empty:
                continue
            latest = df.iloc[-1]
            prev   = df.iloc[-2] if len(df) >= 2 else latest
            price  = float(latest["Close"])
            prev_c = float(prev["Close"])
            chg    = price - prev_c
            chg_p  = chg / prev_c if prev_c else 0
            result[code] = {
                "price":      price,
                "change":     chg,
                "change_pct": chg_p,
                "volume":     float(latest.get("Volume", 0)),
                "trade_date": str(df.index[-1].date()),
            }
        except Exception:
            pass
    return result


@st.cache_data(ttl=300, show_spinner=False)
def fetch_kline_data(code: str) -> pd.DataFrame | None:
    """取得 K 線歷史資料（5 分鐘快取）。"""
    try:
        fetcher = DataFetcher()
        return fetcher.fetch(code, period="6mo", interval="1d")
    except Exception:
        return None


@st.cache_data(ttl=600, show_spinner=False)
def fetch_pool_history(codes: tuple[str, ...]) -> dict:
    """批次下載股池歷史資料（10 分鐘快取）。"""
    fetcher = DataFetcher()
    result = {}
    for code in codes:
        try:
            df = fetcher.fetch(code, period="2y", interval="1d")
            if df is not None and not df.empty:
                result[code] = df
        except Exception:
            pass
    return result


# ═══════════════════════════════════════════════════════════════
#  輔助函數
# ═══════════════════════════════════════════════════════════════

def _tw_color_hex(val: float) -> str:
    if val > 0: return "#B85450"
    if val < 0: return "#5A8A7A"
    return "#9E9E9E"


def _arrow(val: float) -> str:
    if val > 0: return "▲"
    if val < 0: return "▼"
    return "─"


def _mono(text: str) -> str:
    """Wrap text in monospace span."""
    return f'<span class="mono">{text}</span>'


# ═══════════════════════════════════════════════════════════════
#  認證頁
# ═══════════════════════════════════════════════════════════════

def render_auth_page() -> bool:
    if st.session_state.get("_username"):
        return True

    st.markdown("""
    <div style="text-align:center;padding:40px 0 12px;">
      <div style="font-size:3rem;">📈</div>
      <h2 style="color:#383838;margin:10px 0 4px;font-family:'Noto Serif TC',Georgia,serif;">
        台股量化交易平台</h2>
      <p style="color:#7A7A7A;font-size:.85rem;">Professional Quantitative Trading System v5.0</p>
    </div>
    """, unsafe_allow_html=True)

    _, col, _ = st.columns([1, 2, 1])
    with col:
        tab_login, tab_reg = st.tabs(["🔑 登入", "📝 註冊"])

        with tab_login:
            u = st.text_input("帳號", key="login_u", placeholder="輸入帳號")
            p = st.text_input("密碼", type="password", key="login_p", placeholder="輸入密碼")
            if st.button("登入", type="primary", use_container_width=True):
                ok, msg = auth.verify_user(u, p)
                if ok:
                    st.session_state["_username"] = u
                    saved = auth.load_user_settings(u)
                    if saved:
                        st.session_state["_user_settings"] = saved
                    st.rerun()
                else:
                    st.error(msg)

        with tab_reg:
            ru  = st.text_input("新帳號（≥2字）", key="reg_u")
            rp  = st.text_input("密碼（≥6字）",   key="reg_p",  type="password")
            rp2 = st.text_input("確認密碼",        key="reg_p2", type="password")
            if st.button("建立帳號", type="primary", use_container_width=True):
                if rp != rp2:
                    st.error("兩次密碼不一致。")
                else:
                    ok, msg = auth.register_user(ru, rp)
                    st.success(msg) if ok else st.error(msg)

    return False


# ═══════════════════════════════════════════════════════════════
#  頂部控制列（取代 Sidebar）
# ═══════════════════════════════════════════════════════════════

def render_top_controls(username: str) -> dict:
    """
    頂部橫向控制列。
    手機端（<768px）欄位自動垂直堆疊（CSS media query 輔助）。
    """
    saved = st.session_state.get("_user_settings", {})

    # ── 標題列 ────────────────────────────────────────────────
    col_title, col_mode, col_actions = st.columns([5, 2, 2])
    with col_title:
        st.markdown(
            "<h1 style='margin:0;font-size:1.50rem;color:#383838;"
            "font-family:Noto Serif TC,Georgia,serif;'>"
            "📈 台股量化交易平台</h1>"
            "<p style='color:#9E9E9E;font-size:.78rem;margin:3px 0 0;"
            "font-family:Noto Sans TC,Inter,sans-serif;'>"
            f"GA × Monte Carlo × Real-time  ·  v5.0  ·  {username}</p>",
            unsafe_allow_html=True,
        )

    with col_mode:
        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
        short_term_mode = st.toggle(
            "⚡ 短線優先（KDJ + 量能）",
            value=saved.get("short_term_mode", True),
            key="short_term_toggle",
            help="開：短均線 + KDJ + 量能爆發（波段/當沖）\n關：長均線 + 標準技術信號（季線持有）",
        )

    with col_actions:
        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
        ac1, ac2, ac3 = st.columns(3)
        with ac1:
            if st.button("🔄", help="刷新報價"):
                fetch_realtime_quotes.clear()
                fetch_kline_data.clear()
                st.rerun()
        with ac2:
            if st.button("🗑️", help="清除結果"):
                for k in ("_results", "_config", "_has_results"):
                    st.session_state.pop(k, None)
                st.rerun()
        with ac3:
            if st.button("登出", help="登出帳號"):
                for k in ("_username", "_user_settings", "_results", "_config", "_has_results"):
                    st.session_state.pop(k, None)
                st.rerun()

    st.markdown("---")

    # ── 第一行：核心輸入 ──────────────────────────────────────
    st.markdown(
        "<div class='section-bar'>⚙️ 策略設定</div>",
        unsafe_allow_html=True,
    )

    col_cash, col_hold, col_pool = st.columns([1, 2, 2])

    with col_cash:
        st.markdown("##### 💰 可用現金")
        available_cash = st.number_input(
            "cash_input",
            min_value=10_000, max_value=100_000_000,
            value=int(saved.get("available_cash", 500_000)),
            step=10_000, format="%d",
            label_visibility="collapsed",
        )
        st.caption(f"NT$ {available_cash:,}")

    with col_hold:
        st.markdown("##### 📋 現有持股")
        st.caption("填純數字代號，可新增/刪除列")
        saved_holdings = saved.get("current_holdings", {})
        default_rows = (
            [{"代號": c, "買入均價": v["cost"], "持有股數": v["shares"]}
             for c, v in saved_holdings.items()]
            if saved_holdings else
            [{"代號": "2330", "買入均價": 850.0, "持有股數": 1000},
             {"代號": "2317", "買入均價":  95.0, "持有股數": 2000}]
        )
        h_df = st.data_editor(
            pd.DataFrame(default_rows), num_rows="dynamic",
            use_container_width=True,
            column_config={
                "代號":    st.column_config.TextColumn("代號", width="small"),
                "買入均價": st.column_config.NumberColumn("均價", min_value=0.01, format="%.2f"),
                "持有股數": st.column_config.NumberColumn("股數", min_value=0, step=1000, format="%d"),
            },
            key="holdings_editor",
        )
        current_holdings: dict[str, dict] = {}
        for _, row in h_df.dropna(subset=["代號"]).iterrows():
            code = str(row["代號"]).strip()
            if code and float(row.get("買入均價", 0) or 0) > 0:
                current_holdings[code] = {
                    "cost":   float(row["買入均價"]),
                    "shares": int(row.get("持有股數") or 0),
                }

    with col_pool:
        st.markdown("##### 🎯 候選股池")
        auto_pool = generate_asset_driven_pool(available_cash, current_holdings)
        pool_mode = st.radio(
            "pool_mode_radio",
            ["🤖 自動推薦", "✏️ 手動輸入"],
            index=0, horizontal=True, label_visibility="collapsed",
        )
        if pool_mode == "🤖 自動推薦":
            target_pool = auto_pool
            # 顯示帶中文名稱的股池預覽
            preview = "、".join(stock_display(c) for c in target_pool[:5])
            extra   = f"…等 {len(target_pool)} 支" if len(target_pool) > 5 else f"共 {len(target_pool)} 支"
            st.caption(f"`{preview}` {extra}")
            if st.button("🔄 重新生成股池"):
                st.rerun()
        else:
            saved_pool_str = ",".join(saved.get("target_pool", auto_pool))
            pool_text = st.text_area(
                "pool_text", value=saved_pool_str,
                height=72, label_visibility="collapsed",
                placeholder="2330,2317,2454,...",
            )
            target_pool = [c.strip() for c in pool_text.replace("\n", ",").split(",") if c.strip()]
            st.caption(f"共 {len(target_pool)} 支")

    # ── 第二行：進階參數（可折疊） ────────────────────────────
    col_ga, col_mc, col_exec = st.columns([2, 2, 1])

    with col_ga:
        with st.expander("🧬 遺傳演算法參數"):
            pop = st.slider("種群大小", 20, 100, int(saved.get("ga_config", {}).get("population_size", 50)), step=10)
            gen = st.slider("演化代數", 20, 100, int(saved.get("ga_config", {}).get("generations", 50)), step=10)
            cr  = st.slider("交叉率",   0.50, 1.00, float(saved.get("ga_config", {}).get("crossover_rate", 0.80)), step=0.05)
            mr  = st.slider("變異率",   0.05, 0.30, float(saved.get("ga_config", {}).get("mutation_rate", 0.15)), step=0.05)
    # fall-through defaults if expander not open yet
    try: pop
    except NameError:
        pop = int(saved.get("ga_config", {}).get("population_size", 50))
        gen = int(saved.get("ga_config", {}).get("generations", 50))
        cr  = float(saved.get("ga_config", {}).get("crossover_rate", 0.80))
        mr  = float(saved.get("ga_config", {}).get("mutation_rate", 0.15))

    with col_mc:
        with st.expander("🎲 蒙地卡羅參數"):
            n_s = st.select_slider("路徑數", options=[500, 1000, 2000, 5000],
                                   value=saved.get("mc_config", {}).get("n_simulations", 1000))
            n_d = st.select_slider("模擬天數", options=[10, 21, 42, 63, 126],
                                   value=saved.get("mc_config", {}).get("n_days", 21),
                                   format_func=lambda x: f"{x}日")
            if short_term_mode and n_d > 42:
                st.caption("⚡ 短線模式建議 10~21 日")
    try: n_s
    except NameError:
        n_s = saved.get("mc_config", {}).get("n_simulations", 1000)
        n_d = saved.get("mc_config", {}).get("n_days", 21)

    with col_exec:
        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
        top_n = st.slider("📌 選股數", 1, 6, int(saved.get("top_n", 3)))
        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
        save_clicked = st.button("💾 儲存設定", use_container_width=True)
        run_clicked  = st.button("🚀 執行推演", type="primary", use_container_width=True)

    if save_clicked:
        auth.save_user_settings(username, {
            "available_cash":   available_cash,
            "current_holdings": current_holdings,
            "target_pool":      target_pool,
            "ga_config":        {"population_size": pop, "generations": gen,
                                 "crossover_rate": cr, "mutation_rate": mr},
            "mc_config":        {"n_simulations": n_s, "n_days": n_d},
            "top_n":            top_n,
            "short_term_mode":  short_term_mode,
        })
        st.success("✅ 設定已儲存！")

    return {
        "available_cash":   available_cash,
        "current_holdings": current_holdings,
        "target_pool":      target_pool,
        "ga_config":        {"population_size": pop, "generations": gen,
                             "crossover_rate": cr, "mutation_rate": mr},
        "mc_config":        {"n_simulations": n_s, "n_days": n_d},
        "top_n":            top_n,
        "short_term_mode":  short_term_mode,
        "run_clicked":      run_clicked,
    }


# ═══════════════════════════════════════════════════════════════
#  思考鏈（ThinkingLog）標籤顯示
# ═══════════════════════════════════════════════════════════════

# 關鍵字 → (class, emoji)
_THINK_STAGE_MAP = [
    (["下載", "資料", "data", "fetch"],                "data",  "📡"),
    (["GA", "遺傳", "演化", "種群", "適應度"],          "ga",    "🧬"),
    (["蒙地卡羅", "Monte", "模擬", "路徑", "GBM"],      "mc",    "🎲"),
    (["持股", "換股", "建議", "分析", "holdings"],      "hold",  "💼"),
]

def _classify_think(msg: str) -> tuple[str, str]:
    for keywords, cls, emoji in _THINK_STAGE_MAP:
        if any(k in msg for k in keywords):
            return cls, emoji
    return "info", "💡"


class TagThinkingLogger(ThinkingLogger):
    """
    繼承 ThinkingLogger，將每條訊息渲染為小標籤（打字機滾動效果）。
    tags 儲存已累積的 HTML，每次 log() 呼叫追加並刷新容器。
    """
    def __init__(self, container):
        super().__init__(container)
        self._tags_html: list[str] = []

    def log(self, message: str, level: str = "info") -> None:
        cls, emoji = _classify_think(message)
        # 截斷過長訊息
        short = message if len(message) <= 40 else message[:38] + "…"
        tag_html = f'<span class="think-tag {cls}">{emoji} {short}</span>'
        self._tags_html.append(tag_html)
        # 保留最近 40 個標籤，避免過長
        if len(self._tags_html) > 40:
            self._tags_html = self._tags_html[-40:]
        wrap = (
            '<div class="think-wrap">'
            + "".join(self._tags_html)
            + '<span class="think-cursor"></span>'
            + "</div>"
        )
        self._container.markdown(wrap, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
#  推薦理由（技術指標觸發說明）
# ═══════════════════════════════════════════════════════════════

def _build_reason_pills(action: str, signal: float, score: float,
                         short_term: bool) -> str:
    """
    根據 action / signal 組合生成「推薦理由」技術指標標籤 HTML。
    """
    pills: list[tuple[str, str]] = []  # (類型: bull/bear/neutral, 文字)

    if action in ("STOP_LOSS",):
        pills.append(("bear", "⚠️ 信號< -0.15"))
        pills.append(("bear", "虧損> 10%"))
    elif action in ("NEW_BUY", "SWITCH"):
        if signal > 0.30:
            pills.append(("bull", "强勢多頭信號"))
        if short_term:
            pills.append(("bull", "KDJ 黃金交叉"))
            pills.append(("bull", "量能爆發確認"))
        if signal > 0.15:
            pills.append(("bull", "MA 多頭排列"))
        if score > 0.50:
            pills.append(("bull", "RSI 超賣回升"))
        if score > 0.30:
            pills.append(("bull", "布林通道突破"))
    elif action == "HOLD":
        pills.append(("bull", "技術面穩定"))
        pills.append(("bull", f"信號 {signal:+.2f}"))
    elif action == "WATCH":
        pills.append(("", "信號中性觀望"))
        if short_term:
            pills.append(("", "等待 KDJ 交叉"))
    elif action == "CASH":
        pills.append(("", "無明確信號"))
        pills.append(("bear", "保守持現"))

    html_pills = " ".join(
        f'<span class="ind-pill {cls}">{txt}</span>'
        for cls, txt in pills
    )
    return f'<div class="action-reason">{html_pills}</div>'


# ═══════════════════════════════════════════════════════════════
#  Plotly 圖表函數（MC / Distribution / Scores / Fitness）
# ═══════════════════════════════════════════════════════════════

_PLOT_BG   = "rgba(0,0,0,0)"
_PLOT_AREA = "#FDFBF7"
_PLOT_GRID = "rgba(0,0,0,0.04)"
_PLOT_FONT = "#4A4A4A"
_MONO_FONT = "JetBrains Mono, Fira Code, monospace"


def _base_layout(**kwargs) -> dict:
    return dict(
        plot_bgcolor=_PLOT_AREA, paper_bgcolor=_PLOT_BG,
        font=dict(color=_PLOT_FONT, family=_MONO_FONT, size=11),
        **kwargs,
    )


def chart_monte_carlo(mc: dict, cash: float) -> go.Figure:
    paths = mc["paths"]
    n, init, bk_y = mc["simulation_days"], mc["initial_portfolio_value"], cash * 0.30
    x = np.arange(n + 1)
    p05, p25, p50, p75, p95 = [np.percentile(paths, p, axis=0) for p in (5, 25, 50, 75, 95)]

    fig = go.Figure()
    fig.add_hrect(y0=0, y1=bk_y, fillcolor="rgba(184,84,80,0.07)", line_width=0, layer="below",
                  annotation_text="  破產風險區", annotation_position="bottom right",
                  annotation_font=dict(color="rgba(184,84,80,.55)", size=10))
    rng = np.random.default_rng(42)
    for ix in rng.choice(len(paths), min(150, len(paths)), replace=False):
        fig.add_trace(go.Scatter(x=x, y=paths[ix], mode="lines",
                                  line=dict(color="rgba(96,125,139,0.06)", width=0.7),
                                  showlegend=False, hoverinfo="skip"))
    for y_band, fc, name in [
        (np.concatenate([p95, p05[::-1]]), "rgba(96,125,139,0.08)", "90%信心帶"),
        (np.concatenate([p75, p25[::-1]]), "rgba(96,125,139,0.20)", "IQR帶"),
    ]:
        fig.add_trace(go.Scatter(x=np.concatenate([x, x[::-1]]), y=y_band,
                                  fill="toself", fillcolor=fc,
                                  line=dict(color="rgba(0,0,0,0)"), name=name, hoverinfo="skip"))
    for y_l, col, dash, name in [
        (p50, "#607D8B", "solid", "中位數"),
        (p05, "rgba(184,84,80,.85)", "dot", "5th%"),
        (p95, "rgba(90,138,122,.85)", "dot", "95th%"),
    ]:
        fig.add_trace(go.Scatter(x=x, y=y_l, mode="lines",
                                  line=dict(color=col, width=2 if dash == "solid" else 1.5, dash=dash),
                                  name=name))
    fig.add_hline(y=init, line_dash="dash", line_color="rgba(96,125,139,.40)",
                  annotation_text=f"初始 NT${init:,.0f}",
                  annotation_font=dict(color="#607D8B", size=10), annotation_position="top left")
    fig.add_hline(y=bk_y, line_dash="solid", line_color="rgba(184,84,80,.60)",
                  annotation_text=f"破產門檻 NT${bk_y:,.0f}",
                  annotation_font=dict(color="#B85450", size=10), annotation_position="bottom left")
    fig.update_layout(
        **_base_layout(height=480, margin=dict(t=65)),
        title=dict(text=f"蒙地卡羅  {mc['n_simulations']:,} 路徑 × {n} 交易日",
                   font=dict(color=_PLOT_FONT)),
        xaxis=dict(title="交易日", gridcolor=_PLOT_GRID, color=_PLOT_FONT),
        yaxis=dict(title="投組市值 (NT$)", tickformat=",.0f", gridcolor=_PLOT_GRID, color=_PLOT_FONT),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    bgcolor="rgba(255,255,255,0)"),
        hovermode="x unified",
    )
    return fig


def chart_return_dist(mc: dict) -> go.Figure:
    paths = mc["paths"]
    init  = mc["initial_portfolio_value"]
    rets  = (paths[:, -1] - init) / init * 100

    fig = go.Figure()
    for mask, color, name in [
        (rets < 0,  "rgba(184,84,80,.55)",  "虧損路徑"),
        (rets >= 0, "rgba(90,138,122,.55)", "獲利路徑"),
    ]:
        fig.add_trace(go.Histogram(x=rets[mask], nbinsx=40, marker_color=color, name=name))
    for pct, col in [(5, "#B85450"), (50, "#607D8B"), (95, "#5A8A7A")]:
        v = float(np.percentile(rets, pct))
        fig.add_vline(x=v, line_dash="dash", line_color=col, line_width=1.5,
                      annotation_text=f"{pct}th:{v:.1f}%",
                      annotation_font=dict(color=col, size=10), annotation_position="top")
    fig.add_vline(x=0, line_color="rgba(0,0,0,.15)", line_width=1.2)
    fig.update_layout(
        **_base_layout(height=320),
        title=dict(text="季末報酬率分佈", font=dict(color=_PLOT_FONT)),
        barmode="overlay",
        xaxis=dict(title="報酬率 (%)", gridcolor=_PLOT_GRID, color=_PLOT_FONT),
        yaxis=dict(title="模擬次數", gridcolor=_PLOT_GRID, color=_PLOT_FONT),
        legend=dict(orientation="h", y=1.05, bgcolor="rgba(255,255,255,0)"),
    )
    return fig


def chart_scores(sorted_stocks: list) -> go.Figure:
    codes  = [stock_display(c) for c, _ in sorted_stocks]   # 代號+中文名
    scores = [s for _, s in sorted_stocks]

    fig = go.Figure(go.Bar(
        x=scores, y=codes, orientation="h",
        marker_color=["#C97870" if s >= 0 else "#6B9E8E" for s in scores],
        text=[f"{s:+.4f}" for s in scores],
        textposition="outside",
        textfont=dict(color=_PLOT_FONT, size=11, family=_MONO_FONT),
    ))
    fig.add_vline(x=0, line_color="rgba(0,0,0,.15)", line_width=1.2)
    fig.update_layout(
        **_base_layout(height=max(260, len(codes) * 44 + 80),
                       margin=dict(l=120, r=110, t=50, b=40)),
        title=dict(text="GA 評分（紅=看多　綠=看空）", font=dict(color=_PLOT_FONT)),
        xaxis=dict(gridcolor=_PLOT_GRID, color=_PLOT_FONT,
                   range=[min(scores) * 1.4 - 0.1, max(scores) * 1.4 + 0.1]),
        yaxis=dict(autorange="reversed", color=_PLOT_FONT),
    )
    return fig


def chart_fitness(history: list) -> go.Figure:
    if not history:
        return go.Figure()
    gens, best_f, avg_f = zip(*[(h["generation"], h["best_fitness"], h["avg_fitness"])
                                 for h in history])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=gens, y=best_f, mode="lines+markers",
                              line=dict(color="#607D8B", width=2.2),
                              marker=dict(size=3, color="#607D8B"), name="最佳"))
    fig.add_trace(go.Scatter(x=gens, y=avg_f, mode="lines",
                              line=dict(color="rgba(184,150,106,.80)", width=1.5, dash="dot"),
                              name="平均"))
    fig.update_layout(
        **_base_layout(height=300),
        title=dict(text="GA 適應度收斂", font=dict(color=_PLOT_FONT)),
        xaxis=dict(title="代數", gridcolor=_PLOT_GRID, color=_PLOT_FONT),
        yaxis=dict(title="適應度", gridcolor=_PLOT_GRID, color=_PLOT_FONT),
        legend=dict(orientation="h", y=1.08, bgcolor="rgba(255,255,255,0)"),
    )
    return fig


# ═══════════════════════════════════════════════════════════════
#  UI 渲染函數
# ═══════════════════════════════════════════════════════════════

def render_top_kpi_bar(summary: dict) -> None:
    def _card(label, val_str, dlt_str, dlt_num):
        col = _tw_color_hex(dlt_num)
        arr = _arrow(dlt_num)
        return (
            f'<div class="kpi-wrap">'
            f'<div class="kpi-lbl">{label}</div>'
            f'<div class="kpi-val mono">{val_str}</div>'
            f'<div class="kpi-dlt mono" style="color:{col};">{arr} {dlt_str}</div>'
            f"</div>"
        )

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(_card("💰 總資產", f"NT${summary['total_assets']:,.0f}",
                      f"NT${summary['total_pnl']:+,.0f}  未實現損益",
                      summary["total_pnl"]), unsafe_allow_html=True)
    c2.markdown(_card("📦 持倉市值", f"NT${summary['holding_value']:,.0f}",
                      f"成本 NT${summary['total_cost']:,.0f}",
                      summary["holding_value"] - summary["total_cost"]), unsafe_allow_html=True)
    c3.markdown(_card("📅 今日損益", f"NT${summary['today_pnl']:+,.0f}",
                      f"今日報酬 {summary['today_return']:+.2%}",
                      summary["today_pnl"]), unsafe_allow_html=True)
    c4.markdown(_card("📈 投資報酬率", f"{summary['total_return']:+.2%}",
                      f"最新交易日 {summary['last_trade_date']}",
                      summary["total_return"]), unsafe_allow_html=True)
    st.caption(
        f"⏱ 報價快取 5 分鐘  ·  最新交易日：{summary['last_trade_date']}"
        f"  ·  台股配色：紅漲綠跌（Morandi 低飽和）"
    )


def render_holdings_table(rows: list[dict]) -> None:
    st.markdown('<div class="section-bar">📋 HOLDINGS WATCHLIST</div>', unsafe_allow_html=True)
    if not rows:
        st.info("請輸入持股資料。")
        return

    df = pd.DataFrame([{
        "代號":     r["code"],
        "名稱":     get_name(r["code"]),
        "資料日期": r["trade_date"],
        "持有股數": r["shares"],
        "成本均價": r["cost"],
        "現價":     r["current"],
        "漲跌幅(%)": r["change_pct"] * 100,
        "今日損益": r["change"] * r["shares"],
        "未實現損益": r["pnl"],
        "未實現%":  r["pnl_pct"] * 100,
        "總市值":   r["curr_value"],
    } for r in rows])

    def _tw(val):
        if not isinstance(val, (int, float)) or pd.isna(val): return ""
        if val > 0: return "color:#B85450;font-weight:600"
        if val < 0: return "color:#5A8A7A;font-weight:600"
        return "color:#9E9E9E"

    styled = df.style
    color_cols = ["漲跌幅(%)", "今日損益", "未實現損益", "未實現%"]
    try:
        styled = styled.map(_tw, subset=color_cols)
    except AttributeError:
        styled = styled.applymap(_tw, subset=color_cols)

    styled = styled.format({
        "持有股數": "{:,}",
        "成本均價": "NT${:.2f}",
        "現價":     "NT${:.2f}",
        "漲跌幅(%)": "{:+.2f}%",
        "今日損益": "NT${:+,.0f}",
        "未實現損益": "NT${:+,.0f}",
        "未實現%":  "{:+.2f}%",
        "總市值":   "NT${:,.0f}",
    }).hide(axis="index")
    st.dataframe(styled, use_container_width=True, height=min(400, len(rows) * 45 + 52))


_ACTION_STYLE = {
    "STOP_LOSS": ("rgba(184,84,80,0.08)",    "#B85450", "🛑"),
    "SWITCH":    ("rgba(184,150,106,0.10)",  "#B8966A", "⚡"),
    "NEW_BUY":   ("rgba(96,125,139,0.09)",   "#607D8B", "🆕"),
    "HOLD":      ("rgba(90,138,122,0.08)",   "#5A8A7A", "✅"),
    "WATCH":     ("rgba(122,110,158,0.08)",  "#7A6E9E", "👁️"),
    "CASH":      ("rgba(158,158,158,0.08)",  "#9E9E9E", "💰"),
}


def render_daily_guide(guide: list[dict], short_term: bool = False) -> None:
    st.markdown('<div class="section-bar">📋 TODAY\'S ACTION GUIDE</div>', unsafe_allow_html=True)
    st.caption(f"根據 GA 最佳策略 + 即時報價 + 換股成本分析  ·  {datetime.date.today()}")

    if not guide:
        st.success("✅  今日無需操作，持股狀況良好。")
        return

    # 概覽表（含推薦理由欄）
    table_data = []
    for item in guide:
        # 組合技術指標觸發條件文字
        reasons = []
        if item["action"] in ("NEW_BUY", "SWITCH"):
            if item["signal"] > 0.30:    reasons.append("强勢多頭")
            elif item["signal"] > 0.15:  reasons.append("MA多頭")
            if short_term:               reasons.append("KDJ+量能")
            if item.get("excess_return") and item["excess_return"] > 0.10:
                reasons.append("超額報酬↑")
        elif item["action"] == "STOP_LOSS":
            reasons = ["虧損>10%", "信號<-0.15"]
        elif item["action"] == "HOLD":
            reasons = ["信號穩定", f"{item['signal']:+.2f}"]
        else:
            reasons = ["觀望"]

        table_data.append({
            "優先級":  item["priority"],
            "操作":    item["icon"] + "  " + item["label"],
            "信號":    f"{item['signal']:+.3f}",
            "建議金額": f"NT${item['amount']:,.0f}" if item["amount"] else "─",
            "超額年化": f"{item['excess_return']:+.1%}" if item.get("excess_return") else "─",
            "回本時間": (f"{item['breakeven_months']:.1f}月"
                        if item.get("breakeven_months") and item["breakeven_months"] != float("inf")
                        else "─"),
            "推薦理由": "、".join(reasons),
        })

    st.dataframe(
        pd.DataFrame(table_data),
        use_container_width=True, hide_index=True,
        height=min(300, len(guide) * 40 + 42),
    )

    st.markdown("---")
    st.markdown("##### 詳細操作說明")
    for item in guide:
        bg, border, _ = _ACTION_STYLE.get(item["action"], ("rgba(158,158,158,0.07)", "#9E9E9E", "─"))
        reason_html = _build_reason_pills(
            item["action"], item["signal"],
            item.get("detail_score", 0), short_term
        )
        code_name = ""
        if "code" in item:
            code_name = f"（{stock_display(item['code'])}）"
        st.markdown(f"""
        <div class="action-card" style="background:{bg};border-left-color:{border};">
          <div class="action-title">{item['icon']}&nbsp;&nbsp;{item['label']}{code_name}</div>
          <div class="action-body">{item['summary']}</div>
          {reason_html}
        </div>""", unsafe_allow_html=True)

        if item.get("detail"):
            with st.expander("查看詳細計算"):
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("信號強度", f"{item['signal']:+.3f}")
                if item.get("excess_return"):
                    c2.metric("超額年化", f"{item['excess_return']:+.1%}")
                if item.get("breakeven_months") and item["breakeven_months"] != float("inf"):
                    c3.metric("換股回本", f"{item['breakeven_months']:.1f} 個月")
                if item["amount"]:
                    c4.metric("涉及金額", f"NT${item['amount']:,.0f}")
                st.markdown(item["detail"])


# ── K 線模組（@st.fragment 隔離） ─────────────────────────────

def _kline_body(available_codes: list[str], quotes: dict) -> None:
    st.markdown('<div class="section-bar">📊 INDIVIDUAL STOCK ANALYSIS</div>',
                unsafe_allow_html=True)
    if not available_codes:
        st.info("請先輸入股票代號。")
        return

    col_s, col_q = st.columns([2, 5])
    with col_s:
        # Selectbox 顯示「代號 名稱」
        display_options = [stock_display(c) for c in available_codes]
        sel_display = st.selectbox("選擇代號", display_options, key="kline_sel",
                                   label_visibility="collapsed")
        # 反查回純代號
        idx  = display_options.index(sel_display)
        code = available_codes[idx]
        name = get_name(code)
        st.caption(f"**{code} {name}**  近 120 日 K 線")

    with col_q:
        if code in quotes:
            q   = quotes[code]
            col = _tw_color_hex(q["change_pct"])
            arr = _arrow(q["change_pct"])
            st.markdown(
                f"<span class='mono' style='font-size:1.3rem;font-weight:700;color:{col};'>"
                f"NT${q['price']:.2f}&nbsp;{arr}&nbsp;{abs(q['change_pct']):.2%}"
                f"&nbsp;<span style='font-size:.85rem;'>({q['change']:+.2f})</span></span>"
                f"&nbsp;&nbsp;<span style='color:#9E9E9E;font-size:.80rem;'>"
                f"成交量 {q['volume']:,.0f}&nbsp;·&nbsp;{q.get('trade_date','')}</span>",
                unsafe_allow_html=True,
            )

    try:
        with st.spinner(f"載入 {code} K 線..."):
            df = fetch_kline_data(code)
    except Exception as e:
        st.warning(f"⚠️  K 線載入失敗（{e}），請稍後重試。")
        return

    if df is None or df.empty:
        st.warning(f"⚠️  無法取得 {code} 的數據。")
        return

    # ── TradingView Lightweight Charts ──────────────────────
    tradingview_chart(df, code, name, height=520)

    # 統計卡
    recent = df.tail(60)
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("60日最高", f"NT${float(recent['High'].max()):.1f}")
    s2.metric("60日最低", f"NT${float(recent['Low'].min()):.1f}")
    s3.metric("60日均量", f"{float(recent['Volume'].mean()):,.0f}")
    last_c = float(recent["Close"].iloc[-1])
    h60w   = float(recent["High"].max())
    s4.metric("距60日高點", f"{(last_c - h60w) / h60w:.2%}", delta_color="off")


# @st.fragment 隔離：切換 selectbox 只重跑此函數，不干擾推演進程
try:
    @st.fragment
    def render_kline_section(available_codes: list[str], quotes: dict) -> None:
        _kline_body(available_codes, quotes)
except AttributeError:
    def render_kline_section(available_codes: list[str], quotes: dict) -> None:
        _kline_body(available_codes, quotes)


def render_analysis_tabs(results: dict, config: dict) -> None:
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊  市場掃描", "🧬  GA 最佳化", "🎲  蒙地卡羅", "💼  持股分析"
    ])

    with tab1:
        ss    = results["sorted_stocks"]
        sc    = results["selected_codes"]
        n_pos = sum(1 for _, s in ss if s > 0)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("掃描股數", len(ss))
        m2.metric("正向信號", n_pos, delta=f"佔{n_pos/len(ss):.0%}" if ss else "")
        m3.metric("最終選股", len(sc))
        fh = results["fitness_history"]
        m4.metric("GA最佳適應度", f"{fh[-1]['best_fitness']:.4f}" if fh else "─")
        st.markdown("---")
        ca, cb = st.columns([3, 2])
        with ca:
            st.plotly_chart(chart_scores(ss), use_container_width=True)
        with cb:
            st.markdown("##### 評分明細")
            mode_tag = "⚡短線" if config.get("short_term_mode") else "🔵長線"
            st.caption(f"策略模式：{mode_tag}")
            st.dataframe(pd.DataFrame([{
                "#":    i,
                "代號": stock_display(c),   # 顯示中文名稱
                "評分": f"{s:+.4f}",
                "信號": "▲看多" if s > 0.10 else ("▼看空" if s < -0.10 else "─中性"),
                "":     "✅" if c in sc else "",
            } for i, (c, s) in enumerate(ss, 1)]),
                use_container_width=True, hide_index=True,
                height=min(420, len(ss) * 40 + 42))
        if results.get("failed_codes"):
            st.warning(f"⚠️  下載失敗：{results['failed_codes']}")

    with tab2:
        bp = results["best_params"]
        fh = results["fitness_history"]
        if fh:
            i_b, f_b = fh[0]["best_fitness"], fh[-1]["best_fitness"]
            c1, c2, c3 = st.columns(3)
            c1.metric("初代適應度", f"{i_b:.4f}")
            c2.metric("末代適應度", f"{f_b:.4f}")
            c3.metric("演化改進",   f"{f_b - i_b:+.4f}",
                      delta_color="normal" if f_b > i_b else "inverse")
        st.plotly_chart(chart_fitness(fh), use_container_width=True)
        st.markdown("##### 最佳策略參數")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**MA 交叉**")
            st.metric("短期", f"{bp['ma_short']}日")
            st.metric("長期", f"{bp['ma_long']}日")
            st.metric("權重", f"{bp['ma_weight']:.2f}")
        with c2:
            st.markdown("**RSI**")
            st.metric("周期", f"{bp['rsi_period']}日")
            st.metric("超買", f"{bp['rsi_ob']:.0f}")
            st.metric("超賣", f"{bp['rsi_os']:.0f}")
            st.metric("權重", f"{bp['rsi_weight']:.2f}")
        with c3:
            st.markdown("**布林通道**")
            st.metric("周期", f"{bp['bb_period']}日")
            st.metric("標準差", f"{bp['bb_std']:.1f}σ")
            st.metric("權重", f"{bp['bb_weight']:.2f}")
            st.metric("買入閾", f"{bp['buy_threshold']:.2f}")
        with st.expander("完整 JSON"):
            st.json(bp)

    with tab3:
        mc   = results["mc_stats"]
        cash = config["available_cash"]
        win  = mc["win_rate"]
        bk   = mc["bankruptcy_probability"]
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("勝率", mc["win_rate_pct"],
                  delta="正期望" if win > 0.5 else "負期望，謹慎",
                  delta_color="normal" if win > 0.5 else "inverse")
        k2.metric("期望季度報酬", mc["expected_return_pct"],
                  delta=f"年化 {mc['expected_annualized_return_pct']}")
        k3.metric("動態破產機率", mc["bankruptcy_probability_pct"],
                  delta="偏高" if bk > 0.05 else "可控",
                  delta_color="inverse" if bk > 0.05 else "normal")
        k4.metric("平均最大回撤", mc["avg_max_drawdown_pct"])
        if win < 0.5: st.warning("⚠️  勝率低於 50%，建議縮小部位。")
        if bk > 0.05: st.error("🚨  破產機率超過 5%！請設置嚴格停損。")
        st.markdown("---")
        st.plotly_chart(chart_monte_carlo(mc, cash), use_container_width=True)
        lc, rc = st.columns([3, 2])
        with lc:
            st.plotly_chart(chart_return_dist(mc), use_container_width=True)
        with rc:
            st.markdown("##### 報酬率分位數")
            d = mc["return_distribution"]
            st.dataframe(pd.DataFrame({
                "分位":  ["最壞(5%)", "悲觀(25%)", "中位(50%)", "樂觀(75%)", "最佳(95%)"],
                "季報酬": [f"{v:+.2%}" for v in [d["p05"], d["p25"], d["p50"], d["p75"], d["p95"]]],
            }), use_container_width=True, hide_index=True)
            st.markdown("##### 配置明細")
            st.dataframe(pd.DataFrame([{
                "代號": stock_display(c),
                "現價": f"NT${det['price_now']:,.1f}",
                "股數": f"{det['n_shares']:,}",
                "投入": f"NT${det['actual_cost']:,.0f}",
            } for c, det in mc["allocation_detail"].items()]),
                use_container_width=True, hide_index=True)
            st.info(f"總投入 NT${mc['total_invested']:,.0f}\n\n剩餘現金 NT${mc['unused_cash']:,.0f}")

    with tab4:
        ha   = results["holdings_analysis"]
        recs = results["recommendations"]
        if not ha:
            st.info("未輸入持股。")
            return
        tv = sum(i["current_value"]  for i in ha.values())
        tp = sum(i["unrealized_pnl"] for i in ha.values())
        tc = sum(i["cost_value"]      for i in ha.values())
        t1, t2, t3 = st.columns(3)
        t1.metric("持股總市值", f"NT${tv:,.0f}")
        t2.metric("總未實現損益", f"NT${tp:,.0f}",
                  delta=f"{tp/tc:.2%}" if tc > 0 else "",
                  delta_color="normal" if tp >= 0 else "inverse")
        t3.metric("總成本", f"NT${tc:,.0f}")
        st.markdown("---")
        st.dataframe(pd.DataFrame([{
            "代號":   stock_display(c),
            "成本":   f"NT${i['cost_price']:.1f}",
            "現價":   f"NT${i['current_price']:.1f}",
            "股數":   f"{i['shares']:,}",
            "現值":   f"NT${i['current_value']:,.0f}",
            "損益%":  f"{i['unrealized_pnl_pct']:+.2%}",
            "信號":   f"{i['current_signal']:+.3f}",
            "Sharpe": f"{i['sharpe_ratio']:.3f}",
            "期望年化": f"{i['expected_annual_ret']:.2%}",
        } for c, i in ha.items()]),
            use_container_width=True, hide_index=True)
        st.markdown("---")
        st.markdown("##### 換股建議")
        if not recs:
            st.success("✅  持股狀況良好，無換股建議。")
        else:
            st.warning(f"⚡  {len(recs)} 筆建議")
            _ip = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}
            for i, r in enumerate(recs, 1):
                st.markdown(
                    f"**{_ip.get(r['priority'],'─')} 建議{i}**  "
                    f"賣 **{stock_display(r['sell_code'])}** → 買 **{stock_display(r['buy_code'])}**"
                )
                c1, c2, c3, c4 = st.columns(4)
                c1.metric(f"{r['sell_code']} 期望年化", f"{r['sell_expected_ret']:.2%}",
                          delta=f"信號{r['sell_signal']:+.3f}")
                c2.metric(f"{r['buy_code']} 歷史年化",  f"{r['buy_annual_ret']:.2%}",
                          delta=f"GA{r['buy_ga_score']:+.4f}")
                c3.metric("超額年化", f"{r['excess_return_annual']:+.2%}")
                c4.metric("年度機會成本", f"NT${r['opportunity_cost_annual']:,.0f}")
                ca, cb = st.columns(2)
                ca.info(f"換股成本 NT${r['switch_cost_ntd']:,.0f}")
                cb.info(f"預計回本 {r['payback_str']}")
                st.markdown("---")


# ═══════════════════════════════════════════════════════════════
#  主程式
# ═══════════════════════════════════════════════════════════════

def main() -> None:
    # ── 認證閘門 ──────────────────────────────────────────────
    if not render_auth_page():
        return

    username = st.session_state["_username"]

    # ── 頂部控制列（取代 Sidebar）─────────────────────────────
    config = render_top_controls(username)
    ch     = config["current_holdings"]
    pool   = config["target_pool"]
    cash   = config["available_cash"]

    st.markdown("---")

    # ── 即時報價 ──────────────────────────────────────────────
    all_codes = tuple(sorted(set(list(ch.keys()) + pool)))
    with st.spinner("⚡ 獲取即時報價..."):
        quotes = fetch_realtime_quotes(all_codes)

    # ── KPI 列 ────────────────────────────────────────────────
    if ch:
        render_top_kpi_bar(compute_portfolio_summary(ch, quotes, cash))
    else:
        st.info("ℹ️  輸入持股後，此處將顯示即時資產快報。")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── 持股表格 ──────────────────────────────────────────────
    if ch:
        render_holdings_table(build_holdings_rows(ch, quotes))

    st.markdown("<br>", unsafe_allow_html=True)

    # ── K 線圖（Fragment 隔離） ────────────────────────────────
    render_kline_section(list(dict.fromkeys(list(ch.keys()) + pool)), quotes)

    st.markdown("---")

    # ── 執行推演 ──────────────────────────────────────────────
    if config["run_clicked"]:
        if not pool:
            st.error("❌  請先設定候選股池。")
            return
        for k in ("_results", "_config", "_has_results"):
            st.session_state.pop(k, None)

        prog = st.progress(0)
        stat = st.empty()

        # 思考鏈監控視窗
        st.markdown(
            "<div class='section-bar'>🧠 THINKING LOG  ─  AI 推理過程</div>",
            unsafe_allow_html=True,
        )
        think_container = st.empty()
        # 初始化空白標籤區
        think_container.markdown(
            '<div class="think-wrap"><span class="think-cursor"></span></div>',
            unsafe_allow_html=True,
        )
        logger = TagThinkingLogger(think_container)

        try:
            all_hist_codes = tuple(sorted(set(list(ch.keys()) + pool)))
            stock_data     = fetch_pool_history(all_hist_codes)

            results = run_full_pipeline(
                cash, ch, pool, stock_data,
                config["ga_config"], config["mc_config"], config["top_n"],
                config["short_term_mode"],
                prog, stat,
                thinking_logger=logger,
            )
            st.session_state.update({
                "_results": results, "_config": config, "_has_results": True,
            })
            prog.empty(); stat.empty()
            st.success("✅  分析完成！")

        except Exception as e:
            prog.empty(); stat.empty()
            st.error(f"❌  {e}")
            with st.expander("錯誤詳情"):
                st.exception(e)
            return

    # ── 結果展示 ──────────────────────────────────────────────
    if st.session_state.get("_has_results"):
        res = st.session_state["_results"]
        cfg = st.session_state.get("_config", config)

        render_daily_guide(
            generate_daily_guide(
                res["holdings_analysis"], res["stock_scores"],
                cfg["current_holdings"], cfg["available_cash"], quotes,
            ),
            short_term=cfg.get("short_term_mode", True),
        )
        st.markdown("---")
        render_analysis_tabs(res, cfg)
    else:
        st.markdown("""
        <div style="text-align:center;padding:52px 0 36px;color:#9E9E9E;">
          <div style="font-size:2.6rem;margin-bottom:16px;opacity:.50;">🧬</div>
          <p style="color:#7A7A7A;font-size:.95rem;margin:0 0 8px;">
            點擊上方 <b style="color:#607D8B;">🚀 執行推演</b> 啟動 GA 優化與蒙地卡羅模擬</p>
          <p style="color:#AEAEAE;font-size:.80rem;margin:0;">
            系統將依持倉與資金，產出個性化買入建議與換股分析</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.caption("⚠️ 本系統僅供學術研究，不構成投資建議。過去績效不代表未來報酬。")


if __name__ == "__main__":
    main()
