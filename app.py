"""
app.py  ─  v4.0  實戰交易儀表板（UI 層）
─────────────────────────────────────────
架構說明：
  app.py   → Streamlit UI（頁面、圖表、表單）
  engine.py → 業務邏輯（Pipeline、決策引擎、ThinkingLogger）
  auth.py   → 用戶認證（登入/註冊/設定持久化）

台股配色（Morandi）：漲 = 消紅 #B85450   跌 = 消綠 #5A8A7A
"""

# ── 標準函式庫 ────────────────────────────────────────────────
import datetime

# ── 第三方套件 ────────────────────────────────────────────────
import yfinance as yf
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit.components.v1 as components

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
    TW_STOCK_NAMES,
)
from data_fetcher import DataFetcher, get_tw_daily_snapshot, fetch_with_funnel


# ═══════════════════════════════════════════════════════════════
#  頁面設定（第一個 st 指令）
# ═══════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="台股量化交易儀表板",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
/* ══════════════════════════════════════════════════════════════
   Morandi × 日系文青  ─  Design System v4.0
   Palette:
     Linen BG  #F9F6F1  |  Surface #FFFFFF
     Accent    #607D8B  |  Sage    #7A9E87
     Text      #383838  |  Muted   #7A7A7A
     Border    #E6E1D9  |  Rise    #B85450  Fall #5A8A7A
══════════════════════════════════════════════════════════════ */

/* ─ Google Fonts ─────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Noto+Serif+TC:wght@300;400;600&family=Noto+Sans+TC:wght@300;400;500&family=Inter:wght@300;400;500&display=swap');

/* ─ Root ─────────────────────────────────────────────────── */
:root {
    --bg      : #F9F6F1;
    --surface : #FFFFFF;
    --sidebar : #F2EFE8;
    --accent  : #607D8B;
    --sage    : #7A9E87;
    --amber   : #B8966A;
    --text    : #383838;
    --muted   : #7A7A7A;
    --faint   : #AEAEAE;
    --border  : #E6E1D9;
    --shadow  : rgba(56,40,20,.07);
    --rise    : #B85450;
    --fall    : #5A8A7A;
}

/* ─ Global ───────────────────────────────────────────────── */
.stApp {
    background-color: var(--bg) !important;
    font-family: 'Noto Sans TC', 'Inter', sans-serif;
    color: var(--text);
}

/* ─ Sidebar ──────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background-color: var(--sidebar) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] .stMarkdown,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span {
    color: var(--text) !important;
}
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] h4 {
    font-family: 'Noto Serif TC', Georgia, serif !important;
    color: var(--text) !important;
}

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

/* ─ Buttons ──────────────────────────────────────────────── */
[data-testid="stButton"] > button {
    background: transparent !important;
    border: 1.5px solid var(--border) !important;
    border-radius: 24px !important;
    color: var(--muted) !important;
    font-family: 'Noto Sans TC', 'Inter', sans-serif !important;
    font-size: .83rem !important;
    letter-spacing: .04em;
    padding: 7px 20px !important;
    transition: all .2s ease !important;
    box-shadow: none !important;
}
[data-testid="stButton"] > button:hover {
    border-color: var(--accent) !important;
    color: var(--accent) !important;
    background: rgba(96,125,139,.06) !important;
}
/* Primary variant */
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
    box-shadow: 0 6px 24px rgba(56,40,20,.11) !important;
}
[data-testid="metric-label"] > div {
    color: var(--muted) !important;
    font-size: .74rem !important;
    letter-spacing: .07em;
    text-transform: uppercase;
    font-family: 'Noto Sans TC', 'Inter', sans-serif !important;
}
[data-testid="metric-value"] > div {
    color: var(--text) !important;
    font-family: 'Noto Serif TC', Georgia, serif !important;
    font-size: 1.5rem !important;
    font-weight: 600 !important;
}
[data-testid="metric-delta"] {
    font-size: .76rem !important;
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
    padding: 8px 18px !important;
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
    margin: 8px 0 !important;
    overflow: hidden;
}
[data-testid="stExpander"] summary {
    color: var(--text) !important;
    font-family: 'Noto Sans TC', 'Inter', sans-serif !important;
    font-size: .86rem !important;
    padding: 12px 16px !important;
}
[data-testid="stExpander"] summary:hover {
    background: rgba(96,125,139,.04) !important;
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
    font-size: .88rem !important;
}
.stTextInput > div > div > input:focus,
.stNumberInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(96,125,139,.14) !important;
}

/* ─ Select / Radio ───────────────────────────────────────── */
[data-testid="stSelectbox"] > div > div {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
}
.stRadio label { color: var(--text) !important; font-size: .85rem !important; }
.stRadio label span { color: var(--muted) !important; }

/* ─ Toggle ───────────────────────────────────────────────── */
.stToggle label { color: var(--text) !important; }

/* ─ Slider ───────────────────────────────────────────────── */
.stSlider [data-testid="stTickBar"] { color: var(--faint) !important; }

/* ─ Caption / Code ───────────────────────────────────────── */
[data-testid="stCaptionContainer"] p,
.stCaption { color: var(--muted) !important; font-size: .77rem !important; }
code {
    background: rgba(96,125,139,.10) !important;
    color: var(--accent) !important;
    border-radius: 6px !important;
    padding: 2px 7px !important;
    font-size: .81rem !important;
}

/* ─ Alerts / Info boxes ─────────────────────────────────── */
[data-testid="stAlert"] {
    border-radius: 10px !important;
    border: 1px solid var(--border) !important;
}

/* ─ Divider ──────────────────────────────────────────────── */
hr { border: none !important; border-top: 1px solid var(--border) !important; margin: 20px 0 !important; }

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
   Custom Component Styles
═══════════════════════════════════════════════════════════ */

/* ─ KPI Cards ────────────────────────────────────────────── */
.kpi-wrap {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 22px 24px 16px;
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
    font-size: .70rem;
    font-family: 'Noto Sans TC', 'Inter', sans-serif;
    color: var(--muted);
    letter-spacing: .12em;
    text-transform: uppercase;
    margin-bottom: 10px;
}
.kpi-val {
    font-size: 1.55rem;
    font-family: 'Noto Serif TC', Georgia, serif;
    font-weight: 600;
    color: var(--text);
    line-height: 1.2;
}
.kpi-dlt {
    font-size: .76rem;
    font-family: 'Noto Sans TC', 'Inter', sans-serif;
    margin-top: 8px;
    opacity: .85;
}

/* ─ Section Headers ──────────────────────────────────────── */
.section-bar {
    font-size: .68rem;
    font-family: 'Noto Sans TC', 'Inter', sans-serif;
    font-weight: 600;
    color: var(--faint);
    letter-spacing: .20em;
    text-transform: uppercase;
    display: flex;
    align-items: center;
    gap: 12px;
    margin: 28px 0 14px;
}
.section-bar::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
}

/* ─ Thinking Log ─────────────────────────────────────────── */
.think-box {
    background: #F5F2EC;
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent);
    border-radius: 0 10px 10px 0;
    padding: 14px 18px;
    font-family: 'SF Mono', 'Courier New', monospace;
    font-size: .79rem;
    color: var(--muted);
    max-height: 220px;
    overflow-y: auto;
    line-height: 1.85;
}
.think-box::-webkit-scrollbar { width: 3px; }
.think-box::-webkit-scrollbar-track { background: transparent; }
.think-box::-webkit-scrollbar-thumb { background: var(--border); border-radius: 4px; }
</style>
""", unsafe_allow_html=True)


def _stock_label(code: str) -> str:
    """將代號轉為「代號 中文名」顯示標籤，無對應名稱時只返回代號。"""
    name = TW_STOCK_NAMES.get(code, '')
    return f"{code} {name}" if name else code


# ═══════════════════════════════════════════════════════════════
#  快取數據函數（唯一接觸 yfinance 的層）
# ═══════════════════════════════════════════════════════════════

@st.cache_data(ttl=300, show_spinner=False)
def fetch_realtime_quotes(codes: tuple[str, ...]) -> dict[str, dict]:
    """批次獲取即時報價，快取 5 分鐘。"""
    quotes: dict[str, dict] = {}
    for code in codes:
        for suffix in ('.TW', '.TWO'):
            try:
                # auto_adjust=False：使用未還原除權息的原始收盤價，
                # 與看盤軟體顯示一致（不受股利調整影響）
                raw = yf.download(
                    f"{code}{suffix}", period='5d', interval='1d',
                    progress=False, auto_adjust=False,
                )
                if raw.empty:
                    continue
                if isinstance(raw.columns, pd.MultiIndex):
                    raw.columns = [c[0] for c in raw.columns]
                # 優先使用原始 Close（未調整）
                close_col = 'Close'
                raw = raw.dropna(subset=[close_col]).sort_index()
                if len(raw) < 1:
                    continue
                curr  = float(raw[close_col].iloc[-1])
                prev  = float(raw[close_col].iloc[-2]) if len(raw) >= 2 else curr
                chg   = curr - prev
                chg_p = chg / prev if prev != 0 else 0.0
                quotes[code] = {
                    'price': curr, 'prev_close': prev,
                    'change': chg, 'change_pct': chg_p,
                    'volume': float(raw['Volume'].iloc[-1]),
                    'trade_date': raw.index[-1].strftime('%Y-%m-%d'),
                    'suffix': suffix,
                    'name': TW_STOCK_NAMES.get(code, ''),
                }
                break
            except Exception:
                continue
    return quotes


@st.cache_data(ttl=300, show_spinner=False)
def fetch_kline_data(code: str) -> pd.DataFrame | None:
    """獲取個股近 3 個月 K 線數據，快取 5 分鐘。"""
    for suffix in ('.TW', '.TWO'):
        try:
            raw = yf.download(
                f"{code}{suffix}", period='3mo', interval='1d',
                progress=False, auto_adjust=False,
            )
            if raw.empty:
                continue
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = [c[0] for c in raw.columns]
            raw.index = pd.to_datetime(raw.index)
            raw = raw.dropna(subset=['Open', 'High', 'Low', 'Close'])
            if len(raw) >= 5:
                return raw.sort_index()
        except Exception:
            continue
    return None


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_pool_history(codes: tuple[str, ...]) -> dict[str, pd.DataFrame]:
    """下載股票池 2 年歷史數據，快取 1 小時。作為漏斗失敗時的 fallback。"""
    fetcher = DataFetcher(period='2y')
    return fetcher.fetch_multiple(list(codes))


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_funnel_pool(period: str = "2y") -> tuple[list[str], dict, dict]:
    """
    呼叫 TWSE/TPEx API 取得全市場快照，執行三階段漏斗篩選：
      Stage 1 — 成交量 > 1,000 張
      Stage 2 — 收盤價 > 5MA
      Stage 3 — 一次性批次下載 {period} 完整歷史

    快取 1 小時，同一交易日內不重複呼叫 API。

    Returns
    -------
    (passed_codes, stock_data, selection_reasons)
      passed_codes      : 通過兩關篩選的代號列表
      stock_data        : {代號: OHLCV DataFrame}
      selection_reasons : {代號: {'volume_lots', 'ma5_pct', 'reason_str', ...}}

    若 API 失敗（非交易日 / 網路問題 / 盤中資料未就緒），
    回傳空值以觸發 fallback。
    """
    try:
        codes, snapshot_df = get_tw_daily_snapshot(verbose=False)
        if not codes:
            return [], {}, {}
        stock_data, selection_reasons = fetch_with_funnel(
            snapshot_df, period=period, verbose=False
        )
        return list(stock_data.keys()), stock_data, selection_reasons
    except Exception:
        return [], {}, {}


# ═══════════════════════════════════════════════════════════════
#  認證頁面
# ═══════════════════════════════════════════════════════════════

def render_auth_page() -> bool:
    """
    若未登入則顯示登入/註冊頁，返回 True 表示已通過認證。
    """
    if st.session_state.get('_username'):
        return True

    st.markdown("""
    <div style="text-align:center;padding:30px 0 10px;">
      <div style="font-size:2.8rem;">📈</div>
      <h2 style="color:var(--text,#383838);margin:8px 0 4px;font-family:'Noto Serif TC',Georgia,serif;">台股量化交易儀表板</h2>
      <p style="color:var(--muted,#7A7A7A);font-size:.85rem;">請先登入或建立帳號</p>
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
                    st.session_state['_username'] = u
                    saved = auth.load_user_settings(u)
                    if saved:
                        st.session_state['_user_settings'] = saved
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
#  設定標籤、儀表板標籤、分析標籤
# ═══════════════════════════════════════════════════════════════

def render_settings_tab(username: str) -> tuple[dict, bool]:
    """
    全部策略設定（原側邊欄內容移至此處）。
    Returns (config dict, run_clicked bool)。
    """
    saved = st.session_state.get('_user_settings', {})

    st.markdown("### ⚙️ 策略設定")

    # ── 資金 & 持股（水平兩欄）────────────────────────────────
    st.markdown('<div class="section-bar">💰 資金與持股</div>', unsafe_allow_html=True)
    col_cash, col_hold = st.columns([1, 2])

    with col_cash:
        st.markdown("**可用現金（NT$）**")
        available_cash = st.number_input(
            "cash", min_value=10_000, max_value=100_000_000,
            value=int(saved.get('available_cash', 500_000)),
            step=10_000, format="%d", label_visibility='collapsed',
        )
        st.caption(f"NT$ {available_cash:,}")

    with col_hold:
        st.markdown("**現有持股**")
        st.caption("代號填純數字（不含 .TW），可動態新增 / 刪除")
        saved_holdings = saved.get('current_holdings', {})
        default_rows = (
            [{'代號': c, '買入均價': v['cost'], '持有股數': v['shares']}
             for c, v in saved_holdings.items()]
            if saved_holdings else
            [{'代號': '2330', '買入均價': 850.0, '持有股數': 1000},
             {'代號': '2317', '買入均價':  95.0, '持有股數': 2000}]
        )
        h_df = st.data_editor(
            pd.DataFrame(default_rows), num_rows='dynamic', use_container_width=True,
            column_config={
                '代號':    st.column_config.TextColumn('代號', width='small'),
                '買入均價': st.column_config.NumberColumn('均價', min_value=0.01, format="%.2f"),
                '持有股數': st.column_config.NumberColumn('股數', min_value=0, step=1000, format="%d"),
            },
            key='cfg_holdings_editor',
        )

    current_holdings: dict[str, dict] = {}
    for _, row in h_df.dropna(subset=['代號']).iterrows():
        code = str(row['代號']).strip()
        if code and float(row.get('買入均價', 0) or 0) > 0:
            current_holdings[code] = {
                'cost':   float(row['買入均價']),
                'shares': int(row.get('持有股數') or 0),
            }

    st.markdown("---")

    # ── 候選股池 ──────────────────────────────────────────────
    st.markdown('<div class="section-bar">🎯 候選股池</div>', unsafe_allow_html=True)
    # 嘗試漏斗 API，失敗時 fallback 到硬編碼池
    funnel_codes, _funnel_data, _funnel_reasons = fetch_funnel_pool()
    if funnel_codes:
        auto_pool = (
            list(current_holdings.keys())
            + [c for c in funnel_codes if c not in current_holdings]
        )[:50]
        _auto_pool_source = "市場漏斗"
    else:
        auto_pool = generate_asset_driven_pool(available_cash, current_holdings)
        _auto_pool_source = "資產導向"

    pool_mode = st.radio(
        "pool_mode", ["🤖 資產導向（自動）", "✏️ 手動輸入"],
        index=0, label_visibility='collapsed', horizontal=True,
    )
    if pool_mode == "🤖 資產導向（自動）":
        target_pool = auto_pool
        _src_badge = "🌐 全市場漏斗篩選" if _auto_pool_source == "市場漏斗" else "📋 資產導向（API 未就緒）"
        st.caption(
            f"{_src_badge}，共 {len(target_pool)} 支候選：\n"
            f"`{'、'.join(target_pool[:6])}{'…' if len(target_pool) > 6 else ''}`"
        )
        if st.button("🔄 重新生成", key="regen_pool"):
            st.cache_data.clear()
            st.rerun()
    else:
        saved_pool_str = ','.join(saved.get('target_pool', auto_pool))
        pool_text = st.text_area(
            "pool", value=saved_pool_str,
            height=80, label_visibility='collapsed',
        )
        target_pool = [c.strip() for c in pool_text.replace('\n', ',').split(',') if c.strip()]
        st.caption(f"共 {len(target_pool)} 支")

    st.markdown("---")

    # ── 交易策略 & 進階參數（左右兩欄）───────────────────────
    st.markdown('<div class="section-bar">⚡ 策略與參數</div>', unsafe_allow_html=True)
    col_strat, col_params = st.columns([1, 2])

    with col_strat:
        short_term_mode = st.toggle(
            "短線優先（KDJ + 量能強化）",
            value=saved.get('short_term_mode', True),
            help="開啟：短均線 + KDJ + 成交量爆發（波段/當沖）\n"
                 "關閉：長均線 + 技術信號（季線持有）",
        )
        st.caption("🔴 短線模式：GA 縮短週期，KDJ + 量能加權 40%"
                   if short_term_mode else
                   "🔵 長線模式：GA 標準週期，純技術信號")
        st.markdown("")
        top_n = st.slider("📌 最終選股數", 1, 6, int(saved.get('top_n', 3)))

    with col_params:
        with st.expander("🧬 遺傳演算法參數", expanded=False):
            pop = st.slider("種群大小", 20, 100, int(saved.get('ga_config', {}).get('population_size', 50)), step=10)
            gen = st.slider("演化代數", 20, 100, int(saved.get('ga_config', {}).get('generations', 50)), step=10)
            cr  = st.slider("交叉率",   0.50, 1.00, float(saved.get('ga_config', {}).get('crossover_rate', 0.80)), step=0.05)
            mr  = st.slider("變異率",   0.05, 0.30, float(saved.get('ga_config', {}).get('mutation_rate', 0.15)), step=0.05)
        with st.expander("🎲 蒙地卡羅參數", expanded=False):
            n_s = st.select_slider("路徑數",   options=[500, 1000, 2000, 5000],
                                   value=saved.get('mc_config', {}).get('n_simulations', 1000))
            n_d = st.select_slider("模擬天數", options=[10, 21, 42, 63, 126],
                                   value=saved.get('mc_config', {}).get('n_days', 21),
                                   format_func=lambda x: f"{x}日")
            if short_term_mode and n_d > 42:
                st.caption("⚡ 短線模式建議模擬 10~21 日")

    st.markdown("---")

    # ── 操作按鈕列 ────────────────────────────────────────────
    btn1, btn2, btn3 = st.columns([2, 2, 1])
    with btn1:
        if st.button("💾 儲存設定至帳號", use_container_width=True):
            auth.save_user_settings(username, {
                'available_cash':   available_cash,
                'current_holdings': current_holdings,
                'target_pool':      target_pool,
                'ga_config':        {'population_size': pop, 'generations': gen,
                                     'crossover_rate': cr, 'mutation_rate': mr},
                'mc_config':        {'n_simulations': n_s, 'n_days': n_d},
                'top_n':            top_n,
                'short_term_mode':  short_term_mode,
            })
            st.success("✅ 設定已儲存！")
    with btn2:
        run_clicked = st.button("🚀  執行推演", type="primary", use_container_width=True)
    with btn3:
        if st.session_state.get('_has_results'):
            if st.button("🗑️  清除", use_container_width=True):
                for k in ('_results', '_config', '_has_results'):
                    st.session_state.pop(k, None)
                st.rerun()

    config = {
        'available_cash':   available_cash,
        'current_holdings': current_holdings,
        'target_pool':      target_pool,
        'ga_config':        {'population_size': pop, 'generations': gen,
                             'crossover_rate': cr, 'mutation_rate': mr},
        'mc_config':        {'n_simulations': n_s, 'n_days': n_d},
        'top_n':            top_n,
        'short_term_mode':  short_term_mode,
    }
    return config, run_clicked


def render_dashboard_tab(config: dict, quotes: dict) -> None:
    """儀表板：設定摘要、KPI 快報、持股明細、K 線、今日操作指引。"""
    ch   = config['current_holdings']
    pool = config['target_pool']
    cash = config['available_cash']

    # 快速設定摘要（折疊式，預設收合）
    with st.expander("⚙️ 目前設定摘要", expanded=not bool(ch)):
        s1, s2, s3 = st.columns(3)
        s1.metric("可用現金", f"NT${cash:,}")
        s2.metric("持股數量", f"{len(ch)} 支")
        s3.metric("候選股池", f"{len(pool)} 支")
        if ch:
            st.caption(
                "持股：" + "  ·  ".join(
                    f"**{_stock_label(c)}** × {v['shares']:,}股 @ NT${v['cost']:.1f}"
                    for c, v in list(ch.items())[:5]
                ) + ("  …" if len(ch) > 5 else "")
            )
        st.caption("如需修改請切換至 **⚙️ 設定** 頁籤。")

    st.markdown("<br>", unsafe_allow_html=True)

    # KPI 快報
    if ch:
        render_top_kpi_bar(compute_portfolio_summary(ch, quotes, cash))
    else:
        st.info("ℹ️  在「⚙️ 設定」頁輸入持股後，此處將顯示即時資產快報。")

    st.markdown("<br>", unsafe_allow_html=True)

    if ch:
        render_holdings_table(build_holdings_rows(ch, quotes))
        st.markdown("<br>", unsafe_allow_html=True)

    render_kline_section(list(dict.fromkeys(list(ch.keys()) + pool)), quotes)

    # 今日操作指引（有結果時顯示）
    if st.session_state.get('_has_results'):
        res = st.session_state['_results']
        cfg = st.session_state.get('_config', config)
        st.markdown("---")
        render_daily_guide(generate_daily_guide(
            res['holdings_analysis'], res['stock_scores'],
            cfg['current_holdings'], cfg['available_cash'], quotes,
            strategy_reasons=res.get('strategy_reasons'),
        ))


def render_analysis_tab(config: dict) -> None:
    """分析結果頁（GA + 蒙地卡羅 + 持股分析）。"""
    if not st.session_state.get('_has_results'):
        st.markdown("""
        <div style="text-align:center;padding:48px 0 32px;">
          <div style="font-size:2.4rem;margin-bottom:14px;opacity:.55;">🧬</div>
          <p style="color:#7A7A7A;font-size:.95rem;margin:0 0 6px;">
            切換至 <b style="color:#607D8B;">⚙️ 設定</b> 頁籤，點擊
            <b style="color:#B85450;">🚀 執行推演</b> 啟動 GA 優化與蒙地卡羅模擬</p>
          <p style="color:#AEAEAE;font-size:.80rem;margin:0;">
            系統將自動依你的持倉與資金，產出個性化買入建議</p>
        </div>""", unsafe_allow_html=True)
        return

    res = st.session_state['_results']
    cfg = st.session_state.get('_config', config)
    render_analysis_tabs(res, cfg)


# ═══════════════════════════════════════════════════════════════
#  Plotly 圖表函數
# ═══════════════════════════════════════════════════════════════

def _tw_color_hex(val: float) -> str:
    """台股漲跌配色（Morandi 低飽和版）。"""
    if val > 0: return '#B85450'   # 消紅 ─ 漲
    if val < 0: return '#5A8A7A'   # 消綠 ─ 跌
    return '#9E9E9E'               # 中性灰


def _arrow(val: float) -> str:
    if val > 0: return '▲'
    if val < 0: return '▼'
    return '─'


def render_kline_lwc(df: pd.DataFrame, code: str) -> None:
    """
    TradingView Lightweight Charts K 線圖。
    ‣ 支援手機觸控：單指拖移平移、雙指 pinch 縮放
    ‣ 成交量以半透明色塊疊在價格區下方 25%
    ‣ MA5 / MA20 / MA60 折線
    ‣ 十字線懸停顯示 OHLC 數值
    ‣ 無需額外 Python 套件（直接 embed Lightweight Charts v4 CDN）
    """
    import json

    df = df.tail(60).copy()
    df.index = pd.to_datetime(df.index)
    for p, col in [(5, 'MA5'), (20, 'MA20'), (60, 'MA60')]:
        df[col] = df['Close'].rolling(p).mean()

    def _ts(idx):
        return idx.strftime('%Y-%m-%d') if hasattr(idx, 'strftime') else str(idx)[:10]

    candles_js = json.dumps([
        {"time": _ts(idx),
         "open":  round(float(r['Open']),  2),
         "high":  round(float(r['High']),  2),
         "low":   round(float(r['Low']),   2),
         "close": round(float(r['Close']), 2)}
        for idx, r in df.iterrows()
    ])
    volume_js = json.dumps([
        {"time":  _ts(idx),
         "value": float(r['Volume']),
         "color": "#C9787088" if float(r['Close']) >= float(r['Open']) else "#6B9E8E88"}
        for idx, r in df.iterrows()
    ])

    def _ma_js(col):
        return json.dumps([
            {"time": _ts(idx), "value": round(float(v), 2)}
            for idx, v in df[col].dropna().items()
        ])

    ma5_js  = _ma_js('MA5')
    ma20_js = _ma_js('MA20')
    ma60_js = _ma_js('MA60')
    label   = _stock_label(code)

    html = f"""<!DOCTYPE html>
<html><head>
<meta name="viewport" content="width=device-width,initial-scale=1,user-scalable=no">
<script src="https://unpkg.com/lightweight-charts@4.1.3/dist/lightweight-charts.standalone.production.js"></script>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
html,body{{background:#FDFBF7;overflow:hidden;height:100%;width:100%}}
#chart{{width:100%;height:460px}}
#legend{{
  position:absolute;top:8px;left:12px;z-index:10;
  font-size:11px;font-family:'Inter',sans-serif;
  color:#7A7A7A;pointer-events:none;line-height:1.7
}}
</style></head>
<body>
<div style="position:relative">
  <div id="legend"></div>
  <div id="chart"></div>
</div>
<script>
const CANDLES={candles_js};
const VOLUME={volume_js};
const MA5={ma5_js};
const MA20={ma20_js};
const MA60={ma60_js};

const chart = LightweightCharts.createChart(document.getElementById('chart'), {{
  autoSize: true,
  layout: {{
    background: {{type:'solid',color:'#FDFBF7'}},
    textColor: '#4A4A4A',
    fontFamily: "'Inter', sans-serif",
  }},
  grid: {{
    vertLines: {{color:'rgba(0,0,0,0.04)'}},
    horzLines: {{color:'rgba(0,0,0,0.04)'}},
  }},
  crosshair: {{mode: 1}},
  rightPriceScale: {{
    borderColor: 'rgba(0,0,0,0.08)',
    scaleMargins: {{top:0.06, bottom:0.26}},
  }},
  timeScale: {{
    borderColor: 'rgba(0,0,0,0.08)',
    timeVisible: true,
    secondsVisible: false,
    fixRightEdge: true,
  }},
  handleScroll: {{
    mouseWheel: true,
    pressedMouseMove: true,
    horzTouchDrag: true,
    vertTouchDrag: false,
  }},
  handleScale: {{
    axisPressedMouseMove: true,
    mouseWheel: true,
    pinch: true,
  }},
}});

/* ── Candlestick ─────────────────────────────────────────── */
const candle = chart.addCandlestickSeries({{
  upColor:       '#C97870',
  downColor:     '#6B9E8E',
  borderVisible: false,
  wickUpColor:   '#B85450',
  wickDownColor: '#5A8A7A',
}});
candle.setData(CANDLES);

/* ── Volume overlay (bottom 24 % of pane) ───────────────── */
const vol = chart.addHistogramSeries({{
  priceFormat: {{type: 'volume'}},
  priceScaleId: 'vol',
}});
vol.priceScale().applyOptions({{
  scaleMargins: {{top: 0.76, bottom: 0}},
}});
vol.setData(VOLUME);

/* ── MA lines ────────────────────────────────────────────── */
const addMA = (data, color, title) => {{
  const s = chart.addLineSeries({{
    color, lineWidth: 1.4, title,
    crosshairMarkerVisible: false,
    priceLineVisible: false,
    lastValueVisible: true,
  }});
  s.setData(data);
  return s;
}};
addMA(MA5,  '#B8966A', 'MA5');
addMA(MA20, '#607D8B', 'MA20');
addMA(MA60, '#7A9E87', 'MA60');

/* ── Crosshair OHLC legend ───────────────────────────────── */
const legend = document.getElementById('legend');
chart.subscribeCrosshairMove(param => {{
  if (!param.time || !param.seriesData.size) {{
    legend.innerHTML = '';
    return;
  }}
  const d = param.seriesData.get(candle);
  if (!d) return;
  const c = d.close >= d.open ? '#B85450' : '#5A8A7A';
  legend.innerHTML =
    '<span style="color:' + c + '">'
    + 'O\u00a0' + d.open.toFixed(2)
    + '\u00a0\u00a0H\u00a0' + d.high.toFixed(2)
    + '\u00a0\u00a0L\u00a0' + d.low.toFixed(2)
    + '\u00a0\u00a0C\u00a0' + d.close.toFixed(2)
    + '</span>';
}});
</script></body></html>"""

    components.html(html, height=472, scrolling=False)


def chart_monte_carlo(mc: dict, cash: float) -> go.Figure:
    """蒙地卡羅路徑圖（Morandi 淺色主題）。"""
    paths = mc['paths']
    n, init, bk_y = mc['simulation_days'], mc['initial_portfolio_value'], cash * 0.30
    x = np.arange(n + 1)
    p05, p25, p50, p75, p95 = [np.percentile(paths, p, axis=0) for p in (5,25,50,75,95)]

    _bg   = 'rgba(0,0,0,0)'
    _plot = '#FDFBF7'
    _grid = 'rgba(0,0,0,0.05)'
    _font = '#4A4A4A'

    fig = go.Figure()
    fig.add_hrect(y0=0, y1=bk_y, fillcolor='rgba(184,84,80,0.07)', line_width=0, layer='below',
                  annotation_text="  破產風險區", annotation_position="bottom right",
                  annotation_font=dict(color='rgba(184,84,80,0.55)', size=10))
    rng = np.random.default_rng(42)
    for ix in rng.choice(len(paths), min(150, len(paths)), replace=False):
        fig.add_trace(go.Scatter(x=x, y=paths[ix], mode='lines',
                                  line=dict(color='rgba(96,125,139,0.06)', width=0.7),
                                  showlegend=False, hoverinfo='skip'))
    for y_band, fill_color, name in [
        (np.concatenate([p95, p05[::-1]]), 'rgba(96,125,139,0.08)', '90%信心帶'),
        (np.concatenate([p75, p25[::-1]]), 'rgba(96,125,139,0.20)', 'IQR帶'),
    ]:
        fig.add_trace(go.Scatter(x=np.concatenate([x, x[::-1]]), y=y_band,
                                  fill='toself', fillcolor=fill_color,
                                  line=dict(color='rgba(0,0,0,0)'), name=name, hoverinfo='skip'))
    for y_line, color, dash, name in [
        (p50, '#607D8B', 'solid', '中位數'),
        (p05, 'rgba(184,84,80,.85)', 'dot', '5th'),
        (p95, 'rgba(90,138,122,.85)', 'dot', '95th'),
    ]:
        fig.add_trace(go.Scatter(x=x, y=y_line, mode='lines',
                                  line=dict(color=color, width=2 if dash=='solid' else 1.5, dash=dash),
                                  name=name))
    for y_val, dash, color, ann_color, label, pos in [
        (init, 'dash', 'rgba(96,125,139,.40)', '#607D8B', f'初始投入 NT${init:,.0f}', 'top left'),
        (bk_y, 'solid', 'rgba(184,84,80,.60)', '#B85450', f'破產門檻 NT${bk_y:,.0f}', 'bottom left'),
    ]:
        fig.add_hline(y=y_val, line_dash=dash, line_color=color,
                      annotation_text=label, annotation_position=pos,
                      annotation_font=dict(color=ann_color, size=10))
    fig.update_layout(
        title=dict(text=f"蒙地卡羅  {mc['n_simulations']:,} 路徑 × {n} 交易日",
                   font=dict(color=_font)),
        xaxis=dict(title='交易日', gridcolor=_grid, color=_font),
        yaxis=dict(title='投組市值 (NT$)', tickformat=',.0f', gridcolor=_grid, color=_font),
        plot_bgcolor=_plot, paper_bgcolor=_bg, font=dict(color=_font),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1,
                    bgcolor='rgba(255,255,255,0)', font=dict(color=_font)),
        hovermode='x unified', height=480, margin=dict(t=65),
    )
    return fig


def chart_return_dist(mc: dict) -> go.Figure:
    """報酬率分佈直方圖（Morandi 淺色主題）。"""
    paths = mc['paths']
    init  = mc['initial_portfolio_value']
    rets  = (paths[:, -1] - init) / init * 100

    _bg   = 'rgba(0,0,0,0)'
    _plot = '#FDFBF7'
    _grid = 'rgba(0,0,0,0.05)'
    _font = '#4A4A4A'

    fig = go.Figure()
    for mask, color, name in [
        (rets < 0,  'rgba(184,84,80,.55)',  '虧損路徑'),
        (rets >= 0, 'rgba(90,138,122,.55)', '獲利路徑'),
    ]:
        fig.add_trace(go.Histogram(x=rets[mask], nbinsx=40, marker_color=color, name=name))
    for pct, col in [(5,'#B85450'),(50,'#607D8B'),(95,'#5A8A7A')]:
        v = float(np.percentile(rets, pct))
        fig.add_vline(x=v, line_dash='dash', line_color=col, line_width=1.5,
                      annotation_text=f'{pct}th:{v:.1f}%',
                      annotation_font=dict(color=col, size=10), annotation_position='top')
    fig.add_vline(x=0, line_color='rgba(0,0,0,.15)', line_width=1.2)
    fig.update_layout(
        title=dict(text='一季末報酬率分佈', font=dict(color=_font)),
        barmode='overlay',
        xaxis=dict(title='報酬率 (%)', gridcolor=_grid, color=_font),
        yaxis=dict(title='模擬次數', gridcolor=_grid, color=_font),
        plot_bgcolor=_plot, paper_bgcolor=_bg, font=dict(color=_font),
        legend=dict(orientation='h', y=1.05, bgcolor='rgba(255,255,255,0)',
                    font=dict(color=_font)),
        height=320,
    )
    return fig


def chart_scores(sorted_stocks: list) -> go.Figure:
    """GA 評分條形圖（Morandi 淺色主題）。"""
    codes  = [c for c, _ in sorted_stocks]
    scores = [s for _, s in sorted_stocks]

    _bg   = 'rgba(0,0,0,0)'
    _plot = '#FDFBF7'
    _grid = 'rgba(0,0,0,0.05)'
    _font = '#4A4A4A'

    fig = go.Figure(go.Bar(
        x=scores, y=codes, orientation='h',
        marker_color=['#C97870' if s >= 0 else '#6B9E8E' for s in scores],
        text=[f"{s:+.4f}" for s in scores],
        textposition='outside',
        textfont=dict(color=_font, size=11),
    ))
    fig.add_vline(x=0, line_color='rgba(0,0,0,.15)', line_width=1.2)
    fig.update_layout(
        title=dict(text='GA 評分（消紅=看多　消綠=看空）', font=dict(color=_font)),
        xaxis=dict(gridcolor=_grid, color=_font,
                   range=[min(scores)*1.4-.1, max(scores)*1.4+.1]),
        yaxis=dict(autorange='reversed', color=_font),
        plot_bgcolor=_plot, paper_bgcolor=_bg, font=dict(color=_font),
        height=max(260, len(codes)*42+80), margin=dict(l=70, r=110, t=50, b=40),
    )
    return fig


def chart_fitness(history: list) -> go.Figure:
    if not history:
        return go.Figure()
    gens, best_f, avg_f = zip(*[(h['generation'], h['best_fitness'], h['avg_fitness'])
                                 for h in history])

    _bg   = 'rgba(0,0,0,0)'
    _plot = '#FDFBF7'
    _grid = 'rgba(0,0,0,0.05)'
    _font = '#4A4A4A'

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=gens, y=best_f, mode='lines+markers',
                              line=dict(color='#607D8B', width=2.2),
                              marker=dict(size=3, color='#607D8B'), name='最佳'))
    fig.add_trace(go.Scatter(x=gens, y=avg_f, mode='lines',
                              line=dict(color='rgba(184,150,106,.80)', width=1.5, dash='dot'),
                              name='平均'))
    fig.update_layout(
        title=dict(text='GA 適應度收斂', font=dict(color=_font)),
        xaxis=dict(title='代數', gridcolor=_grid, color=_font),
        yaxis=dict(title='適應度', gridcolor=_grid, color=_font),
        plot_bgcolor=_plot, paper_bgcolor=_bg, font=dict(color=_font),
        legend=dict(orientation='h', y=1.08, bgcolor='rgba(255,255,255,0)',
                    font=dict(color=_font)),
        height=300,
    )
    return fig


# ═══════════════════════════════════════════════════════════════
#  UI 渲染函數
# ═══════════════════════════════════════════════════════════════

def render_top_kpi_bar(summary: dict) -> None:
    def _card(label, val_str, dlt_str, dlt_num):
        col = _tw_color_hex(dlt_num)
        arr = _arrow(dlt_num)
        return (f'<div class="kpi-wrap"><div class="kpi-lbl">{label}</div>'
                f'<div class="kpi-val">{val_str}</div>'
                f'<div class="kpi-dlt" style="color:{col};">{arr} {dlt_str}</div></div>')

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(_card("💰 總資產", f"NT${summary['total_assets']:,.0f}",
                      f"NT${summary['total_pnl']:+,.0f}  未實現損益",
                      summary['total_pnl']), unsafe_allow_html=True)
    c2.markdown(_card("📦 持倉市值", f"NT${summary['holding_value']:,.0f}",
                      f"成本 NT${summary['total_cost']:,.0f}",
                      summary['holding_value'] - summary['total_cost']), unsafe_allow_html=True)
    c3.markdown(_card("📅 今日損益", f"NT${summary['today_pnl']:+,.0f}",
                      f"今日報酬 {summary['today_return']:+.2%}",
                      summary['today_pnl']), unsafe_allow_html=True)
    c4.markdown(_card("📈 投資報酬率", f"{summary['total_return']:+.2%}",
                      f"最新交易日 {summary['last_trade_date']}",
                      summary['total_return']), unsafe_allow_html=True)
    st.caption(f"⏱ 報價快取 5 分鐘  ·  最新交易日：{summary['last_trade_date']}  ·  台股配色：紅漲綠跌")


def render_holdings_table(rows: list[dict]) -> None:
    st.markdown('<div class="section-bar">📋 HOLDINGS WATCHLIST</div>', unsafe_allow_html=True)
    if not rows:
        st.info("請在左側側邊欄輸入持股資料。")
        return

    df = pd.DataFrame([{
        '代號': r['code'], '資料日期': r['trade_date'], '持有股數': r['shares'],
        '成本均價': r['cost'], '現價': r['current'],
        '漲跌幅(%)': r['change_pct']*100, '今日損益': r['change']*r['shares'],
        '未實現損益': r['pnl'], '未實現%': r['pnl_pct']*100, '總市值': r['curr_value'],
    } for r in rows])

    def _tw(val):
        if not isinstance(val, (int, float)) or pd.isna(val): return ''
        if val > 0: return 'color:#B85450;font-weight:600'
        if val < 0: return 'color:#5A8A7A;font-weight:600'
        return 'color:#9E9E9E'

    styled = df.style
    color_cols = ['漲跌幅(%)', '今日損益', '未實現損益', '未實現%']
    try:
        styled = styled.map(_tw, subset=color_cols)
    except AttributeError:
        styled = styled.applymap(_tw, subset=color_cols)

    styled = styled.format({
        '持有股數': '{:,}', '成本均價': 'NT${:.2f}', '現價': 'NT${:.2f}',
        '漲跌幅(%)': '{:+.2f}%', '今日損益': 'NT${:+,.0f}',
        '未實現損益': 'NT${:+,.0f}', '未實現%': '{:+.2f}%', '總市值': 'NT${:,.0f}',
    }).hide(axis='index')
    st.dataframe(styled, use_container_width=True, height=min(400, len(rows)*45+50))


_ACTION_STYLE = {
    # (背景色, 左邊框色, emoji)  ── Morandi 暖調淺背景
    'STOP_LOSS': ('rgba(184,84,80,0.08)',   '#B85450', '🛑'),
    'SWITCH':    ('rgba(184,150,106,0.10)', '#B8966A', '⚡'),
    'NEW_BUY':   ('rgba(96,125,139,0.09)',  '#607D8B', '🆕'),
    'HOLD':      ('rgba(90,138,122,0.08)',  '#5A8A7A', '✅'),
    'WATCH':     ('rgba(122,110,158,0.08)', '#7A6E9E', '👁️'),
    'CASH':      ('rgba(158,158,158,0.08)', '#9E9E9E', '💰'),
}


def render_daily_guide(guide: list[dict]) -> None:
    st.markdown('<div class="section-bar">📋 TODAY\'S ACTION GUIDE</div>', unsafe_allow_html=True)
    st.caption(f"根據 GA 最佳策略 + 即時報價 + 換股成本分析  ·  {datetime.date.today()}")

    if not guide:
        st.success("✅  今日無需操作，持股狀況良好。")
        return

    st.dataframe(pd.DataFrame([{
        '優先級': item['priority'],
        '操作':   item['icon'] + '  ' + item['label'],
        '信號':   f"{item['signal']:+.3f}",
        '建議金額': f"NT${item['amount']:,.0f}" if item['amount'] else '─',
        '超額年化': f"{item['excess_return']:+.1%}" if item.get('excess_return') else '─',
        '回本時間': (f"{item['breakeven_months']:.1f}月"
                    if item.get('breakeven_months') and item['breakeven_months'] != float('inf')
                    else '─'),
    } for item in guide]), use_container_width=True, hide_index=True,
        height=min(300, len(guide)*38+40))

    st.markdown("---")
    st.markdown("##### 詳細操作說明")
    for item in guide:
        bg, border, _ = _ACTION_STYLE.get(item['action'], ('rgba(158,158,158,0.07)', '#9E9E9E', '─'))

        # ── 策略依據 badge + tag pills ─────────────────────────
        reason  = item.get('strategy_reason', {})
        pr      = reason.get('primary_reason', '')
        tags    = reason.get('tags', [])
        r_summ  = reason.get('summary', '')

        _reason_html = ''
        if pr and pr not in ('─', 'Data Limited', ''):
            _badge = (
                f'<span style="display:inline-block;background:#607D8B;color:#fff;'
                f'border-radius:20px;padding:2px 10px;font-size:.72rem;font-weight:600;'
                f'letter-spacing:.04em;margin-right:6px;">{pr}</span>'
            )
            _pills = ''.join(
                f'<span style="display:inline-block;background:rgba(96,125,139,0.10);'
                f'color:#607D8B;border:1px solid rgba(96,125,139,0.25);border-radius:12px;'
                f'padding:1px 8px;font-size:.71rem;margin-right:4px;">{t}</span>'
                for t in tags[:4]
            )
            _reason_html = f'<div style="margin-top:10px;line-height:2;">{_badge}{_pills}</div>'

        st.markdown(f"""
        <div style="background:{bg};border-left:4px solid {border};
                    border-radius:0 10px 10px 0;padding:14px 20px;margin:10px 0;">
          <div style="font-size:1.05rem;font-weight:700;color:#383838;margin-bottom:6px;">
            {item['icon']}&nbsp;&nbsp;{item['label']}</div>
          <div style="color:#5A5A5A;font-size:.9rem;line-height:1.6;">{item['summary']}</div>
          {_reason_html}
        </div>""", unsafe_allow_html=True)

        if item.get('detail'):
            with st.expander("查看詳細計算"):
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("信號強度", f"{item['signal']:+.3f}")
                if item.get('excess_return'):
                    c2.metric("超額年化", f"{item['excess_return']:+.1%}")
                if item.get('breakeven_months') and item['breakeven_months'] != float('inf'):
                    c3.metric("換股回本", f"{item['breakeven_months']:.1f} 個月")
                if item['amount']:
                    c4.metric("涉及金額", f"NT${item['amount']:,.0f}")
                st.markdown(item['detail'])

                # 技術指標細節
                sig_d = reason.get('signal_detail', {})
                if sig_d:
                    st.markdown("**技術指標數值**")
                    sd_cols = st.columns(len(sig_d))
                    for (k, v), col in zip(sig_d.items(), sd_cols):
                        col.metric(k, f"{v:.1f}" if k == 'RSI' else f"{v:+.3f}")
                    if r_summ:
                        st.caption(f"📊 {r_summ}")


# ── K 線模組（@st.fragment 隔離）──────────────────────────────

def _kline_body(available_codes: list[str], quotes: dict) -> None:
    st.markdown('<div class="section-bar">📊 INDIVIDUAL STOCK ANALYSIS</div>',
                unsafe_allow_html=True)
    if not available_codes:
        st.info("請先在側邊欄輸入股票代號。")
        return

    col_s, col_q = st.columns([2, 5])
    with col_s:
        code = st.selectbox("選擇代號", available_codes, key='kline_sel',
                             label_visibility='collapsed')
        st.caption(f"**{code}**  近 60 日 K 線")
    with col_q:
        if code in quotes:
            q   = quotes[code]
            col = _tw_color_hex(q['change_pct'])
            arr = _arrow(q['change_pct'])
            st.markdown(
                f"<span style='font-size:1.3rem;font-weight:700;color:{col};'>"
                f"NT${q['price']:.2f}&nbsp;{arr}&nbsp;{abs(q['change_pct']):.2%}"
                f"&nbsp;<span style='font-size:.85rem;'>({q['change']:+.2f})</span></span>"
                f"&nbsp;&nbsp;<span style='color:#9E9E9E;font-size:.82rem;'>"
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

    render_kline_lwc(df, code)

    recent = df.tail(60)
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("60日最高", f"NT${float(recent['High'].max()):.1f}")
    s2.metric("60日最低", f"NT${float(recent['Low'].min()):.1f}")
    s3.metric("60日均量", f"{float(recent['Volume'].mean()):,.0f}")
    last_c = float(recent['Close'].iloc[-1])
    h60w   = float(recent['High'].max())
    s4.metric("距60日高點", f"{(last_c - h60w) / h60w:.2%}", delta_color="off")


# @st.fragment 隔離 K 線區塊：切換 selectbox 只重跑此函數，
# 不觸發完整 app re-run，從根本上防止干擾正在執行的推演。
try:
    @st.fragment
    def render_kline_section(available_codes: list[str], quotes: dict) -> None:
        _kline_body(available_codes, quotes)
except AttributeError:
    # Streamlit < 1.33 降級：直接呼叫（保留 try-except 避免崩潰）
    def render_kline_section(available_codes: list[str], quotes: dict) -> None:
        _kline_body(available_codes, quotes)


def render_analysis_tabs(results: dict, config: dict) -> None:
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊  市場掃描", "🧬  GA 最佳化", "🎲  蒙地卡羅", "💼  持股分析"
    ])

    with tab1:
        ss    = results['sorted_stocks']
        sc    = results['selected_codes']
        n_pos = sum(1 for _, s in ss if s > 0)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("掃描股數", len(ss))
        m2.metric("正向信號", n_pos, delta=f"佔{n_pos/len(ss):.0%}" if ss else "")
        m3.metric("最終選股", len(sc))
        fh = results['fitness_history']
        m4.metric("GA最佳適應度", f"{fh[-1]['best_fitness']:.4f}" if fh else "─")
        st.markdown("---")
        ca, cb = st.columns([3, 2])
        with ca:
            st.plotly_chart(chart_scores(ss), use_container_width=True)
        with cb:
            st.markdown("##### 評分明細")
            mode_tag = "⚡短線" if config.get('short_term_mode') else "🔵長線"
            st.caption(f"策略模式：{mode_tag}")
            _sr_map  = results.get('strategy_reasons', {})
            _sel_map = results.get('selection_reasons', {})
            st.dataframe(pd.DataFrame([{
                '#':      i,
                '代號':   c,
                '評分':   f"{s:+.4f}",
                '信號':   "▲看多" if s > .10 else ("▼看空" if s < -.10 else "─中性"),
                '篩選依據': _sel_map.get(c, {}).get('reason_str', '─'),
                '策略依據': _sr_map.get(c, {}).get('primary_reason', '─'),
                '信號標籤': '  '.join(_sr_map.get(c, {}).get('tags', [])[:3]),
                '':       "✅" if c in sc else "",
            } for i, (c, s) in enumerate(ss, 1)]),
                use_container_width=True, hide_index=True,
                height=min(420, len(ss)*38+40))
        if results.get('failed_codes'):
            st.warning(f"⚠️  下載失敗：{results['failed_codes']}")

    with tab2:
        bp = results['best_params']
        fh = results['fitness_history']
        if fh:
            i_b, f_b = fh[0]['best_fitness'], fh[-1]['best_fitness']
            c1, c2, c3 = st.columns(3)
            c1.metric("初代適應度", f"{i_b:.4f}")
            c2.metric("末代適應度", f"{f_b:.4f}")
            c3.metric("演化改進", f"{f_b-i_b:+.4f}",
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
        mc   = results['mc_stats']
        cash = config['available_cash']
        win  = mc['win_rate']
        bk   = mc['bankruptcy_probability']
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("勝率", mc['win_rate_pct'],
                  delta="正期望" if win>.5 else "負期望，謹慎",
                  delta_color="normal" if win>.5 else "inverse")
        k2.metric("期望季度報酬", mc['expected_return_pct'],
                  delta=f"年化 {mc['expected_annualized_return_pct']}")
        k3.metric("動態破產機率", mc['bankruptcy_probability_pct'],
                  delta="偏高" if bk>.05 else "可控",
                  delta_color="inverse" if bk>.05 else "normal")
        k4.metric("平均最大回撤", mc['avg_max_drawdown_pct'])
        if win < .5: st.warning("⚠️  勝率低於 50%，建議縮小部位。")
        if bk > .05: st.error("🚨  破產機率超過 5%！請設置嚴格停損。")
        st.markdown("---")
        st.plotly_chart(chart_monte_carlo(mc, cash), use_container_width=True)
        lc, rc = st.columns([3, 2])
        with lc:
            st.plotly_chart(chart_return_dist(mc), use_container_width=True)
        with rc:
            st.markdown("##### 報酬率分位數")
            d = mc['return_distribution']
            st.dataframe(pd.DataFrame({
                '分位': ['最壞(5%)', '悲觀(25%)', '中位(50%)', '樂觀(75%)', '最佳(95%)'],
                '季報酬': [f"{v:+.2%}" for v in [d['p05'],d['p25'],d['p50'],d['p75'],d['p95']]],
            }), use_container_width=True, hide_index=True)
            st.markdown("##### 配置明細")
            st.dataframe(pd.DataFrame([{
                '代號': c, '現價': f"NT${det['price_now']:,.1f}",
                '股數': f"{det['n_shares']:,}", '投入': f"NT${det['actual_cost']:,.0f}",
            } for c, det in mc['allocation_detail'].items()]),
                use_container_width=True, hide_index=True)
            st.info(f"總投入 NT${mc['total_invested']:,.0f}\n\n剩餘現金 NT${mc['unused_cash']:,.0f}")

    with tab4:
        ha   = results['holdings_analysis']
        recs = results['recommendations']
        if not ha:
            st.info("未輸入持股。")
            return
        tv = sum(i['current_value']  for i in ha.values())
        tp = sum(i['unrealized_pnl'] for i in ha.values())
        tc = sum(i['cost_value']      for i in ha.values())
        t1, t2, t3 = st.columns(3)
        t1.metric("持股總市值", f"NT${tv:,.0f}")
        t2.metric("總未實現損益", f"NT${tp:,.0f}",
                  delta=f"{tp/tc:.2%}" if tc > 0 else "",
                  delta_color="normal" if tp >= 0 else "inverse")
        t3.metric("總成本", f"NT${tc:,.0f}")
        st.markdown("---")
        st.dataframe(pd.DataFrame([{
            '代號': c, '成本': f"NT${i['cost_price']:.1f}",
            '現價': f"NT${i['current_price']:.1f}", '股數': f"{i['shares']:,}",
            '現值': f"NT${i['current_value']:,.0f}",
            '損益%': f"{i['unrealized_pnl_pct']:+.2%}",
            '信號': f"{i['current_signal']:+.3f}",
            'Sharpe': f"{i['sharpe_ratio']:.3f}",
            '期望年化': f"{i['expected_annual_ret']:.2%}",
        } for c, i in ha.items()]),
            use_container_width=True, hide_index=True)
        st.markdown("---")
        st.markdown("##### 換股建議")
        if not recs:
            st.success("✅  持股狀況良好，無換股建議。")
        else:
            st.warning(f"⚡  {len(recs)} 筆建議")
            _ip = {'HIGH': '🔴', 'MEDIUM': '🟡', 'LOW': '🟢'}
            for i, r in enumerate(recs, 1):
                st.markdown(f"**{_ip.get(r['priority'],'─')} 建議{i}**  "
                             f"賣 **{r['sell_code']}** → 買 **{r['buy_code']}**")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric(f"{r['sell_code']}期望年化", f"{r['sell_expected_ret']:.2%}",
                          delta=f"信號{r['sell_signal']:+.3f}")
                c2.metric(f"{r['buy_code']}歷史年化", f"{r['buy_annual_ret']:.2%}",
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
    # 認證閘門
    if not render_auth_page():
        return

    username = st.session_state['_username']

    # ── 頁首：標題 + 刷新 + 登出 ──────────────────────────────
    col_h, col_btns = st.columns([5, 1])
    with col_h:
        # 從上次執行的 config 或 user_settings 讀取策略模式（供 badge 用）
        _saved   = st.session_state.get('_user_settings', {})
        _cfg     = st.session_state.get('_config', _saved)
        _is_short = _cfg.get('short_term_mode', _saved.get('short_term_mode', True))
        badge = (
            "<span style='background:#B85450;color:#fff;border-radius:20px;"
            "font-size:.70rem;padding:2px 10px;margin-left:10px;letter-spacing:.04em;'>⚡ 短線</span>"
            if _is_short else
            "<span style='background:#607D8B;color:#fff;border-radius:20px;"
            "font-size:.70rem;padding:2px 10px;margin-left:10px;letter-spacing:.04em;'>● 長線</span>"
        )
        st.markdown(
            f"<h1 style='margin:0;font-size:1.55rem;color:#383838;"
            f"font-family:Noto Serif TC,Georgia,serif;'>"
            f"📈 台股量化交易儀表板{badge}</h1>"
            f"<p style='color:#9E9E9E;font-size:.80rem;margin:4px 0 0;"
            f"font-family:Noto Sans TC,Inter,sans-serif;'>"
            f"GA × Monte Carlo × Real-time Dashboard  ·  v4.0  ·  {username}</p>",
            unsafe_allow_html=True,
        )
    with col_btns:
        st.markdown("<br>", unsafe_allow_html=True)
        cb1, cb2 = st.columns(2)
        if cb1.button("🔄", help="刷新報價"):
            fetch_realtime_quotes.clear()
            fetch_kline_data.clear()
            st.rerun()
        if cb2.button("登出"):
            for k in ('_username', '_user_settings', '_results', '_config', '_has_results'):
                st.session_state.pop(k, None)
            st.rerun()

    st.markdown("---")

    # ── 主體三標籤 ────────────────────────────────────────────
    tab_dash, tab_analysis, tab_settings = st.tabs(["📊 儀表板", "📈 分析", "⚙️ 設定"])

    # 設定標籤優先執行（取得最新 config + run_clicked）
    with tab_settings:
        config, run_clicked = render_settings_tab(username)

    ch   = config['current_holdings']
    pool = config['target_pool']
    cash = config['available_cash']

    # 即時報價
    all_codes = tuple(sorted(set(list(ch.keys()) + pool)))
    quotes: dict = {}
    if all_codes:
        with st.spinner("⚡ 獲取即時報價..."):
            quotes = fetch_realtime_quotes(all_codes)

    with tab_dash:
        render_dashboard_tab(config, quotes)

    with tab_analysis:
        render_analysis_tab(config)

    # ── 執行推演（在標籤外執行，進度條顯示於頁底）────────────
    if run_clicked:
        if not pool:
            st.error("❌  請先設定候選股池。")
            return
        for k in ('_results', '_config', '_has_results'):
            st.session_state.pop(k, None)

        prog = st.progress(0)
        stat = st.empty()

        st.markdown(
            "<div class='section-bar'>🧠 THINKING LOG  ─  AI 推理過程監控</div>",
            unsafe_allow_html=True,
        )
        think_container = st.empty()
        logger = ThinkingLogger(think_container)

        try:
            # 嘗試複用漏斗快取的已下載數據（避免重複下載）
            _, funnel_stock_data, funnel_reasons = fetch_funnel_pool()
            if funnel_stock_data:
                # 補下載漏斗未覆蓋的持股（若有）
                missing_codes = tuple(
                    c for c in ch.keys() if c not in funnel_stock_data
                )
                holdings_extra = fetch_pool_history(missing_codes) if missing_codes else {}
                stock_data = {**funnel_stock_data, **holdings_extra}
                # 以實際下載成功的代號作為分析股池
                analysis_pool = [c for c in pool if c in stock_data]
            else:
                # Fallback：下載用戶指定的股池
                all_hist_codes = tuple(sorted(set(list(ch.keys()) + pool)))
                stock_data = fetch_pool_history(all_hist_codes)
                funnel_reasons = {}
                analysis_pool = pool

            results = run_full_pipeline(
                cash, ch, analysis_pool, stock_data,
                config['ga_config'], config['mc_config'], config['top_n'],
                config['short_term_mode'],
                prog, stat,
                thinking_logger=logger,
                funnel_reasons=funnel_reasons,
            )
            st.session_state.update({
                '_results': results, '_config': config, '_has_results': True,
            })
            prog.empty(); stat.empty()
            st.success("✅  分析完成！請切換至 📈 分析 頁籤查看結果。")
            st.rerun()

        except Exception as e:
            prog.empty(); stat.empty()
            st.error(f"❌  {e}")
            with st.expander("錯誤詳情"):
                st.exception(e)
            return

    st.markdown("---")
    st.caption("⚠️ 本系統僅供學術研究，不構成投資建議。過去績效不代表未來報酬。")


if __name__ == '__main__':
    main()
