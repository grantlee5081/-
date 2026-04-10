"""
app.py  ─  v4.0  實戰交易儀表板（UI 層）
─────────────────────────────────────────
架構說明：
  app.py   → Streamlit UI（頁面、圖表、表單）
  engine.py → 業務邏輯（Pipeline、決策引擎、ThinkingLogger）
  auth.py   → 用戶認證（登入/註冊/設定持久化）

台股配色：漲 = 紅 #e84545   跌 = 綠 #26a69a
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
.stApp { background-color: #0a0e1a; }

.kpi-wrap {
    background: linear-gradient(135deg,#131929,#1a2035);
    border:1px solid rgba(76,155,232,.20);
    border-radius:12px; padding:16px 20px 12px;
    text-align:center; position:relative; overflow:hidden;
}
.kpi-wrap::before {
    content:''; position:absolute; top:0; left:0; right:0; height:2px;
    background:linear-gradient(90deg,#4c9be8,#7c4dff);
}
.kpi-lbl { font-size:.75rem; color:#6b7fa8; letter-spacing:.08em; margin-bottom:4px; }
.kpi-val { font-size:1.5rem; font-weight:700; color:#e0e8f8; line-height:1.2; }
.kpi-dlt { font-size:.80rem; margin-top:4px; }
.tw-red  { color:#e84545; }  .tw-green { color:#26a69a; }  .tw-gray { color:#6b7fa8; }

.section-bar {
    font-size:.76rem; font-weight:600; color:#4c9be8;
    letter-spacing:.12em; text-transform:uppercase;
    display:flex; align-items:center; gap:8px; margin:0 0 10px;
}
.section-bar::after {
    content:''; flex:1; height:1px;
    background:linear-gradient(90deg,rgba(76,155,232,.35),transparent);
}

div[data-testid="metric-container"] {
    background:#131929; border:1px solid rgba(76,155,232,.18);
    border-radius:10px; padding:12px 16px;
}

.think-box {
    background:#0d1117; border:1px solid rgba(76,155,232,.25);
    border-radius:8px; padding:12px 16px;
    font-family:'Courier New',monospace; font-size:.82rem;
    color:#7ec8e3; max-height:220px; overflow-y:auto; line-height:1.7;
}
</style>
""", unsafe_allow_html=True)


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
                raw = yf.download(
                    f"{code}{suffix}", period='5d', interval='1d',
                    progress=False, auto_adjust=True,
                )
                if raw.empty:
                    continue
                if isinstance(raw.columns, pd.MultiIndex):
                    raw.columns = [c[0] for c in raw.columns]
                raw = raw.dropna(subset=['Close']).sort_index()
                if len(raw) < 1:
                    continue
                curr  = float(raw['Close'].iloc[-1])
                prev  = float(raw['Close'].iloc[-2]) if len(raw) >= 2 else curr
                chg   = curr - prev
                chg_p = chg / prev if prev != 0 else 0.0
                quotes[code] = {
                    'price': curr, 'prev_close': prev,
                    'change': chg, 'change_pct': chg_p,
                    'volume': float(raw['Volume'].iloc[-1]),
                    'trade_date': raw.index[-1].strftime('%Y-%m-%d'),
                    'suffix': suffix,
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
                progress=False, auto_adjust=True,
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
    """下載股票池 1 年歷史數據（短線模式縮短週期），快取 1 小時。"""
    fetcher = DataFetcher(period='1y')
    return fetcher.fetch_multiple(list(codes))


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
      <h2 style="color:#e0e8f8;margin:8px 0 4px;">台股量化交易儀表板</h2>
      <p style="color:#6b7fa8;font-size:.85rem;">請先登入或建立帳號</p>
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
#  側邊欄（含用戶設定 + 資產導向股池 + 短線模式切換）
# ═══════════════════════════════════════════════════════════════

def render_sidebar(username: str) -> dict:
    with st.sidebar:
        # 用戶資訊列
        col_u, col_out = st.columns([3, 1])
        col_u.markdown(
            f"<div style='color:#4c9be8;font-size:.85rem;font-weight:600;'>"
            f"👤  {username}</div>", unsafe_allow_html=True,
        )
        if col_out.button("登出", key="btn_logout"):
            for k in ('_username', '_user_settings', '_results', '_config', '_has_results'):
                st.session_state.pop(k, None)
            st.rerun()

        st.markdown("---")
        st.markdown("## ⚙️ 策略設定")
        saved = st.session_state.get('_user_settings', {})

        # 可用現金
        st.markdown("#### 💰 可用現金（NT$）")
        available_cash = st.number_input(
            "cash", min_value=10_000, max_value=100_000_000,
            value=int(saved.get('available_cash', 500_000)),
            step=10_000, format="%d", label_visibility='collapsed',
        )
        st.caption(f"NT$ {available_cash:,}")
        st.markdown("---")

        # 現有持股（data_editor）
        st.markdown("#### 📋 現有持股")
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
            key='holdings_editor',
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

        # 候選股池（資產導向自動生成 or 手動）
        st.markdown("#### 🎯 候選股池")
        auto_pool = generate_asset_driven_pool(available_cash, current_holdings)
        pool_mode = st.radio(
            "pool_mode", ["🤖 資產導向（自動）", "✏️ 手動輸入"],
            index=0, label_visibility='collapsed', horizontal=True,
        )
        if pool_mode == "🤖 資產導向（自動）":
            target_pool = auto_pool
            st.caption(
                f"依持倉結構自動推薦 {len(target_pool)} 支：\n"
                f"`{'、'.join(target_pool[:6])}{'…' if len(target_pool) > 6 else ''}`"
            )
            if st.button("🔄 重新生成", key="regen_pool"):
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

        # 交易策略模式
        st.markdown("#### ⚡ 交易策略")
        short_term_mode = st.toggle(
            "短線優先（KDJ + 量能強化）",
            value=saved.get('short_term_mode', True),
            help="開啟：短均線 + KDJ + 成交量爆發（波段/當沖）\n"
                 "關閉：長均線 + 技術信號（季線持有）",
        )
        st.caption("🔴 短線模式：GA 縮短週期，KDJ + 量能加權 40%"
                   if short_term_mode else
                   "🔵 長線模式：GA 標準週期，純技術信號")
        st.markdown("---")

        # 進階參數
        with st.expander("🧬 遺傳演算法參數"):
            pop = st.slider("種群大小", 20, 100, int(saved.get('ga_config', {}).get('population_size', 50)), step=10)
            gen = st.slider("演化代數", 20, 100, int(saved.get('ga_config', {}).get('generations', 50)), step=10)
            cr  = st.slider("交叉率",   0.50, 1.00, float(saved.get('ga_config', {}).get('crossover_rate', 0.80)), step=0.05)
            mr  = st.slider("變異率",   0.05, 0.30, float(saved.get('ga_config', {}).get('mutation_rate', 0.15)), step=0.05)
        with st.expander("🎲 蒙地卡羅參數"):
            n_s = st.select_slider("路徑數",   options=[500, 1000, 2000, 5000],
                                   value=saved.get('mc_config', {}).get('n_simulations', 1000))
            n_d = st.select_slider("模擬天數", options=[10, 21, 42, 63, 126],
                                   value=saved.get('mc_config', {}).get('n_days', 21),
                                   format_func=lambda x: f"{x}日")
            if short_term_mode and n_d > 42:
                st.caption("⚡ 短線模式建議模擬 10~21 日")
        top_n = st.slider("📌 最終選股數", 1, 6, int(saved.get('top_n', 3)))
        st.markdown("---")

        # 儲存設定
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

        run_clicked = st.button("🚀  執行推演", type="primary", use_container_width=True)
        if st.session_state.get('_has_results'):
            if st.button("🗑️  清除結果", use_container_width=True):
                for k in ('_results', '_config', '_has_results'):
                    st.session_state.pop(k, None)
                st.rerun()

    return {
        'available_cash':   available_cash,
        'current_holdings': current_holdings,
        'target_pool':      target_pool,
        'ga_config':        {'population_size': pop, 'generations': gen,
                             'crossover_rate': cr, 'mutation_rate': mr},
        'mc_config':        {'n_simulations': n_s, 'n_days': n_d},
        'top_n':            top_n,
        'short_term_mode':  short_term_mode,
        'run_clicked':      run_clicked,
    }


# ═══════════════════════════════════════════════════════════════
#  Plotly 圖表函數
# ═══════════════════════════════════════════════════════════════

def _tw_color_hex(val: float) -> str:
    if val > 0: return '#e84545'
    if val < 0: return '#26a69a'
    return '#6b7fa8'


def _arrow(val: float) -> str:
    if val > 0: return '▲'
    if val < 0: return '▼'
    return '─'


def chart_kline(df: pd.DataFrame, code: str) -> go.Figure:
    """台股 K 線圖（60日）+ MA5/20/60 + 成交量副圖。"""
    df = df.tail(60).copy()
    for p, c in [(5, 'MA5'), (20, 'MA20'), (60, 'MA60')]:
        df[c] = df['Close'].rolling(p).mean()
    vol_colors = ['#e84545' if c >= o else '#26a69a'
                  for c, o in zip(df['Close'], df['Open'])]

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.75, 0.25], vertical_spacing=0.025)
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'],
        low=df['Low'], close=df['Close'],
        increasing_line_color='#e84545', increasing_fillcolor='#e84545',
        decreasing_line_color='#26a69a', decreasing_fillcolor='#26a69a',
        line_width=1, name='K線', showlegend=False,
    ), row=1, col=1)
    for col_n, color, name in [('MA5','#ffd700','MA5'),('MA20','#4c9be8','MA20'),('MA60','#ff8c00','MA60')]:
        fig.add_trace(go.Scatter(x=df.index, y=df[col_n], mode='lines',
                                  line=dict(color=color, width=1.4), name=name), row=1, col=1)
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], marker_color=vol_colors,
                          marker_line_width=0, name='成交量', showlegend=False, opacity=0.85),
                  row=2, col=1)
    fig.add_hline(y=float(df['Close'].iloc[-1]), row=1, col=1,
                  line_dash='dot', line_color='rgba(255,255,255,0.22)', line_width=1)
    _d = 'rgba(14,17,23,1)'
    fig.update_layout(
        title=dict(text=f'<b>{code}</b>　近 60 日 K 線', font=dict(size=14, color='#e0e8f8')),
        xaxis_rangeslider_visible=False,
        plot_bgcolor=_d, paper_bgcolor='rgba(14,17,23,0)',
        font=dict(color='#c8d4e8', size=11), height=500,
        margin=dict(l=60, r=40, t=50, b=30),
        legend=dict(orientation='h', y=1.01, x=0, bgcolor='rgba(0,0,0,0)', font=dict(size=11)),
        hovermode='x unified',
        yaxis=dict(title='股價 (NT$)', gridcolor='rgba(255,255,255,0.06)', tickformat=',.1f', side='right'),
        yaxis2=dict(title='成交量', gridcolor='rgba(255,255,255,0.06)', tickformat='.2s', side='right'),
    )
    fig.update_xaxes(gridcolor='rgba(255,255,255,0.06)')
    return fig


def chart_monte_carlo(mc: dict, cash: float) -> go.Figure:
    paths = mc['paths']
    n, init, bk_y = mc['simulation_days'], mc['initial_portfolio_value'], cash * 0.30
    x = np.arange(n + 1)
    p05, p25, p50, p75, p95 = [np.percentile(paths, p, axis=0) for p in (5,25,50,75,95)]

    fig = go.Figure()
    fig.add_hrect(y0=0, y1=bk_y, fillcolor='rgba(220,50,50,0.10)', line_width=0, layer='below',
                  annotation_text="  破產風險區", annotation_position="bottom right",
                  annotation_font=dict(color='rgba(255,100,100,0.5)', size=10))
    rng = np.random.default_rng(42)
    for ix in rng.choice(len(paths), min(150, len(paths)), replace=False):
        fig.add_trace(go.Scatter(x=x, y=paths[ix], mode='lines',
                                  line=dict(color='rgba(130,170,230,0.06)', width=0.7),
                                  showlegend=False, hoverinfo='skip'))
    for y_band, fill_color, name in [
        (np.concatenate([p95, p05[::-1]]), 'rgba(76,155,232,0.10)', '90%信心帶'),
        (np.concatenate([p75, p25[::-1]]), 'rgba(76,155,232,0.25)', 'IQR帶'),
    ]:
        fig.add_trace(go.Scatter(x=np.concatenate([x, x[::-1]]), y=y_band,
                                  fill='toself', fillcolor=fill_color,
                                  line=dict(color='rgba(0,0,0,0)'), name=name, hoverinfo='skip'))
    for y_line, color, dash, name in [
        (p50, '#4c9be8', 'solid', '中位數'),
        (p05, 'rgba(232,69,69,.8)', 'dot', '5th'),
        (p95, 'rgba(38,166,154,.8)', 'dot', '95th'),
    ]:
        fig.add_trace(go.Scatter(x=x, y=y_line, mode='lines',
                                  line=dict(color=color, width=2 if dash=='solid' else 1.5, dash=dash),
                                  name=name))
    for y_val, dash, color, label, pos in [
        (init, 'dash', 'rgba(255,255,255,.35)', f'初始投入 NT${init:,.0f}', 'top left'),
        (bk_y, 'solid', 'rgba(232,69,69,.75)', f'破產門檻 NT${bk_y:,.0f}', 'bottom left'),
    ]:
        fig.add_hline(y=y_val, line_dash=dash, line_color=color,
                      annotation_text=label, annotation_position=pos,
                      annotation_font=dict(color=color, size=10))
    _d = 'rgba(14,17,23,1)'
    fig.update_layout(
        title=f"蒙地卡羅  {mc['n_simulations']:,} 路徑 × {n} 交易日",
        xaxis=dict(title='交易日', gridcolor='rgba(255,255,255,.06)'),
        yaxis=dict(title='投組市值 (NT$)', tickformat=',.0f', gridcolor='rgba(255,255,255,.06)'),
        plot_bgcolor=_d, paper_bgcolor='rgba(14,17,23,0)', font=dict(color='white'),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1, bgcolor='rgba(0,0,0,0)'),
        hovermode='x unified', height=480, margin=dict(t=65),
    )
    return fig


def chart_return_dist(mc: dict) -> go.Figure:
    paths = mc['paths']
    init  = mc['initial_portfolio_value']
    rets  = (paths[:, -1] - init) / init * 100
    _d = 'rgba(14,17,23,1)'
    fig = go.Figure()
    for mask, color, name in [(rets < 0, 'rgba(232,69,69,.65)', '虧損路徑'),
                               (rets >= 0, 'rgba(38,166,154,.65)', '獲利路徑')]:
        fig.add_trace(go.Histogram(x=rets[mask], nbinsx=40, marker_color=color, name=name))
    for pct, col in [(5,'#e84545'),(50,'white'),(95,'#26a69a')]:
        v = float(np.percentile(rets, pct))
        fig.add_vline(x=v, line_dash='dash', line_color=col, line_width=1.5,
                      annotation_text=f'{pct}th:{v:.1f}%',
                      annotation_font=dict(color=col, size=10), annotation_position='top')
    fig.add_vline(x=0, line_color='rgba(255,255,255,.25)', line_width=1.2)
    fig.update_layout(
        title='一季末報酬率分佈', barmode='overlay',
        xaxis=dict(title='報酬率 (%)', gridcolor='rgba(255,255,255,.06)'),
        yaxis=dict(title='模擬次數', gridcolor='rgba(255,255,255,.06)'),
        plot_bgcolor=_d, paper_bgcolor='rgba(14,17,23,0)', font=dict(color='white'),
        legend=dict(orientation='h', y=1.05), height=320,
    )
    return fig


def chart_scores(sorted_stocks: list) -> go.Figure:
    codes  = [c for c, _ in sorted_stocks]
    scores = [s for _, s in sorted_stocks]
    _d = 'rgba(14,17,23,1)'
    fig = go.Figure(go.Bar(
        x=scores, y=codes, orientation='h',
        marker_color=['#e84545' if s >= 0 else '#26a69a' for s in scores],
        text=[f"{s:+.4f}" for s in scores], textposition='outside',
    ))
    fig.add_vline(x=0, line_color='rgba(255,255,255,.18)', line_width=1.2)
    fig.update_layout(
        title='GA 評分（紅=看多 綠=看空）',
        xaxis=dict(gridcolor='rgba(255,255,255,.06)',
                   range=[min(scores)*1.4-.1, max(scores)*1.4+.1]),
        yaxis=dict(autorange='reversed'),
        plot_bgcolor=_d, paper_bgcolor='rgba(14,17,23,0)', font=dict(color='white'),
        height=max(260, len(codes)*42+80), margin=dict(l=70, r=110, t=50, b=40),
    )
    return fig


def chart_fitness(history: list) -> go.Figure:
    if not history:
        return go.Figure()
    gens, best_f, avg_f = zip(*[(h['generation'], h['best_fitness'], h['avg_fitness'])
                                 for h in history])
    _d = 'rgba(14,17,23,1)'
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=gens, y=best_f, mode='lines+markers',
                              line=dict(color='#4c9be8', width=2.2), marker=dict(size=3), name='最佳'))
    fig.add_trace(go.Scatter(x=gens, y=avg_f, mode='lines',
                              line=dict(color='rgba(255,215,60,.75)', width=1.5, dash='dot'), name='平均'))
    fig.update_layout(
        title='GA 適應度收斂',
        xaxis=dict(title='代數', gridcolor='rgba(255,255,255,.06)'),
        yaxis=dict(title='適應度', gridcolor='rgba(255,255,255,.06)'),
        plot_bgcolor=_d, paper_bgcolor='rgba(14,17,23,0)', font=dict(color='white'),
        legend=dict(orientation='h', y=1.08), height=300,
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
        if val > 0: return 'color:#e84545;font-weight:600'
        if val < 0: return 'color:#26a69a;font-weight:600'
        return 'color:#6b7fa8'

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
    'STOP_LOSS': ('rgba(50,20,20,0.85)',  '#e84545', '🛑'),
    'SWITCH':    ('rgba(40,35,10,0.85)',  '#ffd700', '⚡'),
    'NEW_BUY':   ('rgba(15,25,45,0.85)', '#4c9be8', '🆕'),
    'HOLD':      ('rgba(15,35,25,0.85)', '#26a69a', '✅'),
    'WATCH':     ('rgba(25,25,35,0.85)', '#8888cc', '👁️'),
    'CASH':      ('rgba(20,22,32,0.85)', '#6b7fa8', '💰'),
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
        bg, border, _ = _ACTION_STYLE.get(item['action'], ('rgba(20,22,32,.85)', '#6b7fa8', '─'))
        st.markdown(f"""
        <div style="background:{bg};border-left:4px solid {border};
                    border-radius:0 10px 10px 0;padding:14px 20px;margin:10px 0;">
          <div style="font-size:1.05rem;font-weight:700;color:#e0e8f8;margin-bottom:6px;">
            {item['icon']}&nbsp;&nbsp;{item['label']}</div>
          <div style="color:#c8d4e8;font-size:.9rem;line-height:1.6;">{item['summary']}</div>
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
                f"&nbsp;&nbsp;<span style='color:#6b7fa8;font-size:.82rem;'>"
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

    st.plotly_chart(chart_kline(df, code), use_container_width=True)

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
            st.dataframe(pd.DataFrame([{
                '#': i, '代號': c, '評分': f"{s:+.4f}",
                '信號': "▲看多" if s > .10 else ("▼看空" if s < -.10 else "─中性"),
                '': "✅" if c in sc else "",
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
    config   = render_sidebar(username)
    ch       = config['current_holdings']
    pool     = config['target_pool']
    cash     = config['available_cash']

    # 即時報價（頁面載入即執行，與 GA 無關）
    all_codes = tuple(sorted(set(list(ch.keys()) + pool)))
    with st.spinner("⚡ 獲取即時報價..."):
        quotes = fetch_realtime_quotes(all_codes)

    # 標題列
    col_h, col_r = st.columns([5, 1])
    with col_h:
        badge = (
            "<span style='background:#e84545;color:#fff;border-radius:4px;"
            "font-size:.72rem;padding:2px 8px;margin-left:8px;'>⚡ 短線</span>"
            if config['short_term_mode'] else
            "<span style='background:#4c9be8;color:#fff;border-radius:4px;"
            "font-size:.72rem;padding:2px 8px;margin-left:8px;'>🔵 長線</span>"
        )
        st.markdown(
            f"<h1 style='margin:0;font-size:1.55rem;color:#e0e8f8;'>"
            f"📈 台股量化交易儀表板{badge}</h1>"
            f"<p style='color:#6b7fa8;font-size:.8rem;margin:2px 0 0;'>"
            f"GA × Monte Carlo × Real-time Dashboard  |  v4.0  ·  {username}</p>",
            unsafe_allow_html=True,
        )
    with col_r:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔄 刷新報價"):
            fetch_realtime_quotes.clear()
            fetch_kline_data.clear()
            st.rerun()

    st.markdown("---")

    if ch:
        render_top_kpi_bar(compute_portfolio_summary(ch, quotes, cash))
    else:
        st.info("ℹ️  在左側輸入持股後，此處將顯示即時資產快報。")

    st.markdown("<br>", unsafe_allow_html=True)

    if ch:
        render_holdings_table(build_holdings_rows(ch, quotes))

    st.markdown("<br>", unsafe_allow_html=True)

    # K 線（Fragment 隔離）
    render_kline_section(list(dict.fromkeys(list(ch.keys()) + pool)), quotes)

    st.markdown("---")

    # 執行推演
    if config['run_clicked']:
        if not pool:
            st.error("❌  請先設定候選股池。")
            return
        for k in ('_results', '_config', '_has_results'):
            st.session_state.pop(k, None)

        prog = st.progress(0)
        stat = st.empty()

        # 思考鏈監控視窗
        st.markdown(
            "<div style='font-size:.78rem;font-weight:600;color:#4c9be8;"
            "letter-spacing:.1em;margin:12px 0 6px;'>"
            "🧠 THINKING LOG  ─  AI 推理過程監控</div>",
            unsafe_allow_html=True,
        )
        think_container = st.empty()
        logger = ThinkingLogger(think_container)

        try:
            # 下載歷史數據（帶快取）
            all_hist_codes = tuple(sorted(set(list(ch.keys()) + pool)))
            stock_data = fetch_pool_history(all_hist_codes)

            results = run_full_pipeline(
                cash, ch, pool, stock_data,
                config['ga_config'], config['mc_config'], config['top_n'],
                config['short_term_mode'],
                prog, stat,
                thinking_logger=logger,
            )
            st.session_state.update({
                '_results': results, '_config': config, '_has_results': True,
            })
            prog.empty(); stat.empty()
            st.success("✅  分析完成！")

        except Exception as e:
            prog.empty(); stat.empty()
            st.error(f"❌  {e}")
            with st.expander("錯誤詳情"):
                st.exception(e)
            return

    # 結果展示
    if st.session_state.get('_has_results'):
        res = st.session_state['_results']
        cfg = st.session_state.get('_config', config)

        render_daily_guide(generate_daily_guide(
            res['holdings_analysis'], res['stock_scores'],
            cfg['current_holdings'], cfg['available_cash'], quotes,
        ))
        st.markdown("---")
        render_analysis_tabs(res, cfg)
    else:
        st.markdown("""
        <div style="text-align:center;padding:40px 0;color:#4a5568;">
          <div style="font-size:2.5rem;margin-bottom:10px;">🧬</div>
          <p>點擊左側 <b style="color:#4c9be8;">🚀 執行推演</b> 啟動 GA 優化與蒙地卡羅模擬</p>
          <p style="font-size:.82rem;">系統將自動依你的持倉與資金，產出個性化買入建議</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.caption("⚠️ 本系統僅供學術研究，不構成投資建議。過去績效不代表未來報酬。")


if __name__ == '__main__':
    main()
