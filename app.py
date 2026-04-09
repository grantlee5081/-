"""
app.py  ─  v2.0  Trading Dashboard Edition
───────────────────────────────────────────
新增功能（v2.0）：
  ① 頂部資產快報      ─ 總資產 / 今日盈虧 / 投資報酬率（台股紅漲綠跌配色）
  ② 持股即時看板      ─ @st.cache_data 快取，含現價 / 漲跌幅 / 損益 / 市值
  ③ 個股 K 線模組     ─ go.Candlestick 60日K線 + MA5/20/60 + 成交量副圖
  ④ 效能優化          ─ 所有 yfinance 呼叫皆有 5 分鐘快取，避免被封鎖

台股配色慣例：
  漲 (price ↑) = 紅色 #e84545
  跌 (price ↓) = 綠色 #26a69a
"""

# ── 標準函式庫 ─────────────────────────────────────────────────
import io
import datetime
from contextlib import redirect_stdout

# ── 第三方套件 ─────────────────────────────────────────────────
import yfinance as yf
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── 後端模組 ───────────────────────────────────────────────────
from data_fetcher import DataFetcher
from genetic_algorithm import GeneticAlgorithm
from monte_carlo import MonteCarloSimulator
from holdings_analyzer import HoldingsAnalyzer
from performance_metrics import PerformanceMetrics
from technical_factors import TechnicalFactors


# ═══════════════════════════════════════════════════════════════
#  頁面設定（必須是第一個 Streamlit 指令）
# ═══════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="台股量化交易儀表板",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Trading Dashboard 專屬 CSS ─────────────────────────────────
st.markdown("""
<style>
/* ── 全域底色 ── */
.stApp { background-color: #0a0e1a; }

/* ── 頂部資產快報卡片 ── */
.kpi-card {
    background: linear-gradient(135deg, #131929 0%, #1a2035 100%);
    border: 1px solid rgba(76,155,232,0.20);
    border-radius: 12px;
    padding: 18px 22px 14px;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #4c9be8, #7c4dff);
}
.kpi-label  { font-size: 0.78rem; color: #6b7fa8; letter-spacing: 0.08em; margin-bottom: 4px; }
.kpi-value  { font-size: 1.55rem; font-weight: 700; color: #e0e8f8; line-height: 1.2; }
.kpi-delta  { font-size: 0.82rem; margin-top: 4px; }
.tw-red     { color: #e84545; }
.tw-green   { color: #26a69a; }
.tw-gray    { color: #6b7fa8; }
.tw-white   { color: #c8d4e8; }

/* ── 持股表格 ── */
.holdings-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.88rem;
}
.holdings-table th {
    background: #131929;
    color: #6b7fa8;
    font-weight: 500;
    padding: 10px 14px;
    border-bottom: 1px solid #1e2b40;
    letter-spacing: 0.05em;
}
.holdings-table th:not(:first-child) { text-align: right; }
.holdings-table td {
    padding: 11px 14px;
    border-bottom: 1px solid #141a2a;
    color: #c8d4e8;
}
.holdings-table td:not(:first-child) { text-align: right; }
.holdings-table tr:hover td { background: rgba(76,155,232,0.05); }
.stock-code {
    font-weight: 700;
    color: #e0e8f8;
    font-size: 0.95rem;
    letter-spacing: 0.02em;
}

/* ── 區塊標題 ── */
.dash-section {
    font-size: 0.78rem;
    font-weight: 600;
    color: #4c9be8;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin: 0 0 10px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.dash-section::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, rgba(76,155,232,0.3), transparent);
}

/* ── Streamlit metric 卡片 ── */
div[data-testid="metric-container"] {
    background: #131929;
    border: 1px solid rgba(76,155,232,0.18);
    border-radius: 10px;
    padding: 14px 18px;
}

/* ── 換股建議卡片 ── */
.rec-card {
    background: #131929;
    border-left: 3px solid #ffd700;
    border-radius: 0 8px 8px 0;
    padding: 14px 18px;
    margin-bottom: 14px;
}

/* ── 分頁容器加底色 ── */
div[data-testid="stTabs"] > div > div > div {
    background: transparent;
}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
#  ① 快取數據函數（@st.cache_data，前端唯一與 Yahoo Finance 的橋接點）
# ═══════════════════════════════════════════════════════════════

@st.cache_data(ttl=300, show_spinner=False)
def fetch_realtime_quotes(codes: tuple[str, ...]) -> dict[str, dict]:
    """
    批次獲取即時報價（快取 5 分鐘）。

    設計重點：
      - 參數使用 tuple 確保可雜湊（hashable），滿足 @st.cache_data 要求
      - 依序嘗試 .TW / .TWO 後綴，成功即跳出
      - 下載最近 5 個交易日資料（period='5d'），
        確保即使遇到週末 / 假日仍有至少兩個有效交易日
      - 返回 dict 包含：price / prev_close / change / change_pct / volume

    Returns
    -------
    {代號: {price, prev_close, change, change_pct, volume}} 或空 dict
    """
    quotes: dict[str, dict] = {}
    for code in codes:
        for suffix in ('.TW', '.TWO'):
            try:
                raw = yf.download(
                    f"{code}{suffix}",
                    period='5d',
                    interval='1d',
                    progress=False,
                    auto_adjust=True,
                )
                if raw.empty:
                    continue
                # 統一多層欄位
                if isinstance(raw.columns, pd.MultiIndex):
                    raw.columns = [c[0] for c in raw.columns]
                raw = raw.dropna(subset=['Close'])
                if len(raw) < 1:
                    continue
                curr  = float(raw['Close'].iloc[-1])
                prev  = float(raw['Close'].iloc[-2]) if len(raw) >= 2 else curr
                chg   = curr - prev
                chg_p = chg / prev if prev != 0 else 0.0
                quotes[code] = {
                    'price':      curr,
                    'prev_close': prev,
                    'change':     chg,
                    'change_pct': chg_p,
                    'volume':     float(raw['Volume'].iloc[-1]),
                    'suffix':     suffix,
                }
                break
            except Exception:
                continue
    return quotes


@st.cache_data(ttl=300, show_spinner=False)
def fetch_kline_data(code: str) -> pd.DataFrame | None:
    """
    獲取個股 K 線原始資料（快取 5 分鐘）。

    下載近 3 個月的日線資料（約 63 個交易日），
    前端只截取最後 60 根 K 棒顯示。
    自動嘗試 .TW / .TWO 後綴。

    Returns
    -------
    OHLCV DataFrame 或 None（找不到時）
    """
    for suffix in ('.TW', '.TWO'):
        try:
            raw = yf.download(
                f"{code}{suffix}",
                period='3mo',
                interval='1d',
                progress=False,
                auto_adjust=True,
            )
            if raw.empty:
                continue
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = [c[0] for c in raw.columns]
            raw.index = pd.to_datetime(raw.index)
            raw = raw.dropna(subset=['Open', 'High', 'Low', 'Close'])
            if len(raw) < 5:
                continue
            return raw
        except Exception:
            continue
    return None


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_pool_history(codes: tuple[str, ...]) -> dict[str, pd.DataFrame]:
    """
    批次下載股票池歷史數據（快取 1 小時）。
    供 GA 優化與蒙地卡羅模擬使用。
    與 DataFetcher 邏輯一致，但加上快取層避免重複下載。
    """
    fetcher = DataFetcher(period='2y')
    return fetcher.fetch_multiple(list(codes))


# ═══════════════════════════════════════════════════════════════
#  ② 數據處理工具函數
# ═══════════════════════════════════════════════════════════════

def _tw_color_class(val: float) -> str:
    """返回台股配色 CSS class：正值=紅(tw-red), 負值=綠(tw-green), 零=灰。"""
    if val > 0:   return 'tw-red'
    if val < 0:   return 'tw-green'
    return 'tw-gray'

def _tw_color_hex(val: float) -> str:
    """返回台股配色十六進位色碼。"""
    if val > 0:   return '#e84545'
    if val < 0:   return '#26a69a'
    return '#6b7fa8'

def _arrow(val: float) -> str:
    if val > 0: return '▲'
    if val < 0: return '▼'
    return '─'


def compute_portfolio_summary(
        current_holdings: dict,
        quotes: dict,
        available_cash: float,
) -> dict:
    """
    計算頂部資產快報所需的四個指標。

    Returns
    -------
    {total_assets, holding_value, total_cost, total_pnl,
     today_pnl, total_return_pct, today_pnl_pct}
    """
    holding_value = 0.0
    total_cost    = 0.0
    today_pnl     = 0.0

    for code, info in current_holdings.items():
        if code not in quotes:
            continue
        q      = quotes[code]
        shares = info['shares']
        cost   = info['cost']
        holding_value += q['price']      * shares
        total_cost    += cost            * shares
        today_pnl     += q['change']     * shares

    total_pnl    = holding_value - total_cost
    total_assets = holding_value + available_cash
    total_return = (total_pnl / total_cost) if total_cost > 0 else 0.0
    today_return = (today_pnl / (holding_value - today_pnl)) if (holding_value - today_pnl) > 0 else 0.0

    return {
        'total_assets':    total_assets,
        'holding_value':   holding_value,
        'total_cost':      total_cost,
        'total_pnl':       total_pnl,
        'today_pnl':       today_pnl,
        'total_return':    total_return,
        'today_return':    today_return,
        'available_cash':  available_cash,
    }


def build_holdings_rows(current_holdings: dict, quotes: dict) -> list[dict]:
    """將持股字典 + 報價字典合併為前端顯示用的 row 列表。"""
    rows = []
    for code, info in current_holdings.items():
        if code not in quotes:
            continue
        q      = quotes[code]
        shares = info['shares']
        cost   = info['cost']
        curr   = q['price']

        cost_val    = cost  * shares
        curr_val    = curr  * shares
        pnl         = curr_val - cost_val
        pnl_pct     = pnl / cost_val if cost_val > 0 else 0.0

        rows.append({
            'code':         code,
            'shares':       shares,
            'cost':         cost,
            'current':      curr,
            'change':       q['change'],
            'change_pct':   q['change_pct'],
            'cost_value':   cost_val,
            'curr_value':   curr_val,
            'pnl':          pnl,
            'pnl_pct':      pnl_pct,
        })
    return rows


# ═══════════════════════════════════════════════════════════════
#  ③ Plotly 圖表函數
# ═══════════════════════════════════════════════════════════════

def chart_kline(df: pd.DataFrame, code: str) -> go.Figure:
    """
    台股式 K 線圖（60日）+ MA5/20/60 + 成交量副圖。

    台股 K 棒顏色慣例：
      陽線（收 > 開）= 紅色 #e84545
      陰線（收 < 開）= 綠色 #26a69a

    圖表結構（make_subplots）：
      Row 1（75%）: Candlestick + MA 均線
      Row 2（25%）: Volume 成交量棒圖
      shared_xaxes=True 確保縮放同步

    MA 顏色：
      MA5  = 金色  #ffd700（短線動能參考）
      MA20 = 藍色  #4c9be8（中線趨勢參考）
      MA60 = 橙色  #ff8c00（長線趨勢參考）
    """
    df = df.tail(60).copy()

    # ── 計算均線 ──
    df['MA5']  = df['Close'].rolling(5).mean()
    df['MA20'] = df['Close'].rolling(20).mean()
    df['MA60'] = df['Close'].rolling(60).mean()

    # ── 成交量顏色：依當日漲跌決定 ──
    vol_colors = [
        '#e84545' if c >= o else '#26a69a'
        for c, o in zip(df['Close'], df['Open'])
    ]

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.75, 0.25],
        vertical_spacing=0.025,
    )

    # ── K 棒 ──
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'], high=df['High'],
        low=df['Low'],   close=df['Close'],
        increasing_line_color='#e84545',
        increasing_fillcolor='#e84545',
        decreasing_line_color='#26a69a',
        decreasing_fillcolor='#26a69a',
        line_width=1,
        name='K線',
        showlegend=False,
    ), row=1, col=1)

    # ── MA 均線 ──
    for col_name, color, name in [
        ('MA5',  '#ffd700', 'MA5'),
        ('MA20', '#4c9be8', 'MA20'),
        ('MA60', '#ff8c00', 'MA60'),
    ]:
        fig.add_trace(go.Scatter(
            x=df.index, y=df[col_name],
            mode='lines',
            line=dict(color=color, width=1.4),
            name=name,
        ), row=1, col=1)

    # ── 成交量棒圖 ──
    fig.add_trace(go.Bar(
        x=df.index, y=df['Volume'],
        marker_color=vol_colors,
        marker_line_width=0,
        name='成交量',
        showlegend=False,
        opacity=0.85,
    ), row=2, col=1)

    # ── 最新收盤標注 ──
    last_close = float(df['Close'].iloc[-1])
    fig.add_hline(
        y=last_close, row=1, col=1,
        line_dash='dot',
        line_color='rgba(255,255,255,0.25)',
        line_width=1,
    )

    _dark = 'rgba(14,17,23,1)'
    fig.update_layout(
        title=dict(
            text=f'<b>{code}</b>　近 60 日 K 線圖',
            font=dict(size=15, color='#e0e8f8'),
        ),
        xaxis_rangeslider_visible=False,
        plot_bgcolor=_dark,
        paper_bgcolor='rgba(14,17,23,0)',
        font=dict(color='#c8d4e8', size=11),
        height=520,
        margin=dict(l=60, r=30, t=55, b=30),
        legend=dict(
            orientation='h',
            yanchor='bottom', y=1.01,
            xanchor='left',   x=0,
            bgcolor='rgba(0,0,0,0)',
            font=dict(size=11),
        ),
        hovermode='x unified',
        xaxis2=dict(
            gridcolor='rgba(255,255,255,0.06)',
            showgrid=True,
        ),
        yaxis=dict(
            title='股價 (NT$)',
            gridcolor='rgba(255,255,255,0.06)',
            tickformat=',.1f',
            side='right',
        ),
        yaxis2=dict(
            title='成交量',
            gridcolor='rgba(255,255,255,0.06)',
            tickformat='.2s',
            side='right',
        ),
    )
    fig.update_xaxes(gridcolor='rgba(255,255,255,0.06)')
    return fig


def chart_monte_carlo(mc_stats: dict, available_cash: float) -> go.Figure:
    """蒙地卡羅路徑圖（含破產風險區與分位數帶）。"""
    paths      = mc_stats['paths']
    n_days     = mc_stats['simulation_days']
    init_val   = mc_stats['initial_portfolio_value']
    bankrupt_y = available_cash * 0.30
    x          = np.arange(n_days + 1)

    p05  = np.percentile(paths, 5,  axis=0)
    p25  = np.percentile(paths, 25, axis=0)
    p50  = np.percentile(paths, 50, axis=0)
    p75  = np.percentile(paths, 75, axis=0)
    p95  = np.percentile(paths, 95, axis=0)

    fig = go.Figure()

    fig.add_hrect(
        y0=0, y1=bankrupt_y,
        fillcolor='rgba(220,50,50,0.10)',
        line_width=0, layer='below',
        annotation_text="  破產風險區（損失 >70%）",
        annotation_position="bottom right",
        annotation_font=dict(color='rgba(255,100,100,0.55)', size=10),
    )

    # 150 條樣本路徑
    rng = np.random.default_rng(seed=42)
    for ix in rng.choice(len(paths), min(150, len(paths)), replace=False):
        fig.add_trace(go.Scatter(
            x=x, y=paths[ix], mode='lines',
            line=dict(color='rgba(130,170,230,0.06)', width=0.7),
            showlegend=False, hoverinfo='skip',
        ))

    fig.add_trace(go.Scatter(
        x=np.concatenate([x, x[::-1]]),
        y=np.concatenate([p95, p05[::-1]]),
        fill='toself', fillcolor='rgba(76,155,232,0.10)',
        line=dict(color='rgba(0,0,0,0)'),
        name='90% 信心帶', hoverinfo='skip',
    ))
    fig.add_trace(go.Scatter(
        x=np.concatenate([x, x[::-1]]),
        y=np.concatenate([p75, p25[::-1]]),
        fill='toself', fillcolor='rgba(76,155,232,0.25)',
        line=dict(color='rgba(0,0,0,0)'),
        name='IQR 帶', hoverinfo='skip',
    ))
    fig.add_trace(go.Scatter(
        x=x, y=p50, mode='lines',
        line=dict(color='#4c9be8', width=2.5),
        name='中位數',
    ))
    fig.add_trace(go.Scatter(
        x=x, y=p05, mode='lines',
        line=dict(color='rgba(232,69,69,0.8)', width=1.5, dash='dot'),
        name='5th pct（悲觀）',
    ))
    fig.add_trace(go.Scatter(
        x=x, y=p95, mode='lines',
        line=dict(color='rgba(38,166,154,0.8)', width=1.5, dash='dot'),
        name='95th pct（樂觀）',
    ))

    fig.add_hline(y=init_val, line_dash='dash',
                  line_color='rgba(255,255,255,0.35)', line_width=1.5,
                  annotation_text=f'初始投入  NT${init_val:,.0f}',
                  annotation_position='top left',
                  annotation_font=dict(color='rgba(255,255,255,0.5)', size=10))
    fig.add_hline(y=bankrupt_y, line_dash='solid',
                  line_color='rgba(232,69,69,0.75)', line_width=2,
                  annotation_text=f'破產門檻  NT${bankrupt_y:,.0f}',
                  annotation_position='bottom left',
                  annotation_font=dict(color='rgba(232,69,69,0.8)', size=10))

    _dark = 'rgba(14,17,23,1)'
    fig.update_layout(
        title=dict(
            text=(f"蒙地卡羅  {mc_stats['n_simulations']:,} 條路徑 × "
                  f"{n_days} 個交易日"),
            font=dict(size=14),
        ),
        xaxis=dict(title='交易日', gridcolor='rgba(255,255,255,0.06)'),
        yaxis=dict(title='投組市值 (NT$)', tickformat=',.0f',
                   gridcolor='rgba(255,255,255,0.06)'),
        plot_bgcolor=_dark, paper_bgcolor='rgba(14,17,23,0)',
        font=dict(color='white'),
        legend=dict(orientation='h', yanchor='bottom', y=1.02,
                    xanchor='right', x=1, bgcolor='rgba(0,0,0,0)'),
        hovermode='x unified', height=500, margin=dict(t=70),
    )
    return fig


def chart_return_distribution(mc_stats: dict) -> go.Figure:
    """期末報酬率分佈直方圖（虧損紅 / 獲利綠，台股慣例）。"""
    paths   = mc_stats['paths']
    init    = mc_stats['initial_portfolio_value']
    rets    = (paths[:, -1] - init) / init * 100.0
    loss    = rets < 0

    fig = go.Figure()
    for mask, color, name in [
        (loss,  'rgba(232,69,69,0.65)',   '虧損路徑'),
        (~loss, 'rgba(38,166,154,0.65)',  '獲利路徑'),
    ]:
        fig.add_trace(go.Histogram(
            x=rets[mask], nbinsx=40,
            marker_color=color,
            marker_line=dict(color=color, width=0.3),
            name=name,
        ))

    for pct, color in [(5, '#e84545'), (50, 'white'), (95, '#26a69a')]:
        val = float(np.percentile(rets, pct))
        fig.add_vline(x=val, line_dash='dash', line_color=color, line_width=1.5,
                      annotation_text=f'{pct}th: {val:.1f}%',
                      annotation_font=dict(color=color, size=10),
                      annotation_position='top')
    fig.add_vline(x=0, line_color='rgba(255,255,255,0.25)', line_width=1.2)

    _dark = 'rgba(14,17,23,1)'
    fig.update_layout(
        title='一季末報酬率分佈',
        barmode='overlay',
        xaxis=dict(title='報酬率 (%)', gridcolor='rgba(255,255,255,0.06)'),
        yaxis=dict(title='模擬次數',   gridcolor='rgba(255,255,255,0.06)'),
        plot_bgcolor=_dark, paper_bgcolor='rgba(14,17,23,0)',
        font=dict(color='white'),
        legend=dict(orientation='h', y=1.05),
        height=340,
    )
    return fig


def chart_stock_scores(sorted_stocks: list) -> go.Figure:
    """GA 評分橫條圖（正分=台股紅看多, 負分=台股綠看空）。"""
    codes  = [c for c, _ in sorted_stocks]
    scores = [s for _, s in sorted_stocks]
    colors = ['#e84545' if s >= 0 else '#26a69a' for s in scores]

    fig = go.Figure(go.Bar(
        x=scores, y=codes, orientation='h',
        marker_color=colors,
        text=[f"{s:+.4f}" for s in scores],
        textposition='outside', textfont=dict(size=11),
    ))
    fig.add_vline(x=0, line_color='rgba(255,255,255,0.20)', line_width=1.2)

    _dark = 'rgba(14,17,23,1)'
    fig.update_layout(
        title='GA 綜合評分排行（紅 = 看多，綠 = 看空）',
        xaxis=dict(title='綜合評分',
                   gridcolor='rgba(255,255,255,0.06)',
                   range=[min(scores)*1.35-0.1, max(scores)*1.35+0.1]),
        yaxis=dict(autorange='reversed'),
        plot_bgcolor=_dark, paper_bgcolor='rgba(14,17,23,0)',
        font=dict(color='white'),
        height=max(280, len(codes) * 42 + 80),
        margin=dict(l=70, r=110, t=50, b=40),
    )
    return fig


def chart_fitness_history(history: list[dict]) -> go.Figure:
    """GA 適應度收斂曲線。"""
    gens   = [h['generation']  for h in history]
    best_f = [h['best_fitness'] for h in history]
    avg_f  = [h['avg_fitness']  for h in history]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=gens, y=best_f, mode='lines+markers',
        line=dict(color='#4c9be8', width=2.2), marker=dict(size=3),
        name='最佳適應度'))
    fig.add_trace(go.Scatter(
        x=gens, y=avg_f, mode='lines',
        line=dict(color='rgba(255,215,60,0.75)', width=1.5, dash='dot'),
        name='種群平均'))

    _dark = 'rgba(14,17,23,1)'
    fig.update_layout(
        title='GA 適應度收斂曲線',
        xaxis=dict(title='演化代數', gridcolor='rgba(255,255,255,0.06)'),
        yaxis=dict(title='適應度',   gridcolor='rgba(255,255,255,0.06)'),
        plot_bgcolor=_dark, paper_bgcolor='rgba(14,17,23,0)',
        font=dict(color='white'),
        legend=dict(orientation='h', y=1.08),
        height=320,
    )
    return fig


# ═══════════════════════════════════════════════════════════════
#  ④ 後端：完整量化分析流程
# ═══════════════════════════════════════════════════════════════

def run_full_pipeline(
        available_cash:   float,
        current_holdings: dict,
        target_pool:      list[str],
        ga_config:        dict,
        mc_config:        dict,
        top_n:            int,
        _progress:        st.delta_generator.DeltaGenerator,
        _status:          st.delta_generator.DeltaGenerator,
) -> dict:
    """
    執行 GA 優化 → 蒙地卡羅模擬 → 持股分析的完整流程。
    所有計算封裝於此，前端只呼叫一次並等待結果。
    """
    results = {}

    # Step 1：下載股票數據（使用快取層）
    _status.text("📡  Step 1/4  正在載入歷史數據...")
    _progress.progress(5)

    all_codes = tuple(sorted(set(list(current_holdings.keys()) + target_pool)))
    stock_data = fetch_pool_history(all_codes)

    if not stock_data:
        raise RuntimeError("無法下載任何股票數據，請確認網路連線或股票代號。")

    results['stock_data']   = stock_data
    results['failed_codes'] = [c for c in all_codes if c not in stock_data]
    _progress.progress(20)

    # Step 2：遺傳演算法優化
    _status.text("🧬  Step 2/4  遺傳演算法優化中（約需 30~90 秒）...")
    _progress.progress(25)

    ga = GeneticAlgorithm(**ga_config)
    pool_data = {c: stock_data[c] for c in target_pool if c in stock_data}
    if not pool_data:
        raise RuntimeError("目標股池無有效數據。")

    _buf = io.StringIO()
    with redirect_stdout(_buf):
        best_params = ga.evolve(pool_data, verbose=True)

    stock_scores  = ga.score_stocks(pool_data, best_params)
    sorted_stocks = sorted(stock_scores.items(), key=lambda x: x[1], reverse=True)

    results['best_params']     = best_params
    results['stock_scores']    = stock_scores
    results['sorted_stocks']   = sorted_stocks
    results['fitness_history'] = ga.fitness_history
    _progress.progress(65)

    # Step 3：選股 + 蒙地卡羅
    _status.text("🎲  Step 3/4  蒙地卡羅模擬中（1,000+ 路徑）...")
    _progress.progress(68)

    positive      = [c for c, s in sorted_stocks if s > 0]
    fallback      = [c for c, _ in sorted_stocks if c not in positive]
    selected_codes = (positive + fallback)[:top_n]
    selected_prices = {
        c: stock_data[c]['Close'].dropna()
        for c in selected_codes if c in stock_data
    }

    simulator = MonteCarloSimulator(
        n_simulations=mc_config['n_simulations'],
        n_days=mc_config['n_days'],
    )
    mc_stats = simulator.simulate_portfolio(
        selected_stocks=selected_prices,
        available_cash=available_cash,
    )

    results['selected_codes']  = selected_codes
    results['mc_stats']        = mc_stats
    _progress.progress(85)

    # Step 4：持股分析
    _status.text("💼  Step 4/4  分析持股機會成本...")
    _progress.progress(90)

    holdings_analysis = {}
    recommendations   = []

    if current_holdings:
        holdings_data = {c: stock_data[c] for c in current_holdings if c in stock_data}
        analyzer = HoldingsAnalyzer()
        holdings_analysis = analyzer.analyze(
            current_holdings=current_holdings,
            stock_data=holdings_data,
            best_params=best_params,
        )
        candidate_scores = {
            c: s for c, s in stock_scores.items()
            if c not in current_holdings and s > 0
        }
        candidate_data = {c: stock_data[c] for c in candidate_scores if c in stock_data}
        recommendations = analyzer.recommend_switches(
            holdings_analysis=holdings_analysis,
            candidate_scores=candidate_scores,
            candidate_data=candidate_data,
        )

    results['holdings_analysis'] = holdings_analysis
    results['recommendations']   = recommendations
    _progress.progress(100)
    _status.text("✅  分析完成！")
    return results


# ═══════════════════════════════════════════════════════════════
#  ⑤ UI 區塊函數
# ═══════════════════════════════════════════════════════════════

# ── 側邊欄 ────────────────────────────────────────────────────

def render_sidebar() -> dict:
    """收集所有使用者輸入，返回設定字典。"""
    with st.sidebar:
        st.markdown("## ⚙️ 策略設定")
        st.markdown("---")

        st.markdown("#### 💰 可用現金（NT$）")
        available_cash = st.number_input(
            "available_cash", min_value=10_000, max_value=100_000_000,
            value=500_000, step=10_000, format="%d",
            label_visibility='collapsed',
        )
        st.caption(f"NT$ {available_cash:,}")
        st.markdown("---")

        st.markdown("#### 📋 現有持股")
        st.caption("代號填純數字（不加 .TW）。動態新增 / 刪除列。")

        _default_h = pd.DataFrame({
            '代號':   ['2330', '2317'],
            '買入均價': [850.0,  95.0],
            '持有股數': [1000,   2000],
        })
        holdings_df = st.data_editor(
            _default_h, num_rows='dynamic', use_container_width=True,
            column_config={
                '代號':   st.column_config.TextColumn('代號',  width='small'),
                '買入均價': st.column_config.NumberColumn('均價', min_value=0.01, format="%.2f"),
                '持有股數': st.column_config.NumberColumn('股數', min_value=0, step=1000, format="%d"),
            },
            key='holdings_editor',
        )

        current_holdings: dict[str, dict] = {}
        for _, row in holdings_df.dropna(subset=['代號']).iterrows():
            code = str(row['代號']).strip()
            if code and float(row.get('買入均價', 0) or 0) > 0:
                current_holdings[code] = {
                    'cost':   float(row['買入均價']),
                    'shares': int(row.get('持有股數') or 0),
                }
        st.markdown("---")

        st.markdown("#### 🎯 目標觀測股池")
        st.caption("逗號分隔，不含後綴")
        pool_text = st.text_area(
            "pool", value="2330,2317,2454,2382,2308,2881,2882,3711,6505,1301",
            height=80, label_visibility='collapsed', key='pool_text',
        )
        target_pool = [c.strip() for c in pool_text.replace('\n', ',').split(',') if c.strip()]
        st.caption(f"共 {len(target_pool)} 支")
        st.markdown("---")

        with st.expander("🧬 遺傳演算法參數"):
            pop_size    = st.slider("種群大小", 20, 100, 50, step=10)
            generations = st.slider("演化代數", 20, 100, 50, step=10)
            cr          = st.slider("交叉率",   0.50, 1.00, 0.80, step=0.05)
            mr          = st.slider("變異率",   0.05, 0.30, 0.15, step=0.05)

        with st.expander("🎲 蒙地卡羅參數"):
            n_sims = st.select_slider("路徑數量", options=[1000, 2000, 5000], value=1000)
            n_days = st.select_slider(
                "模擬天數", options=[21, 42, 63, 126], value=63,
                format_func=lambda x: f"{x} 日（約 {x//21} 個月）",
            )

        top_n = st.slider("📌 最終選股數量", 1, 6, 3)
        st.markdown("---")

        run_clicked = st.button(
            "🚀  執行推演",
            type="primary",
            use_container_width=True,
            help="執行 GA 優化 + 蒙地卡羅模擬（約需 30~90 秒）",
        )

        if st.session_state.get('_has_results'):
            if st.button("🗑️  清除分析結果", use_container_width=True):
                for k in ('_results', '_config', '_has_results'):
                    st.session_state.pop(k, None)
                st.rerun()

    return {
        'available_cash':   available_cash,
        'current_holdings': current_holdings,
        'target_pool':      target_pool,
        'ga_config': {
            'population_size': pop_size,
            'generations':     generations,
            'crossover_rate':  cr,
            'mutation_rate':   mr,
        },
        'mc_config': {'n_simulations': n_sims, 'n_days': n_days},
        'top_n':       top_n,
        'run_clicked': run_clicked,
    }


# ── ① 頂部資產快報 ─────────────────────────────────────────────

def render_top_kpi_bar(summary: dict):
    """
    頁面最頂部的四欄 KPI 卡片列。

    指標：總資產 / 持倉市值 / 今日盈虧 / 整體報酬率
    台股配色：正值 = 紅，負值 = 綠，零 = 灰

    使用原生 HTML 卡片而非 st.metric，
    以取得台股慣例的紅漲綠跌配色（st.metric 的 delta_color 無法精確控制字體色）。
    """
    def _card(label: str, value: str, delta: str, delta_val: float) -> str:
        cls   = _tw_color_class(delta_val)
        arrow = _arrow(delta_val)
        return f"""
        <div class="kpi-card">
          <div class="kpi-label">{label}</div>
          <div class="kpi-value tw-white">{value}</div>
          <div class="kpi-delta {cls}">{arrow} {delta}</div>
        </div>"""

    pnl     = summary['total_pnl']
    t_ret   = summary['total_return']
    today   = summary['today_pnl']
    t_today = summary['today_return']
    cash    = summary['available_cash']

    col1, col2, col3, col4 = st.columns(4)
    col1.markdown(
        _card("💰 總資產（持倉 + 現金）",
              f"NT${summary['total_assets']:,.0f}",
              f"NT${pnl:+,.0f}", pnl),
        unsafe_allow_html=True,
    )
    col2.markdown(
        _card("📦 持倉市值",
              f"NT${summary['holding_value']:,.0f}",
              f"成本 NT${summary['total_cost']:,.0f}", 0),
        unsafe_allow_html=True,
    )
    col3.markdown(
        _card("📅 今日損益",
              f"NT${today:+,.0f}",
              f"{t_today:+.2%}", today),
        unsafe_allow_html=True,
    )
    col4.markdown(
        _card("📈 整體投資報酬率",
              f"{t_ret:+.2%}",
              f"NT${pnl:+,.0f} 未實現", pnl),
        unsafe_allow_html=True,
    )

    # 資料時間戳
    now = datetime.datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
    st.caption(f"⏱ 報價快取 5 分鐘  |  最後更新：{now}  |  台股配色：紅漲綠跌")


# ── ② 持股即時看板 ─────────────────────────────────────────────

def render_holdings_dashboard(rows: list[dict]):
    """
    以 HTML 表格渲染持股看板，支援台股配色。

    欄位：代號 / 持有股數 / 成本均價 / 現價 / 漲跌幅(%) / 未實現損益 / 總市值

    設計選擇 HTML table 而非 st.dataframe：
      - st.dataframe 的 Styler 不支援即時 delta_color 反轉
      - HTML table 可精確控制每格顏色，台股慣例紅漲綠跌
    """
    st.markdown('<div class="dash-section">📋 HOLDINGS WATCHLIST</div>',
                unsafe_allow_html=True)

    if not rows:
        st.info("尚未輸入持股，請在左側側邊欄新增。")
        return

    header = """
    <table class="holdings-table">
    <thead><tr>
      <th style="text-align:left">代號</th>
      <th>持有股數</th>
      <th>成本均價</th>
      <th>現價</th>
      <th>漲跌幅</th>
      <th>今日損益</th>
      <th>未實現損益</th>
      <th>總市值</th>
    </tr></thead><tbody>
    """

    body = ""
    for r in rows:
        # 現價顏色依今日漲跌
        price_color = _tw_color_hex(r['change_pct'])
        # 損益顏色依未實現損益
        pnl_color   = _tw_color_hex(r['pnl'])
        arrow       = _arrow(r['change_pct'])
        today_pnl   = r['change'] * r['shares']

        body += f"""
        <tr>
          <td><span class="stock-code">{r['code']}</span></td>
          <td>{r['shares']:,}</td>
          <td>NT${r['cost']:.2f}</td>
          <td style="color:{price_color}; font-weight:700;">
            NT${r['current']:.2f}
          </td>
          <td style="color:{price_color}; font-weight:600;">
            {arrow} {abs(r['change_pct']):.2%}
            <span style="font-size:0.8em; color:rgba(255,255,255,0.4);">
              ({r['change']:+.2f})
            </span>
          </td>
          <td style="color:{_tw_color_hex(today_pnl)}; font-weight:600;">
            NT${today_pnl:+,.0f}
          </td>
          <td style="color:{pnl_color}; font-weight:600;">
            NT${r['pnl']:+,.0f}
            <span style="font-size:0.82em; color:rgba(255,255,255,0.45);">
              ({r['pnl_pct']:+.2%})
            </span>
          </td>
          <td style="color:#c8d4e8;">NT${r['curr_value']:,.0f}</td>
        </tr>
        """

    footer = "</tbody></table>"
    st.markdown(header + body + footer, unsafe_allow_html=True)
    st.markdown("")


# ── ③ 個股 K 線模組 ────────────────────────────────────────────

def render_kline_section(available_codes: list[str]):
    """
    個股詳細分析區塊。

    使用 st.selectbox 選擇代號 → 呼叫快取的 fetch_kline_data()
    → 渲染 chart_kline()。

    設計重點：
      - available_codes 為側邊欄的持股 + 股池聯集
      - 選股後的 K 線資料已被 @st.cache_data 快取（5 分鐘）
        再次選同一支股票時不會觸發新的網路請求
    """
    st.markdown('<div class="dash-section">📊 INDIVIDUAL STOCK ANALYSIS</div>',
                unsafe_allow_html=True)

    if not available_codes:
        st.info("請先在側邊欄輸入股票代號。")
        return

    col_select, col_info = st.columns([2, 5])

    with col_select:
        selected_code = st.selectbox(
            "選擇股票代號",
            options=available_codes,
            index=0,
            key='kline_selector',
            label_visibility='collapsed',
        )
        st.caption(f"顯示 **{selected_code}** 近 60 日 K 線")

    with col_info:
        # 顯示即時報價小摘要（若有快取報價）
        q_data = st.session_state.get('_quotes', {})
        if selected_code in q_data:
            q = q_data[selected_code]
            chg_col = _tw_color_hex(q['change_pct'])
            arrow   = _arrow(q['change_pct'])
            st.markdown(
                f"<span style='font-size:1.3rem; font-weight:700; "
                f"color:{chg_col};'>"
                f"NT${q['price']:.2f}&nbsp;&nbsp;"
                f"{arrow} {abs(q['change_pct']):.2%}&nbsp;"
                f"<span style='font-size:0.9rem;'>({q['change']:+.2f})</span>"
                f"</span>&nbsp;&nbsp;"
                f"<span style='color:#6b7fa8; font-size:0.82rem;'>"
                f"成交量 {q['volume']:,.0f}</span>",
                unsafe_allow_html=True,
            )

    # 下載並渲染 K 線
    with st.spinner(f"載入 {selected_code} K 線數據..."):
        df = fetch_kline_data(selected_code)

    if df is None or df.empty:
        st.warning(f"⚠️  無法取得 {selected_code} 的 K 線數據，請確認代號是否正確。")
        return

    st.plotly_chart(chart_kline(df, selected_code), use_container_width=True)

    # K 線下方的基本統計
    recent = df.tail(60)
    h52w   = float(recent['High'].max())
    l52w   = float(recent['Low'].min())
    avg_vol = float(recent['Volume'].mean())
    last_close = float(recent['Close'].iloc[-1])

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("60日最高", f"NT${h52w:.1f}")
    s2.metric("60日最低", f"NT${l52w:.1f}")
    s3.metric("60日均量", f"{avg_vol:,.0f}")
    s4.metric("距60日高點",
              f"{(last_close - h52w) / h52w:.2%}",
              delta_color="off")


# ── 量化分析分頁 ───────────────────────────────────────────────

def render_analysis_tabs(results: dict, config: dict):
    """渲染 GA / 蒙地卡羅 / 持股分析四個分頁。"""

    tab_scan, tab_ga, tab_mc, tab_holdings = st.tabs([
        "📊  市場掃描",
        "🧬  GA 最佳化",
        "🎲  蒙地卡羅模擬",
        "💼  持股分析",
    ])

    # ── Tab 1：市場掃描 ─────────────────────────────────────
    with tab_scan:
        sorted_stocks  = results['sorted_stocks']
        selected_codes = results['selected_codes']
        n_pos = sum(1 for _, s in sorted_stocks if s > 0)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("掃描股數", len(sorted_stocks))
        m2.metric("正向信號", n_pos,
                  delta=f"佔 {n_pos/len(sorted_stocks):.0%}" if sorted_stocks else "")
        m3.metric("最終選股", len(selected_codes))
        m4.metric("GA 最佳適應度",
                  f"{results['fitness_history'][-1]['best_fitness']:.4f}"
                  if results['fitness_history'] else "─")

        st.markdown("---")
        col_c, col_t = st.columns([3, 2])
        with col_c:
            st.plotly_chart(chart_stock_scores(sorted_stocks), use_container_width=True)
        with col_t:
            st.markdown("##### 評分明細")
            rows = [{
                '#':   i, '代號': c,
                '評分': f"{s:+.4f}",
                '信號': "▲看多" if s > 0.10 else ("▼看空" if s < -0.10 else "─中性"),
                '狀態': "✅" if c in selected_codes else "",
            } for i, (c, s) in enumerate(sorted_stocks, 1)]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True,
                         height=min(420, len(rows)*38+40))

        if results.get('failed_codes'):
            st.warning(f"⚠️  以下代號無法取得數據：{results['failed_codes']}")

    # ── Tab 2：GA 最佳化 ────────────────────────────────────
    with tab_ga:
        bp = results['best_params']
        fh = results['fitness_history']

        if fh:
            i_best, f_best = fh[0]['best_fitness'], fh[-1]['best_fitness']
            c1, c2, c3 = st.columns(3)
            c1.metric("初代適應度", f"{i_best:.4f}")
            c2.metric("末代適應度", f"{f_best:.4f}")
            c3.metric("演化淨改進",  f"{f_best-i_best:+.4f}",
                      delta_color="normal" if f_best > i_best else "inverse")

        st.plotly_chart(chart_fitness_history(fh), use_container_width=True)

        st.markdown("##### 最佳策略參數")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**MA 交叉**")
            st.metric("短期均線", f"{bp['ma_short']} 日")
            st.metric("長期均線", f"{bp['ma_long']} 日")
            st.metric("MA 權重",  f"{bp['ma_weight']:.2f}")
        with c2:
            st.markdown("**RSI**")
            st.metric("RSI 周期", f"{bp['rsi_period']} 日")
            st.metric("超買閾值",  f"{bp['rsi_ob']:.0f}")
            st.metric("超賣閾值",  f"{bp['rsi_os']:.0f}")
            st.metric("RSI 權重", f"{bp['rsi_weight']:.2f}")
        with c3:
            st.markdown("**布林通道**")
            st.metric("BB 周期",   f"{bp['bb_period']} 日")
            st.metric("標準差倍數", f"{bp['bb_std']:.1f} σ")
            st.metric("BB 權重",   f"{bp['bb_weight']:.2f}")
            st.metric("買入閾值",   f"{bp['buy_threshold']:.2f}")
        with st.expander("📋 完整參數（JSON）"):
            st.json(bp)

    # ── Tab 3：蒙地卡羅 ─────────────────────────────────────
    with tab_mc:
        mc   = results['mc_stats']
        cash = config['available_cash']
        win  = mc['win_rate']
        bkrp = mc['bankruptcy_probability']

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("勝率", mc['win_rate_pct'],
                  delta="正期望值" if win > 0.50 else "負期望，謹慎",
                  delta_color="normal" if win > 0.50 else "inverse")
        k2.metric("期望季度報酬", mc['expected_return_pct'],
                  delta=f"年化 {mc['expected_annualized_return_pct']}")
        k3.metric("動態破產機率", mc['bankruptcy_probability_pct'],
                  delta="偏高" if bkrp > 0.05 else "可控",
                  delta_color="inverse" if bkrp > 0.05 else "normal")
        k4.metric("平均最大回撤", mc['avg_max_drawdown_pct'])

        if win < 0.50:
            st.warning("⚠️  勝率低於 50%，建議縮小部位或觀望。")
        if bkrp > 0.05:
            st.error("🚨  破產機率超過 5%！建議設置嚴格停損。")

        st.markdown("---")
        st.plotly_chart(chart_monte_carlo(mc, cash), use_container_width=True)

        left, right = st.columns([3, 2])
        with left:
            st.plotly_chart(chart_return_distribution(mc), use_container_width=True)
        with right:
            st.markdown("##### 報酬率分位數")
            d = mc['return_distribution']
            st.dataframe(pd.DataFrame({
                '分位數':   ['最壞(5%)', '悲觀(25%)', '中位(50%)', '樂觀(75%)', '最佳(95%)'],
                '季度報酬': [f"{v:+.2%}" for v in [d['p05'],d['p25'],d['p50'],d['p75'],d['p95']]],
            }), use_container_width=True, hide_index=True)

            st.markdown("##### 資金配置")
            alloc_rows = [{
                '代號':     c,
                '現價':     f"NT${det['price_now']:,.1f}",
                '買入股數': f"{det['n_shares']:,}",
                '投入':     f"NT${det['actual_cost']:,.0f}",
            } for c, det in mc['allocation_detail'].items()]
            if alloc_rows:
                st.dataframe(pd.DataFrame(alloc_rows), use_container_width=True, hide_index=True)
            st.info(f"總投入 NT${mc['total_invested']:,.0f}　剩餘現金 NT${mc['unused_cash']:,.0f}")

    # ── Tab 4：持股分析 ──────────────────────────────────────
    with tab_holdings:
        ha   = results['holdings_analysis']
        recs = results['recommendations']

        if not ha:
            st.info("未輸入現有持股。請在左側側邊欄新增。")
            return

        total_val  = sum(i['current_value']  for i in ha.values())
        total_pnl  = sum(i['unrealized_pnl'] for i in ha.values())
        total_cost = sum(i['cost_value']      for i in ha.values())

        t1, t2, t3 = st.columns(3)
        t1.metric("持股總市值",   f"NT${total_val:,.0f}")
        t2.metric("總未實現損益", f"NT${total_pnl:,.0f}",
                  delta=f"{total_pnl/total_cost:.2%}" if total_cost > 0 else "",
                  delta_color="normal" if total_pnl >= 0 else "inverse")
        t3.metric("總持倉成本",   f"NT${total_cost:,.0f}")

        st.markdown("---")
        h_rows = [{
            '代號':     c,
            '成本均價': f"NT${i['cost_price']:.1f}",
            '現價':     f"NT${i['current_price']:.1f}",
            '股數':     f"{i['shares']:,}",
            '現值':     f"NT${i['current_value']:,.0f}",
            '損益%':    f"{i['unrealized_pnl_pct']:+.2%}",
            '信號':     f"{i['current_signal']:+.3f}",
            'Sharpe':   f"{i['sharpe_ratio']:.3f}",
            '期望年化': f"{i['expected_annual_ret']:.2%}",
        } for c, i in ha.items()]
        st.dataframe(pd.DataFrame(h_rows), use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown("##### 換股建議")

        if not recs:
            st.success("✅  持股狀況良好，暫無換股建議。")
        else:
            st.warning(f"⚡  發現 **{len(recs)}** 筆換股機會")
            _icons = {'HIGH': '🔴', 'MEDIUM': '🟡', 'LOW': '🟢'}
            for i, rec in enumerate(recs, 1):
                with st.container():
                    st.markdown(f"**{_icons.get(rec['priority'],'─')} 建議 {i}**（{rec['priority']}）　"
                                f"賣出 **{rec['sell_code']}** → 買入 **{rec['buy_code']}**")
                    r1, r2, r3, r4 = st.columns(4)
                    r1.metric(f"{rec['sell_code']} 期望年化", f"{rec['sell_expected_ret']:.2%}",
                              delta=f"信號 {rec['sell_signal']:+.3f}")
                    r2.metric(f"{rec['buy_code']} 歷史年化", f"{rec['buy_annual_ret']:.2%}",
                              delta=f"GA {rec['buy_ga_score']:+.4f}")
                    r3.metric("超額年化報酬", f"{rec['excess_return_annual']:+.2%}")
                    r4.metric("年度機會成本", f"NT${rec['opportunity_cost_annual']:,.0f}")
                    i1, i2 = st.columns(2)
                    i1.info(f"💸 換股成本 NT${rec['switch_cost_ntd']:,.0f} ({rec['switch_cost_rate']:.3%})")
                    i2.info(f"⏱️ 預計回本 {rec['payback_str']}")
                    st.markdown("---")


# ═══════════════════════════════════════════════════════════════
#  主程式
# ═══════════════════════════════════════════════════════════════

def main():

    # ── 側邊欄（輸入）──
    config = render_sidebar()
    ch     = config['current_holdings']
    pool   = config['target_pool']
    cash   = config['available_cash']

    # ── 即時報價（Dashboard 層，與 GA 無關，頁面載入即執行）──
    all_dashboard_codes = tuple(sorted(set(list(ch.keys()) + pool)))
    with st.spinner("⚡ 獲取即時報價中..."):
        quotes = fetch_realtime_quotes(all_dashboard_codes)
    st.session_state['_quotes'] = quotes   # 供 K 線模組讀取

    # ── 頁面標題 ──
    col_title, col_refresh = st.columns([5, 1])
    with col_title:
        st.markdown(
            "<h1 style='margin:0; font-size:1.6rem; color:#e0e8f8;'>"
            "📈 台股量化交易儀表板</h1>"
            "<p style='color:#6b7fa8; font-size:0.82rem; margin:2px 0 0;'>"
            "Genetic Algorithm × Monte Carlo × Real-time Dashboard"
            "</p>",
            unsafe_allow_html=True,
        )
    with col_refresh:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔄 刷新報價", help="清除快取並重新抓取即時報價"):
            fetch_realtime_quotes.clear()
            fetch_kline_data.clear()
            st.rerun()

    st.markdown("---")

    # ── ① 頂部資產快報 ──
    if ch:
        summary = compute_portfolio_summary(ch, quotes, cash)
        render_top_kpi_bar(summary)
    else:
        st.info("ℹ️  在左側側邊欄輸入持股後，此處將顯示即時資產快報。")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── ② 持股即時看板 ──
    if ch:
        rows = build_holdings_rows(ch, quotes)
        render_holdings_dashboard(rows)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── ③ 個股 K 線模組 ──
    available_codes = list(dict.fromkeys(list(ch.keys()) + pool))  # 保持順序去重
    render_kline_section(available_codes)

    st.markdown("---")

    # ── ④ 量化分析區塊（需點擊執行推演）──
    st.markdown('<div class="dash-section">🔬 QUANTITATIVE ANALYSIS ENGINE</div>',
                unsafe_allow_html=True)

    # ── 執行推演按鈕觸發 ──
    if config['run_clicked']:
        if not pool:
            st.error("❌  請先輸入至少一支股票到目標觀測股池。")
            return

        for k in ('_results', '_config', '_has_results'):
            st.session_state.pop(k, None)

        progress_bar = st.progress(0, text="初始化...")
        status_text  = st.empty()

        try:
            results = run_full_pipeline(
                available_cash   = cash,
                current_holdings = ch,
                target_pool      = pool,
                ga_config        = config['ga_config'],
                mc_config        = config['mc_config'],
                top_n            = config['top_n'],
                _progress        = progress_bar,
                _status          = status_text,
            )
            st.session_state['_results']     = results
            st.session_state['_config']      = config
            st.session_state['_has_results'] = True

            progress_bar.empty()
            status_text.empty()
            st.success("✅  分析完成！查看下方各分頁。")

        except Exception as exc:
            progress_bar.empty()
            status_text.empty()
            st.error(f"❌  執行失敗：{exc}")
            with st.expander("詳細錯誤"):
                st.exception(exc)
            return

    # ── 顯示結果或說明 ──
    if st.session_state.get('_has_results') and '_results' in st.session_state:
        render_analysis_tabs(
            st.session_state['_results'],
            st.session_state.get('_config', config),
        )
    else:
        st.markdown("""
        <div style="text-align:center; padding:40px 0; color:#4a5568;">
            <div style="font-size:2.5rem; margin-bottom:10px;">🧬</div>
            <p style="font-size:1rem;">點擊左側 <b style="color:#4c9be8;">🚀 執行推演</b>
            按鈕，啟動遺傳演算法優化與蒙地卡羅模擬</p>
            <p style="font-size:0.82rem;">約需 30~90 秒</p>
        </div>
        """, unsafe_allow_html=True)

    # ── 頁腳 ──
    st.markdown("---")
    st.caption(
        "⚠️ 本系統僅供學術研究，不構成投資建議。過去績效不代表未來報酬。"
        "  |  資料來源：Yahoo Finance (yfinance)  |  台股配色：紅漲綠跌"
    )


if __name__ == '__main__':
    main()
