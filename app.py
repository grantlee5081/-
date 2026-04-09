"""
app.py  ─  v3.0  實戰交易儀表板
────────────────────────────────
v3.0 重大修正：
  ① 真實數據：頂部指標完全由 yfinance 現價計算，零硬編碼
  ② 修正 HTML 表格：改用 st.dataframe + pandas Styler，杜絕 <tr> 原始碼外露
  ③ 今日操作指南：GA 結果與現持股對比，輸出具體買賣指令
     - 計算換股能否覆蓋手續費與未實現虧損
     - 輸出「維持現狀 / 建議換股 / 止損出場 / 新建部位」
  ④ 最新交易日：fetch 使用 period='5d' 取最末完整交易日收盤

台股配色：漲 = 紅 #e84545   跌 = 綠 #26a69a
"""

# ── 標準函式庫 ────────────────────────────────────────────────
import io
import datetime
from contextlib import redirect_stdout

# ── 第三方套件 ────────────────────────────────────────────────
import yfinance as yf
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── 後端模組 ──────────────────────────────────────────────────
from data_fetcher import DataFetcher
from genetic_algorithm import GeneticAlgorithm
from monte_carlo import MonteCarloSimulator
from holdings_analyzer import HoldingsAnalyzer
from performance_metrics import PerformanceMetrics
from technical_factors import TechnicalFactors

# 台股換股總成本率（賣出 0.3% 證交稅 + 買賣雙邊手續費 0.1425%×2）
SWITCH_COST_RATE = 0.00585


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

/* KPI 卡片 */
.kpi-wrap {
    background: linear-gradient(135deg,#131929,#1a2035);
    border:1px solid rgba(76,155,232,.20);
    border-radius:12px;
    padding:16px 20px 12px;
    text-align:center;
    position:relative; overflow:hidden;
}
.kpi-wrap::before {
    content:''; position:absolute;
    top:0; left:0; right:0; height:2px;
    background:linear-gradient(90deg,#4c9be8,#7c4dff);
}
.kpi-lbl  { font-size:.75rem; color:#6b7fa8; letter-spacing:.08em; margin-bottom:4px; }
.kpi-val  { font-size:1.5rem; font-weight:700; color:#e0e8f8; line-height:1.2; }
.kpi-dlt  { font-size:.80rem; margin-top:4px; }
.tw-red   { color:#e84545; }
.tw-green { color:#26a69a; }
.tw-gray  { color:#6b7fa8; }

/* 操作指南卡片（inline style 控色，避免 CSS class 載入問題）*/
.guide-card {
    border-radius:0 10px 10px 0;
    padding:14px 20px;
    margin:8px 0;
}

/* 今日操作標題 */
.section-bar {
    font-size:.76rem; font-weight:600; color:#4c9be8;
    letter-spacing:.12em; text-transform:uppercase;
    display:flex; align-items:center; gap:8px;
    margin: 0 0 10px;
}
.section-bar::after {
    content:''; flex:1; height:1px;
    background:linear-gradient(90deg,rgba(76,155,232,.35),transparent);
}

/* st.metric 卡片底色 */
div[data-testid="metric-container"] {
    background:#131929;
    border:1px solid rgba(76,155,232,.18);
    border-radius:10px;
    padding:12px 16px;
}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
#  ① 快取數據函數（唯一接觸 yfinance 的層）
# ═══════════════════════════════════════════════════════════════

@st.cache_data(ttl=300, show_spinner=False)
def fetch_realtime_quotes(codes: tuple[str, ...]) -> dict[str, dict]:
    """
    批次獲取即時報價，快取 5 分鐘。

    - 使用 period='5d' 確保跨越週末也能取到最後一個完整交易日收盤
    - 依序嘗試 .TW / .TWO 後綴
    - 返回 {代號: {price, prev_close, change, change_pct, volume, date}}
    """
    quotes: dict[str, dict] = {}
    for code in codes:
        for suffix in ('.TW', '.TWO'):
            try:
                raw = yf.download(
                    f"{code}{suffix}",
                    period='5d', interval='1d',
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
                    'price':      curr,
                    'prev_close': prev,
                    'change':     chg,
                    'change_pct': chg_p,
                    'volume':     float(raw['Volume'].iloc[-1]),
                    'trade_date': raw.index[-1].strftime('%Y-%m-%d'),
                    'suffix':     suffix,
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
                f"{code}{suffix}",
                period='3mo', interval='1d',
                progress=False, auto_adjust=True,
            )
            if raw.empty:
                continue
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = [c[0] for c in raw.columns]
            raw.index = pd.to_datetime(raw.index)
            raw = raw.dropna(subset=['Open','High','Low','Close'])
            if len(raw) >= 5:
                return raw.sort_index()
        except Exception:
            continue
    return None


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_pool_history(codes: tuple[str, ...]) -> dict[str, pd.DataFrame]:
    """下載股票池歷史數據（2年），快取 1 小時，供 GA 使用。"""
    fetcher = DataFetcher(period='2y')
    return fetcher.fetch_multiple(list(codes))


# ═══════════════════════════════════════════════════════════════
#  ② 數據計算函數（純 Python，無 Streamlit 依賴）
# ═══════════════════════════════════════════════════════════════

def _tw_color_hex(val: float) -> str:
    if val > 0: return '#e84545'
    if val < 0: return '#26a69a'
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
    根據 sidebar 輸入的持股清單 + yfinance 現價，計算頂部四個 KPI。
    無任何硬編碼數值，全部來自使用者輸入與即時報價。

    Returns
    -------
    {total_assets, holding_value, total_cost, total_pnl,
     today_pnl, total_return, today_return, available_cash,
     last_trade_date}
    """
    holding_value = 0.0
    total_cost    = 0.0
    today_pnl     = 0.0
    last_date     = "─"

    for code, info in current_holdings.items():
        q = quotes.get(code)
        if q is None:
            continue
        sh = info['shares']
        holding_value += q['price'] * sh
        total_cost    += info['cost'] * sh
        today_pnl     += q['change'] * sh
        last_date = q.get('trade_date', last_date)

    total_pnl    = holding_value - total_cost
    total_assets = holding_value + available_cash
    total_ret    = total_pnl / total_cost if total_cost > 0 else 0.0
    prev_val     = holding_value - today_pnl
    today_ret    = today_pnl / prev_val if prev_val > 0 else 0.0

    return {
        'total_assets':    total_assets,
        'holding_value':   holding_value,
        'total_cost':      total_cost,
        'total_pnl':       total_pnl,
        'today_pnl':       today_pnl,
        'total_return':    total_ret,
        'today_return':    today_ret,
        'available_cash':  available_cash,
        'last_trade_date': last_date,
    }


def build_holdings_rows(current_holdings: dict, quotes: dict) -> list[dict]:
    """合併持股清單與即時報價，生成 st.dataframe 用的行列表。"""
    rows = []
    for code, info in current_holdings.items():
        q = quotes.get(code)
        if q is None:
            continue
        sh     = info['shares']
        cost   = info['cost']
        curr   = q['price']
        cost_v = cost * sh
        curr_v = curr * sh
        pnl    = curr_v - cost_v
        rows.append({
            'code': code, 'shares': sh, 'cost': cost,
            'current': curr,
            'change': q['change'],
            'change_pct': q['change_pct'],
            'cost_value': cost_v,
            'curr_value': curr_v,
            'pnl': pnl,
            'pnl_pct': pnl / cost_v if cost_v > 0 else 0.0,
            'trade_date': q.get('trade_date', '─'),
        })
    return rows


def generate_daily_guide(
        holdings_analysis: dict,
        stock_scores: dict,
        current_holdings: dict,
        available_cash: float,
        quotes: dict,
) -> list[dict]:
    """
    今日操作指南決策引擎。

    決策邏輯（依優先級）
    ─────────────────────
    對現有持股（逐一判斷）：

      ① STOP_LOSS（最高優先）
         條件：未實現虧損 > 10%  且  GA 技術信號 < -0.15
         動作：建議出清止損，計算止損金額

      ② SWITCH（次高優先）
         條件：GA 信號 < -0.10  且  找得到評分更高的候選
         換股門檻：預期超額年化報酬 > 5%
         計算：回本時間 = (未實現虧損 + 換股手續費) / 月超額收益

      ③ HOLD
         條件：GA 信號 > +0.10
         動作：繼續持有

      ④ WATCH
         條件：-0.10 ≤ 信號 ≤ +0.10
         動作：觀察，等待方向確認

    對閒置現金：

      ⑤ NEW_BUY
         條件：available_cash ≥ NT$50,000 且有評分 > 0.20 的未持有標的
         動作：建議買入，分配金額 = min(cash × 40%, 現有最大持倉市值)

      ⑥ CASH
         條件：無明確買入信號
         動作：保留現金，等待機會

    Returns
    -------
    list of action dicts，每筆包含完整的操作說明與計算數據
    """
    guide: list[dict] = []

    # ── 預計算候選股排行（未持有 + 正向信號）──
    candidates = sorted(
        [(c, s) for c, s in stock_scores.items()
         if c not in current_holdings and s > 0.10],
        key=lambda x: x[1], reverse=True,
    )

    # ── 逐一判斷現有持股 ──────────────────────────────────────
    for code, info in holdings_analysis.items():
        q = quotes.get(code)
        if q is None:
            continue

        signal      = info['recent_signal_5d']
        pnl         = info['unrealized_pnl']
        pnl_pct     = info['unrealized_pnl_pct']
        curr_val    = info['current_value']
        exp_ret     = info['expected_annual_ret']
        annual_ret  = info['annual_return']

        # ① STOP_LOSS：深度虧損 + 技術信號持續偏弱
        if pnl_pct < -0.10 and signal < -0.15:
            stop_loss_amount = curr_val
            guide.append({
                'action':     'STOP_LOSS',
                'icon':       '🛑',
                'label':      f'止損出場  {code}',
                'sell_code':  code,
                'buy_code':   None,
                'amount':     stop_loss_amount,
                'signal':     signal,
                'priority':   'URGENT',
                'summary':    (
                    f"已虧損 **{pnl_pct:.2%}**（NT${pnl:+,.0f}），"
                    f"技術信號 {signal:+.3f} 持續偏弱。"
                    f"建議出清 **{code}**，止損金額 NT${stop_loss_amount:,.0f}。"
                ),
                'detail': (
                    f"- 持倉市值：NT${curr_val:,.0f}\n"
                    f"- 未實現虧損：NT${pnl:+,.0f}（{pnl_pct:.2%}）\n"
                    f"- 近5日均信號：{signal:+.3f}（門檻 < -0.15）\n"
                    f"- 執行出清後，資金可重新配置至較強標的"
                ),
                'excess_return':   None,
                'breakeven_months': None,
            })
            continue

        # ② SWITCH：信號偏弱 + 有更佳替代
        if signal < -0.10 and candidates:
            best_code, best_score = candidates[0]
            cand_info = holdings_analysis.get(best_code)

            # 估算候選股年化報酬（從歷史數據或評分代入）
            cand_annual = (
                cand_info['annual_return']
                if cand_info else best_score * 0.30 + 0.05
            )
            excess_annual = cand_annual - exp_ret

            if excess_annual > 0.05:
                switch_cost_ntd = curr_val * SWITCH_COST_RATE
                # 回本計算：需覆蓋目前未實現虧損 + 換股手續費
                total_to_recover = max(0, -pnl) + switch_cost_ntd
                monthly_excess   = curr_val * (excess_annual / 12.0)
                breakeven_m = (
                    total_to_recover / monthly_excess
                    if monthly_excess > 0 else float('inf')
                )
                breakeven_str = (
                    f"{breakeven_m:.1f} 個月"
                    if breakeven_m != float('inf') else "無法估算"
                )

                # 判斷是否「划算」：回本時間 < 12 個月才建議換股
                is_worthwhile = breakeven_m < 12.0

                guide.append({
                    'action':    'SWITCH' if is_worthwhile else 'WATCH',
                    'icon':      '⚡' if is_worthwhile else '👁️',
                    'label':     (
                        f'建議換股  {code} → {best_code}'
                        if is_worthwhile else f'觀察  {code}'
                    ),
                    'sell_code': code,
                    'buy_code':  best_code if is_worthwhile else None,
                    'amount':    curr_val,
                    'signal':    signal,
                    'priority':  'HIGH' if is_worthwhile else 'MEDIUM',
                    'summary': (
                        f"賣出 **{code}**（信號 {signal:+.3f}），"
                        f"買入 **{best_code}**（GA評分 {best_score:+.4f}）。\n"
                        f"預期超額年化報酬 **{excess_annual:+.1%}**，"
                        f"回本時間（含目前虧損 + 手續費）**{breakeven_str}**。"
                    ) if is_worthwhile else (
                        f"**{code}** 信號偏弱（{signal:+.3f}），"
                        f"但換股回本時間 {breakeven_str} 過長，暫不建議行動。持續觀察。"
                    ),
                    'detail': (
                        f"- 現持倉市值：NT${curr_val:,.0f}\n"
                        f"- 未實現損益：NT${pnl:+,.0f}（{pnl_pct:+.2%}）\n"
                        f"- 換股手續費：NT${switch_cost_ntd:,.0f}（{SWITCH_COST_RATE:.3%}）\n"
                        f"- 需回本金額：NT${total_to_recover:,.0f}（虧損 + 手續費）\n"
                        f"- 月超額收益：NT${monthly_excess:,.0f}\n"
                        f"- 預計回本：{breakeven_str}\n"
                        f"- {best_code} 歷史年化報酬：{cand_annual:.2%}　vs　"
                        f"{code} 期望報酬：{exp_ret:.2%}"
                    ),
                    'excess_return':    excess_annual,
                    'breakeven_months': breakeven_m,
                })
                continue

        # ③ HOLD：信號正向
        if signal > 0.10:
            guide.append({
                'action':    'HOLD',
                'icon':      '✅',
                'label':     f'維持持有  {code}',
                'sell_code': None,
                'buy_code':  code,
                'amount':    curr_val,
                'signal':    signal,
                'priority':  'LOW',
                'summary': (
                    f"**{code}** 技術信號正向（{signal:+.3f} ▲），"
                    f"期望年化報酬 {exp_ret:.2%}，建議繼續持有。"
                ),
                'detail': (
                    f"- 近5日均信號：{signal:+.3f}\n"
                    f"- 期望年化報酬：{exp_ret:.2%}\n"
                    f"- 未實現損益：NT${pnl:+,.0f}（{pnl_pct:+.2%}）"
                ),
                'excess_return':    None,
                'breakeven_months': None,
            })
            continue

        # ④ WATCH：信號中性
        guide.append({
            'action':    'WATCH',
            'icon':      '👁️',
            'label':     f'觀察  {code}',
            'sell_code': None,
            'buy_code':  None,
            'amount':    curr_val,
            'signal':    signal,
            'priority':  'LOW',
            'summary': (
                f"**{code}** 信號中性（{signal:+.3f}），"
                f"暫無明確方向，等待突破確認後再行動。"
            ),
            'detail': (
                f"- 近5日均信號：{signal:+.3f}（中性區間 ±0.10）\n"
                f"- 期望年化報酬：{exp_ret:.2%}\n"
                f"- 未實現損益：NT${pnl:+,.0f}（{pnl_pct:+.2%}）"
            ),
            'excess_return':    None,
            'breakeven_months': None,
        })

    # ── 閒置現金判斷 ──────────────────────────────────────────
    # 取最大持倉市值作為單筆建倉上限（不過度集中）
    max_position = max((r['curr_value'] for r in build_holdings_rows(
                        current_holdings, quotes)), default=100_000)
    buy_amount = min(available_cash * 0.40, max_position)

    if available_cash >= 50_000 and candidates:
        best_code, best_score = candidates[0]
        # 若最佳候選已在 SWITCH 建議中，取次佳
        used = {g['buy_code'] for g in guide if g.get('buy_code')}
        fresh = [(c, s) for c, s in candidates if c not in used]
        if fresh:
            best_code, best_score = fresh[0]
            buy_shares_est = int(buy_amount / quotes.get(best_code, {}).get('price', 1) / 1000) * 1000
            price_est      = quotes.get(best_code, {}).get('price', 0)
            actual_spend   = buy_shares_est * price_est if buy_shares_est > 0 else 0

            if buy_shares_est > 0:
                guide.append({
                    'action':    'NEW_BUY',
                    'icon':      '🆕',
                    'label':     f'新建部位  {best_code}',
                    'sell_code': None,
                    'buy_code':  best_code,
                    'amount':    actual_spend,
                    'signal':    best_score,
                    'priority':  'MEDIUM',
                    'summary': (
                        f"閒置現金 NT${available_cash:,.0f}，"
                        f"建議買入 **{best_code}**（GA評分 {best_score:+.4f}）。\n"
                        f"建議投入 NT${actual_spend:,.0f}，"
                        f"可買 {buy_shares_est:,} 股（每股 NT${price_est:.1f}）。"
                    ),
                    'detail': (
                        f"- 閒置現金：NT${available_cash:,.0f}\n"
                        f"- 建議投入（40% / 不超最大持倉）：NT${actual_spend:,.0f}\n"
                        f"- 可買股數：{buy_shares_est:,} 股\n"
                        f"- GA 評分：{best_score:+.4f}\n"
                        f"- 建議分批建倉，不要一次全押"
                    ),
                    'excess_return':    None,
                    'breakeven_months': None,
                })
            else:
                guide.append({
                    'action': 'CASH',
                    'icon': '💰',
                    'label': '保留現金',
                    'sell_code': None, 'buy_code': None,
                    'amount': available_cash,
                    'signal': 0.0,
                    'priority': 'LOW',
                    'summary': (
                        f"閒置現金 NT${available_cash:,.0f}，"
                        f"目前最佳候選 **{best_code}** 現價過高，"
                        f"資金不足買進整張，建議等待回調。"
                    ),
                    'detail': '',
                    'excess_return': None,
                    'breakeven_months': None,
                })
    elif available_cash >= 50_000:
        guide.append({
            'action': 'CASH',
            'icon': '💰',
            'label': '保留現金',
            'sell_code': None, 'buy_code': None,
            'amount': available_cash,
            'signal': 0.0,
            'priority': 'LOW',
            'summary': (
                f"閒置現金 NT${available_cash:,.0f}，"
                f"目前股池中無明確正向信號標的，建議保留現金等待機會。"
            ),
            'detail': '',
            'excess_return': None,
            'breakeven_months': None,
        })

    # 按優先級排序
    _order = {'URGENT': 0, 'HIGH': 1, 'MEDIUM': 2, 'LOW': 3}
    guide.sort(key=lambda x: _order.get(x['priority'], 9))
    return guide


# ═══════════════════════════════════════════════════════════════
#  ③ Plotly 圖表函數
# ═══════════════════════════════════════════════════════════════

def chart_kline(df: pd.DataFrame, code: str) -> go.Figure:
    """台股 K 線圖（60日）+ MA5/20/60 + 成交量副圖。"""
    df = df.tail(60).copy()
    df['MA5']  = df['Close'].rolling(5).mean()
    df['MA20'] = df['Close'].rolling(20).mean()
    df['MA60'] = df['Close'].rolling(60).mean()
    vol_colors = ['#e84545' if c >= o else '#26a69a'
                  for c, o in zip(df['Close'], df['Open'])]

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.75, 0.25], vertical_spacing=0.025,
    )
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'], high=df['High'],
        low=df['Low'],   close=df['Close'],
        increasing_line_color='#e84545', increasing_fillcolor='#e84545',
        decreasing_line_color='#26a69a', decreasing_fillcolor='#26a69a',
        line_width=1, name='K線', showlegend=False,
    ), row=1, col=1)

    for col_n, color, name in [
        ('MA5', '#ffd700', 'MA5'),
        ('MA20', '#4c9be8', 'MA20'),
        ('MA60', '#ff8c00', 'MA60'),
    ]:
        fig.add_trace(go.Scatter(
            x=df.index, y=df[col_n], mode='lines',
            line=dict(color=color, width=1.4), name=name,
        ), row=1, col=1)

    fig.add_trace(go.Bar(
        x=df.index, y=df['Volume'],
        marker_color=vol_colors, marker_line_width=0,
        name='成交量', showlegend=False, opacity=0.85,
    ), row=2, col=1)

    fig.add_hline(y=float(df['Close'].iloc[-1]), row=1, col=1,
                  line_dash='dot', line_color='rgba(255,255,255,0.22)', line_width=1)

    _d = 'rgba(14,17,23,1)'
    fig.update_layout(
        title=dict(text=f'<b>{code}</b>　近 60 日 K 線', font=dict(size=14, color='#e0e8f8')),
        xaxis_rangeslider_visible=False,
        plot_bgcolor=_d, paper_bgcolor='rgba(14,17,23,0)',
        font=dict(color='#c8d4e8', size=11),
        height=500, margin=dict(l=60, r=40, t=50, b=30),
        legend=dict(orientation='h', y=1.01, x=0, bgcolor='rgba(0,0,0,0)', font=dict(size=11)),
        hovermode='x unified',
        yaxis=dict(title='股價 (NT$)', gridcolor='rgba(255,255,255,0.06)',
                   tickformat=',.1f', side='right'),
        yaxis2=dict(title='成交量', gridcolor='rgba(255,255,255,0.06)',
                    tickformat='.2s', side='right'),
    )
    fig.update_xaxes(gridcolor='rgba(255,255,255,0.06)')
    return fig


def chart_monte_carlo(mc: dict, cash: float) -> go.Figure:
    paths = mc['paths']
    n     = mc['simulation_days']
    init  = mc['initial_portfolio_value']
    bk_y  = cash * 0.30
    x     = np.arange(n + 1)
    p05, p25, p50, p75, p95 = [np.percentile(paths, p, axis=0) for p in (5,25,50,75,95)]

    fig = go.Figure()
    fig.add_hrect(y0=0, y1=bk_y, fillcolor='rgba(220,50,50,0.10)',
                  line_width=0, layer='below',
                  annotation_text="  破產風險區", annotation_position="bottom right",
                  annotation_font=dict(color='rgba(255,100,100,0.5)', size=10))

    rng = np.random.default_rng(42)
    for ix in rng.choice(len(paths), min(150, len(paths)), replace=False):
        fig.add_trace(go.Scatter(x=x, y=paths[ix], mode='lines',
                                  line=dict(color='rgba(130,170,230,0.06)', width=0.7),
                                  showlegend=False, hoverinfo='skip'))

    fig.add_trace(go.Scatter(
        x=np.concatenate([x, x[::-1]]), y=np.concatenate([p95, p05[::-1]]),
        fill='toself', fillcolor='rgba(76,155,232,0.10)',
        line=dict(color='rgba(0,0,0,0)'), name='90%信心帶', hoverinfo='skip'))
    fig.add_trace(go.Scatter(
        x=np.concatenate([x, x[::-1]]), y=np.concatenate([p75, p25[::-1]]),
        fill='toself', fillcolor='rgba(76,155,232,0.25)',
        line=dict(color='rgba(0,0,0,0)'), name='IQR帶', hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=x, y=p50, mode='lines',
                              line=dict(color='#4c9be8', width=2.5), name='中位數'))
    fig.add_trace(go.Scatter(x=x, y=p05, mode='lines',
                              line=dict(color='rgba(232,69,69,.8)', width=1.5, dash='dot'), name='5th'))
    fig.add_trace(go.Scatter(x=x, y=p95, mode='lines',
                              line=dict(color='rgba(38,166,154,.8)', width=1.5, dash='dot'), name='95th'))

    fig.add_hline(y=init,  line_dash='dash', line_color='rgba(255,255,255,.35)',
                  annotation_text=f'初始投入 NT${init:,.0f}', annotation_position='top left',
                  annotation_font=dict(color='rgba(255,255,255,.5)', size=10))
    fig.add_hline(y=bk_y, line_dash='solid', line_color='rgba(232,69,69,.75)',
                  annotation_text=f'破產門檻 NT${bk_y:,.0f}', annotation_position='bottom left',
                  annotation_font=dict(color='rgba(232,69,69,.8)', size=10))

    _d = 'rgba(14,17,23,1)'
    fig.update_layout(
        title=f"蒙地卡羅  {mc['n_simulations']:,} 路徑 × {n} 交易日",
        xaxis=dict(title='交易日', gridcolor='rgba(255,255,255,.06)'),
        yaxis=dict(title='投組市值 (NT$)', tickformat=',.0f', gridcolor='rgba(255,255,255,.06)'),
        plot_bgcolor=_d, paper_bgcolor='rgba(14,17,23,0)', font=dict(color='white'),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1,
                    bgcolor='rgba(0,0,0,0)'),
        hovermode='x unified', height=480, margin=dict(t=65),
    )
    return fig


def chart_return_dist(mc: dict) -> go.Figure:
    paths  = mc['paths']
    init   = mc['initial_portfolio_value']
    rets   = (paths[:, -1] - init) / init * 100
    loss   = rets < 0
    _d     = 'rgba(14,17,23,1)'

    fig = go.Figure()
    for mask, color, name in [
        (loss,  'rgba(232,69,69,.65)',  '虧損路徑'),
        (~loss, 'rgba(38,166,154,.65)', '獲利路徑'),
    ]:
        fig.add_trace(go.Histogram(x=rets[mask], nbinsx=40,
                                    marker_color=color, name=name))
    for pct, col in [(5,'#e84545'),(50,'white'),(95,'#26a69a')]:
        v = float(np.percentile(rets, pct))
        fig.add_vline(x=v, line_dash='dash', line_color=col, line_width=1.5,
                      annotation_text=f'{pct}th:{v:.1f}%',
                      annotation_font=dict(color=col, size=10), annotation_position='top')
    fig.add_vline(x=0, line_color='rgba(255,255,255,.25)', line_width=1.2)
    fig.update_layout(
        title='一季末報酬率分佈', barmode='overlay',
        xaxis=dict(title='報酬率 (%)', gridcolor='rgba(255,255,255,.06)'),
        yaxis=dict(title='模擬次數',   gridcolor='rgba(255,255,255,.06)'),
        plot_bgcolor=_d, paper_bgcolor='rgba(14,17,23,0)', font=dict(color='white'),
        legend=dict(orientation='h', y=1.05), height=320,
    )
    return fig


def chart_scores(sorted_stocks: list) -> go.Figure:
    codes  = [c for c, _ in sorted_stocks]
    scores = [s for _, s in sorted_stocks]
    colors = ['#e84545' if s >= 0 else '#26a69a' for s in scores]
    _d = 'rgba(14,17,23,1)'
    fig = go.Figure(go.Bar(
        x=scores, y=codes, orientation='h',
        marker_color=colors,
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
    gens, best_f, avg_f = zip(*[
        (h['generation'], h['best_fitness'], h['avg_fitness']) for h in history
    ]) if history else ([], [], [])
    _d = 'rgba(14,17,23,1)'
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=gens, y=best_f, mode='lines+markers',
                              line=dict(color='#4c9be8', width=2.2), marker=dict(size=3),
                              name='最佳'))
    fig.add_trace(go.Scatter(x=gens, y=avg_f, mode='lines',
                              line=dict(color='rgba(255,215,60,.75)', width=1.5, dash='dot'),
                              name='平均'))
    fig.update_layout(
        title='GA 適應度收斂', xaxis=dict(title='代數', gridcolor='rgba(255,255,255,.06)'),
        yaxis=dict(title='適應度', gridcolor='rgba(255,255,255,.06)'),
        plot_bgcolor=_d, paper_bgcolor='rgba(14,17,23,0)', font=dict(color='white'),
        legend=dict(orientation='h', y=1.08), height=300,
    )
    return fig


# ═══════════════════════════════════════════════════════════════
#  ④ 後端：完整量化分析流程（GA + MC + 持股分析）
# ═══════════════════════════════════════════════════════════════

def run_full_pipeline(
        available_cash, current_holdings, target_pool,
        ga_config, mc_config, top_n, _prog, _stat,
) -> dict:
    results = {}

    _stat.text("📡  Step 1/4  載入歷史數據...")
    _prog.progress(5)
    all_codes  = tuple(sorted(set(list(current_holdings.keys()) + target_pool)))
    stock_data = fetch_pool_history(all_codes)
    if not stock_data:
        raise RuntimeError("無法下載任何股票數據，請確認網路連線與代號。")
    results['stock_data']   = stock_data
    results['failed_codes'] = [c for c in all_codes if c not in stock_data]
    _prog.progress(20)

    _stat.text("🧬  Step 2/4  遺傳演算法優化（約 30~90 秒）...")
    _prog.progress(25)
    ga = GeneticAlgorithm(**ga_config)
    pool_data = {c: stock_data[c] for c in target_pool if c in stock_data}
    if not pool_data:
        raise RuntimeError("目標股池無有效數據。")
    _buf = io.StringIO()
    with redirect_stdout(_buf):
        best_params = ga.evolve(pool_data, verbose=True)
    stock_scores  = ga.score_stocks(pool_data, best_params)
    sorted_stocks = sorted(stock_scores.items(), key=lambda x: x[1], reverse=True)
    results.update({
        'best_params': best_params, 'stock_scores': stock_scores,
        'sorted_stocks': sorted_stocks, 'fitness_history': ga.fitness_history,
    })
    _prog.progress(65)

    _stat.text("🎲  Step 3/4  蒙地卡羅模擬...")
    _prog.progress(68)
    positive       = [c for c, s in sorted_stocks if s > 0]
    fallback       = [c for c, _ in sorted_stocks if c not in positive]
    selected_codes = (positive + fallback)[:top_n]
    sel_prices     = {c: stock_data[c]['Close'].dropna() for c in selected_codes if c in stock_data}
    simulator      = MonteCarloSimulator(**mc_config)
    mc_stats       = simulator.simulate_portfolio(sel_prices, available_cash=available_cash)
    results.update({'selected_codes': selected_codes, 'mc_stats': mc_stats})
    _prog.progress(85)

    _stat.text("💼  Step 4/4  分析持股機會成本...")
    _prog.progress(90)
    holdings_analysis, recommendations = {}, []
    if current_holdings:
        h_data   = {c: stock_data[c] for c in current_holdings if c in stock_data}
        analyzer = HoldingsAnalyzer()
        holdings_analysis = analyzer.analyze(current_holdings, h_data, best_params)
        cand_scores = {c: s for c, s in stock_scores.items() if c not in current_holdings and s > 0}
        cand_data   = {c: stock_data[c] for c in cand_scores if c in stock_data}
        recommendations = analyzer.recommend_switches(holdings_analysis, cand_scores, cand_data)
    results.update({'holdings_analysis': holdings_analysis, 'recommendations': recommendations})
    _prog.progress(100)
    _stat.text("✅  完成！")
    return results


# ═══════════════════════════════════════════════════════════════
#  ⑤ UI 渲染函數
# ═══════════════════════════════════════════════════════════════

# ── 側邊欄 ────────────────────────────────────────────────────

def render_sidebar() -> dict:
    with st.sidebar:
        st.markdown("## ⚙️ 策略設定")
        st.markdown("---")

        st.markdown("#### 💰 可用現金（NT$）")
        available_cash = st.number_input(
            "cash", min_value=10_000, max_value=100_000_000,
            value=500_000, step=10_000, format="%d",
            label_visibility='collapsed',
        )
        st.caption(f"NT$ {available_cash:,}")
        st.markdown("---")

        st.markdown("#### 📋 現有持股")
        st.caption("代號填純數字（不含 .TW），可動態新增 / 刪除")
        _default = pd.DataFrame({'代號': ['2330','2317'], '買入均價': [850.0, 95.0], '持有股數': [1000, 2000]})
        h_df = st.data_editor(
            _default, num_rows='dynamic', use_container_width=True,
            column_config={
                '代號':    st.column_config.TextColumn('代號',   width='small'),
                '買入均價': st.column_config.NumberColumn('均價',  min_value=0.01, format="%.2f"),
                '持有股數': st.column_config.NumberColumn('股數',  min_value=0, step=1000, format="%d"),
            },
            key='holdings_editor',
        )
        current_holdings: dict[str, dict] = {}
        for _, row in h_df.dropna(subset=['代號']).iterrows():
            code = str(row['代號']).strip()
            if code and float(row.get('買入均價', 0) or 0) > 0:
                current_holdings[code] = {
                    'cost': float(row['買入均價']),
                    'shares': int(row.get('持有股數') or 0),
                }
        st.markdown("---")

        st.markdown("#### 🎯 目標觀測股池")
        pool_text = st.text_area(
            "pool", value="2330,2317,2454,2382,2308,2881,2882,3711,6505,1301",
            height=80, label_visibility='collapsed',
        )
        target_pool = [c.strip() for c in pool_text.replace('\n',',').split(',') if c.strip()]
        st.caption(f"共 {len(target_pool)} 支")
        st.markdown("---")

        with st.expander("🧬 遺傳演算法參數"):
            pop  = st.slider("種群大小", 20, 100, 50, step=10)
            gen  = st.slider("演化代數", 20, 100, 50, step=10)
            cr   = st.slider("交叉率",   0.50, 1.00, 0.80, step=0.05)
            mr   = st.slider("變異率",   0.05, 0.30, 0.15, step=0.05)
        with st.expander("🎲 蒙地卡羅參數"):
            n_s  = st.select_slider("路徑數", options=[1000,2000,5000], value=1000)
            n_d  = st.select_slider("模擬天數", options=[21,42,63,126], value=63,
                                     format_func=lambda x: f"{x}日（{x//21}個月）")
        top_n = st.slider("📌 最終選股數", 1, 6, 3)
        st.markdown("---")

        run_clicked = st.button("🚀  執行推演", type="primary", use_container_width=True)
        if st.session_state.get('_has_results'):
            if st.button("🗑️  清除結果", use_container_width=True):
                for k in ('_results','_config','_has_results'):
                    st.session_state.pop(k, None)
                st.rerun()

    return {
        'available_cash': available_cash,
        'current_holdings': current_holdings,
        'target_pool': target_pool,
        'ga_config': {'population_size': pop, 'generations': gen,
                      'crossover_rate': cr, 'mutation_rate': mr},
        'mc_config': {'n_simulations': n_s, 'n_days': n_d},
        'top_n': top_n,
        'run_clicked': run_clicked,
    }


# ── ① 頂部資產快報 ─────────────────────────────────────────────

def render_top_kpi_bar(summary: dict):
    """
    四欄 KPI 卡片，全部由 compute_portfolio_summary() 真實計算。
    使用 inline style 確保台股配色（紅漲綠跌）正確渲染。
    """
    def _card(label, val_str, dlt_str, dlt_num):
        col = _tw_color_hex(dlt_num)
        arr = _arrow(dlt_num)
        return (
            f'<div class="kpi-wrap">'
            f'<div class="kpi-lbl">{label}</div>'
            f'<div class="kpi-val">{val_str}</div>'
            f'<div class="kpi-dlt" style="color:{col};">{arr} {dlt_str}</div>'
            f'</div>'
        )

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(_card(
        "💰 總資產（持倉 + 現金）",
        f"NT${summary['total_assets']:,.0f}",
        f"NT${summary['total_pnl']:+,.0f}  未實現損益",
        summary['total_pnl'],
    ), unsafe_allow_html=True)
    c2.markdown(_card(
        "📦 持倉市值",
        f"NT${summary['holding_value']:,.0f}",
        f"成本 NT${summary['total_cost']:,.0f}",
        summary['holding_value'] - summary['total_cost'],
    ), unsafe_allow_html=True)
    c3.markdown(_card(
        "📅 今日損益",
        f"NT${summary['today_pnl']:+,.0f}",
        f"今日報酬 {summary['today_return']:+.2%}",
        summary['today_pnl'],
    ), unsafe_allow_html=True)
    c4.markdown(_card(
        "📈 整體投資報酬率",
        f"{summary['total_return']:+.2%}",
        f"最新交易日 {summary['last_trade_date']}",
        summary['total_return'],
    ), unsafe_allow_html=True)

    st.caption(
        f"⏱ 報價快取 5 分鐘  ·  最新交易日：{summary['last_trade_date']}  ·  台股配色：紅漲綠跌"
    )


# ── ② 持股即時看板（st.dataframe + pandas Styler，無原始 HTML）──

def render_holdings_table(rows: list[dict]):
    """
    使用 st.dataframe + pandas.Styler 渲染持股表格。

    關鍵修正：
      - 完全捨棄 HTML <table>，使用原生 Streamlit dataframe
      - 以 Styler.map() 對「漲跌幅」「今日損益」「未實現損益」欄位套用台股配色
      - 使用 Styler.format() 統一數字顯示格式
      - 相容 pandas 1.x (applymap) 與 2.x (map)
    """
    st.markdown('<div class="section-bar">📋 HOLDINGS WATCHLIST</div>',
                unsafe_allow_html=True)
    if not rows:
        st.info("請在左側側邊欄輸入持股資料。")
        return

    df = pd.DataFrame([{
        '代號':       r['code'],
        '資料日期':   r['trade_date'],
        '持有股數':   r['shares'],
        '成本均價':   r['cost'],
        '現價':       r['current'],
        '漲跌幅(%)':  r['change_pct'] * 100,
        '今日損益':   r['change'] * r['shares'],
        '未實現損益': r['pnl'],
        '未實現%':    r['pnl_pct'] * 100,
        '總市值':     r['curr_value'],
    } for r in rows])

    def _tw(val):
        """台股配色：正=紅 負=綠，作用在 raw 數字上。"""
        if not isinstance(val, (int, float)) or pd.isna(val):
            return ''
        if val > 0:  return 'color: #e84545; font-weight:600'
        if val < 0:  return 'color: #26a69a; font-weight:600'
        return 'color: #6b7fa8'

    color_cols = ['漲跌幅(%)', '今日損益', '未實現損益', '未實現%']
    styled = df.style
    try:
        styled = styled.map(_tw, subset=color_cols)        # pandas >= 2.1
    except AttributeError:
        styled = styled.applymap(_tw, subset=color_cols)   # pandas < 2.1

    styled = styled.format({
        '持有股數':   '{:,}',
        '成本均價':   'NT${:.2f}',
        '現價':       'NT${:.2f}',
        '漲跌幅(%)':  '{:+.2f}%',
        '今日損益':   'NT${:+,.0f}',
        '未實現損益': 'NT${:+,.0f}',
        '未實現%':    '{:+.2f}%',
        '總市值':     'NT${:,.0f}',
    }).hide(axis='index')

    st.dataframe(styled, use_container_width=True,
                 height=min(400, len(rows) * 45 + 50))


# ── ③ 今日操作指南 ─────────────────────────────────────────────

_ACTION_STYLE = {
    'STOP_LOSS': ('rgba(50,20,20,0.85)',   '#e84545', '🛑'),
    'SWITCH':    ('rgba(40,35,10,0.85)',   '#ffd700', '⚡'),
    'NEW_BUY':   ('rgba(15,25,45,0.85)',   '#4c9be8', '🆕'),
    'HOLD':      ('rgba(15,35,25,0.85)',   '#26a69a', '✅'),
    'WATCH':     ('rgba(25,25,35,0.85)',   '#8888cc', '👁️'),
    'CASH':      ('rgba(20,22,32,0.85)',   '#6b7fa8', '💰'),
}

def render_daily_guide(guide: list[dict]):
    """
    今日操作指南：以帶顏色邊框卡片呈現每條操作建議。

    採用 inline style 確保在 Streamlit Cloud 上正確顯示，
    避免外部 CSS class 因載入順序問題失效。

    欄位設計：
      左側 1 欄：圖標 + 操作名稱
      右側 4 欄：關鍵指標（信號強度 / 超額報酬 / 回本時間 / 建議金額）
    """
    st.markdown('<div class="section-bar">📋 TODAY\'S ACTION GUIDE</div>',
                unsafe_allow_html=True)

    today = datetime.date.today().strftime("%Y-%m-%d")
    st.caption(f"根據 GA 最佳策略 + 即時報價 + 換股成本分析  ·  {today}")

    if not guide:
        st.success("✅  今日無需操作，持股狀況良好，繼續持有。")
        return

    # 摘要表格（一覽全部操作）
    summary_rows = [{
        '優先級':  item['priority'],
        '操作':    item['icon'] + '  ' + item['label'],
        '信號':    f"{item['signal']:+.3f}",
        '建議金額': f"NT${item['amount']:,.0f}" if item['amount'] else '─',
        '超額年化': f"{item['excess_return']:+.1%}" if item.get('excess_return') else '─',
        '回本時間': f"{item['breakeven_months']:.1f}月" if item.get('breakeven_months') and item['breakeven_months'] != float('inf') else '─',
    } for item in guide]

    st.dataframe(
        pd.DataFrame(summary_rows),
        use_container_width=True,
        hide_index=True,
        height=min(300, len(summary_rows) * 38 + 40),
    )

    st.markdown("---")
    st.markdown("##### 詳細操作說明")

    # 逐條詳細卡片
    for item in guide:
        bg, border_color, _icon = _ACTION_STYLE.get(
            item['action'], ('rgba(20,22,32,.85)', '#6b7fa8', '─')
        )

        # 卡片 HTML（使用 inline style，不依賴外部 CSS class）
        card_html = f"""
        <div style="background:{bg}; border-left:4px solid {border_color};
                    border-radius:0 10px 10px 0; padding:14px 20px; margin:10px 0;">
          <div style="font-size:1.05rem; font-weight:700; color:#e0e8f8; margin-bottom:6px;">
            {item['icon']}&nbsp;&nbsp;{item['label']}
          </div>
          <div style="color:#c8d4e8; font-size:.9rem; line-height:1.6;">
            {item['summary']}
          </div>
        </div>
        """
        st.markdown(card_html, unsafe_allow_html=True)

        # 展開：詳細計算過程
        if item.get('detail'):
            with st.expander("查看詳細計算"):
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("信號強度", f"{item['signal']:+.3f}")
                if item.get('excess_return'):
                    c2.metric("超額年化報酬", f"{item['excess_return']:+.1%}")
                if item.get('breakeven_months') and item['breakeven_months'] != float('inf'):
                    c3.metric("換股回本時間", f"{item['breakeven_months']:.1f} 個月")
                if item['amount']:
                    c4.metric("涉及金額", f"NT${item['amount']:,.0f}")
                st.markdown(item['detail'])


# ── ④ K 線模組 ────────────────────────────────────────────────

def render_kline_section(available_codes: list[str], quotes: dict):
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
            q = quotes[code]
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

    with st.spinner(f"載入 {code} K 線..."):
        df = fetch_kline_data(code)

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
    h52w   = float(recent['High'].max())
    s4.metric("距60日高點", f"{(last_c - h52w) / h52w:.2%}", delta_color="off")


# ── 量化分析分頁 ───────────────────────────────────────────────

def render_analysis_tabs(results: dict, config: dict):
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊  市場掃描", "🧬  GA 最佳化", "🎲  蒙地卡羅", "💼  持股分析"
    ])

    with tab1:
        ss = results['sorted_stocks']
        sc = results['selected_codes']
        n_pos = sum(1 for _, s in ss if s > 0)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("掃描股數", len(ss))
        m2.metric("正向信號", n_pos, delta=f"佔{n_pos/len(ss):.0%}" if ss else "")
        m3.metric("最終選股", len(sc))
        fh = results['fitness_history']
        m4.metric("GA最佳適應度", f"{fh[-1]['best_fitness']:.4f}" if fh else "─")
        st.markdown("---")
        ca, cb = st.columns([3,2])
        with ca: st.plotly_chart(chart_scores(ss), use_container_width=True)
        with cb:
            st.markdown("##### 評分明細")
            st.dataframe(pd.DataFrame([{
                '#': i, '代號': c, '評分': f"{s:+.4f}",
                '信號': "▲看多" if s>.10 else ("▼看空" if s<-.10 else "─中性"),
                '': "✅" if c in sc else "",
            } for i,(c,s) in enumerate(ss,1)]),
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
                      delta_color="normal" if f_b>i_b else "inverse")
        st.plotly_chart(chart_fitness(fh), use_container_width=True)
        st.markdown("##### 最佳策略參數")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**MA 交叉**")
            st.metric("短期", f"{bp['ma_short']}日"); st.metric("長期", f"{bp['ma_long']}日")
            st.metric("權重", f"{bp['ma_weight']:.2f}")
        with c2:
            st.markdown("**RSI**")
            st.metric("周期", f"{bp['rsi_period']}日"); st.metric("超買", f"{bp['rsi_ob']:.0f}")
            st.metric("超賣", f"{bp['rsi_os']:.0f}"); st.metric("權重", f"{bp['rsi_weight']:.2f}")
        with c3:
            st.markdown("**布林通道**")
            st.metric("周期", f"{bp['bb_period']}日"); st.metric("標準差", f"{bp['bb_std']:.1f}σ")
            st.metric("權重", f"{bp['bb_weight']:.2f}"); st.metric("買入閾", f"{bp['buy_threshold']:.2f}")
        with st.expander("完整 JSON"):
            st.json(bp)

    with tab3:
        mc   = results['mc_stats']
        cash = config['available_cash']
        win  = mc['win_rate']
        bk   = mc['bankruptcy_probability']
        k1,k2,k3,k4 = st.columns(4)
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
        lc, rc = st.columns([3,2])
        with lc: st.plotly_chart(chart_return_dist(mc), use_container_width=True)
        with rc:
            st.markdown("##### 報酬率分位數")
            d = mc['return_distribution']
            st.dataframe(pd.DataFrame({
                '分位': ['最壞(5%)','悲觀(25%)','中位(50%)','樂觀(75%)','最佳(95%)'],
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
        t1,t2,t3 = st.columns(3)
        t1.metric("持股總市值", f"NT${tv:,.0f}")
        t2.metric("總未實現損益", f"NT${tp:,.0f}",
                  delta=f"{tp/tc:.2%}" if tc>0 else "",
                  delta_color="normal" if tp>=0 else "inverse")
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
        st.markdown("##### 換股建議（HoldingsAnalyzer 模組）")
        if not recs:
            st.success("✅  持股狀況良好，無換股建議。")
        else:
            st.warning(f"⚡  {len(recs)} 筆建議")
            _ip = {'HIGH':'🔴','MEDIUM':'🟡','LOW':'🟢'}
            for i, r in enumerate(recs, 1):
                st.markdown(
                    f"**{_ip.get(r['priority'],'─')} 建議{i}**  "
                    f"賣 **{r['sell_code']}** → 買 **{r['buy_code']}**"
                )
                c1,c2,c3,c4 = st.columns(4)
                c1.metric(f"{r['sell_code']}期望年化", f"{r['sell_expected_ret']:.2%}",
                          delta=f"信號{r['sell_signal']:+.3f}")
                c2.metric(f"{r['buy_code']}歷史年化",  f"{r['buy_annual_ret']:.2%}",
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

def main():
    config = render_sidebar()
    ch     = config['current_holdings']
    pool   = config['target_pool']
    cash   = config['available_cash']

    # 即時報價（頁面載入即執行，與 GA 無關）
    all_codes = tuple(sorted(set(list(ch.keys()) + pool)))
    with st.spinner("⚡ 獲取即時報價..."):
        quotes = fetch_realtime_quotes(all_codes)
    st.session_state['_quotes'] = quotes

    # 標題列
    col_h, col_r = st.columns([5, 1])
    with col_h:
        st.markdown(
            "<h1 style='margin:0;font-size:1.55rem;color:#e0e8f8;'>"
            "📈 台股量化交易儀表板</h1>"
            "<p style='color:#6b7fa8;font-size:.8rem;margin:2px 0 0;'>"
            "GA × Monte Carlo × Real-time Dashboard  |  v3.0</p>",
            unsafe_allow_html=True,
        )
    with col_r:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔄 刷新報價", help="清除快取重新抓取"):
            fetch_realtime_quotes.clear()
            fetch_kline_data.clear()
            st.rerun()

    st.markdown("---")

    # ① 頂部資產快報（僅在有持股時顯示）
    if ch:
        summary = compute_portfolio_summary(ch, quotes, cash)
        render_top_kpi_bar(summary)
    else:
        st.info("ℹ️  在左側輸入持股後，此處將顯示即時資產快報。")

    st.markdown("<br>", unsafe_allow_html=True)

    # ② 持股看板
    if ch:
        rows = build_holdings_rows(ch, quotes)
        render_holdings_table(rows)

    st.markdown("<br>", unsafe_allow_html=True)

    # ③ 個股 K 線
    render_kline_section(list(dict.fromkeys(list(ch.keys()) + pool)), quotes)

    st.markdown("---")

    # 執行推演
    if config['run_clicked']:
        if not pool:
            st.error("❌  請先輸入目標觀測股池。")
            return
        for k in ('_results','_config','_has_results'):
            st.session_state.pop(k, None)
        prog = st.progress(0)
        stat = st.empty()
        try:
            results = run_full_pipeline(
                cash, ch, pool,
                config['ga_config'], config['mc_config'], config['top_n'],
                prog, stat,
            )
            st.session_state.update({'_results': results, '_config': config, '_has_results': True})
            prog.empty(); stat.empty()
            st.success("✅  分析完成！")
        except Exception as e:
            prog.empty(); stat.empty()
            st.error(f"❌  {e}")
            with st.expander("錯誤詳情"):
                st.exception(e)
            return

    # ④ 今日操作指南 + 量化分析分頁
    if st.session_state.get('_has_results'):
        res = st.session_state['_results']
        cfg = st.session_state.get('_config', config)

        # 今日操作指南（GA 完成後立即顯示）
        guide = generate_daily_guide(
            holdings_analysis = res['holdings_analysis'],
            stock_scores      = res['stock_scores'],
            current_holdings  = cfg['current_holdings'],
            available_cash    = cfg['available_cash'],
            quotes            = quotes,
        )
        render_daily_guide(guide)
        st.markdown("---")

        # 量化分析四分頁
        render_analysis_tabs(res, cfg)
    else:
        st.markdown("""
        <div style="text-align:center;padding:40px 0;color:#4a5568;">
          <div style="font-size:2.5rem;margin-bottom:10px;">🧬</div>
          <p>點擊左側 <b style="color:#4c9be8;">🚀 執行推演</b> 啟動 GA 優化與蒙地卡羅模擬</p>
          <p style="font-size:.82rem;">完成後將在此處顯示今日操作指南與完整分析</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.caption("⚠️ 本系統僅供學術研究，不構成投資建議。過去績效不代表未來報酬。")


if __name__ == '__main__':
    main()
