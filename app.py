"""
app.py
──────
台股量化選股與資產配置系統 — Streamlit 網頁介面

本地運行：
    streamlit run app.py

部署至 Streamlit Cloud：
    見檔案末尾的 [部署說明]

架構說明：
    ┌─────────────────────────────────────────────┐
    │  前端（Streamlit）                           │
    │  app.py ─ 輸入收集、進度顯示、圖表渲染       │
    └────────────────┬────────────────────────────┘
                     │ 呼叫
    ┌────────────────▼────────────────────────────┐
    │  後端（封裝於 run_full_pipeline()）           │
    │  data_fetcher → genetic_algorithm            │
    │  → monte_carlo → holdings_analyzer          │
    └─────────────────────────────────────────────┘

    前端只負責呈現，所有資料計算邏輯均在後端完成。
"""

# ── 標準函式庫 ──────────────────────────────────────────────────
import io
from contextlib import redirect_stdout

# ── 第三方套件 ──────────────────────────────────────────────────
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── 後端模組（與 app.py 同目錄）────────────────────────────────
from data_fetcher import DataFetcher
from genetic_algorithm import GeneticAlgorithm
from monte_carlo import MonteCarloSimulator
from holdings_analyzer import HoldingsAnalyzer
from performance_metrics import PerformanceMetrics
from technical_factors import TechnicalFactors


# ═══════════════════════════════════════════════════════════════
#  頁面基礎設定（必須是第一個 Streamlit 指令）
# ═══════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="台股量化選股系統",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "台股量化選股與資產配置系統\n整合 GA × 蒙地卡羅 × 技術因子分析",
    },
)

# ── 自訂 CSS（暗色系卡片 & 標籤樣式）──
st.markdown("""
<style>
/* KPI 卡片 */
div[data-testid="metric-container"] {
    background: #1e2130;
    border: 1px solid rgba(76,155,232,0.25);
    border-radius: 10px;
    padding: 14px 18px;
}
/* 分區標題左側色條 */
.section-title {
    font-size: 1.05rem;
    font-weight: 600;
    border-left: 4px solid #4c9be8;
    padding-left: 10px;
    margin: 18px 0 10px;
    color: #e0e8f8;
}
/* 換股建議區塊 */
.rec-card {
    background: #1e2130;
    border-left: 4px solid #ffd700;
    border-radius: 6px;
    padding: 14px 18px;
    margin-bottom: 12px;
}
/* 破產警示區 */
.danger-box {
    background: rgba(255,75,75,0.12);
    border: 1px solid rgba(255,75,75,0.4);
    border-radius: 8px;
    padding: 12px 16px;
}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
#  後端：完整量化分析流程（前端呼叫此函數，等待結果）
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
    執行完整量化分析流程並回傳所有結果。

    此函數封裝所有後端邏輯（數據獲取 → GA → 蒙地卡羅 → 持股分析），
    前端只需傳入設定，接收結構化的結果字典後負責渲染。

    Parameters
    ----------
    available_cash   : 可用現金（NT$）
    current_holdings : {代號: {cost, shares}}
    target_pool      : 目標股池代號列表
    ga_config        : GA 超參數字典
    mc_config        : 蒙地卡羅超參數字典
    top_n            : 最終選股數量
    _progress        : Streamlit 進度條物件
    _status          : Streamlit 狀態文字物件

    Returns
    -------
    dict：包含所有分析結果的結構化字典
    """
    results = {}

    # ── Step 1 / 4：數據獲取 ──────────────────────────────────
    _status.text("📡  Step 1/4  正在下載股票數據...")
    _progress.progress(5)

    fetcher = DataFetcher(period='2y')
    all_codes = list(set(list(current_holdings.keys()) + target_pool))
    stock_data = fetcher.fetch_multiple(all_codes)

    if not stock_data:
        raise RuntimeError(
            "無法下載任何股票數據，請確認網路連線，"
            "或股票代號是否正確（純數字，不含 .TW 後綴）。"
        )

    results['stock_data']    = stock_data
    results['failed_codes']  = fetcher.get_failed_codes()
    _progress.progress(20)

    # ── Step 2 / 4：遺傳演算法優化 ────────────────────────────
    _status.text("🧬  Step 2/4  遺傳演算法優化中（約需 30~90 秒）...")
    _progress.progress(25)

    ga = GeneticAlgorithm(
        population_size = ga_config['population_size'],
        generations     = ga_config['generations'],
        crossover_rate  = ga_config['crossover_rate'],
        mutation_rate   = ga_config['mutation_rate'],
    )

    pool_data = {c: stock_data[c] for c in target_pool if c in stock_data}
    if not pool_data:
        raise RuntimeError("目標股池中無任何有效數據，請確認股票代號。")

    # 抑制 GA 的 print 輸出，避免污染 Streamlit 畫面
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

    # ── Step 3 / 4：蒙地卡羅模擬 ─────────────────────────────
    _status.text("🎲  Step 3/4  蒙地卡羅模擬中...")
    _progress.progress(68)

    # 選出最終標的（正信號優先；不足時補入負信號的高排名股）
    positive = [c for c, s in sorted_stocks if s > 0]
    fallback = [c for c, _ in sorted_stocks if c not in positive]
    selected_codes  = (positive + fallback)[:top_n]
    selected_prices = {
        c: stock_data[c]['Close'].dropna()
        for c in selected_codes if c in stock_data
    }

    simulator = MonteCarloSimulator(
        n_simulations = mc_config['n_simulations'],
        n_days        = mc_config['n_days'],
    )
    mc_stats = simulator.simulate_portfolio(
        selected_stocks = selected_prices,
        available_cash  = available_cash,
    )

    results['selected_codes']  = selected_codes
    results['selected_prices'] = selected_prices
    results['mc_stats']        = mc_stats
    results['all_positive']    = len(positive) == len(sorted_stocks)
    _progress.progress(85)

    # ── Step 4 / 4：持股分析 ──────────────────────────────────
    _status.text("💼  Step 4/4  分析現有持股...")
    _progress.progress(90)

    holdings_analysis = {}
    recommendations   = []

    if current_holdings:
        holdings_data = {c: stock_data[c] for c in current_holdings if c in stock_data}
        analyzer = HoldingsAnalyzer()

        holdings_analysis = analyzer.analyze(
            current_holdings = current_holdings,
            stock_data       = holdings_data,
            best_params      = best_params,
        )

        candidate_scores = {
            c: s for c, s in stock_scores.items()
            if c not in current_holdings and s > 0
        }
        candidate_data = {c: stock_data[c] for c in candidate_scores if c in stock_data}

        recommendations = analyzer.recommend_switches(
            holdings_analysis = holdings_analysis,
            candidate_scores  = candidate_scores,
            candidate_data    = candidate_data,
        )

    results['holdings_analysis'] = holdings_analysis
    results['recommendations']   = recommendations

    _progress.progress(100)
    _status.text("✅  分析完成！")
    return results


# ═══════════════════════════════════════════════════════════════
#  Plotly 圖表函數（後端資料 → 前端圖表物件）
# ═══════════════════════════════════════════════════════════════

def chart_monte_carlo(mc_stats: dict, available_cash: float) -> go.Figure:
    """
    蒙地卡羅模擬主圖。

    圖層（由下到上）：
      1. 破產風險區：紅色半透明底色
      2. 150 條隨機抽樣路徑：極淡藍灰，傳達路徑密度
      3. 90% 信心帶 (5%~95%)：淡藍填色
      4. IQR 帶 (25%~75%)：中藍填色
      5. 中位數路徑：亮藍粗線
      6. 5th / 95th 邊界線：紅點 / 綠點虛線
      7. 初始資金水平線：白色虛線標注
      8. 破產門檻水平線：紅色實線標注

    破產門檻定義：資金縮水至原始可用現金的 30% 以下
    """
    paths      = mc_stats['paths']
    n_days     = mc_stats['simulation_days']
    init_val   = mc_stats['initial_portfolio_value']
    bankrupt_y = available_cash * 0.30
    x          = np.arange(n_days + 1)

    # 計算分位數路徑（向量化）
    p05 = np.percentile(paths, 5,  axis=0)
    p25 = np.percentile(paths, 25, axis=0)
    p50 = np.percentile(paths, 50, axis=0)
    p75 = np.percentile(paths, 75, axis=0)
    p95 = np.percentile(paths, 95, axis=0)

    fig = go.Figure()

    # ── 1. 破產風險區底色 ──
    fig.add_hrect(
        y0=0, y1=bankrupt_y,
        fillcolor='rgba(220,50,50,0.10)',
        line_width=0,
        layer='below',
        annotation_text="  破產風險區（損失 >70%）",
        annotation_position="bottom right",
        annotation_font=dict(color='rgba(255,100,100,0.6)', size=11),
    )

    # ── 2. 樣本路徑（150 條，半透明）──
    n_sample  = min(150, len(paths))
    sample_ix = np.random.default_rng(seed=42).choice(len(paths), n_sample, replace=False)
    for ix in sample_ix:
        fig.add_trace(go.Scatter(
            x=x, y=paths[ix],
            mode='lines',
            line=dict(color='rgba(130,170,230,0.07)', width=0.7),
            showlegend=False,
            hoverinfo='skip',
        ))

    # ── 3. 90% 信心帶 ──
    fig.add_trace(go.Scatter(
        x=np.concatenate([x, x[::-1]]),
        y=np.concatenate([p95, p05[::-1]]),
        fill='toself',
        fillcolor='rgba(76,155,232,0.12)',
        line=dict(color='rgba(0,0,0,0)'),
        name='90% 信心帶',
        hoverinfo='skip',
    ))

    # ── 4. IQR 帶（25%~75%）──
    fig.add_trace(go.Scatter(
        x=np.concatenate([x, x[::-1]]),
        y=np.concatenate([p75, p25[::-1]]),
        fill='toself',
        fillcolor='rgba(76,155,232,0.28)',
        line=dict(color='rgba(0,0,0,0)'),
        name='IQR 帶 (25%~75%)',
        hoverinfo='skip',
    ))

    # ── 5. 中位數路徑 ──
    fig.add_trace(go.Scatter(
        x=x, y=p50,
        mode='lines',
        line=dict(color='#4c9be8', width=2.5),
        name='中位數路徑',
    ))

    # ── 6a. 5th pct 邊界（悲觀）──
    fig.add_trace(go.Scatter(
        x=x, y=p05,
        mode='lines',
        line=dict(color='rgba(255,110,110,0.85)', width=1.5, dash='dot'),
        name='5th pct（悲觀）',
    ))

    # ── 6b. 95th pct 邊界（樂觀）──
    fig.add_trace(go.Scatter(
        x=x, y=p95,
        mode='lines',
        line=dict(color='rgba(100,220,130,0.85)', width=1.5, dash='dot'),
        name='95th pct（樂觀）',
    ))

    # ── 7. 初始資金線 ──
    fig.add_hline(
        y=init_val,
        line_dash='dash',
        line_color='rgba(255,255,255,0.40)',
        line_width=1.5,
        annotation_text=f'初始投入  NT${init_val:,.0f}',
        annotation_position='top left',
        annotation_font=dict(color='rgba(255,255,255,0.55)', size=11),
    )

    # ── 8. 破產門檻線 ──
    fig.add_hline(
        y=bankrupt_y,
        line_dash='solid',
        line_color='rgba(255,80,80,0.80)',
        line_width=2,
        annotation_text=f'破產門檻  NT${bankrupt_y:,.0f}',
        annotation_position='bottom left',
        annotation_font=dict(color='rgba(255,80,80,0.85)', size=11),
    )

    # ── 版面 ──
    fig.update_layout(
        title=dict(
            text=(f"蒙地卡羅模擬路徑圖  —  "
                  f"{mc_stats['n_simulations']:,} 條路徑 × {n_days} 個交易日（約"
                  f" {n_days // 21} 個月）"),
            font=dict(size=15),
        ),
        xaxis=dict(
            title='交易日',
            gridcolor='rgba(255,255,255,0.07)',
            showgrid=True,
            tickmode='linear',
            dtick=max(1, n_days // 10),
        ),
        yaxis=dict(
            title='投資組合市值（NT$）',
            tickformat=',.0f',
            gridcolor='rgba(255,255,255,0.07)',
            showgrid=True,
        ),
        plot_bgcolor ='rgba(14,17,23,1)',
        paper_bgcolor='rgba(14,17,23,0)',
        font=dict(color='white'),
        legend=dict(
            orientation='h',
            yanchor='bottom', y=1.02,
            xanchor='right',  x=1.0,
            bgcolor='rgba(0,0,0,0)',
        ),
        hovermode='x unified',
        height=540,
        margin=dict(t=80, b=50),
    )
    return fig


def chart_return_distribution(mc_stats: dict) -> go.Figure:
    """
    期末報酬率分佈直方圖。

    左側紅色區域 = 虧損路徑；右側區域 = 獲利路徑。
    三條垂直虛線標示 5th / 中位數 / 95th 分位。
    """
    paths      = mc_stats['paths']
    init_val   = mc_stats['initial_portfolio_value']
    returns_pct = (paths[:, -1] - init_val) / init_val * 100.0

    # 為虧損 / 獲利的 bin 分別著色
    loss_mask  = returns_pct < 0
    profit_mask = ~loss_mask

    fig = go.Figure()
    for mask, color, name in [
        (loss_mask,   'rgba(220,60,60,0.65)',  '虧損路徑'),
        (profit_mask, 'rgba(76,155,232,0.65)', '獲利路徑'),
    ]:
        fig.add_trace(go.Histogram(
            x=returns_pct[mask],
            nbinsx=40,
            marker_color=color,
            marker_line=dict(color=color, width=0.4),
            name=name,
        ))

    # 分位數垂直線
    for pct, label, color in [
        (5,  f"5th: {np.percentile(returns_pct,5):.1f}%",  'rgba(255,110,110,0.9)'),
        (50, f"中位: {np.percentile(returns_pct,50):.1f}%", 'rgba(255,255,255,0.8)'),
        (95, f"95th: {np.percentile(returns_pct,95):.1f}%", 'rgba(100,220,130,0.9)'),
    ]:
        val = float(np.percentile(returns_pct, pct))
        fig.add_vline(
            x=val,
            line_dash='dash',
            line_color=color,
            line_width=1.5,
            annotation_text=label,
            annotation_font=dict(color=color, size=11),
            annotation_position='top',
        )

    fig.add_vline(x=0, line_color='rgba(255,255,255,0.3)', line_width=1.2)

    fig.update_layout(
        title='一季末報酬率分佈（全部路徑）',
        barmode='overlay',
        xaxis=dict(title='報酬率 (%)', gridcolor='rgba(255,255,255,0.07)'),
        yaxis=dict(title='模擬次數',   gridcolor='rgba(255,255,255,0.07)'),
        plot_bgcolor ='rgba(14,17,23,1)',
        paper_bgcolor='rgba(14,17,23,0)',
        font=dict(color='white'),
        legend=dict(orientation='h', y=1.05),
        height=360,
    )
    return fig


def chart_stock_scores(sorted_stocks: list[tuple]) -> go.Figure:
    """
    股票評分橫條圖。正分（看多）顯示青綠色；負分（看空）顯示紅色。
    """
    codes  = [c for c, _ in sorted_stocks]
    scores = [s for _, s in sorted_stocks]
    colors = ['#00d4aa' if s >= 0 else '#ff4b4b' for s in scores]

    fig = go.Figure(go.Bar(
        x=scores,
        y=codes,
        orientation='h',
        marker_color=colors,
        text=[f"{s:+.4f}" for s in scores],
        textposition='outside',
        textfont=dict(size=11),
    ))
    fig.add_vline(x=0, line_color='rgba(255,255,255,0.25)', line_width=1.2)

    fig.update_layout(
        title='GA 綜合評分排行（正值 = 看多信號）',
        xaxis=dict(
            title='綜合評分',
            gridcolor='rgba(255,255,255,0.07)',
            zeroline=False,
            range=[min(scores) * 1.3 - 0.1, max(scores) * 1.3 + 0.1],
        ),
        yaxis=dict(title='', autorange='reversed'),
        plot_bgcolor ='rgba(14,17,23,1)',
        paper_bgcolor='rgba(14,17,23,0)',
        font=dict(color='white'),
        height=max(300, len(codes) * 42 + 80),
        margin=dict(l=70, r=110, t=50, b=40),
    )
    return fig


def chart_fitness_history(history: list[dict]) -> go.Figure:
    """
    GA 適應度收斂曲線。藍線 = 最佳；黃虛線 = 平均。
    曲線越快上升代表演算法收斂速度越快。
    """
    gens   = [h['generation']   for h in history]
    best_f = [h['best_fitness']  for h in history]
    avg_f  = [h['avg_fitness']   for h in history]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=gens, y=best_f,
        mode='lines+markers',
        line=dict(color='#4c9be8', width=2.2),
        marker=dict(size=3),
        name='最佳適應度',
    ))
    fig.add_trace(go.Scatter(
        x=gens, y=avg_f,
        mode='lines',
        line=dict(color='rgba(255,210,60,0.75)', width=1.5, dash='dot'),
        name='種群平均適應度',
    ))
    fig.update_layout(
        title='GA 適應度收斂曲線（越高越好）',
        xaxis=dict(title='演化代數', gridcolor='rgba(255,255,255,0.07)'),
        yaxis=dict(title='適應度分數', gridcolor='rgba(255,255,255,0.07)'),
        plot_bgcolor ='rgba(14,17,23,1)',
        paper_bgcolor='rgba(14,17,23,0)',
        font=dict(color='white'),
        legend=dict(orientation='h', y=1.08),
        height=330,
    )
    return fig


# ═══════════════════════════════════════════════════════════════
#  側邊欄：收集使用者輸入，回傳設定字典
# ═══════════════════════════════════════════════════════════════

def render_sidebar() -> dict:
    """渲染左側邊欄，返回前端收集到的所有設定。"""

    with st.sidebar:
        st.markdown("## ⚙️ 策略設定")
        st.markdown("---")

        # ── 可用現金 ─────────────────────────────────────────
        st.markdown("#### 💰 可用現金（NT$）")
        available_cash = st.number_input(
            "可用現金",
            min_value=10_000,
            max_value=100_000_000,
            value=500_000,
            step=10_000,
            format="%d",
            label_visibility='collapsed',
        )
        st.caption(f"NT$ {available_cash:,}")

        st.markdown("---")

        # ── 現有持股 ──────────────────────────────────────────
        st.markdown("#### 📋 現有持股")
        st.caption("可新增 / 刪除列。代號填純數字（不加 .TW）。")

        _default_h = pd.DataFrame({
            '代號':   ['2330', '2317'],
            '買入均價': [850.0,  95.0],
            '持有股數': [1000,   2000],
        })

        holdings_df = st.data_editor(
            _default_h,
            num_rows='dynamic',
            use_container_width=True,
            column_config={
                '代號':    st.column_config.TextColumn('代號',    width='small'),
                '買入均價': st.column_config.NumberColumn('均價',  min_value=0.01, format="%.2f"),
                '持有股數': st.column_config.NumberColumn('股數',  min_value=0,    step=1000, format="%d"),
            },
            key='holdings_editor',
        )

        # 轉為後端格式
        current_holdings: dict[str, dict] = {}
        for _, row in holdings_df.dropna(subset=['代號']).iterrows():
            code = str(row['代號']).strip()
            if code and float(row.get('買入均價', 0) or 0) > 0:
                current_holdings[code] = {
                    'cost':   float(row['買入均價']),
                    'shares': int(row.get('持有股數') or 0),
                }

        st.markdown("---")

        # ── 目標股池 ──────────────────────────────────────────
        st.markdown("#### 🎯 目標觀測股池")
        st.caption("代號以逗號分隔，不含後綴")

        pool_text = st.text_area(
            "股池",
            value="2330,2317,2454,2382,2308,2881,2882,3711,6505,1301",
            height=90,
            label_visibility='collapsed',
            key='pool_text',
        )
        target_pool = [c.strip() for c in pool_text.replace('\n', ',').split(',') if c.strip()]
        st.caption(f"共 {len(target_pool)} 支股票")

        st.markdown("---")

        # ── 進階參數（折疊）─────────────────────────────────
        with st.expander("🧬 遺傳演算法參數"):
            pop_size = st.slider(
                "種群大小", 20, 100, 50, step=10,
                help="越大搜尋越徹底，但耗時越長",
            )
            generations = st.slider(
                "演化代數", 20, 100, 50, step=10,
                help="越多越收斂，但耗時越長",
            )
            crossover_rate = st.slider("交叉率", 0.50, 1.00, 0.80, step=0.05)
            mutation_rate  = st.slider("變異率", 0.05, 0.30, 0.15, step=0.05)

        with st.expander("🎲 蒙地卡羅參數"):
            n_simulations = st.select_slider(
                "路徑數量",
                options=[1000, 2000, 5000],
                value=1000,
            )
            n_days = st.select_slider(
                "模擬天數",
                options=[21, 42, 63, 126],
                value=63,
                format_func=lambda x: f"{x} 日（約 {x // 21} 個月）",
            )

        top_n = st.slider("📌 最終選股數量", 1, 6, 3)

        st.markdown("---")

        # ── 執行按鈕 ──────────────────────────────────────────
        run_clicked = st.button(
            "🚀  執行推演",
            type="primary",
            use_container_width=True,
            help="點擊後開始執行 GA 優化與蒙地卡羅模擬",
        )

        # 清除結果按鈕（有結果時才顯示）
        if st.session_state.get('_has_results'):
            if st.button("🗑️  清除結果", use_container_width=True):
                st.session_state.pop('_results', None)
                st.session_state.pop('_config',  None)
                st.session_state['_has_results'] = False
                st.rerun()

    return {
        'available_cash':   available_cash,
        'current_holdings': current_holdings,
        'target_pool':      target_pool,
        'ga_config': {
            'population_size': pop_size,
            'generations':     generations,
            'crossover_rate':  crossover_rate,
            'mutation_rate':   mutation_rate,
        },
        'mc_config': {
            'n_simulations': n_simulations,
            'n_days':        n_days,
        },
        'top_n':       top_n,
        'run_clicked': run_clicked,
    }


# ═══════════════════════════════════════════════════════════════
#  主頁渲染：歡迎畫面 / 結果頁
# ═══════════════════════════════════════════════════════════════

def render_welcome():
    """未執行分析時顯示的歡迎與說明畫面。"""
    st.markdown("""
    <div style="text-align:center; padding: 50px 0 30px;">
        <div style="font-size:3.5rem; margin-bottom:12px;">📈</div>
        <h2 style="margin-bottom:6px;">台股量化選股與資產配置系統</h2>
        <p style="color:rgba(255,255,255,0.55); font-size:1.05rem;">
            整合 遺傳演算法（GA）× 蒙地卡羅模擬 × 技術因子分析
        </p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.info("**🧬 遺傳演算法**\n\n"
                "自動搜尋最佳 MA / RSI / 布林通道的參數組合，"
                "對目標股池評分並選出高期望值標的。")
    with c2:
        st.info("**🎲 蒙地卡羅模擬**\n\n"
                "1,000+ 條 GBM 隨機價格路徑，"
                "評估未來一季的勝率分佈、破產機率與信心區間。")
    with c3:
        st.info("**💼 持股診斷**\n\n"
                "計算現有持股的機會成本與技術信號強度，"
                "在發現更佳標的時輸出換股建議與預計回本時間。")

    st.markdown("---")
    st.markdown("""
    #### 🚀 快速開始
    1. 在左側側邊欄輸入 **可用現金**、**現有持股** 與 **目標股池**
    2. 調整 GA / 蒙地卡羅超參數（使用預設值即可快速開始）
    3. 點擊左側 **🚀 執行推演** 按鈕，等待約 30–90 秒
    """)


def render_results(results: dict, config: dict):
    """渲染完整分析結果（四分頁佈局）。"""

    tab_scan, tab_ga, tab_mc, tab_holdings = st.tabs([
        "📊  市場掃描",
        "🧬  GA 最佳化",
        "🎲  蒙地卡羅模擬",
        "💼  持股分析",
    ])

    # ── Tab 1：市場掃描 ──────────────────────────────────────
    with tab_scan:
        st.markdown('<div class="section-title">GA 評分結果 — 目標股池掃描</div>',
                    unsafe_allow_html=True)

        sorted_stocks  = results['sorted_stocks']
        selected_codes = results['selected_codes']
        n_pos = sum(1 for _, s in sorted_stocks if s > 0)

        # KPI
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("掃描股票數",   len(sorted_stocks))
        m2.metric("正向信號",     n_pos,
                  delta=f"佔 {n_pos/len(sorted_stocks):.0%}" if sorted_stocks else "")
        m3.metric("最終選股",     len(selected_codes))
        m4.metric("GA 最佳適應度",
                  f"{results['fitness_history'][-1]['best_fitness']:.4f}" if results['fitness_history'] else "─")

        st.markdown("---")

        col_chart, col_tbl = st.columns([3, 2])

        with col_chart:
            st.plotly_chart(chart_stock_scores(sorted_stocks), use_container_width=True)

        with col_tbl:
            st.markdown("##### 評分明細")
            rows = []
            for rank, (code, score) in enumerate(sorted_stocks, 1):
                rows.append({
                    '#':   rank,
                    '代號': code,
                    '評分': f"{score:+.4f}",
                    '信號': "▲看多" if score > 0.10 else ("▼看空" if score < -0.10 else "─中性"),
                    '狀態': "✅" if code in selected_codes else "",
                })
            st.dataframe(
                pd.DataFrame(rows),
                use_container_width=True,
                hide_index=True,
                height=min(420, len(rows) * 38 + 40),
            )

        if results.get('failed_codes'):
            st.warning(f"⚠️ 以下代號無法取得數據：{results['failed_codes']}")

        if not results.get('all_positive') and n_pos < len(sorted_stocks):
            st.caption(
                "ℹ️ 部分標的信號為負；最終選股仍依排名填滿，"
                "若全池信號皆弱，建議觀望或擴大股池。"
            )

    # ── Tab 2：GA 最佳化 ─────────────────────────────────────
    with tab_ga:
        st.markdown('<div class="section-title">遺傳演算法最佳化結果</div>',
                    unsafe_allow_html=True)

        bp = results['best_params']
        fh = results['fitness_history']

        if fh:
            i_best = fh[0]['best_fitness']
            f_best = fh[-1]['best_fitness']
            m1, m2, m3 = st.columns(3)
            m1.metric("初代最佳適應度",  f"{i_best:.4f}")
            m2.metric("末代最佳適應度",  f"{f_best:.4f}")
            m3.metric("演化淨改進",      f"{f_best - i_best:+.4f}",
                      delta_color="normal" if f_best > i_best else "inverse")

        st.plotly_chart(chart_fitness_history(fh), use_container_width=True)

        st.markdown('<div class="section-title">最佳策略參數</div>',
                    unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown("**MA 交叉因子**")
            st.metric("短期均線", f"{bp['ma_short']} 日")
            st.metric("長期均線", f"{bp['ma_long']} 日")
            st.metric("MA 權重",  f"{bp['ma_weight']:.2f}")

        with c2:
            st.markdown("**RSI 因子**")
            st.metric("RSI 周期", f"{bp['rsi_period']} 日")
            st.metric("超買閾值",  f"{bp['rsi_ob']:.0f}")
            st.metric("超賣閾值",  f"{bp['rsi_os']:.0f}")
            st.metric("RSI 權重", f"{bp['rsi_weight']:.2f}")

        with c3:
            st.markdown("**布林通道因子**")
            st.metric("BB 周期",   f"{bp['bb_period']} 日")
            st.metric("標準差倍數", f"{bp['bb_std']:.1f} σ")
            st.metric("BB 權重",   f"{bp['bb_weight']:.2f}")
            st.metric("買入閾值",   f"{bp['buy_threshold']:.2f}")

        with st.expander("📋 完整參數字典（JSON）"):
            st.json(bp)

    # ── Tab 3：蒙地卡羅模擬 ─────────────────────────────────
    with tab_mc:
        st.markdown('<div class="section-title">蒙地卡羅模擬分析</div>',
                    unsafe_allow_html=True)

        mc    = results['mc_stats']
        cash  = config['available_cash']
        win   = mc['win_rate']
        bkrp  = mc['bankruptcy_probability']

        # KPI
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("勝率",         mc['win_rate_pct'],
                  delta="正期望值" if win > 0.50 else "負期望值，請謹慎",
                  delta_color="normal" if win > 0.50 else "inverse")
        k2.metric("期望季度報酬", mc['expected_return_pct'],
                  delta=f"年化 {mc['expected_annualized_return_pct']}")
        k3.metric("動態破產機率", mc['bankruptcy_probability_pct'],
                  delta="偏高，注意停損" if bkrp > 0.05 else "風險可控",
                  delta_color="inverse" if bkrp > 0.05 else "normal")
        k4.metric("平均最大回撤", mc['avg_max_drawdown_pct'])

        # 風險警示
        if win < 0.50:
            st.warning("⚠️  勝率低於 50%，當前市場環境不利，建議縮小部位或觀望。")
        if bkrp > 0.05:
            st.error("🚨  破產機率超過 5%！建議設置嚴格停損（例如：下跌 15% 出場）。")

        st.markdown("---")

        # 主圖：路徑圖
        st.plotly_chart(chart_monte_carlo(mc, cash), use_container_width=True)

        # 次圖：分佈圖 + 數據表
        left_col, right_col = st.columns([3, 2])

        with left_col:
            st.plotly_chart(chart_return_distribution(mc), use_container_width=True)

        with right_col:
            st.markdown("##### 報酬率分位數")
            d = mc['return_distribution']
            _dist_data = {
                '分位數':   ['最壞 (5%)', '悲觀 (25%)', '中位數 (50%)', '樂觀 (75%)', '最佳 (95%)'],
                '季度報酬': [f"{v:+.2%}" for v in [d['p05'], d['p25'], d['p50'], d['p75'], d['p95']]],
            }
            st.dataframe(pd.DataFrame(_dist_data), use_container_width=True, hide_index=True)

            st.markdown("##### 資金配置明細")
            alloc_rows = [
                {
                    '代號':   code,
                    '現價':   f"NT${det['price_now']:,.1f}",
                    '買入股數': f"{det['n_shares']:,} 股",
                    '投入金額': f"NT${det['actual_cost']:,.0f}",
                }
                for code, det in mc['allocation_detail'].items()
            ]
            if alloc_rows:
                st.dataframe(pd.DataFrame(alloc_rows), use_container_width=True, hide_index=True)

            st.info(
                f"**總投入** NT${mc['total_invested']:,.0f}\n\n"
                f"**剩餘現金** NT${mc['unused_cash']:,.0f}"
                f"（零頭 + 資金邊界保留）"
            )

    # ── Tab 4：持股分析 ──────────────────────────────────────
    with tab_holdings:
        st.markdown('<div class="section-title">現有持股分析</div>',
                    unsafe_allow_html=True)

        ha   = results['holdings_analysis']
        recs = results['recommendations']

        if not ha:
            st.info("ℹ️  未輸入現有持股。\n\n請在左側側邊欄的「現有持股」表格中新增股票。")
            return

        # 總覽
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

        # 各持股明細表
        st.markdown("##### 各持股診斷明細")
        h_rows = []
        for code, info in ha.items():
            sig = info['current_signal']
            h_rows.append({
                '代號':     code,
                '成本均價': f"NT${info['cost_price']:.1f}",
                '現價':     f"NT${info['current_price']:.1f}",
                '持有股數': f"{info['shares']:,}",
                '現值':     f"NT${info['current_value']:,.0f}",
                '損益 %':   f"{info['unrealized_pnl_pct']:+.2%}",
                '技術信號': f"{sig:+.3f} {'▲' if sig>0.1 else ('▼' if sig<-0.1 else '─')}",
                'Sharpe':   f"{info['sharpe_ratio']:.3f}",
                '期望年化報酬': f"{info['expected_annual_ret']:.2%}",
            })
        st.dataframe(pd.DataFrame(h_rows), use_container_width=True, hide_index=True)

        # 換股建議
        st.markdown("---")
        st.markdown('<div class="section-title">換股建議</div>', unsafe_allow_html=True)

        if not recs:
            st.success(
                "✅  目前持股狀況良好 — 技術信號正向，"
                "且期望報酬高於目標股池候選標的，暫無換股建議。"
            )
        else:
            st.warning(f"⚡  發現 **{len(recs)}** 筆換股機會，建議參考下方分析：")
            st.markdown("")

            _priority_icon = {'HIGH': '🔴', 'MEDIUM': '🟡', 'LOW': '🟢'}

            for i, rec in enumerate(recs, 1):
                icon = _priority_icon.get(rec['priority'], '─')
                with st.container():
                    st.markdown(
                        f"**{icon} 建議 {i}**（優先級：{rec['priority']}）　"
                        f"賣出 **{rec['sell_code']}** &nbsp;→&nbsp; 買入 **{rec['buy_code']}**"
                    )

                    r1, r2, r3, r4 = st.columns(4)
                    r1.metric(
                        f"{rec['sell_code']} 期望年化",
                        f"{rec['sell_expected_ret']:.2%}",
                        delta=f"信號 {rec['sell_signal']:+.3f}",
                    )
                    r2.metric(
                        f"{rec['buy_code']} 歷史年化",
                        f"{rec['buy_annual_ret']:.2%}",
                        delta=f"GA 評分 {rec['buy_ga_score']:+.4f}",
                    )
                    r3.metric(
                        "超額年化報酬",
                        f"{rec['excess_return_annual']:+.2%}",
                    )
                    r4.metric(
                        "年度機會成本",
                        f"NT${rec['opportunity_cost_annual']:,.0f}",
                    )

                    i1, i2 = st.columns(2)
                    i1.info(
                        f"💸 **換股成本**\n\n"
                        f"NT${rec['switch_cost_ntd']:,.0f} "
                        f"（{rec['switch_cost_rate']:.3%}，含稅費）"
                    )
                    i2.info(
                        f"⏱️ **預計回本時間**\n\n"
                        f"{rec['payback_str']}"
                    )

                    st.markdown("---")


# ═══════════════════════════════════════════════════════════════
#  主程式入口
# ═══════════════════════════════════════════════════════════════

def main():

    # ── 側邊欄（收集輸入）──
    config = render_sidebar()

    # ── 頁面標題 ──
    st.title("📈 台股量化選股與資產配置系統")
    st.caption(
        "整合 遺傳演算法（GA）× 蒙地卡羅模擬（GBM）× 多因子技術分析　|　"
        "資料來源：Yahoo Finance（yfinance）"
    )
    st.markdown("---")

    # ── 執行推演 ──────────────────────────────────────────────
    if config['run_clicked']:
        if not config['target_pool']:
            st.error("❌  請先在左側側邊欄輸入至少一支股票到「目標觀測股池」。")
            return

        # 清除舊結果
        st.session_state.pop('_results', None)
        st.session_state.pop('_config',  None)

        # 進度顯示元件
        progress_bar = st.progress(0, text="初始化中...")
        status_text  = st.empty()

        try:
            results = run_full_pipeline(
                available_cash   = config['available_cash'],
                current_holdings = config['current_holdings'],
                target_pool      = config['target_pool'],
                ga_config        = config['ga_config'],
                mc_config        = config['mc_config'],
                top_n            = config['top_n'],
                _progress        = progress_bar,
                _status          = status_text,
            )
            # 儲存到 session state（讓頁面重整後仍保留結果）
            st.session_state['_results']     = results
            st.session_state['_config']      = config
            st.session_state['_has_results'] = True

            progress_bar.empty()
            status_text.empty()
            st.success("✅  分析完成！請切換下方各分頁查看結果。")

        except Exception as exc:
            progress_bar.empty()
            status_text.empty()
            st.error(f"❌  執行失敗：{exc}")
            with st.expander("查看詳細錯誤訊息"):
                st.exception(exc)
            return

    # ── 顯示結果 or 歡迎畫面 ──
    if st.session_state.get('_has_results') and '_results' in st.session_state:
        render_results(
            st.session_state['_results'],
            st.session_state.get('_config', config),
        )
    else:
        render_welcome()

    # ── 頁腳免責聲明 ──
    st.markdown("---")
    st.caption(
        "⚠️ **免責聲明**：本系統僅供學術研究與程式教育用途，"
        "所有分析結果不構成任何投資建議。"
        "過去績效不代表未來報酬，投資有風險，請謹慎評估。"
    )


if __name__ == '__main__':
    main()


# ═══════════════════════════════════════════════════════════════
#  部署說明
# ═══════════════════════════════════════════════════════════════
#
# ── 本地端運行 ──────────────────────────────────────────────
#
#   1. 安裝依賴：
#        pip install -r requirements.txt
#
#   2. 啟動：
#        streamlit run app.py
#
#   3. 瀏覽器自動開啟 http://localhost:8501
#
#
# ── 部署到 Streamlit Cloud（免費，可手機查看）────────────────
#
#   Step 1  將整個「投資」資料夾上傳到 GitHub（建立新 Repository）
#           需包含以下檔案：
#             app.py
#             data_fetcher.py
#             technical_factors.py
#             performance_metrics.py
#             genetic_algorithm.py
#             monte_carlo.py
#             holdings_analyzer.py
#             requirements.txt
#             .streamlit/config.toml
#
#   Step 2  前往 https://share.streamlit.io
#           登入 GitHub 帳號
#
#   Step 3  點擊「New app」，填入：
#             Repository: 你的 GitHub repo 名稱
#             Branch:     main
#             Main file:  app.py
#
#   Step 4  點擊「Deploy!」，等待約 2-3 分鐘完成部署
#
#   Step 5  取得公開網址（格式如 https://yourname-reponame.streamlit.app）
#           即可從手機或任何裝置訪問
#
#   注意事項：
#     - Streamlit Cloud 免費方案有資源限制（1 GB RAM）
#     - 建議將 GA 種群大小設為 ≤ 50，代數 ≤ 50，以避免超時
#     - 若需更大運算量，可升級方案或改用 Railway / Render 部署
# ═══════════════════════════════════════════════════════════════
