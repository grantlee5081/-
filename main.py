"""
main.py
───────
台股量化選股與資產配置系統 — 執行入口

使用方式
--------
1. 修改下方 ══ 使用者設定區 ══ 中的三個輸入參數：
     available_cash    可用現金（新台幣）
     current_holdings  現有持股（代號、成本、股數）
     target_stock_pool 目標觀測股池（代號列表）

2. 安裝依賴：
     pip install -r requirements.txt

3. 執行：
     python main.py

系統流程
────────
Step 1  下載股票歷史數據（yfinance，自動判斷 .TW / .TWO）
Step 2  遺傳演算法優化技術因子參數（MA / RSI / Bollinger Bands）
Step 3  依最佳策略對股票池評分並選出前 N 支標的
Step 4  蒙地卡羅模擬（≥1000 次）評估未來一季的勝率與破產機率
Step 5  分析現有持股，計算機會成本並輸出換股建議

架構說明
────────
本系統採用解耦架構（Decoupled Architecture）：
  - 程式邏輯完全與個人數據分離
  - 所有個人資料僅在本檔案的設定區中維護
  - 各功能模組可獨立測試與替換
"""

import sys
import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════
#  使用者設定區 — 請在此輸入您的個人資金與持股數據
#  程式邏輯不在此區，修改此區不影響系統運作邏輯
# ═══════════════════════════════════════════════════════════════

# 可用現金（新台幣元）
AVAILABLE_CASH: float = 500_000.0

# 現有持股
# 格式：{股票代號: {'cost': 買入均價, 'shares': 持有股數}}
CURRENT_HOLDINGS: dict[str, dict] = {
    '2330': {'cost': 850.0,  'shares': 1000},  # 台積電
    '2317': {'cost':  95.0,  'shares': 2000},  # 鴻海
}

# 目標觀測股池（待評估的候選標的）
TARGET_STOCK_POOL: list[str] = [
    '2330',  # 台積電
    '2317',  # 鴻海
    '2454',  # 聯發科
    '2382',  # 廣達
    '2308',  # 台達電
    '2881',  # 富邦金
    '2882',  # 國泰金
    '3711',  # 日月光投控
    '6505',  # 台塑化
    '1301',  # 台塑
]

# ── GA 參數（進階調整，一般使用者無需修改）──
GA_CONFIG = {
    'population_size': 50,    # 種群大小：越大搜尋越徹底，但越慢
    'generations':     50,    # 演化代數：越多越收斂，但越慢
    'crossover_rate':  0.80,  # 交叉率
    'mutation_rate':   0.15,  # 變異率
}

# ── 蒙地卡羅參數 ──
MC_CONFIG = {
    'n_simulations': 1000,  # 模擬路徑數（最低 1000 次）
    'n_days':         63,   # 模擬交易日數（63 日 ≈ 一季）
}

# ── 選股設定 ──
TOP_N_STOCKS: int = 3   # 最終選出的標的數量

# ═══════════════════════════════════════════════════════════════
#  以下為系統邏輯，使用者通常無需修改
# ═══════════════════════════════════════════════════════════════

from data_fetcher import DataFetcher
from genetic_algorithm import GeneticAlgorithm
from monte_carlo import MonteCarloSimulator
from holdings_analyzer import HoldingsAnalyzer
from performance_metrics import PerformanceMetrics


# ── 輸出格式化工具 ────────────────────────────────────────────

def banner(title: str, width: int = 62) -> None:
    """印出帶有標題的分隔線。"""
    print(f"\n{'═' * width}")
    print(f"  {title}")
    print(f"{'═' * width}")


def sub_section(title: str) -> None:
    print(f"\n  ── {title} ──")


def print_mc_result(stats: dict) -> None:
    """格式化輸出蒙地卡羅統計結果。"""
    print(f"  模擬次數    : {stats['n_simulations']:,} 次")
    print(f"  模擬天數    : {stats['simulation_days']} 個交易日（約一季）")
    print(f"  初始投組市值: NT${stats['initial_portfolio_value']:>12,.0f}")
    print(f"  實際投入資金: NT${stats['total_invested']:>12,.0f}")
    print(f"  剩餘現金    : NT${stats['unused_cash']:>12,.0f}")
    print()
    print(f"  【勝率】              {stats['win_rate_pct']}")
    print(f"  【期望季度報酬】      {stats['expected_return_pct']}")
    print(f"  【期望年化報酬】      {stats['expected_annualized_return_pct']}")
    print(f"  【動態破產機率】      {stats['bankruptcy_probability_pct']}")
    print(f"  【平均最大回撤】      {stats['avg_max_drawdown_pct']}")

    sub_section("報酬率分佈（未來一季）")
    d = stats['return_distribution']
    rows = [
        ("最壞情況  (5th pct)", d['p05']),
        ("悲觀情境 (25th pct)", d['p25']),
        ("中位數   (50th pct)", d['p50']),
        ("樂觀情境 (75th pct)", d['p75']),
        ("最佳情況 (95th pct)", d['p95']),
    ]
    for label, val in rows:
        bar = '█' * max(0, int((val + 0.5) * 20))  # 視覺化長條
        print(f"  {label}: {val:>+7.2%}  {bar}")


def print_allocation(stats: dict) -> None:
    """格式化輸出資金配置細節。"""
    sub_section("資金配置明細（資金邊界 = 可用現金）")
    for code, detail in stats['allocation_detail'].items():
        n = detail['n_shares']
        p = detail['price_now']
        cost = detail['actual_cost']
        print(f"  {code}: 每股 NT${p:>8.1f}  ×  {n:>6,} 股  "
              f"=  NT${cost:>12,.0f}")


# ── 主流程 ────────────────────────────────────────────────────

def main() -> None:

    banner("台股量化選股與資產配置系統  v1.0")
    print(f"\n  可用現金  : NT${AVAILABLE_CASH:,.0f}")
    print(f"  現有持股  : {list(CURRENT_HOLDINGS.keys())}")
    print(f"  觀測股池  : {TARGET_STOCK_POOL}")

    # ──────────────────────────────────────────────────────────
    # STEP 1  數據獲取
    # ──────────────────────────────────────────────────────────
    banner("STEP 1  數據獲取")

    fetcher = DataFetcher(period='2y', interval='1d')

    # 合併持股 + 股池，去重後統一下載
    all_codes = list(set(list(CURRENT_HOLDINGS.keys()) + TARGET_STOCK_POOL))
    print(f"\n  下載 {len(all_codes)} 支股票的歷史數據...\n")

    stock_data = fetcher.fetch_multiple(all_codes)

    failed = fetcher.get_failed_codes()
    print(f"\n  成功: {len(stock_data)} 支  |  失敗: {len(failed)} 支")
    if failed:
        print(f"  失敗代號: {failed}")

    if len(stock_data) == 0:
        print("\n  [錯誤] 無法載入任何股票數據，請確認網路連線後重試。")
        sys.exit(1)

    # ──────────────────────────────────────────────────────────
    # STEP 2  遺傳演算法優化
    # ──────────────────────────────────────────────────────────
    banner("STEP 2  遺傳演算法優化技術因子參數")

    ga = GeneticAlgorithm(
        population_size=GA_CONFIG['population_size'],
        generations=GA_CONFIG['generations'],
        crossover_rate=GA_CONFIG['crossover_rate'],
        mutation_rate=GA_CONFIG['mutation_rate'],
    )

    # 只使用目標股池中有數據的標的進行 GA 優化
    pool_data = {c: stock_data[c] for c in TARGET_STOCK_POOL if c in stock_data}
    if not pool_data:
        print("\n  [錯誤] 目標股池無有效數據。")
        sys.exit(1)

    best_params = ga.evolve(pool_data, verbose=True)

    sub_section("最佳策略參數")
    print(f"  MA 交叉   : 短期={best_params['ma_short']}日 / "
          f"長期={best_params['ma_long']}日 / "
          f"權重={best_params['ma_weight']:.2f}")
    print(f"  RSI       : 周期={best_params['rsi_period']}日 / "
          f"超買={best_params['rsi_ob']:.0f} / "
          f"超賣={best_params['rsi_os']:.0f} / "
          f"權重={best_params['rsi_weight']:.2f}")
    print(f"  布林通道  : 周期={best_params['bb_period']}日 / "
          f"標準差={best_params['bb_std']:.1f}倍 / "
          f"權重={best_params['bb_weight']:.2f}")
    print(f"  買入閾值  : {best_params['buy_threshold']:.2f}")
    print(f"  賣出閾值  : {best_params['sell_threshold']:.2f}")

    # 對股票池評分
    stock_scores = ga.score_stocks(pool_data, best_params)
    sorted_stocks = sorted(stock_scores.items(), key=lambda x: x[1], reverse=True)

    sub_section("股票池評分排行")
    for rank, (code, score) in enumerate(sorted_stocks, 1):
        bar = '▓' * max(0, int((score + 1) * 10))
        signal_str = "看多" if score > 0 else "看空"
        print(f"  {rank:2d}. {code}  {score:+.4f}  [{signal_str}]  {bar}")

    # ──────────────────────────────────────────────────────────
    # STEP 3  選出最佳標的
    # ──────────────────────────────────────────────────────────
    banner("STEP 3  選股結果")

    # 優先選信號為正的，不足則補入排名最高的標的
    positive_stocks = [c for c, s in sorted_stocks if s > 0]
    selected_codes = (positive_stocks + [c for c, _ in sorted_stocks
                                          if c not in positive_stocks])[:TOP_N_STOCKS]

    if not selected_codes:
        print("\n  [警告] 目前無任何可選標的，請擴大股票池或調整參數。")
        sys.exit(1)

    print(f"\n  選出 {len(selected_codes)} 支標的: {selected_codes}")
    if not positive_stocks:
        print("  [注意] 當前市場環境偏弱，所有標的信號均為負值，請謹慎操作。")

    # 過濾出已有數據的標的
    selected_prices = {
        c: stock_data[c]['Close'].dropna()
        for c in selected_codes if c in stock_data
    }

    # ──────────────────────────────────────────────────────────
    # STEP 4  蒙地卡羅模擬
    # ──────────────────────────────────────────────────────────
    banner("STEP 4  蒙地卡羅模擬（未來一季）")

    simulator = MonteCarloSimulator(
        n_simulations=MC_CONFIG['n_simulations'],
        n_days=MC_CONFIG['n_days'],
    )

    mc_stats = simulator.simulate_portfolio(
        selected_stocks=selected_prices,
        available_cash=AVAILABLE_CASH,
    )

    print()
    print_allocation(mc_stats)
    print()
    print_mc_result(mc_stats)

    # 風險警示
    if mc_stats['win_rate'] < 0.50:
        print("\n  ⚠  勝率低於 50%，當前策略風險偏高，建議觀望或縮小部位。")
    if mc_stats['bankruptcy_probability'] > 0.05:
        print("  ⚠  破產機率超過 5%，建議嚴格設置停損點。")

    # ──────────────────────────────────────────────────────────
    # STEP 5  現有持股分析
    # ──────────────────────────────────────────────────────────
    if not CURRENT_HOLDINGS:
        banner("STEP 5  現有持股分析")
        print("\n  （無現有持股，跳過此步驟）")
    else:
        banner("STEP 5  現有持股分析")

        # 確保持股數據已下載
        holdings_data = {
            c: stock_data[c] for c in CURRENT_HOLDINGS if c in stock_data
        }

        analyzer = HoldingsAnalyzer()

        # 5-A：分析每支持股
        holdings_analysis = analyzer.analyze(
            current_holdings=CURRENT_HOLDINGS,
            stock_data=holdings_data,
            best_params=best_params,
        )

        total_holding_value = 0.0
        total_unrealized_pnl = 0.0

        for code, info in holdings_analysis.items():
            sub_section(f"{code}  持股分析")
            pnl_sign = "+" if info['unrealized_pnl'] >= 0 else ""
            sig_label = ("看多↑" if info['current_signal'] > 0.10
                         else "看空↓" if info['current_signal'] < -0.10
                         else "中性─")
            print(f"  持有: {info['shares']:,} 股  @  成本 NT${info['cost_price']:.1f}")
            print(f"  現價: NT${info['current_price']:.1f}  "
                  f"現值: NT${info['current_value']:,.0f}")
            print(f"  未實現損益: "
                  f"{pnl_sign}NT${info['unrealized_pnl']:,.0f}  "
                  f"({pnl_sign}{info['unrealized_pnl_pct']:.2%})")
            print(f"  技術信號   : {info['current_signal']:+.3f}  [{sig_label}]")
            print(f"  近 5 日均信號: {info['recent_signal_5d']:+.3f}")
            print(f"  夏普比率   : {info['sharpe_ratio']:.3f}")
            print(f"  最大回撤   : {info['max_drawdown']:.2%}")
            print(f"  歷史年化報酬: {info['annual_return']:.2%}")
            print(f"  期望年化報酬: {info['expected_annual_ret']:.2%}  "
                  f"（信號調整後）")

            total_holding_value += info['current_value']
            total_unrealized_pnl += info['unrealized_pnl']

        print(f"\n  持股總市值  : NT${total_holding_value:,.0f}")
        pnl_sign = "+" if total_unrealized_pnl >= 0 else ""
        print(f"  總未實現損益: {pnl_sign}NT${total_unrealized_pnl:,.0f}")

        # 5-B：換股建議
        banner("STEP 6  換股建議分析")

        # 候選股 = 股票池中評分為正且不在現有持股的標的
        candidate_scores = {
            c: s for c, s in stock_scores.items()
            if c not in CURRENT_HOLDINGS and s > 0
        }
        candidate_data = {
            c: stock_data[c] for c in candidate_scores if c in stock_data
        }

        recommendations = analyzer.recommend_switches(
            holdings_analysis=holdings_analysis,
            candidate_scores=candidate_scores,
            candidate_data=candidate_data,
            min_excess_return=0.05,
        )

        if not recommendations:
            print("\n  目前持股狀況良好，暫無換股建議。")
            print("  （所有持股期望報酬均高於候選標的，或信號仍屬正向）")
        else:
            print(f"\n  發現 {len(recommendations)} 筆換股機會：\n")
            for i, rec in enumerate(recommendations, 1):
                priority_icons = {'HIGH': '🔴', 'MEDIUM': '🟡', 'LOW': '🟢'}
                icon = priority_icons.get(rec['priority'], '─')
                print(f"  [{i}] {icon} 優先級: {rec['priority']}")
                print(f"       賣出: {rec['sell_code']}")
                print(f"         └ 技術信號 {rec['sell_signal']:+.3f}  "
                      f"期望年化報酬 {rec['sell_expected_ret']:.2%}  "
                      f"Sharpe {rec['sell_sharpe']:.3f}")
                print(f"       買入: {rec['buy_code']}")
                print(f"         └ GA評分 {rec['buy_ga_score']:+.4f}  "
                      f"歷史年化報酬 {rec['buy_annual_ret']:.2%}  "
                      f"Sharpe {rec['buy_sharpe']:.3f}")
                print(f"       現有持倉市值  : NT${rec['current_holding_value']:,.0f}")
                print(f"       換股成本      : NT${rec['switch_cost_ntd']:,.0f}"
                      f"  ({rec['switch_cost_rate']:.3%})")
                print(f"       預期超額年化報酬: {rec['excess_return_annual']:+.2%}")
                print(f"       年度機會成本  : NT${rec['opportunity_cost_annual']:,.0f}")
                print(f"       預計回本時間  : {rec['payback_str']}")
                print()

    # ──────────────────────────────────────────────────────────
    # 最終摘要
    # ──────────────────────────────────────────────────────────
    banner("系統分析摘要")
    print(f"\n  可用現金        : NT${AVAILABLE_CASH:,.0f}")
    print(f"  建議投資標的    : {selected_codes}")
    print(f"  實際投入資金    : NT${mc_stats['total_invested']:,.0f}")
    print(f"  一季期望報酬    : {mc_stats['expected_return_pct']}")
    print(f"  年化期望報酬    : {mc_stats['expected_annualized_return_pct']}")
    print(f"  勝率            : {mc_stats['win_rate_pct']}")
    print(f"  動態破產機率    : {mc_stats['bankruptcy_probability_pct']}")
    print(f"  平均最大回撤    : {mc_stats['avg_max_drawdown_pct']}")

    print(f"""
  ─────────────────────────────────────────────────────
  免責聲明：本系統僅供學術研究與程式教育用途。
  過去績效不代表未來報酬，投資有風險，請謹慎評估。
  本系統輸出結果不構成任何投資建議。
  ─────────────────────────────────────────────────────
""")


if __name__ == '__main__':
    main()
