"""
monte_carlo.py
──────────────
蒙地卡羅模擬引擎

核心功能：
  1. 使用幾何布朗運動（GBM）模擬 N 條未來價格路徑
  2. 考慮「資金邊界」：依 available_cash 計算實際可買股數
  3. 彙整投資組合路徑，輸出勝率分佈與破產機率

幾何布朗運動（GBM）公式
──────────────────────────
  S(t+1) = S(t) × exp[(μ - σ²/2)×dt + σ×√dt×Z]

  其中：
    μ  = 歷史對數報酬率的均值（日頻漂移率）
    σ  = 歷史對數報酬率的標準差（日頻波動率）
    dt = 1（日頻模擬，每步一個交易日）
    Z  ~ N(0, 1)，標準常態隨機數

GBM 假設：
  - 股價連續複利增長
  - 報酬率服從對數常態分佈
  - 波動率為常數（歷史波動率代入）
"""

import numpy as np
import pandas as pd


class MonteCarloSimulator:
    """
    蒙地卡羅投資組合模擬器。

    使用範例
    --------
    simulator = MonteCarloSimulator(n_simulations=1000, n_days=63)
    result = simulator.simulate_portfolio(
        selected_stocks={'2330': price_series, ...},
        available_cash=500_000
    )
    """

    # 破產定義：投資組合縮水至初始資金的此比例以下
    BANKRUPTCY_THRESHOLD = 0.30

    def __init__(self,
                 n_simulations: int = 1000,
                 n_days: int = 63):
        """
        Parameters
        ----------
        n_simulations : 模擬路徑數，最少 1,000 次以確保統計穩定性
        n_days        : 模擬天數，63 個交易日 ≈ 一個季度
        """
        self.n_sims = max(n_simulations, 1000)  # 強制最低 1000 次
        self.n_days = n_days

    # ── GBM 參數估計 ──────────────────────────────────────────

    @staticmethod
    def _estimate_gbm_params(prices: pd.Series) -> tuple[float, float]:
        """
        從歷史收盤價估計 GBM 的日頻參數。

        計算方式：
          1. 對數日報酬率 = ln(P_t / P_{t-1})
          2. μ_log = 對數報酬率序列的均值
          3. σ     = 對數報酬率序列的標準差

        Returns
        -------
        (mu_log, sigma)：日頻漂移率、日頻波動率
        """
        log_returns = np.log(prices / prices.shift(1)).dropna()
        mu = float(log_returns.mean())
        sigma = float(log_returns.std())
        return mu, sigma

    # ── 單股路徑模擬 ──────────────────────────────────────────

    def simulate_price_paths(self, prices: pd.Series) -> np.ndarray:
        """
        為單一股票生成 n_simulations 條未來價格路徑。

        GBM 向量化實作：
          1. 一次性生成 (n_sims, n_days) 的標準常態矩陣 Z
          2. 計算每日對數報酬增量 = (μ - σ²/2) + σ×Z
          3. 對增量取指數並累積乘積（等同於指數累積）
          4. 以最後一個收盤價作為起點縮放路徑

        Returns
        -------
        np.ndarray, shape (n_simulations, n_days+1)
          第 0 列為當前價格（所有路徑相同的起點）
        """
        mu, sigma = self._estimate_gbm_params(prices)
        s0 = float(prices.iloc[-1])  # 模擬起點：最新收盤價

        # 生成隨機震動矩陣
        Z = np.random.standard_normal((self.n_sims, self.n_days))

        # 每日對數報酬：確定性漂移 + 隨機擴散
        drift = (mu - 0.5 * sigma ** 2)          # 校正後的漂移項
        diffusion = sigma * Z                     # 擴散項

        log_increments = drift + diffusion        # shape: (n_sims, n_days)

        # 累積對數報酬 → 轉換為價格比例
        cumulative_log = np.cumsum(log_increments, axis=1)  # (n_sims, n_days)
        price_ratios = np.exp(cumulative_log)                # (n_sims, n_days)

        # 建構完整路徑陣列（含起點）
        paths = np.zeros((self.n_sims, self.n_days + 1))
        paths[:, 0] = s0
        paths[:, 1:] = s0 * price_ratios

        return paths

    # ── 投資組合模擬 ──────────────────────────────────────────

    def simulate_portfolio(self,
                           selected_stocks: dict[str, pd.Series],
                           available_cash: float,
                           weights: dict[str, float] | None = None) -> dict:
        """
        模擬由多支股票組成的投資組合的未來價值。

        資金邊界處理（核心邏輯）：
          1. 依權重計算各股票的分配資金
          2. 以「整張（1000股）」為單位向下取整
             → 台股最小交易單位為 1 張 = 1000 股
          3. 實際持倉市值 = 可買股數 × 當前股價
          4. 模擬的是「實際持倉」的未來市值，非抽象的倉位比例
          5. 未使用資金（零頭）計入現金，不加入模擬

        Parameters
        ----------
        selected_stocks : {代號: 收盤價序列}
        available_cash  : 可用現金（新台幣）
        weights         : {代號: 配置比例}，None 則等權重分配

        Returns
        -------
        dict：詳細的模擬統計結果
        """
        if not selected_stocks:
            raise ValueError("selected_stocks 不可為空")

        n_stocks = len(selected_stocks)

        # 等權重配置（若未指定）
        if weights is None:
            weights = {c: 1.0 / n_stocks for c in selected_stocks}

        # ── 資金邊界：計算實際可買股數 ──
        allocation_detail: dict[str, dict] = {}
        total_invested = 0.0

        for code, prices in selected_stocks.items():
            alloc_cash = available_cash * weights.get(code, 1.0 / n_stocks)
            price_now = float(prices.iloc[-1])

            # 台股以 1,000 股（1張）為最小交易單位
            n_shares = int(alloc_cash / price_now / 1000) * 1000
            actual_cost = n_shares * price_now

            allocation_detail[code] = {
                'price_now': price_now,
                'allocated_cash': alloc_cash,
                'n_shares': n_shares,
                'actual_cost': actual_cost,
            }
            total_invested += actual_cost

        unused_cash = available_cash - total_invested

        # ── 生成每支股票的模擬路徑，換算為持倉市值 ──
        portfolio_paths = np.zeros((self.n_sims, self.n_days + 1))

        for code, prices in selected_stocks.items():
            n_shares = allocation_detail[code]['n_shares']
            if n_shares == 0:
                continue  # 資金不足買進此股，跳過
            price_paths = self.simulate_price_paths(prices)   # (n_sims, n_days+1)
            portfolio_paths += price_paths * n_shares          # 市值路徑

        # 加入未使用的現金（固定不增減）
        portfolio_paths += unused_cash

        # ── 計算統計指標 ──
        initial_value = float(portfolio_paths[:, 0].mean())
        stats = self._compute_statistics(portfolio_paths, initial_value, available_cash)

        # 加入配置細節供輸出使用
        stats['allocation_detail'] = allocation_detail
        stats['total_invested'] = total_invested
        stats['unused_cash'] = unused_cash

        return stats

    # ── 統計計算 ──────────────────────────────────────────────

    def _compute_statistics(self,
                             paths: np.ndarray,
                             initial_value: float,
                             available_cash: float) -> dict:
        """
        從模擬路徑陣列計算所有統計指標。

        指標說明
        --------
        勝率
          最終投組價值 > 初始投資金額的模擬比例。

        破產機率
          任意時間點投組價值 < 初始投資 × BANKRUPTCY_THRESHOLD 的模擬比例。
          （動態破產：只要曾觸及門檻即算破產，非僅考慮期末）

        期望報酬率
          所有模擬路徑期末報酬率的算術均值。

        報酬率分位數
          5%, 25%, 50%, 75%, 95% 分位，建構完整分佈輪廓。

        平均最大回撤
          每條路徑的最大回撤（路徑內最嚴重的峰谷跌幅），取均值。
        """
        final_values = paths[:, -1]                          # 期末市值
        returns = (final_values - initial_value) / initial_value  # 期末報酬率

        # 勝率（期末 > 期初）
        win_rate = float(np.mean(final_values > initial_value))

        # 動態破產機率（路徑中任意時點觸及門檻）
        bankruptcy_level = available_cash * self.BANKRUPTCY_THRESHOLD
        ever_bankrupt = np.any(paths < bankruptcy_level, axis=1)
        bankruptcy_prob = float(np.mean(ever_bankrupt))

        # 報酬率分位數
        pcts = np.percentile(returns, [5, 25, 50, 75, 95])

        # 年化期望報酬（基於模擬天數推算全年）
        annualized_returns = (1.0 + returns) ** (252.0 / self.n_days) - 1.0
        exp_annual_ret = float(np.mean(annualized_returns))

        # 每條路徑的最大回撤
        running_max = np.maximum.accumulate(paths, axis=1)
        drawdowns = (paths - running_max) / running_max
        path_mdds = drawdowns.min(axis=1)                    # 每條路徑的 MDD
        avg_mdd = float(np.mean(path_mdds))

        return {
            # 元數據
            'n_simulations':   self.n_sims,
            'simulation_days': self.n_days,
            'initial_portfolio_value': initial_value,
            'available_cash':  available_cash,

            # 核心指標
            'win_rate':             win_rate,
            'win_rate_pct':         f"{win_rate:.1%}",
            'bankruptcy_probability':     bankruptcy_prob,
            'bankruptcy_probability_pct': f"{bankruptcy_prob:.2%}",

            # 報酬率
            'expected_return':     float(np.mean(returns)),
            'expected_return_pct': f"{np.mean(returns):.2%}",
            'expected_annualized_return':     exp_annual_ret,
            'expected_annualized_return_pct': f"{exp_annual_ret:.2%}",

            # 報酬率分佈
            'return_distribution': {
                'p05': float(pcts[0]),
                'p25': float(pcts[1]),
                'p50': float(pcts[2]),
                'p75': float(pcts[3]),
                'p95': float(pcts[4]),
            },

            # 風險指標
            'avg_max_drawdown':     avg_mdd,
            'avg_max_drawdown_pct': f"{avg_mdd:.2%}",
            'best_case_return':     float(np.max(returns)),
            'worst_case_return':    float(np.min(returns)),

            # 原始路徑（供外部視覺化使用）
            'paths': paths,
        }
