"""
performance_metrics.py
──────────────────────
績效指標計算工具模組

提供計算夏普比率、最大回撤、年化報酬等功能。
所有方法皆為靜態方法，可直接呼叫無需實例化。
"""

import numpy as np
import pandas as pd


class PerformanceMetrics:
    """靜態績效指標計算器，供 GA 適應度函數及其他模組使用。"""

    # 台灣無風險利率（年化），以一年期定存為基準
    RISK_FREE_RATE = 0.015

    # ── 基礎指標 ──────────────────────────────────────────────

    @staticmethod
    def daily_returns(prices: pd.Series) -> pd.Series:
        """
        計算日報酬率序列。
        使用 dropna() 去除因 pct_change() 產生的第一列 NaN。
        """
        return prices.pct_change().dropna()

    @staticmethod
    def cumulative_return(prices: pd.Series) -> float:
        """
        計算整段期間的累積報酬率。
        公式：(期末價 / 期初價) - 1
        """
        if len(prices) < 2:
            return 0.0
        return float(prices.iloc[-1] / prices.iloc[0]) - 1.0

    @staticmethod
    def annualized_return(prices: pd.Series) -> float:
        """
        計算年化報酬率（幾何年化）。
        公式：(1 + 累積報酬) ^ (252 / 交易日數) - 1
        """
        n = len(prices)
        if n < 2:
            return 0.0
        cum_ret = PerformanceMetrics.cumulative_return(prices)
        return float((1.0 + cum_ret) ** (252.0 / n) - 1.0)

    @staticmethod
    def annualized_volatility(returns: pd.Series) -> float:
        """
        計算年化波動率。
        公式：日報酬率標準差 × √252
        """
        if len(returns) < 2:
            return 0.0
        return float(returns.std() * np.sqrt(252))

    # ── 風險調整後指標 ────────────────────────────────────────

    @staticmethod
    def sharpe_ratio(returns: pd.Series,
                     risk_free_rate: float = RISK_FREE_RATE) -> float:
        """
        計算夏普比率（年化）。

        公式：(平均超額報酬 / 報酬率標準差) × √252

        Parameters
        ----------
        returns       : 日報酬率序列
        risk_free_rate: 年化無風險利率，預設 1.5%

        Returns
        -------
        float：夏普比率；資料不足或波動率為零時回傳 0
        """
        if len(returns) < 10 or returns.std() == 0:
            return 0.0
        daily_rf = risk_free_rate / 252.0
        excess = returns - daily_rf
        return float(excess.mean() / excess.std() * np.sqrt(252))

    @staticmethod
    def max_drawdown(prices: pd.Series) -> float:
        """
        計算最大回撤（Maximum Drawdown, MDD）。

        演算法：
          1. 計算日報酬率並去除 NaN
          2. 從日報酬率重建累積淨值曲線
          3. 對累積淨值取擴張期最大值（歷史高點）
          4. MDD = (當前值 - 歷史高點) / 歷史高點 的最小值

        Returns
        -------
        float：負值，例如 -0.25 代表最大曾虧損 25%
        """
        rets = PerformanceMetrics.daily_returns(prices)
        if len(rets) < 2:
            return 0.0
        cumulative = (1.0 + rets).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        return float(drawdown.min())

    # ── GA 適應度函數 ─────────────────────────────────────────

    @staticmethod
    def fitness_score(strategy_returns: pd.Series,
                      w_sharpe: float = 0.40,
                      w_mdd: float = 0.30,
                      w_return: float = 0.30) -> float:
        """
        GA 適應度函數：將三個指標加權合併為單一分數。

        各分指標正規化邏輯：
          - Sharpe Score  : sharpe / 3.0，限縮至 [-1, 1]
            （夏普 3 以上視為滿分，負夏普受懲罰）
          - MDD Score     : 1 + mdd，限縮至 [0, 1]
            （mdd = -0.20 → 分數 0.80；mdd = 0 → 1.00）
          - Return Score  : cumulative_return / 1.0，限縮至 [-1, 1]
            （累積 100% 以上視為滿分）

        Parameters
        ----------
        strategy_returns: 策略的日報酬率序列
        w_sharpe / w_mdd / w_return: 三指標權重（需合計為 1）

        Returns
        -------
        float：加權後適應度分數，約在 [-1, 1] 之間
        """
        if len(strategy_returns) < 10:
            return -999.0

        # 過濾全為零的無效策略（策略從未持倉）
        active_days = (strategy_returns != 0).sum()
        if active_days < 5:
            return -1.0

        # 計算三個指標
        sharpe = PerformanceMetrics.sharpe_ratio(strategy_returns)
        cumulative = (1.0 + strategy_returns).cumprod()
        rolling_max = cumulative.expanding().max()
        mdd = float(((cumulative - rolling_max) / rolling_max).min())
        cum_ret = float(cumulative.iloc[-1] - 1.0)

        # 正規化
        sharpe_score = float(np.clip(sharpe / 3.0, -1.0, 1.0))
        mdd_score = float(np.clip(1.0 + mdd, 0.0, 1.0))
        return_score = float(np.clip(cum_ret / 1.0, -1.0, 1.0))

        return w_sharpe * sharpe_score + w_mdd * mdd_score + w_return * return_score

    # ── 輔助工具 ──────────────────────────────────────────────

    @staticmethod
    def expected_annual_return_with_signal(annual_ret: float,
                                           signal: float) -> float:
        """
        根據技術信號調整期望年化報酬率。

        信號範圍 [-1, 1]：
          +1 → 乘數 2.0（期望翻倍）
           0 → 乘數 1.0（維持原始）
          -1 → 乘數 0.0（期望歸零）

        用於持股分析的機會成本估算。
        """
        multiplier = 1.0 + float(np.clip(signal, -1.0, 1.0))
        return annual_ret * multiplier
