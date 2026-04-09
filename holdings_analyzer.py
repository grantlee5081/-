"""
holdings_analyzer.py
────────────────────
現有持股分析與換股建議模組

核心功能：
  1. 分析現有持股的當前績效與技術信號
  2. 計算「機會成本」（持有現股 vs 換入候選股的預期差距）
  3. 輸出換股建議與「預計回本時間」

機會成本計算邏輯
────────────────
  年度機會成本 = 現持倉市值 × (候選股期望年化報酬 - 現股期望年化報酬)

  換股成本（台股）：
    賣出：0.3% 證交稅 + 0.1425% 手續費
    買入：0.1425% 手續費
    合計約 0.585%（保守估算）

  預計回本時間（月）= 換股成本 / 月超額收益率
    月超額收益率 = 年超額收益率 / 12
"""

import numpy as np
import pandas as pd

from technical_factors import TechnicalFactors
from performance_metrics import PerformanceMetrics


# 台股換股的總交易成本（賣出稅費 + 買賣雙邊手續費）
SWITCH_COST_RATE = 0.00585   # 0.585%


class HoldingsAnalyzer:
    """
    現有持股分析器。

    使用範例
    --------
    analyzer = HoldingsAnalyzer()
    analysis = analyzer.analyze(current_holdings, stock_data, best_params)
    recs = analyzer.recommend_switches(analysis, candidate_scores, stock_data)
    """

    # ── 持股分析 ──────────────────────────────────────────────

    def analyze(self,
                current_holdings: dict[str, dict],
                stock_data: dict[str, pd.DataFrame],
                best_params: dict) -> dict[str, dict]:
        """
        分析每支現有持股的財務狀況與技術面信號。

        Parameters
        ----------
        current_holdings : {代號: {'cost': 買入均價, 'shares': 持有股數}}
        stock_data       : {代號: OHLCV DataFrame}
        best_params      : GA 最佳化後的策略參數

        Returns
        -------
        {代號: 分析結果字典}

        分析結果包含：
          - 當前損益（未實現盈虧）
          - 技術信號（由 GA 最佳參數計算）
          - Sharpe / MDD / 年化報酬等績效指標
          - 期望年化報酬（信號調整後）
        """
        analysis: dict[str, dict] = {}

        for code, info in current_holdings.items():
            if code not in stock_data:
                print(f"  [警告] 持股 {code} 無數據，跳過分析")
                continue

            df = stock_data[code]
            if 'Close' not in df.columns:
                continue

            prices = df['Close'].dropna()
            if len(prices) < max(best_params['ma_long'], 20) + 5:
                print(f"  [警告] {code} 數據長度不足，跳過分析")
                continue

            cost_price = float(info.get('cost', 0))
            shares = int(info.get('shares', 0))
            current_price = float(prices.iloc[-1])

            # ── 損益計算 ──
            current_value = current_price * shares
            cost_value = cost_price * shares
            unrealized_pnl = current_value - cost_value
            unrealized_pnl_pct = (
                (current_price - cost_price) / cost_price
                if cost_price > 0 else 0.0
            )

            # ── 技術信號（使用 GA 最佳參數）──
            signal = TechnicalFactors.composite_signal(
                prices,
                ma_short=best_params['ma_short'],
                ma_long=best_params['ma_long'],
                ma_weight=best_params['ma_weight'],
                rsi_period=best_params['rsi_period'],
                rsi_ob=best_params['rsi_ob'],
                rsi_os=best_params['rsi_os'],
                rsi_weight=best_params['rsi_weight'],
                bb_period=best_params['bb_period'],
                bb_std=best_params['bb_std'],
                bb_weight=best_params['bb_weight'],
            )
            current_signal = float(signal.iloc[-1])
            recent_signal = float(signal.iloc[-5:].mean())

            # ── 績效指標 ──
            rets = PerformanceMetrics.daily_returns(prices)
            sharpe = PerformanceMetrics.sharpe_ratio(rets)
            mdd = PerformanceMetrics.max_drawdown(prices)
            annual_ret = PerformanceMetrics.annualized_return(prices)

            # ── 期望報酬（信號調整）──
            # 近 5 日平均信號反映近期動能；以此調整期望年化報酬
            expected_ret = PerformanceMetrics.expected_annual_return_with_signal(
                annual_ret, recent_signal
            )

            analysis[code] = {
                'code':                code,
                'shares':              shares,
                'cost_price':          cost_price,
                'current_price':       current_price,
                'current_value':       current_value,
                'cost_value':          cost_value,
                'unrealized_pnl':      unrealized_pnl,
                'unrealized_pnl_pct':  unrealized_pnl_pct,
                'current_signal':      current_signal,
                'recent_signal_5d':    recent_signal,
                'sharpe_ratio':        sharpe,
                'max_drawdown':        mdd,
                'annual_return':       annual_ret,
                'expected_annual_ret': expected_ret,
            }

        return analysis

    # ── 換股建議 ──────────────────────────────────────────────

    def recommend_switches(self,
                            holdings_analysis: dict[str, dict],
                            candidate_scores: dict[str, float],
                            candidate_data: dict[str, pd.DataFrame],
                            min_excess_return: float = 0.05) -> list[dict]:
        """
        生成換股建議清單。

        觸發換股評估的條件（滿足任一）：
          A. 近 5 日平均技術信號 < -0.10（偏空）
          B. 信號調整後的期望年化報酬 < 5%

        換股閾值：
          候選股期望年化報酬 - 現股期望年化報酬 > min_excess_return（預設 5%）

        Parameters
        ----------
        holdings_analysis  : analyze() 的返回值
        candidate_scores   : GA 對候選股的評分，{代號: 分數}
        candidate_data     : 候選股的 OHLCV DataFrame
        min_excess_return  : 最低超額報酬門檻（年化）

        Returns
        -------
        換股建議列表（按超額報酬率降序排序）
        """
        # 預先計算候選股的績效指標（只算一次）
        candidate_metrics = self._compute_candidate_metrics(
            candidate_scores, candidate_data
        )

        recommendations: list[dict] = []

        for code, info in holdings_analysis.items():
            # 判斷是否需要評估換股
            weak_signal = info['recent_signal_5d'] < -0.10
            low_expected = info['expected_annual_ret'] < 0.05

            if not (weak_signal or low_expected):
                continue  # 持股狀況良好，無需換股評估

            # 找出最佳換股候選（期望超額報酬最高）
            best_candidate = self._find_best_candidate(
                info, candidate_metrics, min_excess_return
            )

            if best_candidate is None:
                continue

            rec = self._build_recommendation(info, best_candidate)
            recommendations.append(rec)

        # 按超額報酬率降序排序
        recommendations.sort(
            key=lambda x: x['excess_return_annual'], reverse=True
        )
        return recommendations

    # ── 私有輔助方法 ──────────────────────────────────────────

    def _compute_candidate_metrics(self,
                                    scores: dict[str, float],
                                    data: dict[str, pd.DataFrame]) -> list[dict]:
        """
        預計算所有候選股的績效指標，避免重複計算。
        只保留信號為正的候選（已有明確買入信號的標的）。
        """
        metrics: list[dict] = []

        for code, score in scores.items():
            if score <= 0 or code not in data:
                continue

            prices = data[code]['Close'].dropna()
            if len(prices) < 30:
                continue

            rets = PerformanceMetrics.daily_returns(prices)
            annual_ret = PerformanceMetrics.annualized_return(prices)
            sharpe = PerformanceMetrics.sharpe_ratio(rets)

            metrics.append({
                'code':        code,
                'score':       score,
                'annual_ret':  annual_ret,
                'sharpe':      sharpe,
                'price_now':   float(prices.iloc[-1]),
            })

        # 按 GA 評分排序，優先推薦評分最高的
        metrics.sort(key=lambda x: x['score'], reverse=True)
        return metrics[:5]  # 最多考慮前 5 名候選

    def _find_best_candidate(self,
                              holding_info: dict,
                              candidates: list[dict],
                              min_excess: float) -> dict | None:
        """
        在候選股中找到相對現股超額報酬最高且超過門檻的最佳選擇。
        """
        holding_exp_ret = holding_info['expected_annual_ret']
        best = None
        best_excess = -np.inf

        for cand in candidates:
            if cand['code'] == holding_info['code']:
                continue  # 跳過與現持股相同的代號

            excess = cand['annual_ret'] - holding_exp_ret
            if excess > min_excess and excess > best_excess:
                best_excess = excess
                best = cand

        return best

    def _build_recommendation(self,
                                holding: dict,
                                candidate: dict) -> dict:
        """
        組建完整的換股建議字典，包含機會成本與預計回本時間。

        預計回本時間計算：
          - 換股成本（絕對值） = 現持倉市值 × SWITCH_COST_RATE
          - 月超額收益       = 現持倉市值 × (超額年化報酬 / 12)
          - 回本月數         = 換股成本 / 月超額收益

        意義：執行換股後，幾個月的超額收益可以彌補稅費成本。
        """
        current_value = holding['current_value']
        excess_annual = candidate['annual_ret'] - holding['expected_annual_ret']

        # 換股成本（新台幣）
        switch_cost_ntd = current_value * SWITCH_COST_RATE

        # 年度機會成本（不換股每年損失的超額報酬）
        opportunity_cost_annual = current_value * excess_annual

        # 預計回本時間
        monthly_excess_income = current_value * (excess_annual / 12.0)
        if monthly_excess_income > 0:
            payback_months = switch_cost_ntd / monthly_excess_income
            payback_str = f"{payback_months:.1f} 個月"
        else:
            payback_months = float('inf')
            payback_str = "無法估算（超額收益為負）"

        # 優先級判斷
        if excess_annual > 0.20:
            priority = 'HIGH'
        elif excess_annual > 0.10:
            priority = 'MEDIUM'
        else:
            priority = 'LOW'

        return {
            'priority':                priority,
            'sell_code':               holding['code'],
            'sell_signal':             holding['recent_signal_5d'],
            'sell_expected_ret':       holding['expected_annual_ret'],
            'sell_sharpe':             holding['sharpe_ratio'],
            'buy_code':                candidate['code'],
            'buy_ga_score':            candidate['score'],
            'buy_annual_ret':          candidate['annual_ret'],
            'buy_sharpe':              candidate['sharpe'],
            'current_holding_value':   current_value,
            'switch_cost_ntd':         switch_cost_ntd,
            'switch_cost_rate':        SWITCH_COST_RATE,
            'excess_return_annual':    excess_annual,
            'opportunity_cost_annual': opportunity_cost_annual,
            'payback_months':          payback_months,
            'payback_str':             payback_str,
        }
