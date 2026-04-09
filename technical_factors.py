"""
technical_factors.py
────────────────────
技術指標計算模組

實作三種技術因子：
  1. MA 交叉（Moving Average Crossover）
  2. RSI 乖離（Relative Strength Index Deviation）
  3. 布林通道（Bollinger Bands）

每個因子均輸出 [-1, +1] 的標準化信號：
  +1 → 強烈看多
   0 → 中性
  -1 → 強烈看空
"""

import numpy as np
import pandas as pd


class TechnicalFactors:
    """技術指標信號計算器（靜態方法集合）。"""

    # ── MA 交叉 ───────────────────────────────────────────────

    @staticmethod
    def ma_signal(prices: pd.Series,
                  short_period: int,
                  long_period: int) -> pd.Series:
        """
        計算 MA 交叉信號。

        演算法：
          1. 計算短期與長期簡單移動平均線（SMA）
          2. 乖離率 = (短均線 - 長均線) / 長均線
          3. 以 3 倍標準差正規化至 [-1, 1]

        當短均線高於長均線（多頭排列）→ 正值信號
        當短均線低於長均線（空頭排列）→ 負值信號

        Returns
        -------
        pd.Series：與 prices 索引對齊的信號序列，NaN 填為 0
        """
        ma_short = prices.rolling(window=short_period, min_periods=short_period).mean()
        ma_long = prices.rolling(window=long_period, min_periods=long_period).mean()

        # 計算相對乖離率
        deviation = (ma_short - ma_long) / ma_long.replace(0, np.nan)

        # 用滾動標準差正規化（避免在不同市場環境下尺度不一）
        rolling_std = deviation.rolling(window=long_period, min_periods=long_period).std()
        signal = deviation / (3.0 * rolling_std.replace(0, np.nan))
        signal = signal.clip(-1.0, 1.0).fillna(0.0)

        return signal

    # ── RSI 乖離 ──────────────────────────────────────────────

    @staticmethod
    def _rsi_raw(prices: pd.Series, period: int) -> pd.Series:
        """
        計算 RSI 原始值（0~100）。

        公式（Wilder 平滑法）：
          RS  = 平均漲幅 / 平均跌幅（rolling mean 近似）
          RSI = 100 - 100 / (1 + RS)
        """
        delta = prices.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)

        avg_gain = gain.rolling(window=period, min_periods=period).mean()
        avg_loss = loss.rolling(window=period, min_periods=period).mean()

        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return rsi.fillna(50.0)  # 無法計算時預設中性 50

    @staticmethod
    def rsi_signal(prices: pd.Series,
                   period: int,
                   overbought: float,
                   oversold: float) -> pd.Series:
        """
        計算 RSI 乖離信號（反向信號）。

        設計邏輯：
          - RSI 遠低於超賣線 → 深度超賣 → 正信號（預期反彈）
          - RSI 遠高於超買線 → 深度超買 → 負信號（預期回調）
          - 中間帶 → 依線性插值給出漸進信號

        信號計算：
          normalized = (中點 - RSI) / 中點
          中點 = (overbought + oversold) / 2

        Returns
        -------
        pd.Series：信號序列，clip 至 [-1, 1]
        """
        rsi = TechnicalFactors._rsi_raw(prices, period)
        mid = (overbought + oversold) / 2.0

        # 相對中點的偏離程度，轉為反向信號
        signal = (mid - rsi) / mid
        return signal.clip(-1.0, 1.0)

    # ── 布林通道 ──────────────────────────────────────────────

    @staticmethod
    def bollinger_bands(prices: pd.Series,
                        period: int,
                        std_dev: float) -> tuple[pd.Series, pd.Series, pd.Series]:
        """
        計算布林通道三條線。

        Returns
        -------
        (中軌 MA, 上軌 Upper, 下軌 Lower)
        """
        ma = prices.rolling(window=period, min_periods=period).mean()
        std = prices.rolling(window=period, min_periods=period).std()
        upper = ma + std_dev * std
        lower = ma - std_dev * std
        return ma, upper, lower

    @staticmethod
    def bb_signal(prices: pd.Series,
                  period: int,
                  std_dev: float) -> pd.Series:
        """
        計算布林通道信號。

        演算法：
          1. 計算價格在通道中的相對位置：
             pos = (price - lower) / (upper - lower)
             pos=0 → 下軌（超賣），pos=1 → 上軌（超買）
          2. 轉換為反向信號：signal = 1 - 2 × pos
             pos=0.0 → +1.0（下軌，強力買入信號）
             pos=0.5 → 0.0（中軌，中性）
             pos=1.0 → -1.0（上軌，強力賣出信號）

        Returns
        -------
        pd.Series：信號序列，clip 至 [-1, 1]，NaN 填 0
        """
        _, upper, lower = TechnicalFactors.bollinger_bands(prices, period, std_dev)
        band_width = (upper - lower).replace(0, np.nan)
        position = (prices - lower) / band_width
        signal = (1.0 - 2.0 * position).clip(-1.0, 1.0).fillna(0.0)
        return signal

    # ── 綜合信號 ──────────────────────────────────────────────

    @staticmethod
    def composite_signal(prices: pd.Series,
                         ma_short: int,
                         ma_long: int,
                         ma_weight: float,
                         rsi_period: int,
                         rsi_ob: float,
                         rsi_os: float,
                         rsi_weight: float,
                         bb_period: int,
                         bb_std: float,
                         bb_weight: float) -> pd.Series:
        """
        計算三因子加權綜合信號。

        公式：
          composite = (w_ma × S_ma + w_rsi × S_rsi + w_bb × S_bb)
                      / (w_ma + w_rsi + w_bb)

        各因子信號均在 [-1, 1] 範圍，加權後仍在 [-1, 1]。

        Returns
        -------
        pd.Series：綜合信號序列
        """
        total_weight = ma_weight + rsi_weight + bb_weight
        if total_weight == 0:
            return pd.Series(0.0, index=prices.index)

        s_ma = TechnicalFactors.ma_signal(prices, ma_short, ma_long)
        s_rsi = TechnicalFactors.rsi_signal(prices, rsi_period, rsi_ob, rsi_os)
        s_bb = TechnicalFactors.bb_signal(prices, bb_period, bb_std)

        composite = (ma_weight * s_ma +
                     rsi_weight * s_rsi +
                     bb_weight * s_bb) / total_weight
        return composite
