"""
technical_factors.py
────────────────────
技術指標計算模組（v2.0 短線強化版）

實作五種技術因子：
  1. MA 交叉（Moving Average Crossover）
  2. RSI 乖離（Relative Strength Index）
  3. 布林通道（Bollinger Bands）
  4. KDJ 隨機指標（短線核心）    ← 新增
  5. 成交量爆發（Volume Burst）  ← 新增

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
        """
        ma_short = prices.rolling(window=short_period, min_periods=short_period).mean()
        ma_long = prices.rolling(window=long_period, min_periods=long_period).mean()

        deviation = (ma_short - ma_long) / ma_long.replace(0, np.nan)
        rolling_std = deviation.rolling(window=long_period, min_periods=long_period).std()
        signal = deviation / (3.0 * rolling_std.replace(0, np.nan))
        return signal.clip(-1.0, 1.0).fillna(0.0)

    # ── RSI 乖離 ──────────────────────────────────────────────

    @staticmethod
    def _rsi_raw(prices: pd.Series, period: int) -> pd.Series:
        """計算 RSI 原始值（0~100），使用 Wilder 平滑法。"""
        delta = prices.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)

        avg_gain = gain.rolling(window=period, min_periods=period).mean()
        avg_loss = loss.rolling(window=period, min_periods=period).mean()

        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return rsi.fillna(50.0)

    @staticmethod
    def rsi_signal(prices: pd.Series,
                   period: int,
                   overbought: float,
                   oversold: float) -> pd.Series:
        """
        計算 RSI 乖離信號（反向信號）。

        RSI 遠低於超賣線 → 正信號（預期反彈）
        RSI 遠高於超買線 → 負信號（預期回調）
        """
        rsi = TechnicalFactors._rsi_raw(prices, period)
        mid = (overbought + oversold) / 2.0
        signal = (mid - rsi) / mid
        return signal.clip(-1.0, 1.0)

    # ── 布林通道 ──────────────────────────────────────────────

    @staticmethod
    def bollinger_bands(prices: pd.Series,
                        period: int,
                        std_dev: float) -> tuple[pd.Series, pd.Series, pd.Series]:
        """計算布林通道三條線：(中軌, 上軌, 下軌)。"""
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

        pos=0（下軌超賣）→ +1.0，pos=1（上軌超買）→ -1.0
        """
        _, upper, lower = TechnicalFactors.bollinger_bands(prices, period, std_dev)
        band_width = (upper - lower).replace(0, np.nan)
        position = (prices - lower) / band_width
        return (1.0 - 2.0 * position).clip(-1.0, 1.0).fillna(0.0)

    # ── KDJ 隨機指標（短線核心）─────────────────────────────────

    @staticmethod
    def kdj_signal(high: pd.Series,
                   low: pd.Series,
                   close: pd.Series,
                   n: int = 9,
                   m1: int = 3,
                   m2: int = 3) -> pd.Series:
        """
        計算 KDJ 隨機指標信號（短線動能核心）。

        KDJ 是台股最廣泛使用的短線指標之一，能快速捕捉
        超買超賣轉折點，適合波段與當沖交易。

        演算法：
          RSV = (Close - 近 N 日最低) / (近 N 日最高 - 近 N 日最低) × 100
          K   = EMA(RSV, m1)           平滑化隨機值
          D   = EMA(K,   m2)           K 的再平滑
          J   = 3K - 2D                超買超賣放大器

        信號邏輯（反向均值回歸）：
          J < 20  → 深度超賣 → 強烈正信號
          J > 80  → 深度超買 → 強烈負信號
          線性插值中間區域

        Parameters
        ----------
        high, low, close : 高低收盤價 Series（索引需一致）
        n  : RSV 計算窗口（預設 9）
        m1 : K 線 EMA span（預設 3）
        m2 : D 線 EMA span（預設 3）

        Returns
        -------
        pd.Series：clip 至 [-1, 1]，NaN 填 0
        """
        ll = low.rolling(n, min_periods=n).min()
        hh = high.rolling(n, min_periods=n).max()
        band = (hh - ll).replace(0, np.nan)
        rsv = ((close - ll) / band * 100.0).fillna(50.0).clip(0.0, 100.0)

        # 使用 EMA 平滑（span = 2m-1 等效 Wilder 週期 m）
        k = rsv.ewm(span=m1 * 2 - 1, min_periods=m1, adjust=False).mean()
        d = k.ewm(span=m2 * 2 - 1, min_periods=m2, adjust=False).mean()
        j = 3.0 * k - 2.0 * d

        # 正規化：J 中點=50，範圍通常 [−20, 120]
        # J < 20 → 超賣 → 正信號；J > 80 → 超買 → 負信號
        signal = (50.0 - j) / 50.0
        return signal.clip(-1.0, 1.0).fillna(0.0)

    # ── 成交量爆發（Volume Burst）────────────────────────────────

    @staticmethod
    def volume_burst_signal(close: pd.Series,
                            volume: pd.Series,
                            period: int = 20) -> pd.Series:
        """
        計算成交量爆發信號（量價共振）。

        短線交易中「量能」是突破確認的關鍵。
        量增價漲代表多方力道強勁；量增價跌代表空方出貨。

        演算法：
          vol_ratio = 當日成交量 / N 日均量（量比）
          price_chg = 當日漲跌幅
          excess    = max(0, vol_ratio - 1.0)   超出均量部分
          signal    = price_chg × excess × 放大係數

        當量比 < 1（縮量）→ signal ≈ 0（不計入）
        當量比 = 2（爆量）且漲停 → signal → +1.0
        當量比 = 2（爆量）且跌停 → signal → -1.0

        Parameters
        ----------
        close  : 收盤價 Series
        volume : 成交量 Series（單位不限，只需前後一致）
        period : 計算均量的滾動窗口（預設 20 日）

        Returns
        -------
        pd.Series：clip 至 [-1, 1]，NaN 填 0
        """
        vol_ma = volume.rolling(period, min_periods=max(5, period // 2)).mean()
        vol_ratio = (volume / vol_ma.replace(0, np.nan)).fillna(1.0)
        price_chg = close.pct_change().fillna(0.0)

        # 超出均量的部分才作為信號放大器（縮量不計）
        excess_vol = (vol_ratio - 1.0).clip(lower=0.0)
        signal = (price_chg * excess_vol * 5.0).clip(-1.0, 1.0)
        return signal.fillna(0.0)

    # ── 綜合信號（GA 使用，Close Only）──────────────────────────

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
        計算三因子加權綜合信號（MA + RSI + BB）。

        此函數保持與 GA 染色體的相容性，僅使用收盤價。
        若需包含 KDJ + 成交量，請使用 swing_composite_signal()。

        公式：
          composite = (w_ma × S_ma + w_rsi × S_rsi + w_bb × S_bb)
                      / (w_ma + w_rsi + w_bb)
        """
        total_weight = ma_weight + rsi_weight + bb_weight
        if total_weight == 0:
            return pd.Series(0.0, index=prices.index)

        s_ma  = TechnicalFactors.ma_signal(prices, ma_short, ma_long)
        s_rsi = TechnicalFactors.rsi_signal(prices, rsi_period, rsi_ob, rsi_os)
        s_bb  = TechnicalFactors.bb_signal(prices, bb_period, bb_std)

        composite = (ma_weight * s_ma +
                     rsi_weight * s_rsi +
                     bb_weight * s_bb) / total_weight
        return composite

    # ── 短線複合信號（含 KDJ + 成交量）──────────────────────────

    @staticmethod
    def swing_composite_signal(df: pd.DataFrame,
                               ma_short: int,
                               ma_long: int,
                               ma_weight: float,
                               rsi_period: int,
                               rsi_ob: float,
                               rsi_os: float,
                               rsi_weight: float,
                               bb_period: int,
                               bb_std: float,
                               bb_weight: float,
                               kdj_weight: float = 0.35,
                               vol_weight: float = 0.20) -> pd.Series:
        """
        短線複合信號（五因子加權）。

        相較於 composite_signal()，此函數額外引入 KDJ 與
        成交量爆發信號，更適合捕捉短期波段交易機會。

        預設權重分配（短線優先）：
          MA    : ma_weight（原始值，會被重整）
          RSI   : rsi_weight
          BB    : bb_weight
          KDJ   : 0.35  ← 高權重（短線核心）
          Volume: 0.20  ← 量能確認

        Parameters
        ----------
        df         : 含 Open/High/Low/Close/Volume 的 OHLCV DataFrame
        其餘參數   : 與 composite_signal() 相同

        Returns
        -------
        pd.Series：五因子加權後的綜合信號，clip 至 [-1, 1]
        """
        close  = df['Close'].dropna()
        high   = df['High'].reindex(close.index).ffill()
        low    = df['Low'].reindex(close.index).ffill()
        volume = df['Volume'].reindex(close.index).fillna(0)

        s_ma  = TechnicalFactors.ma_signal(close, ma_short, ma_long)
        s_rsi = TechnicalFactors.rsi_signal(close, rsi_period, rsi_ob, rsi_os)
        s_bb  = TechnicalFactors.bb_signal(close, bb_period, bb_std)
        s_kdj = TechnicalFactors.kdj_signal(high, low, close)
        s_vol = TechnicalFactors.volume_burst_signal(close, volume)

        total = ma_weight + rsi_weight + bb_weight + kdj_weight + vol_weight
        if total == 0:
            return pd.Series(0.0, index=close.index)

        composite = (ma_weight  * s_ma  +
                     rsi_weight * s_rsi +
                     bb_weight  * s_bb  +
                     kdj_weight * s_kdj +
                     vol_weight * s_vol) / total
        return composite.clip(-1.0, 1.0)
