"""
data_fetcher.py
───────────────
台股數據獲取模組

核心功能：
  - 自動判斷台股後綴（上市 .TW / 上櫃 .TWO）
  - 快取已解析的代號，避免重複網路請求
  - 支援批次下載，並對失敗代號提供明確回報

yfinance 台股代號規則：
  上市股票（TWSE）: 代號 + .TW   例：2330.TW（台積電）
  上櫃股票（TPEx）: 代號 + .TWO  例：6510.TWO（精測電子）
"""

import warnings
import yfinance as yf
import pandas as pd

warnings.filterwarnings("ignore")


class DataFetcher:
    """
    台股歷史數據下載器。

    使用範例
    --------
    fetcher = DataFetcher(period='2y')
    data = fetcher.fetch_multiple(['2330', '2454', '6510'])
    """

    # 嘗試後綴的順序：先上市再上櫃
    _SUFFIXES = ['.TW', '.TWO']

    def __init__(self, period: str = '2y', interval: str = '1d'):
        """
        Parameters
        ----------
        period  : yfinance 歷史資料長度，例如 '1y', '2y', '5y'
        interval: K 線頻率，預設日線 '1d'
        """
        self.period = period
        self.interval = interval
        self._suffix_cache: dict[str, str] = {}   # {代號: 完整ticker}
        self._failed_codes: list[str] = []

    # ── 私有方法 ──────────────────────────────────────────────

    def _resolve_ticker(self, code: str) -> str | None:
        """
        解析股票代號的正確 yfinance ticker。

        步驟：
          1. 檢查快取，若已解析直接回傳
          2. 依序嘗試 .TW、.TWO 後綴
          3. 下載 5 日數據確認是否有效
          4. 成功則寫入快取，失敗回傳 None

        Returns
        -------
        str 或 None：完整 ticker（例如 '2330.TW'）或 None（找不到）
        """
        if code in self._suffix_cache:
            return self._suffix_cache[code]

        for suffix in self._SUFFIXES:
            ticker = f"{code}{suffix}"
            try:
                # 只下載 5 日作為存在性驗證，速度快
                probe = yf.download(
                    ticker,
                    period='5d',
                    interval='1d',
                    progress=False,
                    auto_adjust=True,
                )
                if not probe.empty:
                    self._suffix_cache[code] = ticker
                    return ticker
            except Exception:
                continue

        # 兩種後綴都失敗
        self._failed_codes.append(code)
        return None

    def _download_single(self, code: str) -> pd.DataFrame | None:
        """
        下載單一股票的歷史數據並標準化欄位名稱。

        yfinance 在多層欄位（MultiIndex）與單層欄位間的行為
        因版本不同而有差異，此處統一取第一層欄位名稱。
        """
        ticker = self._resolve_ticker(code)
        if ticker is None:
            return None

        df = yf.download(
            ticker,
            period=self.period,
            interval=self.interval,
            progress=False,
            auto_adjust=True,   # 自動還原除權息
        )

        if df.empty:
            return None

        # 統一處理 MultiIndex 欄位（yfinance 0.2.x 的特性）
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]

        # 確保必要欄位存在
        required = {'Open', 'High', 'Low', 'Close', 'Volume'}
        if not required.issubset(df.columns):
            return None

        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        return df

    # ── 公開方法 ──────────────────────────────────────────────

    def fetch_multiple(self, codes: list[str]) -> dict[str, pd.DataFrame]:
        """
        批次下載多支股票數據。

        Parameters
        ----------
        codes : 台股代號列表，例如 ['2330', '2454']

        Returns
        -------
        dict：{代號: DataFrame}，失敗的代號不會出現在字典中
        """
        result: dict[str, pd.DataFrame] = {}

        for code in codes:
            df = self._download_single(code)
            if df is not None:
                result[code] = df
                ticker = self._suffix_cache.get(code, code)
                print(f"  [OK] {code} ({ticker})  "
                      f"{df.index[0].date()} → {df.index[-1].date()}  "
                      f"共 {len(df)} 筆")
            else:
                print(f"  [FAIL] {code}：無法下載，已跳過")

        return result

    def get_failed_codes(self) -> list[str]:
        """回傳下載失敗的代號列表，供外部診斷使用。"""
        return list(self._failed_codes)
