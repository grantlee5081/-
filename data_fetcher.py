"""
data_fetcher.py
───────────────
台股數據獲取模組 v2.0 — 漏斗策略（Funnel Strategy）

核心改進：
  Stage 1 — 市場快照篩選
    直接調用 TWSE STOCK_DAY_ALL（上市）與 TPEx 日線報價（上櫃）API，
    取得全市場當日資料，篩選：成交量 > 1,000 張。
    約從 1,700 支縮減至 300~500 支，節省 75%+ 的後續下載量。

  Stage 2 — 技術條件篩選
    對量能通過的代號做一次性批次下載（近 20 日），
    在 DataFrame 階段計算 5MA，篩選：收盤價 > 5MA。
    進一步縮減至 100~200 支。

  Stage 3 — 完整歷史下載
    僅對最終篩選清單使用 yf.download(tickers_list, period='2y', group_by='ticker')
    進行一次性批次下載，完全取代舊的逐支迴圈。

yfinance 台股代號規則：
  上市（TWSE）: 代號 + .TW   例：2330.TW
  上櫃（TPEx）: 代號 + .TWO  例：6510.TWO
"""

import warnings
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import yfinance as yf
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

# ── API 端點 ──────────────────────────────────────────────────
_TWSE_SNAPSHOT_URL = (
    "https://www.twse.com.tw/exchangeReport/STOCK_DAY_ALL?response=json"
)
_TPEX_SNAPSHOT_URL = (
    "https://www.tpex.org.tw/web/stock/aftertrading/daily_close_quotes/"
    "stk_quote_result.php?l=zh-tw&o=json"
)
_REQ_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    )
}

# 量能篩選門檻（張 = 1,000 股）
_VOLUME_MIN_LOTS: int = 1_000


# ═══════════════════════════════════════════════════════════════
#  Stage 1：全市場快照 + 量能篩選
# ═══════════════════════════════════════════════════════════════

def get_tw_daily_snapshot(
    volume_min_lots: int = _VOLUME_MIN_LOTS,
    verbose: bool = True,
) -> tuple[list[str], pd.DataFrame]:
    """
    從 TWSE + TPEx 官方 API 取得全市場當日快照。

    第一關篩選：成交量 > volume_min_lots 張。

    由於 API 直接回傳市場別，可省去 .TW / .TWO 後綴探測步驟。

    Parameters
    ----------
    volume_min_lots : 量能門檻（單位：張）
    verbose         : 是否印出篩選過程

    Returns
    -------
    (filtered_codes, snapshot_df)
      filtered_codes : 通過篩選的純數字代號列表
      snapshot_df    : 含 code / name / volume_lots / close / market 的 DataFrame
    """
    rows: list[dict] = []

    # ── TWSE 上市 ──────────────────────────────────────────────
    try:
        resp = requests.get(_TWSE_SNAPSHOT_URL, headers=_REQ_HEADERS, timeout=12)
        resp.raise_for_status()
        js = resp.json()
        # fields: 證券代號(0), 證券名稱(1), 成交股數(2), 成交筆數(3),
        #         開盤價(4), 最高價(5), 最低價(6), 收盤價(7), 漲跌價差(8), 本益比(9)
        for rec in js.get("data", []):
            if len(rec) < 8:
                continue
            code_raw = str(rec[0]).strip()
            if not code_raw.isdigit():
                continue
            try:
                vol_shares = int(str(rec[2]).replace(",", ""))
                close_str  = str(rec[7]).replace(",", "").strip()
                close      = float(close_str) if close_str not in ("", "--") else 0.0
                if close <= 0:
                    continue
            except (ValueError, IndexError):
                continue
            rows.append({
                "code":         code_raw,
                "name":         str(rec[1]).strip(),
                "volume_lots":  vol_shares // 1_000,
                "close":        close,
                "market":       "TWSE",
            })
        if verbose:
            twse_n = sum(1 for r in rows if r["market"] == "TWSE")
            print(f"  [TWSE] 取得 {twse_n} 筆上市資料")
    except Exception as e:
        if verbose:
            print(f"  [TWSE] 快照失敗：{e}")

    # ── TPEx 上櫃 ──────────────────────────────────────────────
    tpex_start = len(rows)
    try:
        resp = requests.get(_TPEX_SNAPSHOT_URL, headers=_REQ_HEADERS, timeout=12)
        resp.raise_for_status()
        js = resp.json()
        # 常見欄位順序：代號(0), 名稱(1), 收盤(2), 漲跌(3), 開盤(4),
        #               最高(5), 最低(6), 均價(7), 成交股數(千股)(8)
        for rec in js.get("aaData", []):
            if len(rec) < 9:
                continue
            code_raw = str(rec[0]).strip()
            if not code_raw.isdigit():
                continue
            try:
                close_str  = str(rec[2]).replace(",", "").strip()
                close      = float(close_str) if close_str not in ("", "--") else 0.0
                vol_k_str  = str(rec[8]).replace(",", "").strip()  # 千股
                vol_shares = int(float(vol_k_str)) * 1_000 if vol_k_str not in ("", "--") else 0
                if close <= 0:
                    continue
            except (ValueError, IndexError):
                continue
            rows.append({
                "code":         code_raw,
                "name":         str(rec[1]).strip(),
                "volume_lots":  vol_shares // 1_000,
                "close":        close,
                "market":       "TPEx",
            })
        if verbose:
            tpex_n = len(rows) - tpex_start
            print(f"  [TPEx] 取得 {tpex_n} 筆上櫃資料")
    except Exception as e:
        if verbose:
            print(f"  [TPEx] 快照失敗：{e}")

    if not rows:
        if verbose:
            print("  [警告] 無法取得任何市場快照，請確認網路連線。")
        return [], pd.DataFrame()

    snap_df = pd.DataFrame(rows).drop_duplicates("code").reset_index(drop=True)

    # ── 第一關：量能篩選 ──────────────────────────────────────
    before = len(snap_df)
    filtered = snap_df[snap_df["volume_lots"] > volume_min_lots].copy()
    if verbose:
        print(
            f"  [量能篩選] 全市場 {before} 支 → "
            f"成交量 > {volume_min_lots:,} 張：{len(filtered)} 支"
        )

    return filtered["code"].tolist(), filtered.reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════
#  內部工具：後綴解析 + 批次下載
# ═══════════════════════════════════════════════════════════════

def _suffix_from_market(market: str) -> str:
    return ".TW" if market == "TWSE" else ".TWO"


def _batch_yf_download(
    ticker_list: list[str],
    period: str,
    interval: str = "1d",
) -> dict[str, pd.DataFrame]:
    """
    一次性批次下載（yf.download + group_by='ticker'），
    回傳 {yfinance ticker: DataFrame}。

    處理 yfinance 新舊版 MultiIndex 欄位差異。
    """
    if not ticker_list:
        return {}

    raw = yf.download(
        ticker_list,
        period=period,
        interval=interval,
        group_by="ticker",
        auto_adjust=True,
        progress=False,
        threads=True,
    )

    result: dict[str, pd.DataFrame] = {}
    required = {"Open", "High", "Low", "Close", "Volume"}

    if raw.empty:
        return result

    if isinstance(raw.columns, pd.MultiIndex):
        # 判斷 MultiIndex 層次順序（yfinance 版本差異）
        lvl0 = raw.columns.get_level_values(0).unique().tolist()
        lvl1 = raw.columns.get_level_values(1).unique().tolist()

        # 若頂層是 ticker（含後綴）→ group_by='ticker' 正常行為
        if any("." in t for t in lvl0):
            for ticker in lvl0:
                try:
                    df = raw[ticker].copy()
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = [c[0] for c in df.columns]
                    df = df.dropna(subset=["Close"]).sort_index()
                    if not df.empty and required.issubset(df.columns):
                        result[ticker] = df
                except Exception:
                    pass
        else:
            # 頂層是欄位名稱，次層是 ticker（某些版本的行為）
            for ticker in lvl1:
                try:
                    df = raw.xs(ticker, axis=1, level=1).copy()
                    df = df.dropna(subset=["Close"]).sort_index()
                    if not df.empty and required.issubset(df.columns):
                        result[ticker] = df
                except Exception:
                    pass
    else:
        # 單一 ticker 下載
        if len(ticker_list) == 1:
            df = raw.copy()
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0] for c in df.columns]
            df = df.dropna(subset=["Close"]).sort_index()
            if not df.empty and required.issubset(df.columns):
                result[ticker_list[0]] = df

    return result


# ═══════════════════════════════════════════════════════════════
#  Stage 2 + 3：完整漏斗下載
# ═══════════════════════════════════════════════════════════════

def fetch_with_funnel(
    snapshot_df: pd.DataFrame,
    period: str = "2y",
    verbose: bool = True,
) -> tuple[dict[str, pd.DataFrame], dict[str, dict]]:
    """
    第二、三關：對量能篩選後的代號進行 MA5 技術篩選，
    再一次性批次下載完整歷史數據。

    Parameters
    ----------
    snapshot_df : get_tw_daily_snapshot() 回傳的 DataFrame
                  需含 code / market / close 欄位
    period      : yfinance 歷史長度（預設 '2y'）
    verbose     : 是否印出進度

    Returns
    -------
    (stock_data, selection_reasons)
      stock_data         : {代號: OHLCV DataFrame}
      selection_reasons  : {代號: {'volume_lots': int, 'ma5_pct': float, ...}}
    """
    if snapshot_df.empty:
        return {}, {}

    # 建立 code → market 映射（利用快照已知市場，省去後綴探測）
    code_market: dict[str, str] = dict(
        zip(snapshot_df["code"], snapshot_df["market"])
    )
    code_close: dict[str, float] = dict(
        zip(snapshot_df["code"], snapshot_df["close"])
    )
    code_volume: dict[str, int] = dict(
        zip(snapshot_df["code"], snapshot_df["volume_lots"])
    )

    # ── Stage 2：快速下載 20 日，計算 MA5 ─────────────────────
    tickers_20d = [
        f"{code}{_suffix_from_market(market)}"
        for code, market in code_market.items()
    ]
    if verbose:
        print(f"\n  [Stage 2] 快速下載 {len(tickers_20d)} 支（近 20 日）計算 5MA...")

    raw_20d = _batch_yf_download(tickers_20d, period="1mo")

    # 在 DataFrame 階段篩選：收盤價 > 5MA
    passed_codes: list[str] = []
    selection_reasons: dict[str, dict] = {}

    for ticker, df in raw_20d.items():
        code = ticker.replace(".TW", "").replace(".TWO", "")
        close_ser = df["Close"].dropna()
        if len(close_ser) < 5:
            continue
        ma5   = float(close_ser.rolling(5).mean().iloc[-1])
        close = float(close_ser.iloc[-1])
        if ma5 <= 0:
            continue
        ma5_pct = (close - ma5) / ma5 * 100.0
        if close > ma5:  # 收盤價 > 5MA
            passed_codes.append(code)
            selection_reasons[code] = {
                "volume_lots":    code_volume.get(code, 0),
                "close":          round(close, 1),
                "ma5":            round(ma5, 1),
                "ma5_pct":        round(ma5_pct, 2),
            }

    if verbose:
        print(
            f"  [MA5 篩選] {len(raw_20d)} 支 → "
            f"收盤 > 5MA：{len(passed_codes)} 支"
        )

    # ── Stage 3：完整歷史批次下載（一次性）─────────────────────
    tickers_full = [
        f"{code}{_suffix_from_market(code_market[code])}"
        for code in passed_codes
        if code in code_market
    ]
    if verbose:
        print(f"  [Stage 3] 批次下載 {len(tickers_full)} 支（{period}）...")

    raw_full = _batch_yf_download(tickers_full, period=period)

    # 重新映射到純數字代號
    stock_data: dict[str, pd.DataFrame] = {}
    for ticker, df in raw_full.items():
        code = ticker.replace(".TW", "").replace(".TWO", "")
        stock_data[code] = df

    if verbose:
        print(f"  [完成] 成功下載 {len(stock_data)} 支股票歷史數據")

    return stock_data, selection_reasons


# ═══════════════════════════════════════════════════════════════
#  DataFetcher 類別（向後相容，保留 app.py 現有介面）
# ═══════════════════════════════════════════════════════════════

class DataFetcher:
    """
    台股歷史數據下載器 v2.0。

    新功能：
      - fetch_multiple() 改用一次性批次下載（yf.download + group_by='ticker'）
        取代舊的逐支迴圈，速度提升約 3~5 倍。
      - 後綴探測（.TW / .TWO）改為並行執行，減少等待時間。

    使用範例
    --------
    fetcher = DataFetcher(period='2y')
    data = fetcher.fetch_multiple(['2330', '2454', '6510'])
    """

    _SUFFIXES = [".TW", ".TWO"]

    def __init__(self, period: str = "2y", interval: str = "1d"):
        self.period   = period
        self.interval = interval
        self._suffix_cache: dict[str, str] = {}   # {代號: 完整 yf ticker}
        self._failed_codes: list[str] = []

    # ── 後綴探測（並行版）─────────────────────────────────────

    def _resolve_ticker(self, code: str) -> str | None:
        """
        解析股票代號對應的 yfinance ticker（含後綴）。
        已解析的代號會快取，避免重複探測。
        """
        if code in self._suffix_cache:
            return self._suffix_cache[code]

        for suffix in self._SUFFIXES:
            ticker = f"{code}{suffix}"
            try:
                probe = yf.download(
                    ticker, period="5d", interval="1d",
                    progress=False, auto_adjust=True,
                )
                if not probe.empty:
                    self._suffix_cache[code] = ticker
                    return ticker
            except Exception:
                continue

        self._failed_codes.append(code)
        return None

    def _resolve_all_parallel(self, codes: list[str]) -> dict[str, str]:
        """
        並行解析多支股票的後綴（最多 8 條線程），
        回傳 {代號: yf ticker}。
        """
        to_resolve = [c for c in codes if c not in self._suffix_cache]
        # 直接用快取的先處理
        resolved: dict[str, str] = {
            c: self._suffix_cache[c]
            for c in codes
            if c in self._suffix_cache
        }

        if not to_resolve:
            return resolved

        with ThreadPoolExecutor(max_workers=8) as ex:
            future_map = {ex.submit(self._resolve_ticker, c): c for c in to_resolve}
            for fut in as_completed(future_map):
                code = future_map[fut]
                ticker = fut.result()
                if ticker:
                    resolved[code] = ticker

        return resolved

    # ── 批次下載（一次性，取代迴圈）─────────────────────────────

    def fetch_multiple(self, codes: list[str]) -> dict[str, pd.DataFrame]:
        """
        批次下載多支股票的 {period} 歷史數據。

        v2.0 改進：
          1. 並行解析後綴（省去序列探測等待）
          2. 一次性 yf.download(group_by='ticker') 批次下載，取代逐支迴圈

        Parameters
        ----------
        codes : 台股代號列表（純數字，不含後綴）

        Returns
        -------
        dict：{代號: DataFrame}
        """
        if not codes:
            return {}

        print(f"  [後綴探測] 並行解析 {len(codes)} 支股票後綴...")
        ticker_map = self._resolve_all_parallel(codes)   # {code: ticker}

        if not ticker_map:
            print("  [錯誤] 所有代號均無法解析")
            return {}

        failed_resolve = [c for c in codes if c not in ticker_map]
        if failed_resolve:
            self._failed_codes.extend(failed_resolve)
            print(f"  [FAIL] 後綴解析失敗 {len(failed_resolve)} 支：{failed_resolve[:5]}...")

        tickers_yf = list(ticker_map.values())
        print(f"  [批次下載] 一次性下載 {len(tickers_yf)} 支（{self.period}）...")

        raw = _batch_yf_download(tickers_yf, period=self.period, interval=self.interval)

        # 將 yf ticker 映射回純數字代號
        inv_map = {v: k for k, v in ticker_map.items()}   # {ticker: code}
        result: dict[str, pd.DataFrame] = {}

        for ticker, df in raw.items():
            code = inv_map.get(ticker)
            if code is None:
                # fallback：去後綴
                code = ticker.replace(".TW", "").replace(".TWO", "")
            result[code] = df
            print(
                f"  [OK] {code} ({ticker})  "
                f"{df.index[0].date()} → {df.index[-1].date()}  "
                f"共 {len(df)} 筆"
            )

        failed_dl = [inv_map.get(t, t) for t in tickers_yf if t not in raw]
        for code in failed_dl:
            if code not in self._failed_codes:
                self._failed_codes.append(code)
            print(f"  [FAIL] {code}：下載失敗")

        return result

    def get_failed_codes(self) -> list[str]:
        """回傳所有失敗代號（探測失敗 + 下載失敗）。"""
        return list(self._failed_codes)
