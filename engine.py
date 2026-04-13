"""
engine.py
─────────
量化交易引擎：業務邏輯層

職責（與 UI 完全解耦）：
  - 股票候選池生成（資產導向）
  - 投組摘要計算
  - 今日操作指南決策樹
  - 完整量化分析 Pipeline（GA → MC → 持股分析）
  - 思考鏈記錄器（ThinkingLogger）

此模組不引用任何 Streamlit API，可獨立測試。
"""

import io
import datetime
from collections import deque
from contextlib import redirect_stdout

import numpy as np
import pandas as pd

from genetic_algorithm import GeneticAlgorithm
from monte_carlo import MonteCarloSimulator
from holdings_analyzer import HoldingsAnalyzer
from technical_factors import TechnicalFactors

# ── 常數 ──────────────────────────────────────────────────────

SWITCH_COST_RATE = 0.00585   # 台股換股總成本率（含證交稅 + 雙邊手續費）

# 台股候選宇宙（資產導向池的篩選來源）
TW_STOCK_UNIVERSE: dict[str, list[str]] = {
    '半導體':   ['2330', '2454', '3711', '2379', '2344', '2303', '2337', '3034'],
    '電子製造': ['2317', '2382', '2308', '2357', '2324', '2327', '2376', '2395'],
    '金融':     ['2881', '2882', '2891', '2884', '2886', '2892', '2885', '2883'],
    '傳產原料': ['1301', '1303', '6505', '2002', '1101', '1216', '2912'],
    '電信服務': ['2412', '3045', '4904'],
    '其他電子': ['2301', '2385', '3008', '2352', '3231'],
}
ALL_UNIVERSE: list[str] = [c for lst in TW_STOCK_UNIVERSE.values() for c in lst]

# 代號 → 中文名稱對照表（涵蓋 TW_STOCK_UNIVERSE 全部成分股）
TW_STOCK_NAMES: dict[str, str] = {
    # 半導體
    '2330': '台積電',   '2454': '聯發科',   '3711': '日月光投控',
    '2379': '瑞昱',     '2344': '華邦電',   '2303': '聯電',
    '2337': '旺宏',     '3034': '聯詠',
    # 電子製造
    '2317': '鴻海',     '2382': '廣達',     '2308': '台達電',
    '2357': '華碩',     '2324': '仁寶',     '2327': '國巨',
    '2376': '技嘉',     '2395': '研華',
    # 金融
    '2881': '富邦金',   '2882': '國泰金',   '2891': '中信金',
    '2884': '玉山金',   '2886': '兆豐金',   '2892': '第一金',
    '2885': '元大金',   '2883': '開發金',
    # 傳產原料
    '1301': '台塑',     '1303': '南亞',     '6505': '台塑化',
    '2002': '中鋼',     '1101': '台泥',     '1216': '統一',
    '2912': '統一超',
    # 電信服務
    '2412': '中華電',   '3045': '台灣大',   '4904': '遠傳',
    # 其他電子
    '2301': '光寶科',   '2385': '群光',     '3008': '大立光',
    '2352': '佳世達',   '3231': '緯創',
}


# ═══════════════════════════════════════════════════════════════
#  思考鏈記錄器
# ═══════════════════════════════════════════════════════════════

class ThinkingLogger:
    """
    即時思考鏈記錄器。

    在推演過程中持續呼叫 log()，以結構化日誌展示 AI
    當前的判斷邏輯，並透過傳入的 Streamlit empty() 容器
    即時更新畫面，避免用戶誤以為程式卡死。

    使用方式：
        logger = ThinkingLogger(st.empty())
        logger.log("技術指標分析中...", kind='signal')
    """

    _ICONS: dict[str, str] = {
        'data':    '📡',
        'ga':      '🧬',
        'mc':      '🎲',
        'holding': '💼',
        'kdj':     '📊',
        'volume':  '📈',
        'signal':  '⚡',
        'done':    '✅',
        'warn':    '⚠️',
        'info':    'ℹ️',
    }

    def __init__(self, container) -> None:
        """
        Parameters
        ----------
        container : st.empty() 容器，用於即時更新顯示內容
        """
        self._container = container
        self._lines: deque[str] = deque(maxlen=40)
        self._start = datetime.datetime.now()

    def log(self, message: str, kind: str = 'info') -> None:
        """
        輸出一條思考日誌並即時更新畫面。

        Parameters
        ----------
        message : 自然語言說明（顯示給用戶）
        kind    : 圖標類型 (data/ga/mc/holding/kdj/volume/signal/done/warn/info)
        """
        icon    = self._ICONS.get(kind, 'ℹ️')
        elapsed = (datetime.datetime.now() - self._start).total_seconds()
        line    = f"{icon}  [{elapsed:5.1f}s]  {message}"
        self._lines.append(line)
        self._render()

    def _render(self) -> None:
        # 使用 Morandi 色系：accent 藍灰顯示 log 文字，在淺色 think-box 上清晰可讀
        _COLORS = {
            'data':    '#607D8B',   # 霧藍
            'ga':      '#7A6E9E',   # 柔紫
            'mc':      '#5A8A7A',   # 消綠
            'holding': '#B8966A',   # 琥珀
            'kdj':     '#6B7C8D',   # 鐵藍
            'volume':  '#7A9E87',   # 鼠尾草
            'signal':  '#607D8B',   # 霧藍
            'done':    '#5A8A7A',   # 消綠
            'warn':    '#B85450',   # 消紅
            'info':    '#8A8A8A',   # 中性灰
        }
        html_parts = []
        for line in self._lines:
            # 根據 line 前綴 icon 猜測 kind → 對應顏色
            color = '#7A7A7A'   # fallback muted
            for k, v in _COLORS.items():
                icon = ThinkingLogger._ICONS.get(k, '')
                if icon and line.startswith(icon):
                    color = v
                    break
            html_parts.append(f'<span style="color:{color};">{line}</span>')
        html = '<br>'.join(html_parts)
        self._container.markdown(
            f'<div class="think-box">{html}</div>',
            unsafe_allow_html=True,
        )


# ═══════════════════════════════════════════════════════════════
#  資產導向股池生成
# ═══════════════════════════════════════════════════════════════

def generate_asset_driven_pool(
        available_cash: float,
        current_holdings: dict,
        max_pool_size: int = 15,
) -> list[str]:
    """
    根據可用資金與現有持股，自動生成候選股池。

    策略
    ----
    1. 持倉股自動納入（確保分析現有資產）
    2. 分析持倉行業分布
    3. 從未持倉行業各補 2~3 支（多元化）
    4. 補足至 max_pool_size，優先流動性高的大型股

    Parameters
    ----------
    available_cash   : 可用資金（NT$），保留供未來價格篩選擴充
    current_holdings : {代號: {'cost':..., 'shares':...}}
    max_pool_size    : 最大股池數量（預設 15）

    Returns
    -------
    list[str]：推薦候選代號（已去重，保持插入順序）
    """
    pool: list[str] = list(current_holdings.keys())

    # 找出持倉所屬行業
    held_sectors: set[str] = set()
    for code in current_holdings:
        for sector, codes in TW_STOCK_UNIVERSE.items():
            if code in codes:
                held_sectors.add(sector)

    # 優先補入未持倉行業
    for sector, candidates in TW_STOCK_UNIVERSE.items():
        if len(pool) >= max_pool_size:
            break
        picks = 3 if sector not in held_sectors else 1
        added = 0
        for c in candidates:
            if c not in pool and added < picks:
                pool.append(c)
                added += 1

    # 若仍有空位，從全宇宙補充
    for c in ALL_UNIVERSE:
        if len(pool) >= max_pool_size:
            break
        if c not in pool:
            pool.append(c)

    return pool[:max_pool_size]


# ═══════════════════════════════════════════════════════════════
#  投組摘要計算
# ═══════════════════════════════════════════════════════════════

def compute_portfolio_summary(
        current_holdings: dict,
        quotes: dict,
        available_cash: float,
) -> dict:
    """
    根據持股清單 + 即時報價，計算頂部四個 KPI。

    Returns
    -------
    dict 包含：total_assets, holding_value, total_cost,
               total_pnl, today_pnl, total_return,
               today_return, available_cash, last_trade_date
    """
    holding_value = total_cost = today_pnl = 0.0
    last_date = "─"

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
    """合併持股清單與即時報價，生成表格用的列資料。"""
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
            'code':       code,
            'shares':     sh,
            'cost':       cost,
            'current':    curr,
            'change':     q['change'],
            'change_pct': q['change_pct'],
            'cost_value': cost_v,
            'curr_value': curr_v,
            'pnl':        pnl,
            'pnl_pct':    pnl / cost_v if cost_v > 0 else 0.0,
            'trade_date': q.get('trade_date', '─'),
        })
    return rows


# ═══════════════════════════════════════════════════════════════
#  今日操作指南決策引擎
# ═══════════════════════════════════════════════════════════════

def generate_daily_guide(
        holdings_analysis: dict,
        stock_scores: dict,
        current_holdings: dict,
        available_cash: float,
        quotes: dict,
        strategy_reasons: dict | None = None,
) -> list[dict]:
    """
    今日操作指南決策引擎。

    決策邏輯（依優先級）
    ─────────────────────
    對現有持股（逐一判斷）：
      ① STOP_LOSS  — 未實現虧損 > 10% 且信號 < -0.15
      ② SWITCH     — 信號 < -0.10 且有超額報酬 > 5% 的候選
      ③ HOLD       — 信號 > +0.10
      ④ WATCH      — -0.10 ≤ 信號 ≤ +0.10

    對閒置現金：
      ⑤ NEW_BUY    — 現金 ≥ NT$50,000 且有評分 > 0.20 的未持有標的
      ⑥ CASH       — 無明確買入信號，保留現金

    strategy_reasons : {code: build_strategy_reason() 返回值}，
                       可為 None（向下相容）

    Returns
    -------
    list of action dicts，按優先級排序（URGENT → HIGH → MEDIUM → LOW）
    每個 dict 新增 'strategy_reason' key。
    """
    _sr    = strategy_reasons or {}
    _empty = {"primary_reason": "─", "tags": [], "signal_detail": {}, "summary": ""}
    guide: list[dict] = []

    candidates = sorted(
        [(c, s) for c, s in stock_scores.items()
         if c not in current_holdings and s > 0.10],
        key=lambda x: x[1], reverse=True,
    )

    for code, info in holdings_analysis.items():
        q = quotes.get(code)
        if q is None:
            continue

        signal     = info['recent_signal_5d']
        pnl        = info['unrealized_pnl']
        pnl_pct    = info['unrealized_pnl_pct']
        curr_val   = info['current_value']
        exp_ret    = info['expected_annual_ret']
        annual_ret = info['annual_return']

        # ① STOP_LOSS
        if pnl_pct < -0.10 and signal < -0.15:
            guide.append({
                'action': 'STOP_LOSS', 'icon': '🛑',
                'label':  f'止損出場  {code}',
                'sell_code': code, 'buy_code': None,
                'amount': curr_val, 'signal': signal, 'priority': 'URGENT',
                'summary': (
                    f"已虧損 **{pnl_pct:.2%}**（NT${pnl:+,.0f}），"
                    f"技術信號 {signal:+.3f} 持續偏弱。"
                    f"建議出清 **{code}**，止損金額 NT${curr_val:,.0f}。"
                ),
                'detail': (
                    f"- 持倉市值：NT${curr_val:,.0f}\n"
                    f"- 未實現虧損：NT${pnl:+,.0f}（{pnl_pct:.2%}）\n"
                    f"- 近5日均信號：{signal:+.3f}（門檻 < -0.15）"
                ),
                'excess_return': None, 'breakeven_months': None,
                'strategy_reason': _sr.get(code, _empty),
            })
            continue

        # ② SWITCH
        if signal < -0.10 and candidates:
            best_code, best_score = candidates[0]
            cand_info   = holdings_analysis.get(best_code)
            cand_annual = (cand_info['annual_return'] if cand_info
                           else best_score * 0.30 + 0.05)
            excess_annual = cand_annual - exp_ret

            if excess_annual > 0.05:
                switch_cost_ntd  = curr_val * SWITCH_COST_RATE
                total_to_recover = max(0, -pnl) + switch_cost_ntd
                monthly_excess   = curr_val * (excess_annual / 12.0)
                breakeven_m = (total_to_recover / monthly_excess
                               if monthly_excess > 0 else float('inf'))
                breakeven_str = (f"{breakeven_m:.1f} 個月"
                                 if breakeven_m != float('inf') else "無法估算")
                is_worthwhile = breakeven_m < 12.0

                guide.append({
                    'action':  'SWITCH' if is_worthwhile else 'WATCH',
                    'icon':    '⚡' if is_worthwhile else '👁️',
                    'label':   (f'建議換股  {code} → {best_code}'
                                if is_worthwhile else f'觀察  {code}'),
                    'sell_code': code,
                    'buy_code':  best_code if is_worthwhile else None,
                    'amount':    curr_val,
                    'signal':    signal,
                    'priority':  'HIGH' if is_worthwhile else 'MEDIUM',
                    'summary': (
                        f"賣出 **{code}**（信號 {signal:+.3f}），"
                        f"買入 **{best_code}**（GA評分 {best_score:+.4f}）。\n"
                        f"預期超額年化報酬 **{excess_annual:+.1%}**，"
                        f"回本 **{breakeven_str}**。"
                    ) if is_worthwhile else (
                        f"**{code}** 信號偏弱（{signal:+.3f}），"
                        f"換股回本 {breakeven_str} 過長，暫不建議行動。"
                    ),
                    'detail': (
                        f"- 持倉市值：NT${curr_val:,.0f}\n"
                        f"- 未實現損益：NT${pnl:+,.0f}（{pnl_pct:+.2%}）\n"
                        f"- 換股手續費：NT${switch_cost_ntd:,.0f}\n"
                        f"- 預計回本：{breakeven_str}"
                    ),
                    'excess_return': excess_annual,
                    'breakeven_months': breakeven_m,
                    'strategy_reason': _sr.get(best_code if is_worthwhile else code, _empty),
                })
                continue

        # ③ HOLD
        if signal > 0.10:
            guide.append({
                'action': 'HOLD', 'icon': '✅',
                'label':  f'維持持有  {code}',
                'sell_code': None, 'buy_code': code,
                'amount': curr_val, 'signal': signal, 'priority': 'LOW',
                'summary': (
                    f"**{code}** 技術信號正向（{signal:+.3f} ▲），"
                    f"期望年化報酬 {exp_ret:.2%}，建議繼續持有。"
                ),
                'detail': (
                    f"- 近5日均信號：{signal:+.3f}\n"
                    f"- 期望年化報酬：{exp_ret:.2%}\n"
                    f"- 未實現損益：NT${pnl:+,.0f}（{pnl_pct:+.2%}）"
                ),
                'excess_return': None, 'breakeven_months': None,
                'strategy_reason': _sr.get(code, _empty),
            })
            continue

        # ④ WATCH
        guide.append({
            'action': 'WATCH', 'icon': '👁️',
            'label':  f'觀察  {code}',
            'sell_code': None, 'buy_code': None,
            'amount': curr_val, 'signal': signal, 'priority': 'LOW',
            'summary': (
                f"**{code}** 信號中性（{signal:+.3f}），等待突破確認。"
            ),
            'detail': (
                f"- 近5日均信號：{signal:+.3f}（中性區間 ±0.10）\n"
                f"- 未實現損益：NT${pnl:+,.0f}（{pnl_pct:+.2%}）"
            ),
            'excess_return': None, 'breakeven_months': None,
            'strategy_reason': _sr.get(code, _empty),
        })

    # ── 閒置現金判斷 ──────────────────────────────────────────
    max_position = max(
        (r['curr_value'] for r in build_holdings_rows(current_holdings, quotes)),
        default=100_000,
    )
    buy_amount = min(available_cash * 0.40, max_position)

    if available_cash >= 50_000 and candidates:
        best_code, best_score = candidates[0]
        used  = {g['buy_code'] for g in guide if g.get('buy_code')}
        fresh = [(c, s) for c, s in candidates if c not in used]
        if fresh:
            best_code, best_score = fresh[0]
            price_est      = quotes.get(best_code, {}).get('price', 1) or 1
            buy_shares_est = int(buy_amount / price_est / 1000) * 1000
            actual_spend   = buy_shares_est * price_est if buy_shares_est > 0 else 0

            if buy_shares_est > 0:
                guide.append({
                    'action': 'NEW_BUY', 'icon': '🆕',
                    'label':  f'新建部位  {best_code}',
                    'sell_code': None, 'buy_code': best_code,
                    'amount': actual_spend, 'signal': best_score, 'priority': 'MEDIUM',
                    'summary': (
                        f"閒置現金 NT${available_cash:,.0f}，"
                        f"建議買入 **{best_code}**（GA評分 {best_score:+.4f}）。\n"
                        f"建議投入 NT${actual_spend:,.0f}，"
                        f"可買 {buy_shares_est:,} 股（每股 NT${price_est:.1f}）。"
                    ),
                    'detail': (
                        f"- 閒置現金：NT${available_cash:,.0f}\n"
                        f"- 建議投入（40% / 不超最大持倉）：NT${actual_spend:,.0f}\n"
                        f"- 建議分批建倉，不要一次全押"
                    ),
                    'excess_return': None, 'breakeven_months': None,
                    'strategy_reason': _sr.get(best_code, _empty),
                })
            else:
                guide.append({
                    'action': 'CASH', 'icon': '💰', 'label': '保留現金',
                    'sell_code': None, 'buy_code': None,
                    'amount': available_cash, 'signal': 0.0, 'priority': 'LOW',
                    'summary': (
                        f"目前最佳候選 **{best_code}** 現價過高，"
                        f"資金不足買整張，等待回調。"
                    ),
                    'detail': '', 'excess_return': None, 'breakeven_months': None,
                    'strategy_reason': _empty,
                })
    elif available_cash >= 50_000:
        guide.append({
            'action': 'CASH', 'icon': '💰', 'label': '保留現金',
            'sell_code': None, 'buy_code': None,
            'amount': available_cash, 'signal': 0.0, 'priority': 'LOW',
            'summary': (
                f"閒置現金 NT${available_cash:,.0f}，"
                f"目前無明確正向信號標的，保留現金等待機會。"
            ),
            'detail': '', 'excess_return': None, 'breakeven_months': None,
            'strategy_reason': _empty,
        })

    _order = {'URGENT': 0, 'HIGH': 1, 'MEDIUM': 2, 'LOW': 3}
    guide.sort(key=lambda x: _order.get(x['priority'], 9))
    return guide


# ═══════════════════════════════════════════════════════════════
#  策略依據診斷
# ═══════════════════════════════════════════════════════════════

def build_strategy_reason(
        df: pd.DataFrame,
        params: dict,
        ga_score: float,
        short_term_mode: bool = True,
) -> dict:
    """
    根據個股最新技術指標，診斷本次推薦的核心依據。

    Returns
    -------
    dict with keys:
      primary_reason : str   主要依據標籤（英文，供 badge 顯示）
      tags           : list  所有觸發標籤（中文）
      signal_detail  : dict  各指標最新數值（MA / RSI / BB / KDJ / Vol）
      summary        : str   自然語言摘要（中文）
    """
    prices = df['Close'].dropna()
    min_len = max(params.get('ma_long', 20) + 5, 30)
    if len(prices) < min_len:
        return {"primary_reason": "Data Limited", "tags": [], "signal_detail": {}, "summary": "歷史數據不足"}

    # ── 計算各技術信號（取近 3 日均值，降低噪訊）──────────────
    ma_s  = TechnicalFactors.ma_signal(prices, params['ma_short'], params['ma_long'])
    rsi_s = TechnicalFactors.rsi_signal(prices, params['rsi_period'], params['rsi_ob'], params['rsi_os'])
    bb_s  = TechnicalFactors.bb_signal(prices, params['bb_period'], params['bb_std'])
    rsi_r = TechnicalFactors._rsi_raw(prices, params['rsi_period'])

    ma_val  = float(ma_s.iloc[-3:].mean())
    rsi_val = float(rsi_s.iloc[-3:].mean())
    bb_val  = float(bb_s.iloc[-3:].mean())
    rsi_lvl = float(rsi_r.iloc[-1])

    signal_detail: dict = {
        'MA':      round(ma_val,  3),
        'RSI_sig': round(rsi_val, 3),
        'RSI':     round(rsi_lvl, 1),
        'BB':      round(bb_val,  3),
    }

    kdj_val = vol_val = 0.0
    if short_term_mode and all(c in df.columns for c in ['High', 'Low', 'Volume']):
        kdj_s   = TechnicalFactors.kdj_signal(df['High'], df['Low'], df['Close'])
        vol_s   = TechnicalFactors.volume_burst_signal(df['Close'], df['Volume'])
        kdj_val = float(kdj_s.iloc[-3:].mean())
        vol_val = float(vol_s.iloc[-3:].mean())
        signal_detail['KDJ'] = round(kdj_val, 3)
        signal_detail['Vol'] = round(vol_val, 3)

    # ── 觸發標籤 ──────────────────────────────────────────────
    tags: list[str] = []
    os_thr = params.get('rsi_os', 30)

    if   ma_val > 0.20: tags.append("均線強勢突破")
    elif ma_val > 0.08: tags.append("均線多頭")

    if   rsi_lvl < os_thr:         tags.append(f"RSI 超賣 ({rsi_lvl:.0f})")
    elif rsi_lvl < os_thr + 8:     tags.append(f"RSI 低檔 ({rsi_lvl:.0f})")
    elif rsi_val > 0.15:            tags.append("RSI 回升動能")

    if   bb_val > 0.35: tags.append("布林下軌反彈")
    elif bb_val > 0.15: tags.append("布林帶偏多")

    if   kdj_val > 0.35: tags.append("KDJ 超賣反彈")
    elif kdj_val > 0.12: tags.append("KDJ 偏多")

    if   vol_val > 0.25: tags.append("量能爆發")
    elif vol_val > 0.10: tags.append("量能放大")

    if   ga_score > 0.30: tags.append("GA 高評分")
    if not tags:          tags.append("綜合信號偏多")

    # ── 主要依據（最強因子優先）──────────────────────────────
    if   vol_val  > 0.25:         primary = "Volume Breakout"
    elif kdj_val  > 0.35:         primary = "KDJ Oversold"
    elif ma_val   > 0.20:         primary = "Technical Breakout"
    elif rsi_lvl  < os_thr:       primary = f"Low RSI ({rsi_lvl:.0f})"
    elif bb_val   > 0.35:         primary = "BB Rebound"
    elif rsi_val  > 0.15:         primary = "RSI Recovery"
    elif ga_score > 0.25:         primary = "High GA Score"
    else:                          primary = "Mixed Signals"

    # ── 自然語言摘要 ─────────────────────────────────────────
    parts: list[str] = []
    if ma_val  > 0.08:         parts.append(f"均線多頭 {ma_val:+.2f}")
    if rsi_lvl < os_thr + 8:   parts.append(f"RSI {rsi_lvl:.0f}")
    if bb_val  > 0.15:         parts.append(f"布林帶 {bb_val:+.2f}")
    if kdj_val > 0.12:         parts.append(f"KDJ {kdj_val:+.2f}")
    if vol_val > 0.10:         parts.append(f"量能 {vol_val:+.2f}")
    summary = "  ·  ".join(parts) if parts else f"GA {ga_score:+.4f}"

    return {
        "primary_reason": primary,
        "tags":           tags,
        "signal_detail":  signal_detail,
        "summary":        summary,
    }


# ═══════════════════════════════════════════════════════════════
#  選股篩選依據計算
# ═══════════════════════════════════════════════════════════════

def compute_selection_reasons(
    stock_data: dict[str, pd.DataFrame],
    funnel_reasons: dict[str, dict] | None = None,
) -> dict[str, dict]:
    """
    為每支股票計算「篩選依據」資料，提供透明的選股說明。

    對每支已下載的股票計算：
      - avg_volume_lots : 近 5 日平均成交量（張）
      - ma5_pct         : 收盤價偏離 5MA 的百分比（正值 = 高於均線）
      - close           : 最新收盤價
      - ma5             : 最新 5MA

    若有漏斗快照資料（funnel_reasons），優先使用快照中的成交量數字；
    否則從下載的 Volume 欄位自行計算。

    Parameters
    ----------
    stock_data     : {代號: OHLCV DataFrame}
    funnel_reasons : fetch_with_funnel() 回傳的篩選元數據（可為 None）

    Returns
    -------
    dict：{代號: {'avg_volume_lots', 'ma5_pct', 'close', 'ma5', 'reason_str'}}
      reason_str 為供 UI 直接顯示的中文摘要，例如：
      "成交量 5,200 張  均線↑1.2%"
    """
    reasons: dict[str, dict] = {}

    for code, df in stock_data.items():
        try:
            close_ser = df['Close'].dropna()
            vol_ser   = df['Volume'].dropna() if 'Volume' in df.columns else pd.Series(dtype=float)

            if len(close_ser) < 5:
                continue

            latest_close = float(close_ser.iloc[-1])
            ma5_val      = float(close_ser.rolling(5).mean().iloc[-1])
            ma5_pct      = (latest_close - ma5_val) / ma5_val * 100.0 if ma5_val > 0 else 0.0

            # 成交量：優先用漏斗快照（當日精確值），其次用 5 日均量
            if funnel_reasons and code in funnel_reasons:
                avg_vol_lots = funnel_reasons[code].get('volume_lots', 0)
            elif len(vol_ser) >= 5:
                avg_vol_lots = int(vol_ser.iloc[-5:].mean() / 1_000)
            else:
                avg_vol_lots = 0

            trend = "↑" if ma5_pct >= 0 else "↓"
            reason_str = f"量 {avg_vol_lots:,}張  均線{trend}{abs(ma5_pct):.1f}%"

            reasons[code] = {
                'avg_volume_lots': avg_vol_lots,
                'ma5_pct':         round(ma5_pct, 2),
                'close':           round(latest_close, 1),
                'ma5':             round(ma5_val, 1),
                'reason_str':      reason_str,
            }
        except Exception:
            reasons[code] = {
                'avg_volume_lots': 0,
                'ma5_pct':         0.0,
                'close':           0.0,
                'ma5':             0.0,
                'reason_str':      "─",
            }

    return reasons


# ═══════════════════════════════════════════════════════════════
#  完整量化分析 Pipeline
# ═══════════════════════════════════════════════════════════════

def run_full_pipeline(
        available_cash: float,
        current_holdings: dict,
        target_pool: list[str],
        stock_data: dict,           # 由 app.py 的 @st.cache_data 函數提供
        ga_config: dict,
        mc_config: dict,
        top_n: int,
        short_term_mode: bool,
        _prog,                      # st.progress 物件
        _stat,                      # st.empty 物件
        thinking_logger: ThinkingLogger | None = None,
        funnel_reasons: dict | None = None,   # fetch_with_funnel() 回傳的篩選元數據
) -> dict:
    """
    執行完整量化分析流程：GA → MC → 持股分析。

    與 Streamlit 的耦合僅限於 _prog / _stat 兩個進度顯示物件；
    核心運算邏輯完全不依賴 Streamlit，可單獨測試。

    Parameters
    ----------
    available_cash   : 可用資金（NT$）
    current_holdings : {代號: {'cost':..., 'shares':...}}
    target_pool      : 目標股票池代號列表
    stock_data       : 已下載的 OHLCV 字典（由 app 層快取提供）
    ga_config        : GeneticAlgorithm 初始化參數
    mc_config        : MonteCarloSimulator 初始化參數
    top_n            : 最終選股數量
    short_term_mode  : True = 短線模式（KDJ + 量能強化）
    _prog            : st.progress 物件
    _stat            : st.empty 物件
    thinking_logger  : ThinkingLogger 實例（可為 None）

    Returns
    -------
    dict 包含：stock_data, failed_codes, best_params, stock_scores,
               sorted_stocks, fitness_history, selected_codes,
               mc_stats, holdings_analysis, recommendations
    """

    def _think(msg: str, kind: str = 'info') -> None:
        if thinking_logger:
            thinking_logger.log(msg, kind)

    results: dict = {}

    # ── Step 1：使用已快取的數據 ──────────────────────────────
    _think("初始化數據，檢查股票歷史資料...", 'data')
    _stat.text("📡  Step 1/4  載入歷史數據...")
    _prog.progress(5)

    if not stock_data:
        raise RuntimeError("無法下載任何股票數據，請確認網路連線與代號。")

    all_codes = list(set(list(current_holdings.keys()) + target_pool))
    failed    = [c for c in all_codes if c not in stock_data]
    _think(f"數據就緒：成功 {len(stock_data)} 支 / 失敗 {len(failed)} 支", 'data')
    if failed:
        _think(f"失敗代號：{failed}，將從分析中排除", 'warn')

    results['stock_data']   = stock_data
    results['failed_codes'] = failed
    _prog.progress(20)

    # ── Step 2：遺傳演算法優化 ───────────────────────────────
    mode_label = "短線波段（KDJ + 量能強化）" if short_term_mode else "長線持有"
    _think(f"啟動遺傳演算法，策略模式：{mode_label}", 'ga')
    _think(f"種群大小 {ga_config['population_size']}，演化 {ga_config['generations']} 代", 'ga')
    _stat.text("🧬  Step 2/4  遺傳演算法優化（約 30~90 秒）...")
    _prog.progress(25)

    ga_mode = 'short_term' if short_term_mode else 'long_term'
    # 從 ga_config 中移除可能衝突的 mode 鍵（防禦性處理）
    _clean_cfg = {k: v for k, v in ga_config.items()
                  if k in ('population_size', 'generations', 'crossover_rate', 'mutation_rate')}
    ga = GeneticAlgorithm(**_clean_cfg, mode=ga_mode)
    pool_data = {c: stock_data[c] for c in target_pool if c in stock_data}
    if not pool_data:
        raise RuntimeError("目標股池無有效數據。")

    _think("開始 GA 演化，評估染色體適應度（MA / RSI / BB 參數組合）...", 'ga')

    _buf = io.StringIO()
    with redirect_stdout(_buf):
        best_params = ga.evolve(pool_data, verbose=True)

    _think(f"GA 收斂，最佳適應度 {ga.fitness_history[-1]['best_fitness']:+.4f}", 'ga')
    _think(f"MA {best_params['ma_short']}日 / {best_params['ma_long']}日，"
           f"RSI 周期 {best_params['rsi_period']}日", 'signal')

    if short_term_mode:
        _think("短線模式：融入 KDJ 隨機指標進行評分調整...", 'kdj')
        _think("短線模式：融入成交量爆發信號進行評分調整...", 'volume')

    stock_scores  = ga.score_stocks(pool_data, best_params)
    sorted_stocks = sorted(stock_scores.items(), key=lambda x: x[1], reverse=True)
    n_pos = sum(1 for _, s in sorted_stocks if s > 0)
    _think(f"評分完成：正向信號 {n_pos} 支 / 負向信號 {len(sorted_stocks) - n_pos} 支", 'signal')

    results.update({
        'best_params':     best_params,
        'stock_scores':    stock_scores,
        'sorted_stocks':   sorted_stocks,
        'fitness_history': ga.fitness_history,
    })

    # ── 策略依據診斷（每支評分股計算一次）──────────────────────
    _think("計算各股技術因子，生成策略依據標籤...", 'signal')
    strategy_reasons: dict = {}
    for _code, _score in stock_scores.items():
        if _code in pool_data:
            try:
                strategy_reasons[_code] = build_strategy_reason(
                    pool_data[_code], best_params, _score, short_term_mode,
                )
            except Exception:
                strategy_reasons[_code] = {
                    "primary_reason": "─", "tags": [], "signal_detail": {}, "summary": "",
                }
    results['strategy_reasons'] = strategy_reasons

    # ── 選股篩選依據（量能 + MA5 偏差）──────────────────────────
    _think("計算各股量能與均線偏差，生成篩選透明度標籤...", 'data')
    selection_reasons = compute_selection_reasons(pool_data, funnel_reasons)
    results['selection_reasons'] = selection_reasons
    _prog.progress(65)

    # ── Step 3：蒙地卡羅模擬 ─────────────────────────────────
    _think("啟動蒙地卡羅模擬，根據歷史波動率生成未來路徑...", 'mc')
    _stat.text("🎲  Step 3/4  蒙地卡羅模擬...")
    _prog.progress(68)

    positive       = [c for c, s in sorted_stocks if s > 0]
    fallback       = [c for c, _ in sorted_stocks if c not in positive]
    selected_codes = (positive + fallback)[:top_n]
    sel_prices     = {c: stock_data[c]['Close'].dropna()
                      for c in selected_codes if c in stock_data}

    _think(f"選出 {len(selected_codes)} 支標的：{selected_codes}", 'mc')
    _think(f"執行 {mc_config['n_simulations']:,} 路徑 × {mc_config['n_days']} 交易日...", 'mc')

    simulator = MonteCarloSimulator(**mc_config)
    mc_stats  = simulator.simulate_portfolio(sel_prices, available_cash=available_cash)

    _think(f"模擬完成，勝率 {mc_stats['win_rate_pct']}，"
           f"期望季報酬 {mc_stats['expected_return_pct']}", 'mc')

    results.update({'selected_codes': selected_codes, 'mc_stats': mc_stats})
    _prog.progress(85)

    # ── Step 4：持股分析 ──────────────────────────────────────
    _think("分析現有持股機會成本與換股可行性...", 'holding')
    _stat.text("💼  Step 4/4  分析持股機會成本...")
    _prog.progress(90)

    holdings_analysis: dict = {}
    recommendations:   list = []

    if current_holdings:
        h_data   = {c: stock_data[c] for c in current_holdings if c in stock_data}
        analyzer = HoldingsAnalyzer()
        holdings_analysis = analyzer.analyze(current_holdings, h_data, best_params)

        for code, info in holdings_analysis.items():
            sig = info['current_signal']
            sig_label = '看多' if sig > 0.1 else ('看空' if sig < -0.1 else '中性')
            _think(f"  {code}：信號 {sig:+.3f}（{sig_label}），"
                   f"期望年化 {info['expected_annual_ret']:.1%}", 'holding')

        cand_scores = {c: s for c, s in stock_scores.items()
                       if c not in current_holdings and s > 0}
        cand_data   = {c: stock_data[c] for c in cand_scores if c in stock_data}
        recommendations = analyzer.recommend_switches(
            holdings_analysis, cand_scores, cand_data
        )
        # 注入策略依據到每筆換股建議
        _empty_r = {"primary_reason": "─", "tags": [], "signal_detail": {}, "summary": ""}
        for _rec in recommendations:
            _rec['strategy_reason'] = strategy_reasons.get(_rec['buy_code'], _empty_r)
        if recommendations:
            _think(f"發現 {len(recommendations)} 筆換股機會", 'warn')

    results.update({
        'holdings_analysis': holdings_analysis,
        'recommendations':   recommendations,
    })
    _prog.progress(100)
    _think("全部分析完成！", 'done')
    _stat.text("✅  完成！")
    return results
