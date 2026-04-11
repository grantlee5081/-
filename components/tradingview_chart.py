"""
tradingview_chart.py — TradingView Lightweight Charts 嵌入元件
────────────────────────────────────────────────────────────────
使用 TradingView Lightweight Charts v4（CDN）透過 st.components.v1.html 嵌入。
支援：
  ✓ 手機端手勢縮放 / 左右滑動（原生支援）
  ✓ MA5 / MA20 / MA60 均線
  ✓ 成交量副圖（紅漲綠跌 Morandi 色）
  ✓ 台股 Morandi 配色（漲 #B85450 / 跌 #5A8A7A）
  ✓ 自適應寬度 / 深淺色主題（預設淺色）
"""

import json
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components


def _compute_ma(series: pd.Series, period: int) -> list[dict]:
    """計算 MA 並回傳 Lightweight Charts 接受的 [{time, value}] 格式。"""
    ma = series.rolling(period).mean()
    result = []
    for date, val in zip(series.index, ma):
        if pd.notna(val):
            result.append({
                "time": date.strftime("%Y-%m-%d"),
                "value": round(float(val), 2),
            })
    return result


def tradingview_chart(
    df: pd.DataFrame,
    code: str,
    name: str = "",
    height: int = 520,
) -> None:
    """
    在 Streamlit 中嵌入 TradingView Lightweight Charts K 線圖。

    Parameters
    ----------
    df    : OHLCV DataFrame（index=DatetimeIndex，欄位：Open/High/Low/Close/Volume）
    code  : 股票代號（顯示在圖表標題）
    name  : 股票中文名稱（選填）
    height: 圖表高度（px，含副圖）
    """
    if df is None or df.empty:
        st.warning(f"⚠️  {code} 無可用數據")
        return

    df = df.tail(120).copy()
    df.index = pd.to_datetime(df.index)

    # ── OHLCV ──────────────────────────────────────────────────
    candle_data = []
    volume_data = []
    for date, row in df.iterrows():
        ts = date.strftime("%Y-%m-%d")
        candle_data.append({
            "time":  ts,
            "open":  round(float(row["Open"]),  2),
            "high":  round(float(row["High"]),  2),
            "low":   round(float(row["Low"]),   2),
            "close": round(float(row["Close"]), 2),
        })
        is_up = float(row["Close"]) >= float(row["Open"])
        volume_data.append({
            "time":  ts,
            "value": int(row["Volume"]),
            "color": "rgba(184,84,80,0.65)" if is_up else "rgba(90,138,122,0.65)",
        })

    ma5_data  = _compute_ma(df["Close"], 5)
    ma20_data = _compute_ma(df["Close"], 20)
    ma60_data = _compute_ma(df["Close"], 60)

    title = f"{code}　{name}" if name else code

    html = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=5.0">
<script src="https://unpkg.com/lightweight-charts@4.1.3/dist/lightweight-charts.standalone.production.js"></script>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: transparent; font-family: 'Inter', 'Noto Sans TC', sans-serif; }}
  #wrapper {{
    width: 100%;
    background: #FDFBF7;
    border-radius: 14px;
    border: 1px solid #E6E1D9;
    overflow: hidden;
    padding: 12px 0 0;
  }}
  #title {{
    padding: 0 16px 8px;
    font-size: 13px;
    font-weight: 600;
    color: #383838;
    letter-spacing: 0.02em;
  }}
  #legend {{
    padding: 4px 16px 8px;
    font-size: 11px;
    color: #7A7A7A;
    display: flex;
    gap: 16px;
    align-items: center;
    flex-wrap: wrap;
  }}
  .leg {{ display: flex; align-items: center; gap: 4px; }}
  .leg-dot {{ width: 10px; height: 2px; border-radius: 1px; }}
  #chart-main {{ width: 100%; }}
  #chart-vol  {{ width: 100%; border-top: 1px solid #E6E1D9; }}
</style>
</head>
<body>
<div id="wrapper">
  <div id="title">📊 {title}　近 120 日 K 線</div>
  <div id="legend">
    <span class="leg"><span class="leg-dot" style="background:#B8966A;height:2px;width:24px;"></span>MA5</span>
    <span class="leg"><span class="leg-dot" style="background:#607D8B;height:2px;width:24px;"></span>MA20</span>
    <span class="leg"><span class="leg-dot" style="background:#7A9E87;height:2px;width:24px;"></span>MA60</span>
    <span style="color:#B85450;">▲漲</span>
    <span style="color:#5A8A7A;">▼跌</span>
  </div>
  <div id="chart-main"></div>
  <div id="chart-vol"></div>
</div>

<script>
(function() {{
  const CANDLE = {json.dumps(candle_data)};
  const VOLUME = {json.dumps(volume_data)};
  const MA5    = {json.dumps(ma5_data)};
  const MA20   = {json.dumps(ma20_data)};
  const MA60   = {json.dumps(ma60_data)};

  const baseOpts = {{
    layout: {{
      background: {{ type: 'solid', color: 'rgba(0,0,0,0)' }},
      textColor: '#4A4A4A',
      fontSize: 11,
    }},
    grid: {{
      vertLines: {{ color: 'rgba(0,0,0,0.04)' }},
      horzLines: {{ color: 'rgba(0,0,0,0.04)' }},
    }},
    crosshair: {{ mode: LightweightCharts.CrosshairMode.Normal }},
    rightPriceScale: {{ borderColor: 'rgba(0,0,0,0.08)' }},
    timeScale: {{
      borderColor: 'rgba(0,0,0,0.08)',
      timeVisible: false,
      secondsVisible: false,
    }},
    handleScroll: {{ mouseWheel: true, pressedMouseMove: true, horzTouchDrag: true, vertTouchDrag: false }},
    handleScale: {{ axisPressedMouseMove: true, pinch: true, mouseWheel: true }},
  }};

  // ── Main chart ────────────────────────────────────────────
  const mainEl = document.getElementById('chart-main');
  const mainH  = {height - 140};
  const mainChart = LightweightCharts.createChart(mainEl, {{
    ...baseOpts,
    width: mainEl.clientWidth,
    height: mainH,
  }});

  const candleSeries = mainChart.addCandlestickSeries({{
    upColor:         '#C97870',
    downColor:       '#6B9E8E',
    borderUpColor:   '#B85450',
    borderDownColor: '#5A8A7A',
    wickUpColor:     '#B85450',
    wickDownColor:   '#5A8A7A',
  }});
  candleSeries.setData(CANDLE);

  const ma5Series = mainChart.addLineSeries({{
    color: '#B8966A', lineWidth: 1.5, title: 'MA5',
    lastValueVisible: false, priceLineVisible: false,
  }});
  ma5Series.setData(MA5);

  const ma20Series = mainChart.addLineSeries({{
    color: '#607D8B', lineWidth: 1.5, title: 'MA20',
    lastValueVisible: false, priceLineVisible: false,
  }});
  ma20Series.setData(MA20);

  const ma60Series = mainChart.addLineSeries({{
    color: '#7A9E87', lineWidth: 1.5, title: 'MA60',
    lastValueVisible: false, priceLineVisible: false,
  }});
  ma60Series.setData(MA60);

  mainChart.timeScale().fitContent();

  // ── Volume chart ──────────────────────────────────────────
  const volEl = document.getElementById('chart-vol');
  const volChart = LightweightCharts.createChart(volEl, {{
    ...baseOpts,
    width: volEl.clientWidth,
    height: 110,
    timeScale: {{ ...baseOpts.timeScale, visible: true }},
  }});

  const volSeries = volChart.addHistogramSeries({{
    color: 'rgba(96,125,139,0.5)',
    priceFormat: {{ type: 'volume' }},
    priceScaleId: 'vol',
  }});
  volSeries.priceScale().applyOptions({{
    scaleMargins: {{ top: 0.1, bottom: 0 }},
  }});
  volSeries.setData(VOLUME);
  volChart.timeScale().fitContent();

  // ── Sync crosshair between charts ─────────────────────────
  function syncCrosshair(source, target, series) {{
    source.subscribeCrosshairMove(param => {{
      if (param.time) {{
        target.setCrosshairPosition(param.seriesData.get(series)?.value ?? 0, param.time, volSeries);
      }} else {{
        target.clearCrosshairPosition();
      }}
    }});
  }}
  syncCrosshair(mainChart, volChart, candleSeries);

  // ── Responsive resize ─────────────────────────────────────
  function resizeCharts() {{
    mainChart.applyOptions({{ width: mainEl.clientWidth }});
    volChart.applyOptions({{ width: volEl.clientWidth }});
  }}
  window.addEventListener('resize', resizeCharts);
  // Also sync timeScale on scroll
  mainChart.timeScale().subscribeVisibleLogicalRangeChange(range => {{
    if (range) volChart.timeScale().setVisibleLogicalRange(range);
  }});
  volChart.timeScale().subscribeVisibleLogicalRangeChange(range => {{
    if (range) mainChart.timeScale().setVisibleLogicalRange(range);
  }});
}})();
</script>
</body>
</html>
"""

    components.html(html, height=height + 10, scrolling=False)
