"""
股票技術分析程式 - 完整功能示範測試
（使用合成資料，繞過 Yahoo Finance 網路限制）
"""
import os
import warnings
import matplotlib
matplotlib.use("Agg")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from stock_analysis import indicators as ind
from stock_analysis import trend as tr
from stock_analysis import chart as ch
from stock_analysis import report as rp
from stock_analysis.screener import _check_conditions

os.makedirs("charts", exist_ok=True)


# ── 合成資料產生器 ────────────────────────────────────────────
def make_synthetic(seed: int, label: str, has_breakout: bool = True) -> pd.DataFrame:
    np.random.seed(seed)
    n = 160
    dates = pd.bdate_range("2022-01-03", periods=n, freq="W-MON")

    # 下降段（前 100 根）
    close = np.array([120 - i * 0.5 + np.random.randn() * 2 for i in range(n)], dtype=float)

    if has_breakout:
        # 後 60 根底部反彈並突破
        for i in range(100, n):
            close[i] = 70 + (i - 100) * 0.6 + np.random.randn() * 2

    open_ = close + np.random.uniform(-0.5, 0.5, n)
    high  = np.maximum(close, open_) + np.abs(np.random.randn(n)) * 1.5
    low   = np.minimum(close, open_) - np.abs(np.random.randn(n)) * 1.5

    # 突破段成交量放大
    vol = np.abs(np.random.randn(n)) * 4e6 + 14e6
    if has_breakout:
        vol[105:] += 16e6

    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": vol},
        index=dates,
    )


# ── 單支股票分析 ──────────────────────────────────────────────
def analyze(ticker: str, df: pd.DataFrame, save_chart: bool = True) -> dict | None:
    df = ind.compute_all(df)
    sh   = tr.find_swing_highs(df, window=7)
    desc = tr.filter_descending_swing_highs(sh, min_points=2)
    tl   = tr.fit_descending_trendline(desc, min_r_squared=0.4)
    brs  = tr.detect_breakout(df, tl, lookback=50) if tl else []

    levels = None
    if brs:
        levels = tr.compute_entry_levels(df, brs[-1], sh)
        bar = df.iloc[brs[-1].index]

        def _gt(a, b):
            try:
                return bool(float(a) > float(b))
            except Exception:
                return False

        conds = {
            "close_above_trendline": _gt(brs[-1].close, brs[-1].trendline_value),
            "volume_above_ma20":     _gt(bar["Volume"], bar["Volume_MA20"]),
            "dif_above_dea":         _gt(bar["DIF"], bar["DEA"]),
            "close_above_ma20":      _gt(bar["Close"], bar["MA20"]),
        }
        levels["conditions"]   = conds
        levels["signal_valid"] = all(conds.values())

    rp.print_report(ticker, levels, tl, brs, df)

    if save_chart:
        out = f"charts/{ticker.replace('.', '_')}.png"
        ch.plot_analysis(df, ticker, tl, brs, levels, sh, output_path=out, show=False)

    return levels


# ── 台股篩選器測試 ────────────────────────────────────────────
def test_screener():
    print("\n" + "=" * 55)
    print("  台股篩選器條件邏輯測試")
    print("=" * 55)

    np.random.seed(99)
    n = 120
    dates_d = pd.bdate_range("2024-01-01", periods=n, freq="B")
    dates_w = pd.bdate_range("2023-01-01", periods=60, freq="W-MON")

    close_d = np.linspace(80, 95, n) + np.random.randn(n)
    vol_d   = np.ones(n) * 25e6 + np.random.randn(n) * 2e6

    daily = pd.DataFrame({
        "Close": close_d,
        "Volume": vol_d,
        "Volume_MA20": pd.Series(vol_d).rolling(20).mean().values,
        "Volume_MA60": pd.Series(vol_d * 0.8).rolling(60).mean().values,
    }, index=dates_d)

    close_w = np.linspace(75, 95, 60) + np.random.randn(60)
    weekly = pd.DataFrame({
        "Close": close_w,
        "SMA5":  pd.Series(close_w).rolling(5).mean().values,
        "SMA20": pd.Series(close_w).rolling(20).mean().values,
    }, index=dates_w)

    conds = _check_conditions(daily, weekly)
    for name, passed in conds.items():
        status = "✔" if passed else "✗"
        print(f"  {status}  {name}")

    all_pass = all(conds.values())
    print(f"\n  結論：{'★ 全部條件通過' if all_pass else '部分條件未達標'}")
    print("=" * 55)


# ── 主程式 ────────────────────────────────────────────────────
if __name__ == "__main__":
    scenarios = [
        ("2330.TW", 42, True,  "台積電（模擬突破）"),
        ("2317.TW", 77, True,  "鴻海（模擬突破）"),
        ("2454.TW", 13, False, "聯發科（無突破）"),
    ]

    print("\n" + "★" * 20 + " 下降趨勢線突破掃描 " + "★" * 20)
    batch_results = []

    for ticker, seed, has_br, desc in scenarios:
        print(f"\n{'─'*55}")
        print(f"  {desc}（{ticker}）")
        df = make_synthetic(seed, ticker, has_breakout=has_br)
        levels = analyze(ticker, df, save_chart=True)
        batch_results.append((ticker, levels))

    # 批次摘要
    print("\n" + "=" * 62)
    print("  批次掃描摘要")
    print("=" * 62)
    print(f"  {'代號':<12} {'訊號':<10} {'進場':>8} {'停損':>8} {'壓力':>8} {'風報比':>6}")
    print("─" * 62)
    for ticker, levels in batch_results:
        if levels and levels.get("signal_valid"):
            row = (f"  {ticker:<12} {'★ 買進':<10}"
                   f" {levels['entry_price']:>8.2f}"
                   f" {levels['stop_loss']:>8.2f}"
                   f" {levels['resistance_price']:>8.2f}"
                   f" {levels['risk_reward']:>6.2f}")
        else:
            row = f"  {ticker:<12} {'— 觀察':<10}"
        print(row)
    print("=" * 62)

    # 篩選器測試
    test_screener()

    # 確認圖表
    charts = os.listdir("charts")
    print(f"\n  產生圖表：{sorted(charts)}")
    print("\n  ✔ 所有功能測試完成\n")
