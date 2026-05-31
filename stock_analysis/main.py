"""
股票下降趨勢線突破掃描器

使用方式：
    python -m stock_analysis.main --tickers 2330.TW 2317.TW AAPL
    python -m stock_analysis.main --tickers 2330.TW --period 3y --window 5
    python -m stock_analysis.main --file tickers.txt --no-show
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

import matplotlib
import pandas as pd
import yfinance as yf

from . import indicators as ind
from . import trend as tr
from . import chart as ch
from . import report as rp

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


def run_analysis(
    ticker: str,
    period: str = "3y",
    swing_window: int = 5,
    lookback: int = 20,
    min_swing_highs: int = 3,
    output_path: str | None = None,
    show_chart: bool = True,
) -> dict | None:
    logger.info("▶ 分析 %s ...", ticker)

    try:
        raw = yf.download(ticker, period=period, interval="1wk",
                          auto_adjust=True, progress=False)
    except Exception as e:
        logger.error("下載 %s 失敗：%s", ticker, e)
        return None

    if raw is None or raw.empty:
        logger.error("%s：無法取得資料", ticker)
        return None

    # yfinance 多層欄位時攤平
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.droplevel(1)

    df = raw.dropna(subset=["Close", "Open", "High", "Low", "Volume"]).copy()

    if len(df) < 30:
        logger.warning("%s：資料不足（%d 根），略過", ticker, len(df))
        return None

    try:
        df = ind.compute_all(df)
    except ValueError as e:
        logger.error("%s：%s", ticker, e)
        return None

    swing_highs = tr.find_swing_highs(df, window=swing_window)
    desc_highs = tr.filter_descending_swing_highs(swing_highs, min_points=min_swing_highs)
    trendline = tr.fit_descending_trendline(desc_highs)
    breakouts = tr.detect_breakout(df, trendline, lookback=lookback) if trendline else []

    entry_levels = None
    if breakouts:
        br = breakouts[-1]
        entry_levels = tr.compute_entry_levels(df, br, swing_highs)
        bar = df.iloc[br.index]

        def _safe_gt(a, b) -> bool:
            try:
                return bool(a > b)
            except Exception:
                return False

        conditions = {
            "close_above_trendline": _safe_gt(br.close, br.trendline_value),
            "volume_above_ma20": _safe_gt(bar.get("Volume"), bar.get("Volume_MA20")),
            "dif_above_dea": _safe_gt(bar.get("DIF"), bar.get("DEA")),
            "close_above_ma20": _safe_gt(bar.get("Close"), bar.get("MA20")),
        }
        entry_levels["conditions"] = conditions
        entry_levels["signal_valid"] = all(conditions.values())

    rp.print_report(ticker, entry_levels, trendline, breakouts, df)

    if output_path or show_chart:
        ch.plot_analysis(
            df, ticker, trendline, breakouts, entry_levels,
            swing_highs, output_path=output_path, show=show_chart,
        )

    return entry_levels


def _load_tickers(args) -> list[str]:
    tickers = []
    if args.tickers:
        tickers.extend(args.tickers)
    if args.file:
        try:
            with open(args.file) as f:
                tickers.extend(line.strip() for line in f if line.strip())
        except FileNotFoundError:
            logger.error("找不到檔案：%s", args.file)
            sys.exit(1)
    return list(dict.fromkeys(tickers))  # 去重保序


def main() -> None:
    parser = argparse.ArgumentParser(description="股票下降趨勢線突破掃描器")
    parser.add_argument("--tickers", nargs="+", help="Yahoo Finance 股票代號（可多個）")
    parser.add_argument("--file", help="股票代號清單檔案（每行一個）")
    parser.add_argument("--period", default="3y", help="yfinance 期間（預設 3y）")
    parser.add_argument("--window", type=int, default=5, help="波段高點滾動視窗（預設 5）")
    parser.add_argument("--lookback", type=int, default=20, help="掃描最近幾根 K 線（預設 20）")
    parser.add_argument("--min-swing-highs", type=int, default=3, help="最少下降高點數（預設 3）")
    parser.add_argument("--output-dir", default=None, help="圖表存放目錄")
    parser.add_argument("--csv-report", default=None, help="批次結果 CSV 路徑")
    parser.add_argument("--no-show", action="store_true", help="不顯示圖表視窗")
    args = parser.parse_args()

    if not args.tickers and not args.file:
        parser.error("請提供 --tickers 或 --file")

    if args.no_show:
        matplotlib.use("Agg")

    tickers = _load_tickers(args)
    if not tickers:
        logger.error("股票代號清單為空")
        sys.exit(1)

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    csv_rows = [rp.CSV_HEADER]
    results = []

    for ticker in tickers:
        out_path = None
        if args.output_dir:
            out_path = os.path.join(args.output_dir, f"{ticker.replace('.', '_')}.png")

        try:
            entry = run_analysis(
                ticker,
                period=args.period,
                swing_window=args.window,
                lookback=args.lookback,
                min_swing_highs=args.min_swing_highs,
                output_path=out_path,
                show_chart=not args.no_show,
            )
            results.append((ticker, entry))
        except Exception as e:
            logger.error("%s 分析失敗：%s", ticker, e)
            results.append((ticker, None))

    # 批次摘要
    print("\n" + "=" * 60)
    print("  批次掃描摘要")
    print("=" * 60)
    header = f"  {'代號':<12} {'訊號':<8} {'進場':>8} {'停損':>8} {'壓力':>8} {'風報比':>6}"
    print(header)
    print("-" * 60)
    for ticker, entry in results:
        if entry and entry.get("signal_valid"):
            row = (f"  {ticker:<12} {'★ 買進':<8}"
                   f" {entry['entry_price']:>8.2f}"
                   f" {entry['stop_loss']:>8.2f}"
                   f" {entry['resistance_price']:>8.2f}"
                   f" {entry['risk_reward']:>6.2f}")
        else:
            row = f"  {ticker:<12} {'—':<8}"
        print(row)
    print("=" * 60)

    if args.csv_report:
        with open(args.csv_report, "w", encoding="utf-8") as f:
            f.write("\n".join(csv_rows))
        print(f"\n  CSV 已存：{args.csv_report}")


if __name__ == "__main__":
    main()
