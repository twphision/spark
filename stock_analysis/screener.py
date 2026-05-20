"""
台股每日篩選器

篩選條件（依序套用）：
  1. 20 ≤ 日收盤 ≤ 200
  2. 日量 MA20 > 日量 MA60
  3. 日成交量 > 20,000,000 股（2 萬張）
  4. 日收盤 > 週 SMA20
  5. 週 SMA5 > 週 SMA20
"""
from __future__ import annotations

import datetime
import io
import json
import logging
import os
import time

import pandas as pd
import requests
import yfinance as yf

logger = logging.getLogger(__name__)

TWSE_ISIN_URL = "https://isin.twse.com.tw/isin/C_public.jsp?strMode=2"
VOLUME_THRESHOLD = 20_000_000  # 2 萬張


def fetch_taiwan_tickers(fallback_file: str = "tickers.txt") -> list[str]:
    try:
        resp = requests.get(TWSE_ISIN_URL, timeout=15,
                            headers={"User-Agent": "Mozilla/5.0"})
        resp.encoding = "big5"
        tables = pd.read_html(io.StringIO(resp.text))
        df = tables[0]
        df.columns = df.iloc[0]
        df = df.iloc[1:].reset_index(drop=True)

        code_col = df.columns[0]
        codes = df[code_col].astype(str).str.extract(r"^(\d{4,6})\s")[0].dropna()
        tickers = [f"{c}.TW" for c in codes if c.isdigit() and len(c) in (4, 5, 6)]
        logger.info("從 TWSE 取得 %d 支股票", len(tickers))
        return tickers

    except Exception as e:
        logger.warning("TWSE 清單抓取失敗（%s），嘗試本地 fallback：%s", e, fallback_file)

    if os.path.exists(fallback_file):
        with open(fallback_file) as f:
            tickers = [line.strip() for line in f if line.strip()]
        logger.info("從 %s 讀取 %d 支股票", fallback_file, len(tickers))
        return tickers

    logger.error("找不到台股清單，請提供 %s", fallback_file)
    return []


def _download_safe(ticker: str, period: str, interval: str) -> pd.DataFrame | None:
    try:
        raw = yf.download(ticker, period=period, interval=interval,
                          auto_adjust=True, progress=False)
        if raw is None or raw.empty:
            return None
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.droplevel(1)
        return raw.dropna(subset=["Close"])
    except Exception:
        return None


def _check_conditions(daily: pd.DataFrame, weekly: pd.DataFrame) -> dict:
    d = daily.iloc[-1]
    w = weekly.iloc[-1]

    close_d = float(d.get("Close", float("nan")))
    vol_d = float(d.get("Volume", 0))
    vol_ma20 = float(d.get("Volume_MA20", float("nan")))
    vol_ma60 = float(d.get("Volume_MA60", float("nan")))
    sma5_w = float(w.get("SMA5", float("nan")))
    sma20_w = float(w.get("SMA20", float("nan")))

    import math

    def ok(a, b):
        return not math.isnan(a) and not math.isnan(b) and a > b

    return {
        "price_range": 20 <= close_d <= 200,
        "vol_ma20_gt_ma60": ok(vol_ma20, vol_ma60),
        "volume_gt_threshold": vol_d > VOLUME_THRESHOLD,
        "close_gt_weekly_sma20": ok(close_d, sma20_w),
        "weekly_sma5_gt_sma20": ok(sma5_w, sma20_w),
    }


def screen_daily(
    tickers: list[str],
    output_csv: str,
    period_daily: str = "6mo",
    period_weekly: str = "1y",
    sleep_sec: float = 0.3,
) -> pd.DataFrame:
    rows = []
    today = datetime.date.today().isoformat()
    passed = 0

    for i, ticker in enumerate(tickers):
        if i % 50 == 0:
            logger.info("進度 %d/%d ...", i, len(tickers))

        daily = _download_safe(ticker, period_daily, "1d")
        if daily is None or len(daily) < 60:
            time.sleep(sleep_sec)
            continue

        weekly = _download_safe(ticker, period_weekly, "1wk")
        if weekly is None or len(weekly) < 20:
            time.sleep(sleep_sec)
            continue

        # 計算指標
        daily["Volume_MA20"] = daily["Volume"].rolling(20).mean()
        daily["Volume_MA60"] = daily["Volume"].rolling(60).mean()
        weekly["SMA5"] = weekly["Close"].rolling(5).mean()
        weekly["SMA20"] = weekly["Close"].rolling(20).mean()

        conds = _check_conditions(daily, weekly)

        if all(conds.values()):
            d = daily.iloc[-1]
            w = weekly.iloc[-1]
            rows.append({
                "ticker": ticker,
                "date": today,
                "close": round(float(d["Close"]), 2),
                "volume": int(d["Volume"]),
                "vol_ma20": round(float(d["Volume_MA20"]), 0),
                "vol_ma60": round(float(d["Volume_MA60"]), 0),
                "weekly_sma5": round(float(w["SMA5"]), 2),
                "weekly_sma20": round(float(w["SMA20"]), 2),
                "conditions_passed": json.dumps(conds, ensure_ascii=False),
            })
            passed += 1

        time.sleep(sleep_sec)

    result_df = pd.DataFrame(rows)

    if not result_df.empty:
        result_df.to_csv(output_csv, index=False, encoding="utf-8-sig")
        logger.info("篩選完成：%d / %d 通過，已存至 %s", passed, len(tickers), output_csv)
    else:
        logger.info("篩選完成：無股票通過所有條件")
        # 寫出空 CSV 含表頭
        pd.DataFrame(columns=[
            "ticker", "date", "close", "volume", "vol_ma20", "vol_ma60",
            "weekly_sma5", "weekly_sma20", "conditions_passed"
        ]).to_csv(output_csv, index=False, encoding="utf-8-sig")

    return result_df


def run_screener_job(output_dir: str = "./screener_output", fallback_file: str = "tickers.txt") -> None:
    os.makedirs(output_dir, exist_ok=True)
    today = datetime.date.today().isoformat()
    output_csv = os.path.join(output_dir, f"screener_{today}.csv")

    logger.info("=== 台股每日篩選開始：%s ===", today)
    tickers = fetch_taiwan_tickers(fallback_file=fallback_file)

    if not tickers:
        logger.error("股票清單為空，中止")
        return

    result_df = screen_daily(tickers, output_csv)
    print(f"\n=== 篩選結果摘要（{today}）===")
    print(f"掃描股票數：{len(tickers)}")
    print(f"通過條件數：{len(result_df)}")
    if not result_df.empty:
        print(result_df[["ticker", "close", "volume", "weekly_sma5", "weekly_sma20"]].to_string(index=False))
    print(f"\nCSV 檔案：{output_csv}")
