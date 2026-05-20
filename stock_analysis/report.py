from __future__ import annotations

import math
import pandas as pd


def _fmt(val, decimals: int = 2) -> str:
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return "N/A"
    return f"{val:.{decimals}f}"


def print_report(ticker: str, entry_levels: dict | None, trendline, breakouts: list, df: pd.DataFrame) -> None:
    sep = "=" * 55
    print(sep)
    print(f"  股票代號 : {ticker}")
    print(f"  期間     : {df.index[0].date()} ～ {df.index[-1].date()}")
    print(f"  K 線數量 : {len(df)} 根（週線）")
    print("-" * 55)

    if trendline is None:
        print("  ⚠  無法建立下降趨勢線（波段高點不足或斜率非負）")
    else:
        sh_dates = "、".join(str(sh.date.date()) for sh in trendline.swing_highs)
        print(f"  趨勢線斜率  : {trendline.slope:.4f}  R²={trendline.r_squared:.2f}")
        print(f"  使用高點    : {sh_dates}")

    print("-" * 55)

    if not breakouts:
        print("  ✗ 觀察區間內未偵測到突破訊號")
    else:
        br = breakouts[-1]
        symbol = "✔ 突破" if br.touch_type == "breakout" else "◎ 接近"
        print(f"  {symbol} 偵測")
        print(f"    日期       : {br.date.date()}")
        print(f"    收盤價     : {_fmt(br.close)}")
        print(f"    趨勢線價位 : {_fmt(br.trendline_value)}")
        print(f"    類型       : {br.touch_type}")

    print("-" * 55)

    if entry_levels is None:
        print("  進場建議   : 無（條件未滿足）")
    else:
        conds = entry_levels.get("conditions", {})
        valid = entry_levels.get("signal_valid", False)
        signal_str = "★ 全部條件成立，建議觀察進場" if valid else "△ 部分條件未滿足"
        print(f"  {signal_str}")
        print(f"    進場價    : {_fmt(entry_levels['entry_price'])}")
        print(f"    停損價    : {_fmt(entry_levels['stop_loss'])}")
        print(f"    壓力價    : {_fmt(entry_levels['resistance_price'])}")
        print(f"    風報比    : {_fmt(entry_levels['risk_reward'])}")
        if conds:
            print("  條件明細：")
            labels = {
                "close_above_trendline": "  收盤 > 趨勢線",
                "volume_above_ma20":     "  成交量 > MA20",
                "dif_above_dea":         "  DIF > DEA",
                "close_above_ma20":      "  收盤 > MA20",
            }
            for k, label in labels.items():
                status = "✔" if conds.get(k) else "✗"
                print(f"    {status} {label}")

    print(sep)
    print()


def format_report_csv(ticker: str, entry_levels: dict | None, trendline, breakouts: list) -> str:
    if not breakouts:
        br_date = ""
        br_type = ""
    else:
        br = breakouts[-1]
        br_date = str(br.date.date())
        br_type = br.touch_type

    slope = _fmt(trendline.slope, 4) if trendline else ""
    r2 = _fmt(trendline.r_squared) if trendline else ""

    if entry_levels:
        entry = _fmt(entry_levels["entry_price"])
        stop = _fmt(entry_levels["stop_loss"])
        res = _fmt(entry_levels["resistance_price"])
        rr = _fmt(entry_levels["risk_reward"])
        valid = str(entry_levels.get("signal_valid", False))
    else:
        entry = stop = res = rr = valid = ""

    return ",".join([ticker, br_date, br_type, entry, stop, res, rr, slope, r2, valid])


CSV_HEADER = "ticker,breakout_date,touch_type,entry,stop_loss,resistance,risk_reward,trendline_slope,r_squared,signal_valid"
