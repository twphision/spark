from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class SwingHigh:
    date: pd.Timestamp
    index: int
    high: float


@dataclass
class Trendline:
    slope: float
    intercept: float
    r_squared: float
    swing_highs: list = field(default_factory=list)

    def value_at(self, idx: int) -> float:
        return self.slope * idx + self.intercept


@dataclass
class BreakoutResult:
    date: pd.Timestamp
    index: int
    close: float
    trendline_value: float
    touch_type: str  # "breakout" | "approach"


def find_swing_highs(df: pd.DataFrame, window: int = 5) -> list:
    if window % 2 == 0:
        window += 1
        logger.warning("window 調整為奇數：%d", window)

    if len(df) < window * 2:
        logger.warning("資料量不足以偵測波段高點（需要 %d 根）", window * 2)
        return []

    roll_max = df["High"].rolling(2 * window + 1, center=True, min_periods=1).max()
    is_peak = (df["High"] == roll_max) & (df["High"] > df["High"].shift(1))

    candidates: list[SwingHigh] = []
    for pos, (idx, row) in enumerate(df.iterrows()):
        if is_peak.iloc[pos]:
            candidates.append(SwingHigh(date=idx, index=pos, high=row["High"]))

    # 去重：相鄰 window//2 bars 內只保留最高者
    if not candidates:
        return []

    merged: list[SwingHigh] = [candidates[0]]
    for c in candidates[1:]:
        if c.index - merged[-1].index <= window // 2:
            if c.high > merged[-1].high:
                merged[-1] = c
        else:
            merged.append(c)

    return merged


def filter_descending_swing_highs(swing_highs: list, min_points: int = 3) -> list:
    if not swing_highs:
        return []

    def _filter(tolerance: float) -> list:
        result = [swing_highs[0]]
        for sh in swing_highs[1:]:
            if sh.high < result[-1].high * (1 + tolerance):
                result.append(sh)
        return result

    result = _filter(0.02)
    if len(result) < min_points:
        result = _filter(0.05)
        if len(result) < min_points:
            logger.warning("下降波段高點不足 %d 個（僅找到 %d 個）", min_points, len(result))
    return result


def fit_descending_trendline(swing_highs: list, min_r_squared: float = 0.6) -> Trendline | None:
    if len(swing_highs) < 2:
        return None

    xs = np.array([sh.index for sh in swing_highs], dtype=float)
    ys = np.array([sh.high for sh in swing_highs], dtype=float)

    try:
        slope, intercept, r_value, _, _ = stats.linregress(xs, ys)
    except ValueError as e:
        logger.warning("趨勢線擬合失敗：%s", e)
        return None

    if slope >= 0:
        logger.info("趨勢線斜率為正（非下降趨勢），略過")
        return None

    r_squared = r_value ** 2
    if r_squared < min_r_squared:
        logger.info("趨勢線 R²=%.2f 低於門檻 %.2f，略過", r_squared, min_r_squared)
        return None

    return Trendline(slope=slope, intercept=intercept, r_squared=r_squared, swing_highs=swing_highs)


def detect_breakout(
    df: pd.DataFrame,
    trendline: Trendline,
    approach_pct: float = 0.02,
    lookback: int = 20,
) -> list:
    if trendline is None:
        return []

    results: list[BreakoutResult] = []
    start = max(1, len(df) - lookback)

    for i in range(start, len(df)):
        tl_now = trendline.value_at(i)
        tl_prev = trendline.value_at(i - 1)
        close = df["Close"].iloc[i]
        prev_close = df["Close"].iloc[i - 1]

        if prev_close <= tl_prev and close > tl_now:
            results.append(BreakoutResult(
                date=df.index[i],
                index=i,
                close=close,
                trendline_value=tl_now,
                touch_type="breakout",
            ))
        elif abs(close - tl_now) / tl_now <= approach_pct:
            results.append(BreakoutResult(
                date=df.index[i],
                index=i,
                close=close,
                trendline_value=tl_now,
                touch_type="approach",
            ))

    return results


def compute_entry_levels(
    df: pd.DataFrame,
    breakout: BreakoutResult,
    swing_highs: list,
    stop_lookback: int = 3,
) -> dict:
    entry = breakout.close
    i = breakout.index
    stop_start = max(0, i - stop_lookback)
    stop_loss = df["Low"].iloc[stop_start:i].min()

    resistance = float("nan")
    res_label = "historical_max"
    for sh in sorted(swing_highs, key=lambda s: s.high):
        if sh.high > entry and sh.date > breakout.date:
            resistance = sh.high
            res_label = "swing_high"
            break

    if np.isnan(resistance):
        resistance = df["High"].max()

    risk = entry - stop_loss
    reward = resistance - entry
    rr = reward / risk if risk > 0 else float("nan")

    return {
        "entry_price": entry,
        "stop_loss": stop_loss,
        "resistance_price": resistance,
        "resistance_label": res_label,
        "breakout_date": breakout.date,
        "risk_reward": rr,
    }
