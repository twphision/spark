from __future__ import annotations

import math

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _draw_candlesticks(ax: plt.Axes, df: pd.DataFrame) -> None:
    for i, (idx, row) in enumerate(df.iterrows()):
        open_, high, low, close = row["Open"], row["High"], row["Low"], row["Close"]
        color = "#26a69a" if close >= open_ else "#ef5350"

        body_bot = min(open_, close)
        body_top = max(open_, close)
        body_h = body_top - body_bot if body_top != body_bot else 0.01

        ax.bar(i, body_h, bottom=body_bot, color=color, width=0.6, linewidth=0)
        ax.vlines(i, low, high, color=color, linewidth=0.8)


def _resolve_font() -> None:
    import matplotlib.font_manager as fm
    cjk_keywords = ["noto", "wqy", "cjk", "chinese", "han", "pingfang", "heiti", "source han"]
    available = [f.name for f in fm.fontManager.ttflist
                 if any(k in f.name.lower() for k in cjk_keywords)]
    if available:
        matplotlib.rcParams["font.family"] = [available[0], "DejaVu Sans", "sans-serif"]
    else:
        matplotlib.rcParams["font.family"] = ["DejaVu Sans", "sans-serif"]
    matplotlib.rcParams["axes.unicode_minus"] = False


def _label(zh: str, en: str) -> str:
    """在沒有 CJK 字體時自動 fallback 為英文標籤。"""
    import matplotlib.font_manager as fm
    cjk_available = any(
        any(k in f.name.lower() for k in ["noto", "wqy", "cjk", "chinese", "han"])
        for f in fm.fontManager.ttflist
    )
    return zh if cjk_available else en


def plot_analysis(
    df: pd.DataFrame,
    ticker: str,
    trendline,
    breakouts: list,
    entry_levels: dict | None,
    swing_highs: list,
    output_path: str | None = None,
    show: bool = True,
) -> None:
    _resolve_font()

    fig, (ax_candle, ax_vol, ax_macd) = plt.subplots(
        3, 1,
        figsize=(16, 12),
        gridspec_kw={"height_ratios": [3, 1, 1]},
        sharex=True,
    )
    fig.patch.set_facecolor("#131722")
    for ax in (ax_candle, ax_vol, ax_macd):
        ax.set_facecolor("#131722")
        ax.tick_params(colors="#aaaaaa")
        ax.spines["bottom"].set_color("#444444")
        ax.spines["top"].set_color("#444444")
        ax.spines["left"].set_color("#444444")
        ax.spines["right"].set_color("#444444")
        ax.yaxis.label.set_color("#aaaaaa")

    n = len(df)
    x = np.arange(n)

    # --- Panel 1: K 線圖 ---
    _draw_candlesticks(ax_candle, df)

    ma_cfg = [("MA5", "dodgerblue"), ("MA20", "hotpink"), ("MA60", "darkorange")]
    for col, color in ma_cfg:
        if col in df.columns and not df[col].isna().all():
            ax_candle.plot(x, df[col].values, color=color, linewidth=1.0, label=col)

    # 下降趨勢線（黃色虛線）
    if trendline is not None:
        tl_y = trendline.slope * x + trendline.intercept
        ax_candle.plot(x, tl_y, "--", color="yellow", linewidth=1.5,
                       label=_label("下降趨勢線", "Descending Trendline"), zorder=4)

        # 波段高點三角標記
        for sh in trendline.swing_highs:
            ax_candle.scatter(sh.index, sh.high, marker="^", color="gray", s=40, zorder=5)

    # 突破 / 接近標記（橘色圓點）
    for br in breakouts:
        color = "orange" if br.touch_type == "breakout" else "yellow"
        size = 100 if br.touch_type == "breakout" else 60
        ax_candle.scatter(br.index, br.trendline_value, color=color, s=size, zorder=6,
                          label=f"{br.touch_type} ({br.date.date()})")

    # 進場/停損/壓力線
    if entry_levels:
        ax_candle.axhline(entry_levels["entry_price"], color="lime", linestyle="--", linewidth=1.0,
                          label=f"{_label('進場', 'Entry')} {entry_levels['entry_price']:.2f}")
        ax_candle.axhline(entry_levels["stop_loss"], color="red", linestyle="--", linewidth=1.0,
                          label=f"{_label('停損', 'Stop')} {entry_levels['stop_loss']:.2f}")
        ax_candle.axhline(entry_levels["resistance_price"], color="cyan", linestyle="--", linewidth=1.0,
                          label=f"{_label('壓力', 'Resist')} {entry_levels['resistance_price']:.2f}")

    ax_candle.set_ylabel(_label("價格", "Price"), color="#aaaaaa")
    ax_candle.legend(loc="upper right", fontsize=7, framealpha=0.3,
                     facecolor="#131722", labelcolor="#cccccc")

    # --- Panel 2: 成交量 ---
    vol_colors = [
        "#26a69a" if df["Close"].iloc[i] >= df["Open"].iloc[i] else "#ef5350"
        for i in range(n)
    ]
    ax_vol.bar(x, df["Volume"].values, color=vol_colors, alpha=0.7, width=0.6)

    if "Volume_MA20" in df.columns and not df["Volume_MA20"].isna().all():
        ax_vol.plot(x, df["Volume_MA20"].values, color="darkorange", linewidth=1.0, label="Vol MA20")

    ax_vol.set_ylabel(_label("成交量", "Volume"), color="#aaaaaa")
    ax_vol.legend(loc="upper right", fontsize=7, framealpha=0.3,
                  facecolor="#131722", labelcolor="#cccccc")

    # --- Panel 3: MACD ---
    if "MACD_hist" in df.columns and not df["MACD_hist"].isna().all():
        hist = df["MACD_hist"].values
        hist_colors = ["#26a69a" if v >= 0 else "#ef5350" for v in hist]
        ax_macd.bar(x, hist, color=hist_colors, alpha=0.8, width=0.6)

    if "DIF" in df.columns and not df["DIF"].isna().all():
        ax_macd.plot(x, df["DIF"].values, color="cyan", linewidth=1.0, label="DIF")
    if "DEA" in df.columns and not df["DEA"].isna().all():
        ax_macd.plot(x, df["DEA"].values, color="magenta", linewidth=1.0, label="DEA")

    ax_macd.axhline(0, color="#555555", linewidth=0.8)
    ax_macd.set_ylabel("MACD", color="#aaaaaa")  # MACD is always ASCII
    ax_macd.legend(loc="upper right", fontsize=7, framealpha=0.3,
                   facecolor="#131722", labelcolor="#cccccc")

    # X 軸日期標籤
    tick_step = max(1, n // 10)
    tick_positions = list(range(0, n, tick_step))
    tick_labels = [df.index[i].strftime("%Y-%m") for i in tick_positions]
    ax_macd.set_xticks(tick_positions)
    ax_macd.set_xticklabels(tick_labels, rotation=30, ha="right", fontsize=7, color="#aaaaaa")

    fig.suptitle(f"{ticker} — {_label('週線技術分析', 'Weekly Technical Analysis')}",
                 fontsize=13, color="#eeeeee")
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"  圖表已存：{output_path}")

    if show:
        plt.show()

    plt.close(fig)
