import pandas as pd
import logging

logger = logging.getLogger(__name__)


def add_moving_averages(df: pd.DataFrame, windows: list = None) -> pd.DataFrame:
    if windows is None:
        windows = [5, 20, 60]
    for w in windows:
        df[f"MA{w}"] = df["Close"].rolling(w).mean()
    return df


def add_volume_ma(df: pd.DataFrame, windows: list = None) -> pd.DataFrame:
    if windows is None:
        windows = [20, 60]
    for w in windows:
        df[f"Volume_MA{w}"] = df["Volume"].rolling(w).mean()
    return df


def add_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    if len(df) < slow * 2:
        logger.warning("資料不足以計算 MACD（需要至少 %d 根）", slow * 2)
        df["DIF"] = float("nan")
        df["DEA"] = float("nan")
        df["MACD_hist"] = float("nan")
        return df

    ema_fast = df["Close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["Close"].ewm(span=slow, adjust=False).mean()
    df["DIF"] = ema_fast - ema_slow
    df["DEA"] = df["DIF"].ewm(span=signal, adjust=False).mean()
    df["MACD_hist"] = (df["DIF"] - df["DEA"]) * 2
    return df


def compute_all(df: pd.DataFrame) -> pd.DataFrame:
    if df["Close"].isna().all():
        raise ValueError("收盤價全為 NaN，資料無效")
    df = add_moving_averages(df)
    df = add_volume_ma(df)
    df = add_macd(df)
    return df
