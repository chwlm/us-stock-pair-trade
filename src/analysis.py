from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from statsmodels.api import add_constant, OLS
from statsmodels.stats.diagnostic import acorr_ljungbox, het_breuschpagan, normal_ad
from statsmodels.tsa.stattools import adfuller, kpss

from src.data_loader import load_data
from src.utils import config


@st.cache_data(show_spinner=False)
def fetch_price_data(symbols: list[str], start_date: str, end_date: str) -> pd.DataFrame:
    api_key = config.get("webull_api.api_key")
    app_secret = config.get("webull_api.app_secret")

    if not api_key or not app_secret:
        st.info("Webull credentials are not configured. Showing synthetic demo data instead.")
        return create_synthetic_price_data(symbols, start_date, end_date)

    try:
        dataframe = load_data(symbols=symbols, start_date=start_date, end_date=end_date)
    except Exception as exc:
        st.warning(f"Webull data unavailable: {exc}. Falling back to synthetic data for the demo dashboard.")
        return create_synthetic_price_data(symbols, start_date, end_date)

    if dataframe.empty or "close" not in dataframe.columns:
        return create_synthetic_price_data(symbols, start_date, end_date)

    dataframe = dataframe.copy()
    dataframe["date"] = pd.to_datetime(dataframe["date"])
    dataframe = dataframe.sort_values(["date", "symbol"])
    return dataframe


def create_synthetic_price_data(symbols: list[str], start_date: str, end_date: str) -> pd.DataFrame:
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    index = pd.date_range(start=start, end=end, freq="B")
    records = []
    base = {symbol: 100 + idx * 0.7 for idx, symbol in enumerate(symbols)}

    for idx, date in enumerate(index):
        for symbol in symbols:
            drift = idx * 0.18
            season = math.sin(idx / 6) * 2.8
            noise = (idx % 5) * 0.25
            price = base[symbol] + drift + season + noise
            records.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "close": round(price, 4),
                    "open": round(price - 0.6, 4),
                    "high": round(price + 0.8, 4),
                    "low": round(price - 0.8, 4),
                    "volume": int(1000000 + idx * 1500),
                }
            )

    return pd.DataFrame(records)


def prepare_price_matrix(raw_data: pd.DataFrame, asset_a: str, asset_b: str) -> pd.DataFrame:
    matrix = raw_data.pivot_table(index="date", columns="symbol", values="close").sort_index()
    if asset_a not in matrix.columns or asset_b not in matrix.columns:
        raise ValueError("Selected symbols are not available in the loaded price data.")
    return matrix[[asset_a, asset_b]].rename(columns={asset_a: "asset_a", asset_b: "asset_b"})


def fit_pair_ols(price_matrix: pd.DataFrame) -> Dict[str, object]:
    log_df = np.log(price_matrix[["asset_a", "asset_b"]].astype(float))
    model = OLS(log_df["asset_b"], add_constant(log_df["asset_a"])).fit()
    beta = float(model.params["asset_a"])
    alpha = float(model.params["const"])
    spread = log_df["asset_b"] - beta * log_df["asset_a"] - alpha
    residuals = model.resid
    r_squared = float(model.rsquared)

    jb_stat, jb_p = normal_ad(residuals)
    bp_stat, bp_p, _, _ = het_breuschpagan(residuals, add_constant(log_df["asset_a"]))
    lb_df = acorr_ljungbox(residuals, lags=[1], return_df=True)
    lb_p = float(lb_df["lb_pvalue"].iloc[0])

    adf_stat, adf_p, *_ = adfuller(spread.dropna())
    kpss_stat, kpss_p, *_ = kpss(spread.dropna(), regression="c", nlags="auto")

    return {
        "model": model,
        "beta": beta,
        "alpha": alpha,
        "spread": spread,
        "residuals": residuals,
        "assumption_tests": {
            "r_squared": {
                "stat": r_squared,
                "p_value": np.nan,
                "interpretation": "Strong fit" if r_squared >= 0.5 else "Weak fit",
            },
            "normality": {
                "stat": float(jb_stat),
                "p_value": float(jb_p), 
                "interpretation": "Normal" if jb_p >= 0.05 else "Non-normal"
            },
            "heteroskedasticity": {
                "stat": float(bp_stat), 
                "p_value": float(bp_p), 
                "interpretation": "Homoscedastic" if bp_p >= 0.05 else "Heteroskedastic"
            },
            "independence": {
                "stat": float(lb_df["lb_stat"].iloc[0]), 
                "p_value": float(lb_p), 
                "interpretation": "Independent" if lb_p >= 0.05 else "Serially dependent"
            },
        },
        "stationarity": {
            "adf": {"stat": float(adf_stat), "p_value": float(adf_p), "interpretation": "Stationary" if adf_p < 0.05 else "Non-stationary"},
            "kpss": {"stat": float(kpss_stat), "p_value": float(kpss_p), "interpretation": "Stationary" if kpss_p >= 0.05 else "Non-stationary"},
        },
    }


def get_period_window(
    start_date: pd.Timestamp,
    training_months: int,
    validation_months: int,
    test_months: int,
) -> Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    train_end = start_date + pd.DateOffset(months=training_months)
    validation_end = train_end + pd.DateOffset(months=validation_months)
    test_end = validation_end + pd.DateOffset(months=test_months)
    return start_date, train_end, validation_end, test_end

@dataclass
class BacktestMetrics:
    total_trades: int
    pnl: float
    return_pct: float
    max_drawdown: float
    sharpe_ratio: float
    equity_curve: pd.Series


def run_bollinger_backtest(
    spread_series: pd.Series,
    price_a: pd.Series,
    price_b: pd.Series,
    beta: float,
    window: int,
    n_std: float,
    initial_money: float = 1.0,
    evaluation_start: pd.Timestamp | None = None,
) -> Tuple[BacktestMetrics, pd.DataFrame]:

    series = pd.concat([spread_series, price_a.rename("asset_a"), price_b.rename("asset_b")], axis=1)
    series = series.dropna()

    if len(series) <= window:
        raise ValueError("Not enough observations to run the Bollinger-band backtest.")

    hb = series.copy()
    hb["ma"] = hb["spread"].rolling(window=window).mean()
    hb["std"] = hb["spread"].rolling(window=window).std(ddof=1)
    hb["upper"] = hb["ma"] + n_std * hb["std"]
    hb["lower"] = hb["ma"] - n_std * hb["std"]
    hb = hb.dropna().copy()

    cash = initial_money
    position = 0
    total_trades = 0
    a_shares = 0.0
    b_shares = 0.0

    equity_series = []
    signal_points = []
    evaluation_started = evaluation_start is None

    for index, row in hb.iterrows():
        if evaluation_start is not None and index < evaluation_start and not evaluation_started:
            equity_series.append(cash)
            continue

        if not evaluation_started:
            cash = initial_money
            position = 0
            total_trades = 0
            a_shares = 0.0
            b_shares = 0.0
            signal_points = []
            evaluation_started = True

        price_a_value = float(row["asset_a"])
        price_b_value = float(row["asset_b"])
        spread_value = float(row["spread"])

        if position == 0:
            if spread_value > row["upper"]:
                position = 1
                total_trades += 1
                dollar_a = cash / (1 + beta)
                dollar_b = (beta * cash) / (1 + beta)
                
                a_shares = dollar_a / price_a_value
                b_shares = dollar_b / price_b_value
                
                cash = cash - (a_shares * price_a_value) + (b_shares * price_b_value)
                signal_points.append((index, "Buy A / Sell B"))
                
            elif spread_value < row["lower"]:
                position = -1
                total_trades += 1
                
                dollar_a = cash / (1 + beta)
                dollar_b = (beta * cash) / (1 + beta)
                
                a_shares = dollar_a / price_a_value
                b_shares = dollar_b / price_b_value
                
                cash = cash + (a_shares * price_a_value) - (b_shares * price_b_value)
                signal_points.append((index, "Sell A / Buy B"))
                
        else:
            if (position == 1 and spread_value <= row["ma"]) or (position == -1 and spread_value >= row["ma"]):
                if position == 1:
                    cash = cash + (a_shares * price_a_value) - (b_shares * price_b_value)
                elif position == -1:
                    cash = cash - (a_shares * price_a_value) + (b_shares * price_b_value)
                
                position = 0
                a_shares = 0.0
                b_shares = 0.0
                signal_points.append((index, "Exit spread"))

        if position == 1:
            current_equity = cash + (a_shares * price_a_value) - (b_shares * price_b_value)
        elif position == -1:
            current_equity = cash - (a_shares * price_a_value) + (b_shares * price_b_value)
        else:
            current_equity = cash
            
        equity_series.append(current_equity)

    equity_curve = pd.Series(equity_series, index=hb.index)
    daily_returns = equity_curve.pct_change().fillna(0)
    
    std_returns = daily_returns.std(ddof=1)
    if std_returns > 0:
        sharpe_ratio = float((daily_returns.mean() / std_returns) * np.sqrt(252))
    else:
        sharpe_ratio = 0.0
        
    final_equity = float(equity_curve.iloc[-1]) if not equity_curve.empty else initial_money
    pnl = final_equity - initial_money
    return_pct = pnl / initial_money if initial_money else 0.0
    
    peak = equity_curve.cummax()
    drawdown = ((equity_curve - peak) / peak).min() if not equity_curve.empty else 0.0

    metrics = BacktestMetrics(
        total_trades=total_trades,
        pnl=pnl,
        return_pct=return_pct,
        max_drawdown=float(drawdown),
        sharpe_ratio=sharpe_ratio,
        equity_curve=equity_curve,
    )

    result_frame = hb.copy()
    result_frame["signal"] = np.nan
    for date, signal in signal_points:
        result_frame.loc[date, "signal"] = signal
        
    return metrics, result_frame


def show_diagnostics_table(test_results: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    rows = []
    for test_name, values in test_results.items():
        rows.append(
            {
                "Test": test_name,
                "Statistic": values["stat"],
                "p-value": values["p_value"],
                "Interpretation": values["interpretation"],
            }
        )

    diagnostics_df = pd.DataFrame(rows)
    diagnostics_df["Statistic"] = pd.to_numeric(diagnostics_df["Statistic"], errors="coerce")
    diagnostics_df["p-value"] = pd.to_numeric(diagnostics_df["p-value"], errors="coerce")
    return diagnostics_df
