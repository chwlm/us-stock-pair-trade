from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from src.analysis import (
    fetch_price_data,
    fit_pair_ols,
    get_period_window,
    prepare_price_matrix,
    run_bollinger_backtest,
    show_diagnostics_table,
)
from src.utils import config
from src.visuals import build_ols_fit_plot, build_period_plot


def main() -> None:
    st.set_page_config(page_title="Pair Trading Dashboard", layout="wide")
    st.title("Pair Trading Dashboard")

    with st.form("pipeline_form"):
        asset_universe = config.get("asset_universe", [])
        asset_a = st.selectbox("Select asset A", asset_universe, index=asset_universe.index("AAPL") if "AAPL" in asset_universe else 0)
        asset_b = st.selectbox("Select asset B", asset_universe, index=asset_universe.index("MSFT") if "MSFT" in asset_universe else 1)

        start_date = st.date_input("Start date", value=pd.Timestamp("2025-01-01").date())
        training_months = st.number_input("Training period (months)", min_value=1, max_value=60, value=12)
        validation_months = st.number_input("Validation period (months)", min_value=1, max_value=60, value=3)
        test_months = st.number_input("Test period (months)", min_value=1, max_value=60, value=3)

        val_window = st.slider("Validation window size", min_value=5, max_value=60, value=20, key="validation_window")
        val_std = st.slider("Validation number of std for Bollinger band", min_value=1.0, max_value=4.0, value=2.0, step=0.1, key="validation_std")

        submitted = st.form_submit_button("Run pipeline", use_container_width=True)

    if not submitted:
        st.info("Choose the assets and periods, then press Run pipeline to execute the analysis.")
        return

    start_ts = pd.Timestamp(start_date)
    train_end, validation_end, test_end = get_period_window(start_ts, training_months, validation_months, test_months)[1:]
    all_end = test_end

    raw_data = fetch_price_data([asset_a, asset_b], start_date=start_ts.strftime("%Y-%m-%d"), end_date=all_end.strftime("%Y-%m-%d"))
    price_matrix = prepare_price_matrix(raw_data, asset_a, asset_b)

    train_slice = price_matrix.loc[(price_matrix.index >= start_ts) & (price_matrix.index < train_end)]
    validation_slice = price_matrix.loc[(price_matrix.index >= train_end) & (price_matrix.index < validation_end)]
    test_slice = price_matrix.loc[(price_matrix.index >= validation_end) & (price_matrix.index < test_end)]

    if train_slice.empty or validation_slice.empty or test_slice.empty:
        st.warning("Selected period windows are too short for a meaningful signal test. Please widen the date range or reduce the period sizes.")
        return

    train_regression = fit_pair_ols(train_slice)
    beta = train_regression["beta"]
    alpha = train_regression["alpha"]

    st.subheader("Training period")
    train_spread = train_regression["spread"]
    st.plotly_chart(build_period_plot(train_slice, train_spread, asset_a, asset_b, "Training"), use_container_width=True)
    st.metric("Training R-squared", round(train_regression["assumption_tests"]["r_squared"]["stat"], 4))
    st.subheader("Model diagnostics")
    st.plotly_chart(build_ols_fit_plot(train_slice, asset_a, asset_b, beta, alpha, "Training"), use_container_width=True)
    st.dataframe(show_diagnostics_table(train_regression["assumption_tests"]))
    st.subheader("Stationarity diagnostics")
    st.dataframe(show_diagnostics_table(train_regression["stationarity"]))

    st.subheader("Validation period")
    validation_first_idx = price_matrix.index.get_loc(validation_slice.index.min())
    validation_warmup_idx = max(0, validation_first_idx - val_window)
    validation_warmup_start = price_matrix.index[validation_warmup_idx]
    validation_backtest_slice = price_matrix.loc[(price_matrix.index >= validation_warmup_start) & (price_matrix.index < validation_end)]
    validation_spread = (
        np.log(validation_backtest_slice["asset_b"]) - beta * np.log(validation_backtest_slice["asset_a"]) - alpha
    )
    val_with_prices = pd.concat([validation_backtest_slice, validation_spread.rename("spread")], axis=1)
    val_with_prices = val_with_prices.dropna().sort_index()
    
    if len(val_with_prices) <= val_window:
        st.warning(
            f"Validation period contains only {len(val_with_prices)} observations, which is too few for a Bollinger window of {val_window}. "
            "Please widen the validation window or reduce the selected window size."
        )
        return

    validation_metrics, validation_signal_frame = run_bollinger_backtest(
        spread_series=val_with_prices["spread"],
        price_a=val_with_prices["asset_a"],
        price_b=val_with_prices["asset_b"],
        beta=beta,
        window=val_window,
        n_std=val_std,
        evaluation_start=validation_slice.index.min(),
    )

    validation_plot = build_period_plot(
        validation_slice,
        validation_spread.loc[validation_slice.index].rename("spread"),
        asset_a,
        asset_b,
        "Validation",
        signal_frame=validation_signal_frame.loc[validation_signal_frame.index >= validation_slice.index.min()],
    )
    st.plotly_chart(validation_plot, use_container_width=True)

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Trade", validation_metrics.total_trades)
    col2.metric("PnL", round(validation_metrics.pnl, 4))
    col3.metric("Return", f"{validation_metrics.return_pct:.2%}")
    col4.metric("Max Drawdown", f"{validation_metrics.max_drawdown:.2%}")
    col5.metric("Sharpe Ratio", f"{validation_metrics.sharpe_ratio:.2f}")

    st.subheader("Validation backtest results by window and std")
    best_rows = []
    window_candidates = range(5, 61, 5)
    std_candidates = [x / 2 for x in range(2, 9)]
    for current_window in window_candidates:
        for current_std in std_candidates:
            if len(val_with_prices) <= current_window:
                continue
            try:
                metrics, _ = run_bollinger_backtest(
                    spread_series=val_with_prices["spread"],
                    price_a=val_with_prices["asset_a"],
                    price_b=val_with_prices["asset_b"],
                    beta=beta,
                    window=current_window,
                    n_std=current_std,
                    evaluation_start=validation_slice.index.min(),
                )
            except ValueError:
                continue
            best_rows.append(
                {
                    "Window": current_window,
                    "Std": current_std,
                    "Total Trade": metrics.total_trades,
                    "PnL": round(metrics.pnl, 4),
                    "Return": metrics.return_pct,
                    "Max Drawdown": metrics.max_drawdown,
                    "Sharpe Ratio": metrics.sharpe_ratio,
                }
            )

    best_df = pd.DataFrame(best_rows)
    best_df["Return"] = pd.to_numeric(best_df["Return"], errors="coerce")
    best_df["Max Drawdown"] = pd.to_numeric(best_df["Max Drawdown"], errors="coerce")
    best_df["Sharpe Ratio"] = pd.to_numeric(best_df["Sharpe Ratio"], errors="coerce")
    best_df = best_df.sort_values(["Sharpe Ratio"], ascending=False)
    st.dataframe(best_df)

    st.subheader("Test period")
    test_first_idx = price_matrix.index.get_loc(test_slice.index.min())
    test_warmup_idx = max(0, test_first_idx - val_window)
    test_warmup_start = price_matrix.index[test_warmup_idx]
    test_backtest_slice = price_matrix.loc[(price_matrix.index >= test_warmup_start) & (price_matrix.index < test_end)]
    test_spread = (
        np.log(test_backtest_slice["asset_b"]) - beta * np.log(test_backtest_slice["asset_a"]) - alpha
    )
    if len(test_backtest_slice) <= val_window:
        st.warning(
            f"Test period contains only {len(test_slice)} observations, which is too few for a Bollinger window of {val_window}."
        )
        return

    test_metrics, test_signal_frame = run_bollinger_backtest(
        spread_series=test_spread.rename("spread"),
        price_a=test_backtest_slice["asset_a"],
        price_b=test_backtest_slice["asset_b"],
        beta=beta,
        window=val_window,
        n_std=val_std,
        evaluation_start=test_slice.index.min(),
    )

    test_plot = build_period_plot(
        test_slice,
        test_spread.loc[test_slice.index].rename("spread"),
        asset_a,
        asset_b,
        "Test",
        signal_frame=test_signal_frame.loc[test_signal_frame.index >= test_slice.index.min()],
    )
    st.plotly_chart(test_plot, use_container_width=True)
    tc1, tc2, tc3, tc4, tc5 = st.columns(5)
    tc1.metric("Total Trade", test_metrics.total_trades)
    tc2.metric("PnL", round(test_metrics.pnl, 4))
    tc3.metric("Return", f"{test_metrics.return_pct:.2%}")
    tc4.metric("Max Drawdown", f"{test_metrics.max_drawdown:.2%}")
    tc5.metric("Sharpe Ratio", f"{test_metrics.sharpe_ratio:.2f}")


if __name__ == "__main__":
    main()
