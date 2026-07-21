from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def build_ols_fit_plot(
    price_matrix: pd.DataFrame,
    asset_a: str,
    asset_b: str,
    beta: float,
    alpha: float,
    period_label: str,
) -> go.Figure:
    asset_a_col = asset_a if asset_a in price_matrix.columns else "asset_a"
    asset_b_col = asset_b if asset_b in price_matrix.columns else "asset_b"

    log_df = np.log(price_matrix[[asset_a_col, asset_b_col]].astype(float))
    x_sorted = np.sort(log_df[asset_a_col].dropna())
    fitted_y = alpha + beta * x_sorted

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=log_df[asset_a_col],
            y=log_df[asset_b_col],
            mode="markers",
            name="Log price points",
            marker=dict(color="#1f77b4", opacity=0.6, size=6),
        )
    )
    figure.add_trace(
        go.Scatter(
            x=x_sorted,
            y=fitted_y,
            mode="lines",
            name="OLS fitted line",
            line=dict(color="#d62728", width=3),
        )
    )

    figure.update_layout(
        title=f"{period_label} log-price OLS fit: ln({asset_b}) vs ln({asset_a})",
        template="plotly_white",
        xaxis_title=f"ln({asset_a})",
        yaxis_title=f"ln({asset_b})",
        hovermode="closest",
        height=420,
    )
    return figure


def build_period_plot(
    price_matrix: pd.DataFrame,
    spread_series: pd.Series,
    asset_a: str,
    asset_b: str,
    period_label: str,
    signal_frame: pd.DataFrame | None = None,
) -> go.Figure:
    figure = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.3, 0.25, 0.25, 0.2],
        subplot_titles=(f"{asset_a} price", f"{asset_b} price", "Spread", "Drawdown"),
    )

    figure.add_trace(go.Scatter(x=price_matrix.index, y=price_matrix["asset_a"], mode="lines", name=asset_a, line=dict(color="#1f77b4")), row=1, col=1)
    figure.add_trace(go.Scatter(x=price_matrix.index, y=price_matrix["asset_b"], mode="lines", name=asset_b, line=dict(color="#ff7f0e")), row=2, col=1)
    figure.add_trace(go.Scatter(x=spread_series.index, y=spread_series.values, mode="lines", name="spread", line=dict(color="#2ca02c")), row=3, col=1)

    # drawdown plot: use drawdown from signal_frame if available, otherwise skip
    if signal_frame is not None and "drawdown" in signal_frame.columns:
        dd = signal_frame["drawdown"].loc[spread_series.index.min(): spread_series.index.max()]
        figure.add_trace(
            go.Scatter(
                x=dd.index,
                y=dd.values,
                mode="lines",
                name="drawdown",
                line=dict(color="#d62728"),
                fill="tozeroy",
                fillcolor="rgba(214,39,40,0.15)",
            ),
            row=4,
            col=1,
        )

    if signal_frame is not None and not signal_frame.empty:
        if {"ma", "upper", "lower"}.issubset(signal_frame.columns):
            figure.add_trace(
                go.Scatter(
                    x=signal_frame.index,
                    y=signal_frame["ma"],
                    mode="lines",
                    name="MA",
                    line=dict(color="#6c757d", width=2),
                ),
                row=3,
                col=1,
            )
            figure.add_trace(
                go.Scatter(
                    x=signal_frame.index,
                    y=signal_frame["upper"],
                    mode="lines",
                    name="Upper band",
                    line=dict(color="#f39c12", width=1.5, dash="dash"),
                ),
                row=3,
                col=1,
            )
            figure.add_trace(
                go.Scatter(
                    x=signal_frame.index,
                    y=signal_frame["lower"],
                    mode="lines",
                    name="Lower band",
                    line=dict(color="#f39c12", width=1.5, dash="dash"),
                ),
                row=3,
                col=1,
            )

        signal_plot_frame = signal_frame.dropna(subset=["signal"])
        signal_lookup = {
            "Sell A / Buy B": dict(symbol="triangle-up", color="green", size=15),
            "Buy A / Sell B": dict(symbol="triangle-down", color="red", size=15),
            "Exit spread": dict(symbol="diamond", color="orange", size=12),
        }
        for signal_name, signal_style in signal_lookup.items():
            signal_points = signal_plot_frame[signal_plot_frame["signal"] == signal_name]
            if not signal_points.empty:
                figure.add_trace(
                    go.Scatter(
                        x=signal_points.index,
                        y=signal_points["spread"],
                        mode="markers",
                        marker=dict(symbol=signal_style["symbol"], color=signal_style["color"], size=signal_style["size"]),
                        name=signal_name,
                    ),
                    row=3,
                    col=1,
                )

                if "asset_a" in signal_points.columns:
                    figure.add_trace(
                        go.Scatter(
                            x=signal_points.index,
                            y=signal_points["asset_a"],
                            mode="markers",
                            marker=dict(symbol=signal_style["symbol"], color=signal_style["color"], size=signal_style["size"]),
                            name=f"{signal_name} (A)",
                            showlegend=False,
                        ),
                        row=1,
                        col=1,
                    )

                if "asset_b" in signal_points.columns:
                    figure.add_trace(
                        go.Scatter(
                            x=signal_points.index,
                            y=signal_points["asset_b"],
                            mode="markers",
                            marker=dict(symbol=signal_style["symbol"], color=signal_style["color"], size=signal_style["size"]),
                            name=f"{signal_name} (B)",
                            showlegend=False,
                        ),
                        row=2,
                        col=1,
                    )

    figure.update_layout(
        title=f"{period_label} period analysis",
        template="plotly_white",
        hovermode="x unified",
        height=900,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    figure.update_xaxes(rangeslider_visible=False)
    return figure
