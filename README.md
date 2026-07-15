# US Stock Pair Trade Dashboard

A Streamlit-based pair-trading dashboard for analyzing two U.S. equities, fitting an OLS-based spread model, and running a Bollinger-band backtest across training, validation, and test windows.

## What this project does

This app lets you:

- choose a pair of assets
- pick a start date and period lengths for training, validation, and test windows
- load price data for the selected assets
- fit a log-price OLS relationship between the two symbols
- inspect model diagnostics and stationarity tests
- run a Bollinger-band backtest on the validation and test windows
- review the resulting performance metrics and Plotly charts

## Main workflow

1. Select asset A and asset B.
2. Set the start date and the training / validation / test period lengths.
3. Press the Run pipeline button.
4. The app:
   - loads historical data
   - builds the price matrix
   - fits the training-period OLS relationship
   - displays the training diagnostics and OLS fit plot
   - uses the fitted parameters for validation and test backtesting
   - visualizes the result with Plotly charts and summary metrics

## Project structure

- `src/main.py` – Streamlit entrypoint and dashboard UI
- `src/analysis.py` – data preparation, OLS fitting, diagnostics, and Bollinger backtest logic
- `src/visuals.py` – Plotly chart builders for price, spread, and OLS relation plots
- `src/data_loader.py` – Webull API data-loading wrapper
- `src/utils.py` – application configuration access
- `conf/token.txt` – local token file used by the Webull client when credentials are available

## Installation

Create and activate a virtual environment, then install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running the app

From the project root:

```bash
streamlit run src/main.py
```

If Streamlit is not available in the current environment, you can also run it with:

```bash
python -m streamlit run src/main.py
```

## Data source behavior

The dashboard uses the Webull API loader when valid credentials are configured.

If the Webull credentials are missing or the API call fails, the app automatically falls back to synthetic demo data so the dashboard can still be used for development and demonstration.

## Notes on the methodology

- The model uses log prices for the pair regression.
- The spread is calculated from the fitted OLS coefficients.
- The backtest uses rolling Bollinger bands with a warm-up window so the initial target period has enough history to compute indicator values.
- The diagnostics include:
  - OLS R-squared
  - normality diagnostics
  - heteroskedasticity diagnostics
  - independence diagnostics
  - stationarity diagnostics

## Requirements

The project dependencies are listed in `requirements.txt` and include:

- `streamlit`
- `plotly`
- `pandas`
- `numpy`
- `statsmodels`
- `webull-openapi-python-sdk`

## Important caveat

The Webull API integration may require valid credentials and network access. If the live loader is unavailable, the application will continue in fallback mode using synthetic price data.
