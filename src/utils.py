from typing import Any, Dict

from dotenv import load_dotenv
import os

load_dotenv()


class Config:
    def __init__(self):
        self.config = self._default_config()

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key."""
        current: Any = self.config
        for part in key.split("."):
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return default
        return current

    def _default_config(self) -> Dict[str, Any]:
        """Return default configuration."""
        return {
            "webull_api": {
                "api_key": os.getenv("WEBULL_API_KEY"),
                "app_secret": os.getenv("WEBULL_APP_SECRET"),
            },
            "asset_universe": [
                "MSFT",
                "AAPL",
                "GOOGL",
                "META",
                "NVDA",
                "AMD",
                "AVGO",
                "QCOM",
                "INTC",
                "MU",
                "TSM",
                "ASML",
                "AMAT",
                "LRCX",
                "PANW",
                "CRWD",
                "PLTR",
                "AMZN",
                "SNOW",
                "ORCL",
                "JPM",
                "BAC",
                "WFC",
                "C",
                "GS",
                "MS",
                "BLK",
                "BRK B",
                "AXP",
                "V",
                "MA",
                "PYPL",
                "XYZ",
                "SCHW",
                "IBKR",
                "XOM",
                "CVX",
                "COP",
                "SLB",
                "EOG",
                "CAT",
                "DE",
                "GE",
                "HON",
                "LMT",
                "RTX",
                "UNP",
                "UPS",
                "FDX",
                "FCX",
                "WMT",
                "COST",
                "PG",
                "KO",
                "PEP",
                "MCD",
                "SBUX",
                "HD",
                "LOW",
                "DIS",
                "NFLX",
                "TSLA",
                "LLY",
                "UNH",
                "JNJ",
                "MRK",
                "ABBV",
                "PFE",
                "TMO",
                "NKE",
                "NEE",
                "DUK",
                "SO",
                "AEP",
                "EXC",
                "SRE",
                "D",
                # "CEG",
                "VST",
                "XEL",
                "VZ",
                "T",
                "TMUS",
                "CMCSA",
                "CHTR",
                "AMT",
                "CCI",
                "SBAC",
                "EQIX",
                "DLR",
                "PLD",
                "SPG",
                "O",
                "MAR",
                "HLT",
                "DAL",
                "UAL",
                "AAL",
                "BKNG",
                "ABNB",
            ],
            "chunk_size": 15,
        }


config = Config()
