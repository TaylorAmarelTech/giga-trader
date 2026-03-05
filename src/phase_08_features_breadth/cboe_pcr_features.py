"""
Wave N1: CBOE Put/Call Ratio Direct Features — Actual CBOE equity P/C data.

Downloads the official CBOE equity put/call ratio CSV (free, no API key).
Enhances existing pcr_ proxy features with actual institutional data.

Data source chain:
  L1: CBOE CSV (cdn.cboe.com) — free, no key, daily updated
  L2: VIX-based proxy (realized vol from SPY close)
  L3: Zero-fill

Prefix: cboe_
Default: ON
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from src.core.feature_base import FeatureModuleBase

logger = logging.getLogger(__name__)

# CBOE equity P/C ratio CSV URL
CBOE_EQUITY_PCR_URL = (
    "https://cdn.cboe.com/api/global/us_options/market_statistics/daily/"
)
CBOE_TOTAL_PCR_URL = (
    "https://cdn.cboe.com/resources/options/volume_and_call_put_ratios/"
)

MIN_DATA_POINTS = 10


class CBOEPutCallFeatures(FeatureModuleBase):
    """CBOE direct put/call ratio features from official CSV data."""
    FEATURE_NAMES = ["cboe_equity_pcr", "cboe_total_pcr", "cboe_pcr_5d_ma", "cboe_pcr_20d_ma", "cboe_pcr_zscore", "cboe_pcr_trend", "cboe_extreme_signal", "cboe_regime"]


    REQUIRED_COLS = {"close"}

    def __init__(self):
        self._data: Optional[pd.DataFrame] = None
        self._data_source: str = "none"

    # ------------------------------------------------------------------
    def download_cboe_data(
        self, start: str, end: str
    ) -> Optional[pd.DataFrame]:
        """Download CBOE equity put/call ratio CSV.

        Tries multiple CBOE CSV endpoints. Falls back to None on failure.
        """
        import requests

        urls_to_try = [
            f"{CBOE_EQUITY_PCR_URL}equity_pcr.csv",
            f"{CBOE_TOTAL_PCR_URL}totalpc.csv",
            f"{CBOE_TOTAL_PCR_URL}equitypc.csv",
        ]

        for url in urls_to_try:
            try:
                resp = requests.get(url, timeout=15)
                if resp.status_code != 200:
                    continue

                # Try parsing the CSV
                from io import StringIO
                raw = pd.read_csv(StringIO(resp.text))

                if raw.empty or len(raw) < MIN_DATA_POINTS:
                    continue

                # Normalize column names
                raw.columns = [c.strip().lower().replace(" ", "_") for c in raw.columns]

                # Look for a date column
                date_col = None
                for candidate in ["trade_date", "date", "tradedate"]:
                    if candidate in raw.columns:
                        date_col = candidate
                        break

                if date_col is None:
                    # First column might be the date
                    date_col = raw.columns[0]

                raw["date"] = pd.to_datetime(raw[date_col], errors="coerce")
                raw = raw.dropna(subset=["date"])

                if len(raw) < MIN_DATA_POINTS:
                    continue

                # Look for P/C ratio columns
                pcr_col = None
                for candidate in ["p/c_ratio", "put/call_ratio", "pc_ratio",
                                  "pcr", "ratio", "p_c_ratio"]:
                    if candidate in raw.columns:
                        pcr_col = candidate
                        break

                if pcr_col is None:
                    # Try to compute from put_volume / call_volume
                    put_col = None
                    call_col = None
                    for c in raw.columns:
                        if "put" in c and "vol" in c:
                            put_col = c
                        if "call" in c and "vol" in c:
                            call_col = c
                    if put_col and call_col:
                        raw["pcr"] = (
                            pd.to_numeric(raw[put_col], errors="coerce")
                            / pd.to_numeric(raw[call_col], errors="coerce").replace(0, np.nan)
                        )
                        pcr_col = "pcr"
                    else:
                        # Use second numeric column as ratio
                        numeric_cols = raw.select_dtypes(include=[np.number]).columns.tolist()
                        if numeric_cols:
                            pcr_col = numeric_cols[0]

                if pcr_col is None:
                    continue

                result = pd.DataFrame({
                    "date": raw["date"],
                    "pcr": pd.to_numeric(raw[pcr_col], errors="coerce"),
                })
                result = result.dropna(subset=["pcr"])
                result = result.sort_values("date").reset_index(drop=True)

                # Filter to date range
                start_ts = pd.to_datetime(start)
                end_ts = pd.to_datetime(end)
                result = result[
                    (result["date"] >= start_ts) & (result["date"] <= end_ts)
                ]

                if len(result) < MIN_DATA_POINTS:
                    continue

                self._data = result
                self._data_source = "cboe_csv"
                logger.info(
                    f"[CBOE_PCR] Downloaded {len(result)} days from {url.split('/')[-1]}"
                )
                return result

            except Exception as e:
                logger.debug(f"[CBOE_PCR] URL {url} failed: {e}")
                continue

        logger.info("[CBOE_PCR] All CBOE CSV sources failed — will use proxy")
        self._data_source = "proxy"
        return None

    # ------------------------------------------------------------------
    def create_cboe_pcr_features(
        self, df_daily: pd.DataFrame
    ) -> pd.DataFrame:
        """Create CBOE P/C ratio features. Routes to full or proxy."""
        df = df_daily.copy()

        if self._data is not None and len(self._data) >= MIN_DATA_POINTS:
            df = self._create_full_features(df)
        else:
            df = self._create_proxy_features(df)

        # NaN/Inf cleanup
        for col in self._all_feature_names():
            if col in df.columns:
                df[col] = df[col].fillna(0.0).replace([np.inf, -np.inf], 0.0)
            else:
                df[col] = 0.0

        return df

    # ------------------------------------------------------------------
    def _create_full_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features from actual CBOE data."""
        pcr_data = self._data.copy()
        pcr_data["date"] = pd.to_datetime(pcr_data["date"]).dt.normalize()

        # Merge on date
        df["_merge_date"] = pd.to_datetime(df["date"]).dt.normalize()
        df = df.merge(
            pcr_data[["date", "pcr"]].rename(columns={"date": "_merge_date"}),
            on="_merge_date",
            how="left",
        )
        # Forward-fill for non-trading days
        df["pcr"] = df["pcr"].ffill()
        df.drop(columns=["_merge_date"], inplace=True, errors="ignore")

        # Features
        df["cboe_equity_pcr"] = df["pcr"].fillna(0.85)
        df["cboe_total_pcr"] = df["pcr"].fillna(0.85)  # Same source for now
        df["cboe_pcr_5d_ma"] = df["cboe_equity_pcr"].rolling(5, min_periods=1).mean()
        df["cboe_pcr_20d_ma"] = df["cboe_equity_pcr"].rolling(20, min_periods=1).mean()

        roll_mean = df["cboe_equity_pcr"].rolling(60, min_periods=10).mean()
        roll_std = df["cboe_equity_pcr"].rolling(60, min_periods=10).std().replace(0, np.nan)
        df["cboe_pcr_zscore"] = ((df["cboe_equity_pcr"] - roll_mean) / roll_std).fillna(0.0)

        df["cboe_pcr_trend"] = df["cboe_equity_pcr"].diff(5).fillna(0.0)

        # Contrarian extremes: PCR > 1.2 → bearish extremes (bullish contrarian)
        df["cboe_extreme_signal"] = 0.0
        df.loc[df["cboe_equity_pcr"] > 1.2, "cboe_extreme_signal"] = 1.0
        df.loc[df["cboe_equity_pcr"] < 0.5, "cboe_extreme_signal"] = -1.0

        # Regime
        df["cboe_regime"] = 0.0
        df.loc[df["cboe_pcr_zscore"] > 1.0, "cboe_regime"] = 1.0   # Fear (contrarian bullish)
        df.loc[df["cboe_pcr_zscore"] < -1.0, "cboe_regime"] = -1.0  # Greed (contrarian bearish)

        df.drop(columns=["pcr"], inplace=True, errors="ignore")
        return df

    # ------------------------------------------------------------------
    def _create_proxy_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Proxy: use realized volatility as P/C ratio proxy."""
        if "close" in df.columns:
            ret = df["close"].pct_change().fillna(0.0)
            rvol = ret.rolling(20, min_periods=5).std() * np.sqrt(252)

            # Map realized vol to PCR range (~0.5 to ~1.5)
            df["cboe_equity_pcr"] = 0.5 + rvol.fillna(0.15) * 3.0
            df["cboe_total_pcr"] = df["cboe_equity_pcr"]
        else:
            df["cboe_equity_pcr"] = 0.85
            df["cboe_total_pcr"] = 0.85

        df["cboe_pcr_5d_ma"] = df["cboe_equity_pcr"].rolling(5, min_periods=1).mean()
        df["cboe_pcr_20d_ma"] = df["cboe_equity_pcr"].rolling(20, min_periods=1).mean()

        roll_mean = df["cboe_equity_pcr"].rolling(60, min_periods=10).mean()
        roll_std = df["cboe_equity_pcr"].rolling(60, min_periods=10).std().replace(0, np.nan)
        df["cboe_pcr_zscore"] = ((df["cboe_equity_pcr"] - roll_mean) / roll_std).fillna(0.0)

        df["cboe_pcr_trend"] = df["cboe_equity_pcr"].diff(5).fillna(0.0)
        df["cboe_extreme_signal"] = 0.0
        df["cboe_regime"] = 0.0

        return df

    # ------------------------------------------------------------------
    def analyze_current_cboe(
        self, df_daily: pd.DataFrame
    ) -> Optional[Dict]:
        """Analyze current CBOE P/C ratio conditions."""
        if df_daily.empty:
            return None

        last = df_daily.iloc[-1]
        pcr = float(last.get("cboe_equity_pcr", 0.0))
        zscore = float(last.get("cboe_pcr_zscore", 0.0))

        if zscore > 1.5:
            regime = "EXTREME_FEAR"
        elif zscore > 0.5:
            regime = "FEAR"
        elif zscore < -1.5:
            regime = "EXTREME_GREED"
        elif zscore < -0.5:
            regime = "GREED"
        else:
            regime = "NEUTRAL"

        return {
            "regime": regime,
            "pcr": pcr,
            "zscore": zscore,
            "source": self._data_source,
        }

    # ------------------------------------------------------------------
    @staticmethod
    def _all_feature_names() -> List[str]:
        """Return all feature column names."""
        return [
            "cboe_equity_pcr",
            "cboe_total_pcr",
            "cboe_pcr_5d_ma",
            "cboe_pcr_20d_ma",
            "cboe_pcr_zscore",
            "cboe_pcr_trend",
            "cboe_extreme_signal",
            "cboe_regime",
        ]
