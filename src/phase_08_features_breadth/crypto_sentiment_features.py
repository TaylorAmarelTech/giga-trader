"""
GIGA TRADER - Crypto Fear & Greed Features
=============================================
Download and engineer features from the Alternative.me Crypto Fear & Greed Index.

The Crypto Fear & Greed Index (0-100) measures crypto market sentiment.
It serves as a risk-on/risk-off proxy for equity markets because:
  - Crypto sentiment often leads equity risk appetite shifts
  - Extreme crypto fear often precedes broader risk-off events
  - Crypto greed extremes correlate with speculative excess

Data source: Alternative.me API (free, no API key required).
API endpoint: https://api.alternative.me/fng/

Features generated (prefix: crypto_):
  - crypto_fg_index: Raw index value (0-100)
  - crypto_fg_zscore: 60-day z-score
  - crypto_fg_chg_5d: 5-day change
  - crypto_fg_regime: Categorical (0=extreme_fear to 4=extreme_greed)
  - crypto_risk_proxy: Binary risk-on proxy (1 if greed, 0 if fear)
"""

import logging
import warnings
from datetime import datetime, timedelta
from typing import Dict, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

logger = logging.getLogger("CRYPTO_SENTIMENT")


class CryptoSentimentFeatures:
    """
    Download crypto fear/greed data from Alternative.me and create features.

    Pattern: download → compute → merge (same as EconomicFeatures).
    """

    def __init__(self):
        self.data: pd.DataFrame = pd.DataFrame()

    def download_crypto_data(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> pd.DataFrame:
        """
        Download Crypto Fear & Greed Index historical data.

        Alternative.me provides up to ~1 year of historical data for free.
        Returns empty DataFrame on failure.
        """
        print("\n[CRYPTO] Downloading Crypto Fear & Greed Index data...")

        try:
            import requests
        except ImportError:
            print("  [WARN] requests package not available")
            return pd.DataFrame()

        try:
            # Request enough days to cover the date range
            days_needed = (end_date - start_date).days + 30
            days_needed = min(days_needed, 365 * 2)  # API limit ~2 years

            url = f"https://api.alternative.me/fng/?limit={days_needed}&format=json"
            headers = {"User-Agent": "GigaTrader/1.0 (research)"}
            resp = requests.get(url, headers=headers, timeout=30)

            if resp.status_code != 200:
                logger.info(f"  Alternative.me API returned status {resp.status_code}")
                return pd.DataFrame()

            data = resp.json()
            entries = data.get("data", [])

            if not entries:
                print("  [WARN] No data from Alternative.me API")
                return pd.DataFrame()

            records = []
            for entry in entries:
                try:
                    timestamp = int(entry.get("timestamp", 0))
                    score = int(entry.get("value", 0))
                    dt = datetime.fromtimestamp(timestamp)
                    records.append({
                        "date": pd.Timestamp(dt.date()),
                        "crypto_score": float(score),
                    })
                except (ValueError, TypeError, OSError):
                    continue

            if not records:
                return pd.DataFrame()

            df = pd.DataFrame(records)
            df = df.drop_duplicates(subset="date", keep="last")
            df = df.sort_values("date").reset_index(drop=True)

            # Filter to requested date range
            mask = (df["date"] >= pd.to_datetime(start_date)) & (
                df["date"] <= pd.to_datetime(end_date)
            )
            df = df[mask].reset_index(drop=True)

            self.data = df
            print(f"  [CRYPTO] {len(df)} days of Crypto Fear & Greed data loaded")
            return df

        except Exception as e:
            logger.warning(f"  Alternative.me API failed: {e}")
            return pd.DataFrame()

    def create_crypto_features(
        self,
        spy_daily: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Create Crypto Fear & Greed features and merge into spy_daily.

        Produces 5 features with crypto_ prefix.
        Returns original spy_daily unchanged if no data available.
        """
        if self.data.empty:
            return spy_daily

        print("\n[CRYPTO] Engineering crypto sentiment features...")

        features = spy_daily.copy()
        crypto = self.data.copy()
        crypto["date"] = pd.to_datetime(crypto["date"])

        # Ensure numeric
        crypto["crypto_score"] = pd.to_numeric(crypto["crypto_score"], errors="coerce")
        crypto = crypto.dropna(subset=["crypto_score"])

        if crypto.empty:
            return spy_daily

        crypto = crypto.sort_values("date").reset_index(drop=True)

        # 1. Raw index
        crypto["crypto_fg_index"] = crypto["crypto_score"]

        # 2. 60-day z-score
        crypto["crypto_fg_zscore"] = (
            (crypto["crypto_score"] - crypto["crypto_score"].rolling(60, min_periods=10).mean())
            / (crypto["crypto_score"].rolling(60, min_periods=10).std() + 1e-10)
        )

        # 3. 5-day change
        crypto["crypto_fg_chg_5d"] = crypto["crypto_score"].diff(5)

        # 4. Regime classification
        def _classify_regime(score):
            if score < 20:
                return 0  # extreme_fear
            elif score < 40:
                return 1  # fear
            elif score <= 60:
                return 2  # neutral
            elif score <= 80:
                return 3  # greed
            else:
                return 4  # extreme_greed

        crypto["crypto_fg_regime"] = crypto["crypto_score"].apply(_classify_regime)

        # 5. Risk-on proxy (simplified: greed = risk-on, fear = risk-off)
        crypto["crypto_risk_proxy"] = (crypto["crypto_score"] > 50).astype(int)

        # Select feature columns for merge
        crypto_feature_cols = [
            "crypto_fg_index", "crypto_fg_zscore", "crypto_fg_chg_5d",
            "crypto_fg_regime", "crypto_risk_proxy",
        ]

        merge_df = crypto[["date"] + crypto_feature_cols].copy()
        features = features.merge(merge_df, on="date", how="left")

        # Fill NaN with 0
        for col in crypto_feature_cols:
            if col in features.columns:
                features[col] = features[col].fillna(0)

        print(f"  [CRYPTO] Added {len(crypto_feature_cols)} crypto sentiment features")
        return features

    def analyze_current_crypto(self, spy_daily: pd.DataFrame) -> Optional[dict]:
        """Analyze current crypto sentiment for dashboard display."""
        if self.data.empty:
            return None

        crypto = self.data.sort_values("date")
        if crypto.empty:
            return None

        latest = crypto.iloc[-1]
        score = float(latest["crypto_score"])

        regimes = {0: "EXTREME_FEAR", 1: "FEAR", 2: "NEUTRAL", 3: "GREED", 4: "EXTREME_GREED"}
        regime_val = 0 if score < 20 else (1 if score < 40 else (2 if score <= 60 else (3 if score <= 80 else 4)))

        return {
            "crypto_fg_score": score,
            "crypto_regime": regimes[regime_val],
            "is_risk_on": score > 50,
            "date": str(latest["date"]),
        }
