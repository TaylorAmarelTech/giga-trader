"""
GIGA TRADER - Sentiment Features
==================================
Derive market sentiment signals from freely available data.

No paid API keys required. All features are derived from:
  - VIX data (already downloaded by EconomicFeatures)
  - Put/call ratio proxies (VIX vs realized vol, term structure)
  - Options-implied fear/greed indicators
  - Cross-asset risk appetite signals

Optional Alpha Vantage news sentiment (if ALPHA_VANTAGE_API_KEY is set).

Features generated (prefix: sent_):
  VIX-derived sentiment (always available):
    - sent_fear_greed: VIX vs 60d realized vol ratio (fear gauge)
    - sent_fear_greed_z: Z-score of fear_greed
    - sent_vix_mean_revert: VIX distance from 252d mean (reversion signal)
    - sent_vix_acceleration: VIX 5d change minus 20d change (momentum shift)
    - sent_risk_appetite: JNK-SHY spread momentum (credit risk appetite)
    - sent_risk_appetite_z: Z-score of risk appetite
    - sent_flight_to_safety: GLD+TIP momentum vs SPY momentum
    - sent_equity_put_call_proxy: VIX/realized_vol ratio as P/C proxy

  Cross-asset sentiment (always available):
    - sent_safe_haven_flow: Aggregate safe haven demand (GLD + TIP + SHY)
    - sent_risk_on_flow: Aggregate risk-on demand (JNK + XLF + USO)
    - sent_risk_rotation: risk_on - safe_haven (regime indicator)
    - sent_risk_rotation_z: Z-score of rotation

  Optional news sentiment (requires ALPHA_VANTAGE_API_KEY):
    - sent_news_score: Aggregated headline sentiment
    - sent_news_volume: Article count (activity)
    - sent_news_dispersion: Sentiment dispersion (disagreement)
"""

import logging
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from src.core.feature_base import FeatureModuleBase

warnings.filterwarnings("ignore")

logger = logging.getLogger("SENTIMENT_FEATURES")


class SentimentFeatures(FeatureModuleBase):
    """
    Derive market sentiment features from VIX, cross-asset flows,
    and optional news APIs.

    Pattern: download → compute → merge (same as EconomicFeatures).
    """
    FEATURE_NAMES = ["sent_news_score", "sent_news_volume", "sent_news_dispersion"]


    # yfinance symbols needed for sentiment computation
    # (many are already downloaded by EconomicFeatures — we reuse if available)
    REQUIRED_SOURCES = {
        "^VIX": "CBOE Volatility Index",
        "SPY": "S&P 500 ETF",
        "JNK": "High Yield Bonds",
        "SHY": "Short-Term Treasury",
        "GLD": "Gold",
        "TIP": "TIPS",
        "XLF": "Financials",
        "USO": "Oil",
    }

    def __init__(self):
        self._prices: Optional[pd.DataFrame] = None

    def download_sentiment_data(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> pd.DataFrame:
        """Download required price data for sentiment computation."""
        import yfinance as yf

        print("\n[SENTIMENT] Downloading data for sentiment features...")

        symbols = list(self.REQUIRED_SOURCES.keys())
        results = {}

        for symbol in symbols:
            try:
                data = yf.download(
                    symbol,
                    start=start_date.strftime("%Y-%m-%d"),
                    end=(end_date + timedelta(days=1)).strftime("%Y-%m-%d"),
                    auto_adjust=True,
                    progress=False,
                )
                if data.empty:
                    continue

                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.get_level_values(0)

                close = data["Close"].copy()
                close.index = pd.to_datetime(close.index).tz_localize(None)
                results[symbol] = close
            except Exception as e:
                logger.warning(f"  {symbol}: ERROR - {e}")

        if not results:
            print("  [WARN] No sentiment data downloaded")
            return pd.DataFrame()

        self._prices = pd.DataFrame(results)
        print(f"  [SENTIMENT] {len(self._prices)} days, {len(self._prices.columns)} sources")
        return self._prices

    def create_sentiment_features(
        self,
        spy_daily: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Create sentiment features and merge into spy_daily.

        Works with pre-loaded prices (from download_sentiment_data)
        or gracefully returns original df if no data available.
        """
        if self._prices is None or self._prices.empty:
            return spy_daily

        print("\n[SENTIMENT] Engineering sentiment features...")

        features = spy_daily.copy()
        prices = self._prices
        n_added = 0

        # ── VIX-Derived Sentiment ──────────────────────────────────────────

        if "^VIX" in prices.columns and "SPY" in prices.columns:
            vix = prices["^VIX"]
            spy_close = prices["SPY"]
            spy_ret = spy_close.pct_change()

            # Realized volatility (20-day annualized)
            realized_vol = spy_ret.rolling(20).std() * np.sqrt(252) * 100

            # Fear/Greed: VIX vs realized vol
            # VIX > realized = market pricing more fear than warranted
            fear_greed = vix / (realized_vol + 1e-10)
            fear_greed_z = (
                (fear_greed - fear_greed.rolling(60).mean())
                / (fear_greed.rolling(60).std() + 1e-10)
            )

            # VIX mean reversion signal
            vix_252_mean = vix.rolling(252, min_periods=60).mean()
            vix_mean_revert = (vix - vix_252_mean) / (vix_252_mean + 1e-10)

            # VIX acceleration (momentum shift)
            vix_chg_5 = vix.pct_change(5)
            vix_chg_20 = vix.pct_change(20)
            vix_accel = vix_chg_5 - vix_chg_20

            # Equity put/call proxy: VIX / realized vol
            put_call_proxy = vix / (realized_vol + 1e-10)

            feat = pd.DataFrame({
                "sent_fear_greed": fear_greed,
                "sent_fear_greed_z": fear_greed_z,
                "sent_vix_mean_revert": vix_mean_revert,
                "sent_vix_acceleration": vix_accel,
                "sent_equity_put_call_proxy": put_call_proxy,
            })
            feat["date"] = pd.to_datetime(feat.index.date)
            merged = features.merge(feat.reset_index(drop=True), on="date", how="left")
            for col in feat.columns:
                if col != "date":
                    features[col] = merged[col]
            n_added += 5

        # ── Credit Risk Appetite ───────────────────────────────────────────

        if "JNK" in prices.columns and "SHY" in prices.columns:
            jnk_mom = prices["JNK"].pct_change().rolling(5).sum()
            shy_mom = prices["SHY"].pct_change().rolling(5).sum()
            risk_appetite = jnk_mom - shy_mom
            risk_appetite_z = (
                (risk_appetite - risk_appetite.rolling(60).mean())
                / (risk_appetite.rolling(60).std() + 1e-10)
            )

            feat = pd.DataFrame({
                "sent_risk_appetite": risk_appetite,
                "sent_risk_appetite_z": risk_appetite_z,
            })
            feat["date"] = pd.to_datetime(feat.index.date)
            merged = features.merge(feat.reset_index(drop=True), on="date", how="left")
            for col in feat.columns:
                if col != "date":
                    features[col] = merged[col]
            n_added += 2

        # ── Flight to Safety ───────────────────────────────────────────────

        safe_haven_cols = [c for c in ["GLD", "TIP", "SHY"] if c in prices.columns]
        risk_on_cols = [c for c in ["JNK", "XLF", "USO"] if c in prices.columns]

        if len(safe_haven_cols) >= 2 and len(risk_on_cols) >= 2 and "SPY" in prices.columns:
            # Safe haven aggregate momentum
            safe_mom = pd.DataFrame({
                c: prices[c].pct_change().rolling(10).sum()
                for c in safe_haven_cols
            }).mean(axis=1)

            # Risk-on aggregate momentum
            risk_mom = pd.DataFrame({
                c: prices[c].pct_change().rolling(10).sum()
                for c in risk_on_cols
            }).mean(axis=1)

            # SPY momentum for relative comparison
            spy_mom = prices["SPY"].pct_change().rolling(10).sum()

            # Flight to safety: safe haven strength vs SPY
            flight = safe_mom - spy_mom

            # Risk rotation: risk-on vs safe-haven
            rotation = risk_mom - safe_mom
            rotation_z = (
                (rotation - rotation.rolling(60).mean())
                / (rotation.rolling(60).std() + 1e-10)
            )

            feat = pd.DataFrame({
                "sent_flight_to_safety": flight,
                "sent_safe_haven_flow": safe_mom,
                "sent_risk_on_flow": risk_mom,
                "sent_risk_rotation": rotation,
                "sent_risk_rotation_z": rotation_z,
            })
            feat["date"] = pd.to_datetime(feat.index.date)
            merged = features.merge(feat.reset_index(drop=True), on="date", how="left")
            for col in feat.columns:
                if col != "date":
                    features[col] = merged[col]
            n_added += 5

        # ── Optional: Alpha Vantage News Sentiment ─────────────────────────

        news_count = self._add_news_sentiment(features)
        n_added += news_count

        # Fill NaN with 0
        sent_cols = [c for c in features.columns if c.startswith("sent_")]
        features[sent_cols] = features[sent_cols].fillna(0)

        print(f"  [SENTIMENT] Added {n_added} sentiment features ({len(sent_cols)} columns)")
        return features

    def _add_news_sentiment(self, features: pd.DataFrame) -> int:
        """
        Add news sentiment from Alpha Vantage (optional).

        Returns count of features added (0 if API key not available).
        """
        import os
        try:
            from dotenv import load_dotenv
            from pathlib import Path
            env_path = Path(__file__).resolve().parent.parent.parent / ".env"
            if env_path.is_file():
                load_dotenv(env_path)
        except ImportError:
            pass

        api_key = os.environ.get("ALPHA_VANTAGE_API_KEY", "")
        if not api_key:
            return 0

        try:
            import requests
        except ImportError:
            return 0

        print("  [NEWS] Fetching Alpha Vantage news sentiment...")
        try:
            url = (
                f"https://www.alphavantage.co/query?"
                f"function=NEWS_SENTIMENT&tickers=SPY"
                f"&limit=200&apikey={api_key}"
            )
            resp = requests.get(url, timeout=30)
            data = resp.json()

            if "feed" not in data:
                logger.warning(f"  [NEWS] No feed in response: {list(data.keys())}")
                return 0

            # Parse articles
            articles = data["feed"]
            records = []
            for article in articles:
                pub_time = article.get("time_published", "")
                if len(pub_time) >= 8:
                    try:
                        dt = datetime.strptime(pub_time[:8], "%Y%m%d")
                    except ValueError:
                        continue
                else:
                    continue

                # Find SPY-specific sentiment
                score = float(article.get("overall_sentiment_score", 0))
                for ticker_data in article.get("ticker_sentiment", []):
                    if ticker_data.get("ticker") == "SPY":
                        score = float(ticker_data.get("ticker_sentiment_score", score))
                        break

                records.append({"date": dt.date(), "score": score})

            if not records:
                return 0

            news_df = pd.DataFrame(records)
            daily_news = news_df.groupby("date").agg(
                sent_news_score=("score", "mean"),
                sent_news_volume=("score", "count"),
                sent_news_dispersion=("score", "std"),
            ).reset_index()
            daily_news["date"] = pd.to_datetime(daily_news["date"])
            daily_news["sent_news_dispersion"] = daily_news["sent_news_dispersion"].fillna(0)

            merged = features.merge(daily_news, on="date", how="left")
            for col in ["sent_news_score", "sent_news_volume", "sent_news_dispersion"]:
                features[col] = merged[col]

            print(f"    [NEWS] Added 3 news sentiment features from {len(records)} articles")
            return 3

        except Exception as e:
            logger.warning(f"  [NEWS] Error fetching news: {e}")
            return 0

    def analyze_current_sentiment(self, spy_daily: pd.DataFrame) -> Optional[dict]:
        """Analyze current sentiment conditions for dashboard display."""
        if self._prices is None or self._prices.empty:
            return None

        conditions = {}
        prices = self._prices

        # Fear/Greed level
        if "^VIX" in prices.columns and "SPY" in prices.columns:
            vix = prices["^VIX"].dropna()
            spy_ret = prices["SPY"].pct_change().dropna()
            realized_vol = spy_ret.rolling(20).std().iloc[-1] * np.sqrt(252) * 100

            if len(vix) > 0 and realized_vol > 0:
                fg_ratio = float(vix.iloc[-1] / realized_vol)
                conditions["fear_greed_ratio"] = fg_ratio
                if fg_ratio > 1.3:
                    conditions["sentiment_regime"] = "FEAR"
                elif fg_ratio < 0.8:
                    conditions["sentiment_regime"] = "GREED"
                else:
                    conditions["sentiment_regime"] = "NEUTRAL"

        # Risk rotation
        safe_cols = [c for c in ["GLD", "TIP", "SHY"] if c in prices.columns]
        risk_cols = [c for c in ["JNK", "XLF", "USO"] if c in prices.columns]

        if safe_cols and risk_cols:
            safe_mom = np.mean([
                prices[c].pct_change().rolling(10).sum().iloc[-1]
                for c in safe_cols if len(prices[c].dropna()) > 10
            ])
            risk_mom = np.mean([
                prices[c].pct_change().rolling(10).sum().iloc[-1]
                for c in risk_cols if len(prices[c].dropna()) > 10
            ])
            rotation = risk_mom - safe_mom
            conditions["risk_rotation"] = float(rotation)
            conditions["risk_appetite"] = "RISK_ON" if rotation > 0.005 else (
                "RISK_OFF" if rotation < -0.005 else "NEUTRAL"
            )

        return conditions if conditions else None
