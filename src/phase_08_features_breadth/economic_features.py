"""
GIGA TRADER - Economic Features
=================================
Add features from treasury yields, volatility indices, credit spreads,
and derived economic indicators.

Downloads data via yfinance (no API key needed). Each source produces:
  - Level (raw value or close price)
  - Daily change
  - 5-day change
  - 20-day change
  - Z-score (rolling 60-day standardized)
  - Percentile rank (rolling 252-day)

Derived features from combinations:
  - yield_curve_slope: 10Y - 5Y yield
  - yield_curve_steep: 10Y - 13W yield
  - credit_spread: JNK return - SHY return (5d rolling)
  - real_yield_proxy: TNX z-score - TIP 20d return
  - vix_regime: VIX z-score relative to 20d MA
  - oil_fin_divergence: USO 5d return - XLF 5d return
  - vix_term_ratio: VIX / VXV (contango vs backwardation)
  - gold_equity_signal: GLD 5d return - XLF 5d return (risk-off rotation)
  - bond_equity_rotation: AGG 5d return - XLF 5d return
  - commodity_breadth: DBC 20d return - USO 20d return

Optional FRED integration (requires fredapi + FRED_API_KEY):
  - T10Y2Y: 10Y-2Y yield spread (recession indicator)
  - DFF: Fed Funds Rate
  - BAMLH0A0HYM2: High Yield OAS (credit spread)
  - ICSA: Initial Jobless Claims
  - UMCSENT: Michigan Consumer Sentiment
"""

import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


class EconomicFeatures:
    """
    Download economic data via yfinance and create predictive features.

    Follows the same pattern as CrossAssetFeatures: download → feature engineer → merge.
    """

    # Sources to download via yfinance
    SOURCES = {
        # Volatility complex
        "^VIX": {"desc": "CBOE Volatility Index", "category": "volatility", "prefix": "vix"},
        "^VXV": {"desc": "3-Month VIX (term structure)", "category": "volatility", "prefix": "vxv"},
        # Treasury yields
        "^TNX": {"desc": "10-Year Treasury Yield", "category": "rates", "prefix": "tnx"},
        "^TYX": {"desc": "30-Year Treasury Yield", "category": "rates", "prefix": "tyx"},
        "^FVX": {"desc": "5-Year Treasury Yield", "category": "rates", "prefix": "fvx"},
        "^IRX": {"desc": "13-Week Treasury Bill", "category": "rates", "prefix": "irx"},
        # Fixed income ETFs
        "SHY": {"desc": "1-3 Year Treasury ETF", "category": "bonds", "prefix": "shy"},
        "LQD": {"desc": "Investment Grade Corp Bonds", "category": "bonds", "prefix": "lqd"},
        "JNK": {"desc": "High Yield (Junk) Bonds", "category": "bonds", "prefix": "jnk"},
        "TIP": {"desc": "TIPS (Inflation-Protected)", "category": "bonds", "prefix": "tip"},
        "AGG": {"desc": "US Aggregate Bond ETF", "category": "bonds", "prefix": "agg"},
        # Commodities
        "USO": {"desc": "Oil ETF", "category": "commodities", "prefix": "uso"},
        "DBC": {"desc": "Broad Commodities ETF", "category": "commodities", "prefix": "dbc"},
        "GLD": {"desc": "Gold ETF", "category": "commodities", "prefix": "gld"},
        # Sector / equity
        "XLF": {"desc": "Financials Sector ETF", "category": "sector", "prefix": "xlf"},
    }

    # Optional FRED series (downloaded only if fredapi + FRED_API_KEY are available)
    FRED_SERIES = {
        "T10Y2Y": {"desc": "10Y-2Y Yield Spread (recession indicator)", "prefix": "fred_t10y2y"},
        "DFF": {"desc": "Fed Funds Rate", "prefix": "fred_dff"},
        "BAMLH0A0HYM2": {"desc": "High Yield OAS (credit spread)", "prefix": "fred_hy_oas"},
        "ICSA": {"desc": "Initial Jobless Claims (weekly)", "prefix": "fred_claims"},
        "UMCSENT": {"desc": "Michigan Consumer Sentiment (monthly)", "prefix": "fred_sentiment"},
    }

    def __init__(self, sources: Optional[List[str]] = None):
        """
        Args:
            sources: List of yfinance symbols to use. Defaults to all SOURCES.
        """
        if sources is not None:
            self.sources = {k: v for k, v in self.SOURCES.items() if k in sources}
        else:
            self.sources = dict(self.SOURCES)

        self._prices: Optional[pd.DataFrame] = None
        self._fred_data: pd.DataFrame = pd.DataFrame()

    def download_economic_data(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> pd.DataFrame:
        """Download all economic data via yfinance."""
        import yfinance as yf

        print("\n[ECONOMIC] Downloading economic indicator data via yfinance...")

        symbols = list(self.sources.keys())
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
                    print(f"  {symbol}: NO DATA")
                    continue

                # Handle multi-level columns from yfinance
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.get_level_values(0)

                close = data["Close"].copy()
                close.index = pd.to_datetime(close.index).tz_localize(None)
                results[symbol] = close
                print(f"  {symbol}: {len(close)} days - {self.sources[symbol]['desc']}")
            except Exception as e:
                print(f"  {symbol}: ERROR - {e}")

        if not results:
            print("  [WARN] No economic data downloaded")
            return pd.DataFrame()

        self._prices = pd.DataFrame(results)
        print(f"  yfinance: {len(self._prices)} days, {len(self._prices.columns)} sources")

        # ── Optional FRED downloads ──
        self._fred_data = self._download_fred(start_date, end_date)

        print(f"  Total: {len(self._prices)} days, {len(self._prices.columns)} yfinance "
              f"+ {len(self._fred_data.columns) if not self._fred_data.empty else 0} FRED sources")
        return self._prices

    def _download_fred(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Download FRED series if fredapi + FRED_API_KEY are available."""
        import os
        api_key = os.environ.get("FRED_API_KEY", "")
        if not api_key:
            return pd.DataFrame()

        try:
            from fredapi import Fred
        except ImportError:
            return pd.DataFrame()

        print("  [FRED] Downloading FRED economic series...")
        fred = Fred(api_key=api_key)
        results = {}
        for series_id, meta in self.FRED_SERIES.items():
            try:
                data = fred.get_series(
                    series_id,
                    observation_start=start_date.strftime("%Y-%m-%d"),
                    observation_end=end_date.strftime("%Y-%m-%d"),
                )
                if data is None or len(data) == 0:
                    continue
                data.index = pd.to_datetime(data.index).tz_localize(None)
                results[series_id] = data
                print(f"    {series_id}: {len(data)} obs - {meta['desc']}")
            except Exception as e:
                print(f"    {series_id}: ERROR - {e}")

        return pd.DataFrame(results) if results else pd.DataFrame()

    def create_economic_features(
        self,
        spy_daily: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Create features from downloaded economic data and merge into spy_daily.

        For each source, creates 6 features:
          - {prefix}_level: Raw level (z-scored 252d)
          - {prefix}_chg_1d: Daily change
          - {prefix}_chg_5d: 5-day change
          - {prefix}_chg_20d: 20-day change
          - {prefix}_zscore: 60-day z-score
          - {prefix}_pctile: 252-day percentile rank

        Plus derived combination features.
        """
        if self._prices is None or self._prices.empty:
            return spy_daily

        print("\n[ECONOMIC] Engineering features...")

        features = spy_daily.copy()
        n_features_added = 0

        for symbol, meta in self.sources.items():
            if symbol not in self._prices.columns:
                continue

            prices = self._prices[symbol].dropna()
            if len(prices) < 60:
                continue

            prefix = meta["prefix"]
            returns = prices.pct_change()

            asset_features = pd.DataFrame(index=prices.index)

            # 1. Level z-scored (252d)
            asset_features[f"econ_{prefix}_level_z"] = (
                (prices - prices.rolling(252, min_periods=60).mean())
                / (prices.rolling(252, min_periods=60).std() + 1e-10)
            )

            # 2. Daily change
            asset_features[f"econ_{prefix}_chg_1d"] = returns

            # 3. 5-day change
            asset_features[f"econ_{prefix}_chg_5d"] = returns.rolling(5).sum()

            # 4. 20-day change
            asset_features[f"econ_{prefix}_chg_20d"] = returns.rolling(20).sum()

            # 5. 60-day z-score
            asset_features[f"econ_{prefix}_zscore"] = (
                (prices - prices.rolling(60).mean()) / (prices.rolling(60).std() + 1e-10)
            )

            # 6. 252-day percentile rank
            asset_features[f"econ_{prefix}_pctile"] = prices.rolling(252, min_periods=60).apply(
                lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
            )

            # Convert index to date for merge
            asset_features["date"] = pd.to_datetime(asset_features.index.date)

            features = features.merge(
                asset_features.reset_index(drop=True),
                on="date",
                how="left",
            )
            n_features_added += 6

        # ── FRED features (forward-filled to daily) ──
        fred_count = self._add_fred_features(features)
        n_features_added += fred_count

        # ── Derived Combination Features ──
        derived_count = self._add_derived_features(features)
        n_features_added += derived_count

        # Fill NaN with 0 for economic features
        econ_cols = [c for c in features.columns if c.startswith("econ_")]
        features[econ_cols] = features[econ_cols].fillna(0)

        print(f"  Added {n_features_added} economic features ({len(econ_cols)} columns)")
        return features

    def _add_fred_features(self, features: pd.DataFrame) -> int:
        """Add features from FRED data (forward-filled to daily). Returns count."""
        if not hasattr(self, "_fred_data") or self._fred_data is None or self._fred_data.empty:
            return 0

        count = 0
        # Build a daily date index from features
        feature_dates = pd.to_datetime(features["date"])

        for series_id, meta in self.FRED_SERIES.items():
            if series_id not in self._fred_data.columns:
                continue

            raw = self._fred_data[series_id].dropna()
            if len(raw) < 10:
                continue

            prefix = meta["prefix"]

            # Forward-fill to daily frequency
            daily = raw.reindex(pd.date_range(raw.index.min(), raw.index.max(), freq="B")).ffill()
            if len(daily) < 60:
                continue

            # Features: level z-score, change, percentile
            feat = pd.DataFrame(index=daily.index)
            feat[f"econ_{prefix}_zscore"] = (
                (daily - daily.rolling(60, min_periods=20).mean())
                / (daily.rolling(60, min_periods=20).std() + 1e-10)
            )
            feat[f"econ_{prefix}_chg_5d"] = daily.pct_change(5)
            feat[f"econ_{prefix}_chg_20d"] = daily.pct_change(20)

            feat["date"] = pd.to_datetime(feat.index.date)
            features_merged = features.merge(feat.reset_index(drop=True), on="date", how="left")
            for col in [f"econ_{prefix}_zscore", f"econ_{prefix}_chg_5d", f"econ_{prefix}_chg_20d"]:
                features[col] = features_merged[col]
            count += 3

        return count

    def _add_derived_features(self, features: pd.DataFrame) -> int:
        """Add derived combination features. Returns count of features added."""
        count = 0
        prices = self._prices
        if prices is None:
            return 0

        # Yield curve slope: 10Y - 5Y
        if "^TNX" in prices.columns and "^FVX" in prices.columns:
            slope = prices["^TNX"] - prices["^FVX"]
            slope_z = (slope - slope.rolling(60).mean()) / (slope.rolling(60).std() + 1e-10)
            feat = pd.DataFrame({"econ_yield_curve_10_5": slope_z})
            feat["date"] = pd.to_datetime(prices.index.date)
            features_merged = features.merge(feat.reset_index(drop=True), on="date", how="left")
            features["econ_yield_curve_10_5"] = features_merged["econ_yield_curve_10_5"]
            count += 1

        # Yield curve steepness: 10Y - 13W
        if "^TNX" in prices.columns and "^IRX" in prices.columns:
            steep = prices["^TNX"] - prices["^IRX"]
            steep_z = (steep - steep.rolling(60).mean()) / (steep.rolling(60).std() + 1e-10)
            feat = pd.DataFrame({"econ_yield_curve_10_13w": steep_z})
            feat["date"] = pd.to_datetime(prices.index.date)
            features_merged = features.merge(feat.reset_index(drop=True), on="date", how="left")
            features["econ_yield_curve_10_13w"] = features_merged["econ_yield_curve_10_13w"]
            count += 1

        # Credit spread proxy: JNK 5d return - SHY 5d return
        if "JNK" in prices.columns and "SHY" in prices.columns:
            jnk_5d = prices["JNK"].pct_change().rolling(5).sum()
            shy_5d = prices["SHY"].pct_change().rolling(5).sum()
            spread = jnk_5d - shy_5d
            feat = pd.DataFrame({"econ_credit_spread": spread})
            feat["date"] = pd.to_datetime(prices.index.date)
            features_merged = features.merge(feat.reset_index(drop=True), on="date", how="left")
            features["econ_credit_spread"] = features_merged["econ_credit_spread"]
            count += 1

        # Real yield proxy: TNX z-score - TIP 20d return
        if "^TNX" in prices.columns and "TIP" in prices.columns:
            tnx_z = (prices["^TNX"] - prices["^TNX"].rolling(60).mean()) / (
                prices["^TNX"].rolling(60).std() + 1e-10
            )
            tip_ret = prices["TIP"].pct_change(20)
            real_yield = tnx_z - tip_ret * 100  # Scale TIP return
            feat = pd.DataFrame({"econ_real_yield_proxy": real_yield})
            feat["date"] = pd.to_datetime(prices.index.date)
            features_merged = features.merge(feat.reset_index(drop=True), on="date", how="left")
            features["econ_real_yield_proxy"] = features_merged["econ_real_yield_proxy"]
            count += 1

        # VIX regime: z-score of VIX vs 20d MA
        if "^VIX" in prices.columns:
            vix = prices["^VIX"]
            vix_regime = (vix - vix.rolling(20).mean()) / (vix.rolling(20).std() + 1e-10)
            feat = pd.DataFrame({"econ_vix_regime": vix_regime})
            feat["date"] = pd.to_datetime(prices.index.date)
            features_merged = features.merge(feat.reset_index(drop=True), on="date", how="left")
            features["econ_vix_regime"] = features_merged["econ_vix_regime"]
            count += 1

        # Oil-Financials divergence
        if "USO" in prices.columns and "XLF" in prices.columns:
            uso_5d = prices["USO"].pct_change(5)
            xlf_5d = prices["XLF"].pct_change(5)
            div = uso_5d - xlf_5d
            feat = pd.DataFrame({"econ_oil_fin_diverge": div})
            feat["date"] = pd.to_datetime(prices.index.date)
            features_merged = features.merge(feat.reset_index(drop=True), on="date", how="left")
            features["econ_oil_fin_diverge"] = features_merged["econ_oil_fin_diverge"]
            count += 1

        # VIX term structure: VIX / VXV ratio (contango = bullish, backwardation = bearish)
        if "^VIX" in prices.columns and "^VXV" in prices.columns:
            vix = prices["^VIX"]
            vxv = prices["^VXV"]
            # Ratio: < 1.0 = contango (normal), > 1.0 = backwardation (fear)
            ratio = vix / (vxv + 1e-10)
            ratio_z = (ratio - ratio.rolling(60).mean()) / (ratio.rolling(60).std() + 1e-10)
            feat = pd.DataFrame({
                "econ_vix_term_ratio": ratio,
                "econ_vix_term_zscore": ratio_z,
            })
            feat["date"] = pd.to_datetime(prices.index.date)
            features_merged = features.merge(feat.reset_index(drop=True), on="date", how="left")
            features["econ_vix_term_ratio"] = features_merged["econ_vix_term_ratio"]
            features["econ_vix_term_zscore"] = features_merged["econ_vix_term_zscore"]
            count += 2

        # Gold-to-equity signal: GLD return vs XLF return (risk-off detector)
        if "GLD" in prices.columns and "XLF" in prices.columns:
            gld_5d = prices["GLD"].pct_change().rolling(5).sum()
            xlf_5d = prices["XLF"].pct_change().rolling(5).sum()
            gold_eq = gld_5d - xlf_5d  # Positive = risk-off rotation
            feat = pd.DataFrame({"econ_gold_equity_signal": gold_eq})
            feat["date"] = pd.to_datetime(prices.index.date)
            features_merged = features.merge(feat.reset_index(drop=True), on="date", how="left")
            features["econ_gold_equity_signal"] = features_merged["econ_gold_equity_signal"]
            count += 1

        # Bond-equity rotation: AGG return vs XLF return
        if "AGG" in prices.columns and "XLF" in prices.columns:
            agg_5d = prices["AGG"].pct_change().rolling(5).sum()
            xlf_5d = prices["XLF"].pct_change().rolling(5).sum()
            bond_eq = agg_5d - xlf_5d
            feat = pd.DataFrame({"econ_bond_equity_rotation": bond_eq})
            feat["date"] = pd.to_datetime(prices.index.date)
            features_merged = features.merge(feat.reset_index(drop=True), on="date", how="left")
            features["econ_bond_equity_rotation"] = features_merged["econ_bond_equity_rotation"]
            count += 1

        # Commodity breadth: DBC momentum vs oil-only (diversification signal)
        if "DBC" in prices.columns and "USO" in prices.columns:
            dbc_20d = prices["DBC"].pct_change(20)
            uso_20d = prices["USO"].pct_change(20)
            comm_breadth = dbc_20d - uso_20d  # Positive = broad commodity strength
            feat = pd.DataFrame({"econ_commodity_breadth": comm_breadth})
            feat["date"] = pd.to_datetime(prices.index.date)
            features_merged = features.merge(feat.reset_index(drop=True), on="date", how="left")
            features["econ_commodity_breadth"] = features_merged["econ_commodity_breadth"]
            count += 1

        return count

    def analyze_current_conditions(self, spy_daily: pd.DataFrame) -> Optional[dict]:
        """Analyze current economic conditions for dashboard display."""
        if self._prices is None or self._prices.empty:
            return None

        conditions = {}

        # VIX level
        if "^VIX" in self._prices.columns:
            vix = self._prices["^VIX"].dropna()
            if len(vix) > 0:
                current_vix = vix.iloc[-1]
                vix_pctile = (vix.iloc[-252:] <= current_vix).mean() if len(vix) >= 252 else 0.5
                conditions["vix_level"] = float(current_vix)
                conditions["vix_percentile"] = float(vix_pctile)
                if vix_pctile > 0.8:
                    conditions["vix_regime"] = "HIGH_VOL"
                elif vix_pctile < 0.2:
                    conditions["vix_regime"] = "LOW_VOL"
                else:
                    conditions["vix_regime"] = "NORMAL"

        # Yield curve
        if "^TNX" in self._prices.columns and "^FVX" in self._prices.columns:
            tnx = self._prices["^TNX"].dropna()
            fvx = self._prices["^FVX"].dropna()
            if len(tnx) > 0 and len(fvx) > 0:
                slope = float(tnx.iloc[-1] - fvx.iloc[-1])
                conditions["yield_curve_10_5"] = slope
                conditions["yield_curve_signal"] = "INVERTED" if slope < 0 else "NORMAL"

        # Credit conditions
        if "JNK" in self._prices.columns and "SHY" in self._prices.columns:
            jnk_ret5 = self._prices["JNK"].pct_change().rolling(5).sum().dropna()
            shy_ret5 = self._prices["SHY"].pct_change().rolling(5).sum().dropna()
            if len(jnk_ret5) > 0 and len(shy_ret5) > 0:
                spread = float(jnk_ret5.iloc[-1] - shy_ret5.iloc[-1])
                conditions["credit_spread_5d"] = spread
                conditions["credit_signal"] = "RISK_ON" if spread > 0 else "RISK_OFF"

        # VIX term structure
        if "^VIX" in self._prices.columns and "^VXV" in self._prices.columns:
            vix = self._prices["^VIX"].dropna()
            vxv = self._prices["^VXV"].dropna()
            if len(vix) > 0 and len(vxv) > 0:
                ratio = float(vix.iloc[-1] / (vxv.iloc[-1] + 1e-10))
                conditions["vix_term_ratio"] = ratio
                if ratio > 1.05:
                    conditions["vix_term_structure"] = "BACKWARDATION"  # Fear
                elif ratio < 0.90:
                    conditions["vix_term_structure"] = "STEEP_CONTANGO"  # Complacency
                else:
                    conditions["vix_term_structure"] = "CONTANGO"  # Normal

        return conditions if conditions else None
