"""
GIGA TRADER - Anti-Overfit Integration
========================================
Main integration function that combines all anti-overfitting measures:
  1. Component Streak Breadth Features
  2. Cross-Asset Features
  3. MAG Market Breadth Features (MAG3/5/6/7/10/15)
  4. Sector Breadth Features (S&P 500 Sectors)
  5. Volatility Regime Features (VXX-based)
  6. Economic Indicator Features (yields, VIX, credit spreads)
  7. Synthetic SPY Universes
"""

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional

from src.phase_08_features_breadth.streak_features import ComponentStreakFeatures
from src.phase_08_features_breadth.cross_asset_features import CrossAssetFeatures
from src.phase_08_features_breadth.mag7_breadth import Mag7BreadthFeatures
from src.phase_08_features_breadth.sector_breadth import SectorBreadthFeatures
from src.phase_08_features_breadth.volatility_regime import VolatilityRegimeFeatures
from src.phase_03_synthetic_data.synthetic_universe import SyntheticSPYGenerator
from src.phase_08_features_breadth.economic_features import EconomicFeatures


def integrate_anti_overfit(
    df_daily: pd.DataFrame,
    spy_1min: pd.DataFrame = None,
    use_synthetic: bool = True,
    use_cross_assets: bool = True,
    use_breadth_streaks: bool = True,
    use_mag_breadth: bool = True,  # MAG3/5/6/7/10/15 features
    use_sector_breadth: bool = True,  # Sector rotation features
    use_vol_regime: bool = True,  # Volatility regime features
    use_economic_features: bool = True,  # Economic indicators (yields, VIX, credit)
    synthetic_weight: float = 0.4,  # Weight for synthetic data (real = 1 - synthetic)
    use_bear_universes: bool = True,  # Bear market synthetic series
    bear_mean_shift_bps: Optional[List[int]] = None,
    bear_vol_amplify_factor: float = 1.5,
    bear_vol_dampen_factor: float = 0.7,
    use_multiscale_bootstrap: bool = True,  # Multi-timescale regime bootstrap
) -> Tuple[pd.DataFrame, Dict]:
    """
    Integrate all anti-overfitting measures.

    Args:
        df_daily: Daily feature dataframe
        spy_1min: 1-minute SPY data (optional)
        use_synthetic: Generate synthetic SPY universes
        use_cross_assets: Add TLT, QQQ, GLD features
        use_breadth_streaks: Add component streak breadth features
        use_mag_breadth: Add MAG3/5/6/7/10/15 market breadth features
        use_sector_breadth: Add sector rotation and breadth features
        use_vol_regime: Add volatility regime features (VXX-based)
        synthetic_weight: Weight for synthetic data (0.4 = 40% synthetic)

    Returns:
        Augmented dataframe and metadata
    """
    print("\n" + "=" * 70)
    print("ANTI-OVERFITTING INTEGRATION")
    print("=" * 70)

    # Convert date column to pd.Timestamp for consistent merging
    # (spy_daily may have datetime.date objects which don't merge with Timestamps)
    df_daily = df_daily.copy()
    try:
        df_daily["date"] = pd.to_datetime(df_daily["date"], errors='coerce')
        # Remove any rows with invalid dates
        df_daily = df_daily.dropna(subset=["date"])
    except Exception as e:
        print(f"[ANTI-OVERFIT] Warning: Date conversion issue: {e}")

    metadata = {}

    # Safely get date range with validation
    try:
        start_date = pd.to_datetime(df_daily["date"].min())
        end_date = pd.to_datetime(df_daily["date"].max())

        # Validate dates are reasonable (not too far in past or future)
        min_valid_date = pd.Timestamp("2000-01-01")
        max_valid_date = pd.Timestamp.now() + pd.Timedelta(days=1)

        if start_date < min_valid_date or end_date > max_valid_date:
            print(f"[ANTI-OVERFIT] Warning: Date range seems invalid: {start_date} to {end_date}")
            print("[ANTI-OVERFIT] Clamping to valid range")
            start_date = max(start_date, min_valid_date)
            end_date = min(end_date, max_valid_date)
    except Exception as e:
        print(f"[ANTI-OVERFIT] Error getting date range: {e}")
        # Default to last 5 years if date extraction fails
        end_date = pd.Timestamp.now()
        start_date = end_date - pd.Timedelta(days=365 * 5)

    # 1. Component Streak Breadth Features
    if use_breadth_streaks:
        streak_features = ComponentStreakFeatures(max_streak=7)
        component_prices = streak_features.download_component_data(start_date, end_date)

        if not component_prices.empty:
            df_daily = streak_features.compute_streak_features(component_prices, df_daily)
            metadata["streak_features"] = True
            metadata["components_tracked"] = len(component_prices.columns)

            # Analyze current breadth
            signal = streak_features.analyze_breadth_signal(df_daily)
            if signal:
                print(f"  Current breadth: {signal.get('interpretation', 'N/A')}")

    # 2. Cross-Asset Features
    if use_cross_assets:
        cross_assets = CrossAssetFeatures()
        cross_data = cross_assets.download_cross_assets(start_date, end_date)

        if not cross_data.empty:
            df_daily = cross_assets.create_cross_asset_features(cross_data, df_daily)
            metadata["cross_assets"] = list(cross_data.columns)

    # 3. MAG Market Breadth Features (MAG3/5/6/7/10/15)
    if use_mag_breadth:
        mag_breadth = Mag7BreadthFeatures()
        mag_data = mag_breadth.download_mag_data(start_date, end_date)

        if not mag_data.empty:
            df_daily = mag_breadth.create_mag_features(df_daily)
            metadata["mag_features"] = True
            metadata["mag_tickers"] = mag_breadth.ALL_MAG_TICKERS
            metadata["mag_groups"] = ["MAG3", "MAG5", "MAG6", "MAG7", "MAG10", "MAG15"]

            # Analyze current MAG leadership
            mag_signal = mag_breadth.analyze_mag_leadership(df_daily)
            if mag_signal:
                print(f"  MAG7 Leadership: {mag_signal.get('interpretation', 'N/A')}")
                print(f"    Bias: {mag_signal.get('bias', 'N/A')}")
                print(f"    Tech Rotation: {mag_signal.get('rotation', 'N/A')}")
                if mag_signal.get('breadth_warning') != 'NONE':
                    print(f"    Breadth Warning: {mag_signal.get('breadth_warning')}")

    # 4. Sector Breadth Features (S&P 500 Sectors)
    if use_sector_breadth:
        sector_breadth = SectorBreadthFeatures()
        sector_data = sector_breadth.download_sector_data(start_date, end_date)

        if not sector_data.empty:
            df_daily = sector_breadth.create_sector_features(df_daily)
            metadata["sector_features"] = True
            metadata["sector_etfs"] = list(sector_breadth.SECTOR_ETFS.keys())

            # Analyze current sector rotation
            sector_signal = sector_breadth.analyze_sector_rotation(df_daily)
            if sector_signal:
                print(f"  Sector Rotation: {sector_signal.get('rotation', 'N/A')}")
                print(f"    Market Phase: {sector_signal.get('market_phase', 'N/A')}")
                print(f"    Risk Sentiment: {sector_signal.get('risk_sentiment', 'N/A')}")

    # 5. Volatility Regime Features (VXX-based)
    if use_vol_regime:
        vol_regime = VolatilityRegimeFeatures()
        vol_data = vol_regime.download_vol_data(start_date, end_date)

        if not vol_data.empty:
            df_daily = vol_regime.create_vol_features(df_daily)
            metadata["vol_regime_features"] = True

            # Analyze current volatility regime
            vol_signal = vol_regime.analyze_vol_regime(df_daily)
            if vol_signal:
                print(f"  Volatility Regime: {vol_signal.get('regime', 'N/A')}")
                print(f"    Market Condition: {vol_signal.get('market_condition', 'N/A')}")
                print(f"    VXX Percentile: {vol_signal.get('vxx_percentile', 0):.1%}")

    # 6. Economic Indicator Features (yields, VIX, credit spreads)
    if use_economic_features:
        econ_features = EconomicFeatures()
        econ_data = econ_features.download_economic_data(start_date, end_date)

        if not econ_data.empty:
            df_daily = econ_features.create_economic_features(df_daily)
            metadata["economic_features"] = True
            metadata["economic_sources"] = list(econ_data.columns)

            # Analyze current conditions
            econ_signal = econ_features.analyze_current_conditions(df_daily)
            if econ_signal:
                if "vix_regime" in econ_signal:
                    print(f"  VIX Regime: {econ_signal['vix_regime']} "
                          f"(level={econ_signal.get('vix_level', 0):.1f})")
                if "yield_curve_signal" in econ_signal:
                    print(f"  Yield Curve: {econ_signal['yield_curve_signal']} "
                          f"(10Y-5Y={econ_signal.get('yield_curve_10_5', 0):.2f})")
                if "credit_signal" in econ_signal:
                    print(f"  Credit: {econ_signal['credit_signal']}")

    # 7. Synthetic SPY Universes (do last since it multiplies data)
    if use_synthetic:
        real_weight = 1 - synthetic_weight
        synth_gen = SyntheticSPYGenerator(
            n_universes=20,
            real_weight=real_weight,
            use_bear_universes=use_bear_universes,
            bear_mean_shift_bps=bear_mean_shift_bps,
            bear_vol_amplify_factor=bear_vol_amplify_factor,
            bear_vol_dampen_factor=bear_vol_dampen_factor,
            use_multiscale_bootstrap=use_multiscale_bootstrap,
        )

        # Reuse component prices if available, otherwise download
        if "component_prices" not in dir() or component_prices.empty:
            component_prices = synth_gen.download_spy_components(start_date, end_date)

        if not component_prices.empty:
            # Generate filter-based universes
            universes = synth_gen.generate_universes(component_prices, df_daily)

            # Generate "SPY minus component" universes (new feature)
            component_universes = synth_gen.generate_spy_minus_component_universes(
                component_prices, df_daily
            )
            universes.extend(component_universes)

            df_daily = synth_gen.create_augmented_dataset(df_daily, universes)
            n_bear = sum(
                1 for u in universes
                if len(u) > 0 and "universe_type" in u.columns
                and str(u["universe_type"].iloc[0]).startswith("bear_")
            )
            metadata["n_universes"] = len(universes)
            metadata["n_bear_universes"] = n_bear
            metadata["real_weight"] = real_weight
            metadata["synthetic_weight"] = synthetic_weight
            metadata["component_modified_universes"] = len(component_universes)

    print(f"\n[ANTI-OVERFIT] Final dataset: {len(df_daily)} samples")
    print(f"  Metadata: {metadata}")

    return df_daily, metadata
