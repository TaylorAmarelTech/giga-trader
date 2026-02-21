"""
GIGA TRADER - Data Source Analysis Script
==========================================
Downloads freely available financial data sources and evaluates their
predictive value for SPY direction forecasting.

Sources tested:
  - yfinance: ^VIX, ^TNX, ^TYX, ^FVX, ^IRX, SHY, LQD, JNK, TIP, USO, XLF
  - FRED (optional, needs fredapi + FRED_API_KEY): T10Y2Y, DFF, BAMLH0A0HYM2, etc.

For each source, computes:
  - Correlation with SPY next-day return
  - Information Coefficient (rank correlation with 1/5/20 day forward returns)
  - Granger causality F-statistic
  - Data coverage (% of trading days with non-NaN)

Usage:
    python scripts/analyze_data_sources.py
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

# ─── yfinance Sources ─────────────────────────────────────────────────────────

YFINANCE_SOURCES = {
    # Volatility complex
    "^VIX": {"desc": "CBOE Volatility Index", "category": "volatility"},
    "^VXV": {"desc": "3-Month VIX (term structure)", "category": "volatility"},
    # Treasury yields
    "^TNX": {"desc": "10-Year Treasury Yield", "category": "rates"},
    "^TYX": {"desc": "30-Year Treasury Yield", "category": "rates"},
    "^FVX": {"desc": "5-Year Treasury Yield", "category": "rates"},
    "^IRX": {"desc": "13-Week Treasury Bill", "category": "rates"},
    # Fixed income ETFs
    "SHY": {"desc": "1-3 Year Treasury ETF", "category": "bonds"},
    "LQD": {"desc": "Investment Grade Corp Bonds", "category": "bonds"},
    "JNK": {"desc": "High Yield (Junk) Bonds", "category": "bonds"},
    "TIP": {"desc": "TIPS (Inflation-Protected)", "category": "bonds"},
    "AGG": {"desc": "US Aggregate Bond ETF", "category": "bonds"},
    # Commodities
    "USO": {"desc": "Oil ETF", "category": "commodities"},
    "DBC": {"desc": "Broad Commodities ETF", "category": "commodities"},
    "GLD": {"desc": "Gold ETF", "category": "commodities"},
    # Sector
    "XLF": {"desc": "Financials Sector ETF", "category": "sector"},
}

# FRED sources (optional)
FRED_SOURCES = {
    "T10Y2Y": {"desc": "10Y-2Y Yield Spread (recession indicator)", "category": "rates"},
    "DFF": {"desc": "Fed Funds Rate", "category": "rates"},
    "BAMLH0A0HYM2": {"desc": "High Yield OAS (credit spread)", "category": "credit"},
    "TEDRATE": {"desc": "TED Spread (interbank risk)", "category": "credit"},
    "UMCSENT": {"desc": "U of Michigan Consumer Sentiment", "category": "sentiment"},
    "ICSA": {"desc": "Initial Jobless Claims", "category": "employment"},
    "UNRATE": {"desc": "Unemployment Rate", "category": "employment"},
}


def download_spy_returns(start: str = "2014-01-01") -> pd.Series:
    """Download SPY daily returns via yfinance."""
    import yfinance as yf

    print("[SPY] Downloading SPY daily data...")
    spy = yf.download("SPY", start=start, auto_adjust=True, progress=False)
    if spy.empty:
        raise RuntimeError("Failed to download SPY data")

    # Handle multi-level columns from yfinance
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.get_level_values(0)

    spy_ret = spy["Close"].pct_change().dropna()
    spy_ret.index = pd.to_datetime(spy_ret.index).tz_localize(None)
    print(f"  SPY: {len(spy_ret)} trading days, {spy_ret.index.min().date()} to {spy_ret.index.max().date()}")
    return spy_ret


def download_yfinance_sources(start: str = "2014-01-01") -> pd.DataFrame:
    """Download all yfinance sources, return daily close prices."""
    import yfinance as yf

    print("\n[YFINANCE] Downloading data sources...")
    results = {}
    for symbol, meta in YFINANCE_SOURCES.items():
        try:
            data = yf.download(symbol, start=start, auto_adjust=True, progress=False)
            if data.empty:
                print(f"  {symbol}: NO DATA")
                continue
            # Handle multi-level columns
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            close = data["Close"].copy()
            close.index = pd.to_datetime(close.index).tz_localize(None)
            results[symbol] = close
            print(f"  {symbol}: {len(close)} days - {meta['desc']}")
        except Exception as e:
            print(f"  {symbol}: ERROR - {e}")

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)
    return df


def download_fred_sources(start: str = "2014-01-01") -> pd.DataFrame:
    """Download FRED sources (requires fredapi + FRED_API_KEY)."""
    api_key = os.environ.get("FRED_API_KEY", "")
    if not api_key:
        print("\n[FRED] FRED_API_KEY not set, skipping FRED sources")
        return pd.DataFrame()

    try:
        from fredapi import Fred
    except ImportError:
        print("\n[FRED] fredapi not installed. Install with: pip install fredapi")
        return pd.DataFrame()

    print("\n[FRED] Downloading FRED sources...")
    fred = Fred(api_key=api_key)
    results = {}
    for series_id, meta in FRED_SOURCES.items():
        try:
            data = fred.get_series(series_id, observation_start=start)
            if data is None or len(data) == 0:
                print(f"  {series_id}: NO DATA")
                continue
            data.index = pd.to_datetime(data.index).tz_localize(None)
            results[series_id] = data
            print(f"  {series_id}: {len(data)} observations - {meta['desc']}")
        except Exception as e:
            print(f"  {series_id}: ERROR - {e}")

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)
    return df


def compute_forward_returns(spy_ret: pd.Series) -> pd.DataFrame:
    """Compute 1, 5, 20 day forward returns for SPY."""
    fwd = pd.DataFrame(index=spy_ret.index)
    fwd["spy_ret"] = spy_ret
    fwd["fwd_1d"] = spy_ret.shift(-1)
    fwd["fwd_5d"] = spy_ret.rolling(5).sum().shift(-5)
    fwd["fwd_20d"] = spy_ret.rolling(20).sum().shift(-20)
    return fwd


def compute_granger_f_stat(source: pd.Series, target: pd.Series, max_lag: int = 5) -> float:
    """
    Compute Granger causality F-statistic.
    Tests whether lagged source values help predict target beyond target's own lags.
    """
    from numpy.linalg import lstsq

    # Align and clean
    combined = pd.DataFrame({"source": source, "target": target}).dropna()
    if len(combined) < max_lag + 30:
        return 0.0

    y = combined["target"].values[max_lag:]
    n = len(y)

    # Restricted model: target ~ own lags only
    X_restricted = np.column_stack([
        combined["target"].shift(i).values[max_lag:] for i in range(1, max_lag + 1)
    ])

    # Unrestricted model: target ~ own lags + source lags
    X_unrestricted = np.column_stack([
        X_restricted,
        *[combined["source"].shift(i).values[max_lag:].reshape(-1, 1) for i in range(1, max_lag + 1)]
    ])

    # Remove NaN rows
    mask = ~(np.isnan(X_unrestricted).any(axis=1) | np.isnan(y))
    y = y[mask]
    X_restricted = X_restricted[mask]
    X_unrestricted = X_unrestricted[mask]
    n = len(y)

    if n < max_lag * 2 + 10:
        return 0.0

    # Add constant
    X_r = np.column_stack([np.ones(n), X_restricted])
    X_u = np.column_stack([np.ones(n), X_unrestricted])

    try:
        # Solve via least squares
        beta_r, _, _, _ = lstsq(X_r, y, rcond=None)
        beta_u, _, _, _ = lstsq(X_u, y, rcond=None)

        rss_r = np.sum((y - X_r @ beta_r) ** 2)
        rss_u = np.sum((y - X_u @ beta_u) ** 2)

        p = max_lag  # Number of restrictions
        df_u = n - X_u.shape[1]

        if rss_u <= 0 or df_u <= 0:
            return 0.0

        f_stat = ((rss_r - rss_u) / p) / (rss_u / df_u)
        return max(f_stat, 0.0)
    except Exception:
        return 0.0


def analyze_source(
    source_name: str,
    source_prices: pd.Series,
    spy_ret: pd.Series,
    fwd_returns: pd.DataFrame,
    meta: dict,
) -> dict:
    """Analyze a single data source's predictive value."""
    # Compute daily returns for the source
    source_ret = source_prices.pct_change().dropna()

    # Compute derived features
    source_features = pd.DataFrame(index=source_prices.index)
    source_features["level"] = source_prices
    source_features["return_1d"] = source_ret
    source_features["return_5d"] = source_ret.rolling(5).sum()
    source_features["return_20d"] = source_ret.rolling(20).sum()
    source_features["zscore_60d"] = (
        (source_prices - source_prices.rolling(60).mean()) / (source_prices.rolling(60).std() + 1e-10)
    )
    source_features["pctile_252d"] = source_prices.rolling(252).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )

    # Align with SPY
    aligned = fwd_returns.join(source_features, how="inner").dropna(subset=["spy_ret", "return_1d"])
    n_aligned = len(aligned)
    coverage = n_aligned / len(spy_ret) if len(spy_ret) > 0 else 0.0

    if n_aligned < 100:
        return {
            "source": source_name,
            "description": meta["desc"],
            "category": meta["category"],
            "n_days": n_aligned,
            "coverage": coverage,
            "corr_1d": np.nan,
            "ic_1d": np.nan,
            "ic_5d": np.nan,
            "ic_20d": np.nan,
            "granger_f": np.nan,
            "best_feature": "",
            "best_ic": np.nan,
            "verdict": "INSUFFICIENT DATA",
        }

    # Correlation: source return → SPY next-day return
    corr_1d = aligned["return_1d"].corr(aligned["fwd_1d"])

    # Information Coefficient (rank correlation) for each horizon
    from scipy.stats import spearmanr

    best_ic = 0.0
    best_feature = ""
    ic_results = {}

    for feat_name in ["return_1d", "return_5d", "return_20d", "zscore_60d", "pctile_252d", "level"]:
        if feat_name not in aligned.columns:
            continue
        feat_vals = aligned[feat_name].dropna()
        for horizon_name in ["fwd_1d", "fwd_5d", "fwd_20d"]:
            common = aligned[[feat_name, horizon_name]].dropna()
            if len(common) < 50:
                continue
            rho, pval = spearmanr(common[feat_name], common[horizon_name])
            key = f"{feat_name}__{horizon_name}"
            ic_results[key] = {"ic": rho, "pval": pval}
            if abs(rho) > abs(best_ic):
                best_ic = rho
                best_feature = key

    # Standard ICs for 1d return → forward returns
    ic_1d = ic_results.get("return_1d__fwd_1d", {}).get("ic", np.nan)
    ic_5d = ic_results.get("return_1d__fwd_5d", {}).get("ic", np.nan)
    ic_20d = ic_results.get("return_1d__fwd_20d", {}).get("ic", np.nan)

    # Granger causality
    granger_f = compute_granger_f_stat(aligned["return_1d"], aligned["spy_ret"])

    # Verdict
    abs_best_ic = abs(best_ic) if not np.isnan(best_ic) else 0.0
    if abs_best_ic > 0.06 and coverage > 0.8:
        verdict = "STRONG"
    elif abs_best_ic > 0.03 and coverage > 0.7:
        verdict = "MODERATE"
    elif abs_best_ic > 0.02 and coverage > 0.5:
        verdict = "WEAK"
    else:
        verdict = "MINIMAL"

    return {
        "source": source_name,
        "description": meta["desc"],
        "category": meta["category"],
        "n_days": n_aligned,
        "coverage": coverage,
        "corr_1d": corr_1d,
        "ic_1d": ic_1d,
        "ic_5d": ic_5d,
        "ic_20d": ic_20d,
        "granger_f": granger_f,
        "best_feature": best_feature,
        "best_ic": best_ic,
        "verdict": verdict,
    }


def derive_combined_features(
    yf_prices: pd.DataFrame,
    spy_ret: pd.Series,
    fwd_returns: pd.DataFrame,
) -> list:
    """Compute derived features from combinations of sources."""
    from scipy.stats import spearmanr

    results = []
    combos = {}

    # Yield curve slope: 10Y - 2Y (approximate with 10Y - 5Y and 10Y - 13W)
    if "^TNX" in yf_prices.columns and "^FVX" in yf_prices.columns:
        combos["yield_curve_10_5"] = yf_prices["^TNX"] - yf_prices["^FVX"]

    if "^TNX" in yf_prices.columns and "^IRX" in yf_prices.columns:
        combos["yield_curve_10_13w"] = yf_prices["^TNX"] - yf_prices["^IRX"]

    # Credit spread proxy: JNK return - SHY return (risk appetite)
    if "JNK" in yf_prices.columns and "SHY" in yf_prices.columns:
        combos["credit_spread_proxy"] = (
            yf_prices["JNK"].pct_change().rolling(5).sum()
            - yf_prices["SHY"].pct_change().rolling(5).sum()
        )

    # Real yield proxy: TNX level - TIP return (inflation adjustment)
    if "^TNX" in yf_prices.columns and "TIP" in yf_prices.columns:
        tnx_zscore = (yf_prices["^TNX"] - yf_prices["^TNX"].rolling(60).mean()) / (
            yf_prices["^TNX"].rolling(60).std() + 1e-10
        )
        tip_ret_20 = yf_prices["TIP"].pct_change(20)
        combos["real_yield_proxy"] = tnx_zscore - tip_ret_20 * 100  # Scale

    # VIX term structure proxy: VIX level vs 20d MA
    if "^VIX" in yf_prices.columns:
        vix = yf_prices["^VIX"]
        combos["vix_vs_ma20"] = (vix - vix.rolling(20).mean()) / (vix.rolling(20).std() + 1e-10)

    # Oil-Financials divergence
    if "USO" in yf_prices.columns and "XLF" in yf_prices.columns:
        combos["oil_fin_diverge"] = (
            yf_prices["USO"].pct_change(5) - yf_prices["XLF"].pct_change(5)
        )

    for combo_name, combo_series in combos.items():
        aligned = fwd_returns.join(combo_series.rename("combo"), how="inner").dropna(
            subset=["spy_ret", "combo"]
        )
        if len(aligned) < 100:
            results.append({
                "source": combo_name,
                "description": f"Derived: {combo_name}",
                "category": "derived",
                "n_days": len(aligned),
                "coverage": len(aligned) / len(spy_ret),
                "corr_1d": np.nan,
                "ic_1d": np.nan, "ic_5d": np.nan, "ic_20d": np.nan,
                "granger_f": np.nan,
                "best_feature": "", "best_ic": np.nan,
                "verdict": "INSUFFICIENT DATA",
            })
            continue

        ics = {}
        for h in ["fwd_1d", "fwd_5d", "fwd_20d"]:
            sub = aligned[["combo", h]].dropna()
            if len(sub) >= 50:
                rho, _ = spearmanr(sub["combo"], sub[h])
                ics[h] = rho

        best_h = max(ics, key=lambda k: abs(ics[k])) if ics else ""
        best_ic = ics.get(best_h, np.nan)
        abs_best = abs(best_ic) if not np.isnan(best_ic) else 0

        results.append({
            "source": combo_name,
            "description": f"Derived: {combo_name}",
            "category": "derived",
            "n_days": len(aligned),
            "coverage": len(aligned) / len(spy_ret),
            "corr_1d": aligned["combo"].corr(aligned["fwd_1d"]),
            "ic_1d": ics.get("fwd_1d", np.nan),
            "ic_5d": ics.get("fwd_5d", np.nan),
            "ic_20d": ics.get("fwd_20d", np.nan),
            "granger_f": compute_granger_f_stat(aligned["combo"], aligned["spy_ret"]),
            "best_feature": f"combo__{best_h}",
            "best_ic": best_ic,
            "verdict": "STRONG" if abs_best > 0.06 else ("MODERATE" if abs_best > 0.03 else "WEAK"),
        })

    return results


def main():
    print("=" * 70)
    print("GIGA TRADER - Data Source Analysis")
    print("=" * 70)

    # 1. Download SPY
    spy_ret = download_spy_returns(start="2014-01-01")
    fwd_returns = compute_forward_returns(spy_ret)

    # 2. Download yfinance sources
    yf_prices = download_yfinance_sources(start="2014-01-01")

    # 3. Download FRED sources (optional)
    fred_data = download_fred_sources(start="2014-01-01")

    # 4. Analyze each yfinance source
    print("\n" + "=" * 70)
    print("ANALYZING PREDICTIVE VALUE")
    print("=" * 70)

    all_results = []

    for symbol in yf_prices.columns:
        meta = YFINANCE_SOURCES.get(symbol, {"desc": symbol, "category": "unknown"})
        result = analyze_source(symbol, yf_prices[symbol], spy_ret, fwd_returns, meta)
        all_results.append(result)
        print(f"  {symbol:8s} | IC(best)={result['best_ic']:+.4f} | "
              f"Granger F={result['granger_f']:.2f} | "
              f"Coverage={result['coverage']:.1%} | {result['verdict']}")

    # 5. Analyze FRED sources
    if not fred_data.empty:
        for series_id in fred_data.columns:
            meta = FRED_SOURCES.get(series_id, {"desc": series_id, "category": "unknown"})
            # Forward-fill FRED data to daily (many are weekly/monthly)
            fred_daily = fred_data[series_id].reindex(spy_ret.index, method="ffill")
            result = analyze_source(series_id, fred_daily, spy_ret, fwd_returns, meta)
            all_results.append(result)
            print(f"  {series_id:20s} | IC(best)={result['best_ic']:+.4f} | "
                  f"Granger F={result['granger_f']:.2f} | "
                  f"Coverage={result['coverage']:.1%} | {result['verdict']}")

    # 6. Analyze derived/combined features
    print("\n[DERIVED] Analyzing combined features...")
    derived_results = derive_combined_features(yf_prices, spy_ret, fwd_returns)
    all_results.extend(derived_results)
    for r in derived_results:
        print(f"  {r['source']:25s} | IC(best)={r['best_ic']:+.4f} | "
              f"Granger F={r['granger_f']:.2f} | {r['verdict']}")

    # 7. Create results DataFrame and rank
    df_results = pd.DataFrame(all_results)
    df_results["abs_best_ic"] = df_results["best_ic"].abs()
    df_results = df_results.sort_values("abs_best_ic", ascending=False)

    # 8. Print summary report
    print("\n" + "=" * 70)
    print("RANKED DATA SOURCE REPORT")
    print("=" * 70)
    print(f"{'Rank':>4} | {'Source':25s} | {'Category':12s} | {'Best IC':>8} | "
          f"{'Granger F':>9} | {'Coverage':>8} | {'Verdict':12s}")
    print("-" * 100)

    for i, (_, row) in enumerate(df_results.iterrows(), 1):
        print(f"{i:4d} | {row['source']:25s} | {row['category']:12s} | "
              f"{row['best_ic']:+8.4f} | {row['granger_f']:9.2f} | "
              f"{row['coverage']:8.1%} | {row['verdict']:12s}")

    # 9. Recommendations
    strong = df_results[df_results["verdict"] == "STRONG"]
    moderate = df_results[df_results["verdict"] == "MODERATE"]

    print(f"\n{'='*70}")
    print("RECOMMENDATIONS")
    print(f"{'='*70}")
    print(f"  STRONG sources ({len(strong)}): {', '.join(strong['source'].tolist())}")
    print(f"  MODERATE sources ({len(moderate)}): {', '.join(moderate['source'].tolist())}")
    print(f"\n  Recommended for integration: all STRONG + MODERATE sources")
    print(f"  These should be added to EconomicFeatures class")

    # 10. Save results
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    output_path = reports_dir / "data_source_analysis.csv"
    df_results.to_csv(output_path, index=False)
    print(f"\n  Results saved to: {output_path}")

    return df_results


if __name__ == "__main__":
    main()
