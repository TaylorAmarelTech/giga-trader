"""
Tests for Triple Barrier Labeling.

Validates the TripleBarrierLabeler from Lopez de Prado, AFML Ch. 3.

Tests cover:
  - Simple uptrend (all TP labels)
  - Simple downtrend (all SL labels)
  - Flat series (all expiry labels)
  - Mixed series (TP, SL, and expiry)
  - Binary mode (only 0 and 1)
  - Ternary mode (-1, 0, 1)
  - Continuous mode (float returns)
  - touch_type column correctness
  - entry_price and touch_price population
  - Events parameter for selective labeling
  - get_stats correctness
  - label_with_volatility dynamic barrier scaling
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.phase_05_targets.triple_barrier import TripleBarrierLabeler


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def uptrend_prices():
    """Strong uptrend: price rises 0.5% per day."""
    n = 50
    dates = pd.bdate_range("2025-01-02", periods=n)
    # Start at 100, each day +0.5%
    prices = 100.0 * np.cumprod(np.full(n, 1.005))
    return pd.Series(prices, index=dates, name="close")


@pytest.fixture
def downtrend_prices():
    """Strong downtrend: price drops 0.5% per day."""
    n = 50
    dates = pd.bdate_range("2025-01-02", periods=n)
    prices = 100.0 * np.cumprod(np.full(n, 0.995))
    return pd.Series(prices, index=dates, name="close")


@pytest.fixture
def flat_prices():
    """Perfectly flat series (no movement at all)."""
    n = 50
    dates = pd.bdate_range("2025-01-02", periods=n)
    prices = np.full(n, 100.0)
    return pd.Series(prices, index=dates, name="close")


@pytest.fixture
def mixed_prices():
    """Series with distinct up, down, and flat segments."""
    n = 60
    dates = pd.bdate_range("2025-01-02", periods=n)
    prices = np.empty(n)
    prices[0] = 100.0
    # First 20 days: strong up (+0.8% per day -> hits 1% TP in ~2 days)
    for i in range(1, 20):
        prices[i] = prices[i - 1] * 1.008
    # Days 20-40: strong down (-0.8% per day -> hits 1% SL in ~2 days)
    for i in range(20, 40):
        prices[i] = prices[i - 1] * 0.992
    # Days 40-60: flat (tiny oscillation around mean)
    for i in range(40, 60):
        prices[i] = prices[39] * (1.0 + 0.0001 * ((-1) ** i))
    return pd.Series(prices, index=dates, name="close")


@pytest.fixture
def default_labeler():
    """Default TripleBarrierLabeler with 1% TP/SL, 5-day hold, binary mode."""
    return TripleBarrierLabeler(
        tp_pct=0.01, sl_pct=0.01, max_holding_days=5, label_mode="binary"
    )


# ===========================================================================
# 1. TestSimpleUptrend
# ===========================================================================

class TestSimpleUptrend:
    """In a strong uptrend, all labels should be take-profit (1)."""

    def test_all_labels_are_tp(self, uptrend_prices):
        """Every observation in a consistent uptrend should hit TP."""
        labeler = TripleBarrierLabeler(
            tp_pct=0.01, sl_pct=0.01, max_holding_days=5, label_mode="binary"
        )
        result = labeler.label(uptrend_prices)
        assert len(result) > 0
        # All labels should be 1 (TP hit or positive at expiry)
        assert (result["label"] == 1).all(), (
            f"Expected all labels=1 in uptrend, got: "
            f"{result['label'].value_counts().to_dict()}"
        )

    def test_touch_type_is_tp(self, uptrend_prices):
        """Touch type should be 'tp' for strong uptrend (except last few near end)."""
        labeler = TripleBarrierLabeler(
            tp_pct=0.01, sl_pct=0.01, max_holding_days=5, label_mode="binary"
        )
        result = labeler.label(uptrend_prices)
        # With 0.5%/day, TP (1%) is hit in ~2 days. All except the last
        # few entries (which may lack enough forward data) should be "tp".
        # Check that at least 90% are TP (last 1-2 may be expiry).
        tp_ratio = (result["touch_type"] == "tp").mean()
        assert tp_ratio >= 0.90, f"Expected mostly 'tp', got {tp_ratio:.2%}"


# ===========================================================================
# 2. TestSimpleDowntrend
# ===========================================================================

class TestSimpleDowntrend:
    """In a strong downtrend, all labels should be stop-loss."""

    def test_binary_all_zero(self, downtrend_prices):
        """Binary mode: all labels 0 in downtrend."""
        labeler = TripleBarrierLabeler(
            tp_pct=0.01, sl_pct=0.01, max_holding_days=5, label_mode="binary"
        )
        result = labeler.label(downtrend_prices)
        assert len(result) > 0
        assert (result["label"] == 0).all()

    def test_ternary_all_negative(self, downtrend_prices):
        """Ternary mode: nearly all labels -1 in downtrend (last few may expire)."""
        labeler = TripleBarrierLabeler(
            tp_pct=0.01, sl_pct=0.01, max_holding_days=5, label_mode="ternary"
        )
        result = labeler.label(downtrend_prices)
        assert len(result) > 0
        # Last 1-2 entries may not have enough forward data to hit SL,
        # so they can expire as 0. At least 90% should be -1.
        sl_ratio = (result["label"] == -1).mean()
        assert sl_ratio >= 0.90, f"Expected mostly -1, got {sl_ratio:.2%}"


# ===========================================================================
# 3. TestFlatSeries
# ===========================================================================

class TestFlatSeries:
    """In a flat series, all barriers should expire."""

    def test_all_labels_are_expiry(self, flat_prices):
        """All touch types should be 'expiry' for a flat series."""
        labeler = TripleBarrierLabeler(
            tp_pct=0.01, sl_pct=0.01, max_holding_days=5, label_mode="ternary"
        )
        result = labeler.label(flat_prices)
        assert len(result) > 0
        assert (result["touch_type"] == "expiry").all()

    def test_ternary_labels_are_zero(self, flat_prices):
        """Ternary mode should give 0 for all expiry events."""
        labeler = TripleBarrierLabeler(
            tp_pct=0.01, sl_pct=0.01, max_holding_days=5, label_mode="ternary"
        )
        result = labeler.label(flat_prices)
        assert (result["label"] == 0).all()


# ===========================================================================
# 4. TestMixedSeries
# ===========================================================================

class TestMixedSeries:
    """Mixed series should produce a combination of TP, SL, and expiry."""

    def test_has_tp_and_sl(self, mixed_prices):
        """Mixed series produces both TP and SL labels."""
        labeler = TripleBarrierLabeler(
            tp_pct=0.01, sl_pct=0.01, max_holding_days=5, label_mode="ternary"
        )
        result = labeler.label(mixed_prices)
        touch_types = set(result["touch_type"].unique())
        assert "tp" in touch_types, f"Expected 'tp' in {touch_types}"
        assert "sl" in touch_types, f"Expected 'sl' in {touch_types}"

    def test_uptrend_segment_has_tp(self, mixed_prices):
        """Labels from the uptrend segment (days 0-15) should be TP."""
        labeler = TripleBarrierLabeler(
            tp_pct=0.01, sl_pct=0.01, max_holding_days=5, label_mode="ternary"
        )
        result = labeler.label(mixed_prices)
        # First ~15 days (allowing room for forward window into downtrend)
        early_labels = result.iloc[:10]
        assert (early_labels["touch_type"] == "tp").all()


# ===========================================================================
# 5. TestBinaryMode
# ===========================================================================

class TestBinaryMode:
    """Binary mode should produce only 0 and 1."""

    def test_binary_only_zero_one(self, mixed_prices):
        labeler = TripleBarrierLabeler(
            tp_pct=0.01, sl_pct=0.01, max_holding_days=5, label_mode="binary"
        )
        result = labeler.label(mixed_prices)
        unique_labels = set(result["label"].unique())
        assert unique_labels.issubset({0, 1}), (
            f"Binary mode should only produce 0/1, got {unique_labels}"
        )

    def test_binary_dtype_is_int(self, uptrend_prices):
        labeler = TripleBarrierLabeler(
            tp_pct=0.01, sl_pct=0.01, max_holding_days=5, label_mode="binary"
        )
        result = labeler.label(uptrend_prices)
        assert result["label"].dtype in (np.int64, np.int32, int)


# ===========================================================================
# 6. TestTernaryMode
# ===========================================================================

class TestTernaryMode:
    """Ternary mode should produce -1, 0, 1."""

    def test_ternary_values(self, mixed_prices):
        labeler = TripleBarrierLabeler(
            tp_pct=0.01, sl_pct=0.01, max_holding_days=5, label_mode="ternary"
        )
        result = labeler.label(mixed_prices)
        unique_labels = set(result["label"].unique())
        assert unique_labels.issubset({-1, 0, 1}), (
            f"Ternary mode should only produce -1/0/1, got {unique_labels}"
        )

    def test_ternary_dtype_is_int(self, downtrend_prices):
        labeler = TripleBarrierLabeler(
            tp_pct=0.01, sl_pct=0.01, max_holding_days=5, label_mode="ternary"
        )
        result = labeler.label(downtrend_prices)
        assert result["label"].dtype in (np.int64, np.int32, int)


# ===========================================================================
# 7. TestContinuousMode
# ===========================================================================

class TestContinuousMode:
    """Continuous mode should produce float returns."""

    def test_continuous_returns_float(self, mixed_prices):
        labeler = TripleBarrierLabeler(
            tp_pct=0.01, sl_pct=0.01, max_holding_days=5, label_mode="continuous"
        )
        result = labeler.label(mixed_prices)
        assert result["label"].dtype == np.float64

    def test_continuous_tp_returns_positive(self, uptrend_prices):
        """Continuous returns for TP should be >= tp_pct."""
        labeler = TripleBarrierLabeler(
            tp_pct=0.01, sl_pct=0.01, max_holding_days=5, label_mode="continuous"
        )
        result = labeler.label(uptrend_prices)
        tp_labels = result.loc[result["touch_type"] == "tp", "label"]
        assert (tp_labels >= 0.01 - 1e-9).all(), (
            f"TP continuous labels should be >= tp_pct, min was {tp_labels.min()}"
        )

    def test_continuous_sl_returns_negative(self, downtrend_prices):
        """Continuous returns for SL should be <= -sl_pct."""
        labeler = TripleBarrierLabeler(
            tp_pct=0.01, sl_pct=0.01, max_holding_days=5, label_mode="continuous"
        )
        result = labeler.label(downtrend_prices)
        sl_labels = result.loc[result["touch_type"] == "sl", "label"]
        assert (sl_labels <= -0.01 + 1e-9).all(), (
            f"SL continuous labels should be <= -sl_pct, max was {sl_labels.max()}"
        )


# ===========================================================================
# 8. TestTouchTypeColumn
# ===========================================================================

class TestTouchTypeColumn:
    """Verify touch_type column has correct values."""

    def test_touch_type_values(self, mixed_prices):
        labeler = TripleBarrierLabeler(
            tp_pct=0.01, sl_pct=0.01, max_holding_days=5, label_mode="ternary"
        )
        result = labeler.label(mixed_prices)
        valid_types = {"tp", "sl", "expiry"}
        actual_types = set(result["touch_type"].unique())
        assert actual_types.issubset(valid_types), (
            f"Unexpected touch types: {actual_types - valid_types}"
        )

    def test_tp_touch_type_consistent_with_label(self, uptrend_prices):
        """When touch_type is 'tp', ternary label should be 1."""
        labeler = TripleBarrierLabeler(
            tp_pct=0.01, sl_pct=0.01, max_holding_days=5, label_mode="ternary"
        )
        result = labeler.label(uptrend_prices)
        tp_rows = result[result["touch_type"] == "tp"]
        assert (tp_rows["label"] == 1).all()

    def test_sl_touch_type_consistent_with_label(self, downtrend_prices):
        """When touch_type is 'sl', ternary label should be -1."""
        labeler = TripleBarrierLabeler(
            tp_pct=0.01, sl_pct=0.01, max_holding_days=5, label_mode="ternary"
        )
        result = labeler.label(downtrend_prices)
        sl_rows = result[result["touch_type"] == "sl"]
        assert (sl_rows["label"] == -1).all()


# ===========================================================================
# 9. TestEntryAndTouchPrice
# ===========================================================================

class TestEntryAndTouchPrice:
    """Verify entry_price and touch_price columns."""

    def test_entry_price_populated(self, uptrend_prices):
        labeler = TripleBarrierLabeler(
            tp_pct=0.01, sl_pct=0.01, max_holding_days=5, label_mode="binary"
        )
        result = labeler.label(uptrend_prices)
        assert result["entry_price"].notna().all()
        assert (result["entry_price"] > 0).all()

    def test_touch_price_populated(self, uptrend_prices):
        labeler = TripleBarrierLabeler(
            tp_pct=0.01, sl_pct=0.01, max_holding_days=5, label_mode="binary"
        )
        result = labeler.label(uptrend_prices)
        assert result["touch_price"].notna().all()
        assert (result["touch_price"] > 0).all()

    def test_entry_price_matches_price_series(self, uptrend_prices):
        """entry_price should match the price at the event index."""
        labeler = TripleBarrierLabeler(
            tp_pct=0.01, sl_pct=0.01, max_holding_days=5, label_mode="binary"
        )
        result = labeler.label(uptrend_prices)
        for idx in result.index[:5]:  # Check first 5
            np.testing.assert_almost_equal(
                result.loc[idx, "entry_price"],
                uptrend_prices.loc[idx],
                decimal=6,
            )

    def test_return_at_touch_consistent(self, uptrend_prices):
        """return_at_touch should equal (touch_price - entry_price) / entry_price."""
        labeler = TripleBarrierLabeler(
            tp_pct=0.01, sl_pct=0.01, max_holding_days=5, label_mode="binary"
        )
        result = labeler.label(uptrend_prices)
        computed_returns = (
            (result["touch_price"] - result["entry_price"]) / result["entry_price"]
        )
        np.testing.assert_array_almost_equal(
            result["return_at_touch"].values,
            computed_returns.values,
            decimal=10,
        )


# ===========================================================================
# 10. TestEventsParameter
# ===========================================================================

class TestEventsParameter:
    """Test that events parameter labels only specified dates."""

    def test_events_subset(self, uptrend_prices):
        """Passing events should only label those specific dates."""
        labeler = TripleBarrierLabeler(
            tp_pct=0.01, sl_pct=0.01, max_holding_days=5, label_mode="binary"
        )
        # Select every 5th date
        selected = uptrend_prices.index[::5]
        # Exclude dates too close to end
        selected = selected[selected < uptrend_prices.index[-1]]
        result = labeler.label(uptrend_prices, events=selected)
        assert len(result) <= len(selected)
        # All result indices should be in selected
        for idx in result.index:
            assert idx in selected

    def test_events_empty_returns_empty(self, uptrend_prices):
        """Empty events index should return empty DataFrame."""
        labeler = TripleBarrierLabeler(
            tp_pct=0.01, sl_pct=0.01, max_holding_days=5, label_mode="binary"
        )
        empty_events = pd.DatetimeIndex([])
        result = labeler.label(uptrend_prices, events=empty_events)
        assert len(result) == 0

    def test_events_reduces_output_count(self, uptrend_prices):
        """Specifying a subset of events should produce fewer labels."""
        labeler = TripleBarrierLabeler(
            tp_pct=0.01, sl_pct=0.01, max_holding_days=5, label_mode="binary"
        )
        full_result = labeler.label(uptrend_prices)
        # Label only first 5 dates
        subset_events = uptrend_prices.index[:5]
        subset_result = labeler.label(uptrend_prices, events=subset_events)
        assert len(subset_result) < len(full_result)
        assert len(subset_result) == 5


# ===========================================================================
# 11. TestGetStats
# ===========================================================================

class TestGetStats:
    """Test get_stats returns correct counts and averages."""

    def test_stats_counts_match(self, mixed_prices):
        labeler = TripleBarrierLabeler(
            tp_pct=0.01, sl_pct=0.01, max_holding_days=5, label_mode="ternary"
        )
        result = labeler.label(mixed_prices)
        stats = labeler.get_stats(result)

        n_total = len(result)
        assert stats["n_tp"] + stats["n_sl"] + stats["n_expiry"] == n_total

    def test_stats_percentages_sum_to_one(self, mixed_prices):
        labeler = TripleBarrierLabeler(
            tp_pct=0.01, sl_pct=0.01, max_holding_days=5, label_mode="ternary"
        )
        result = labeler.label(mixed_prices)
        stats = labeler.get_stats(result)

        total_pct = stats["pct_tp"] + stats["pct_sl"] + stats["pct_expiry"]
        np.testing.assert_almost_equal(total_pct, 1.0, decimal=10)

    def test_stats_avg_return_tp_positive(self, uptrend_prices):
        labeler = TripleBarrierLabeler(
            tp_pct=0.01, sl_pct=0.01, max_holding_days=5, label_mode="ternary"
        )
        result = labeler.label(uptrend_prices)
        stats = labeler.get_stats(result)
        assert stats["avg_return_tp"] > 0

    def test_stats_avg_return_sl_negative(self, downtrend_prices):
        labeler = TripleBarrierLabeler(
            tp_pct=0.01, sl_pct=0.01, max_holding_days=5, label_mode="ternary"
        )
        result = labeler.label(downtrend_prices)
        stats = labeler.get_stats(result)
        assert stats["avg_return_sl"] < 0

    def test_stats_empty_input(self):
        labeler = TripleBarrierLabeler()
        empty_df = pd.DataFrame(
            columns=["label", "touch_type", "touch_idx",
                     "return_at_touch", "entry_price", "touch_price"]
        )
        stats = labeler.get_stats(empty_df)
        assert stats["n_tp"] == 0
        assert stats["n_sl"] == 0
        assert stats["n_expiry"] == 0

    def test_stats_avg_touch_idx(self, flat_prices):
        """For flat series with max_holding_days=5, avg touch_idx should be near 5."""
        labeler = TripleBarrierLabeler(
            tp_pct=0.01, sl_pct=0.01, max_holding_days=5, label_mode="ternary"
        )
        result = labeler.label(flat_prices)
        stats = labeler.get_stats(result)
        # All expiry, so touch_idx should be max_holding_days (or up to it for last rows)
        assert stats["avg_touch_idx"] > 0


# ===========================================================================
# 12. TestLabelWithVolatility
# ===========================================================================

class TestLabelWithVolatility:
    """Test that label_with_volatility scales barriers dynamically."""

    def test_high_vol_widens_barriers(self, uptrend_prices):
        """High volatility should widen barriers, possibly changing touch types."""
        labeler = TripleBarrierLabeler(
            tp_pct=0.01, sl_pct=0.01, max_holding_days=5, label_mode="ternary"
        )
        # Low vol: barriers are 1%
        low_vol = pd.Series(1.0, index=uptrend_prices.index)
        result_low = labeler.label_with_volatility(
            uptrend_prices, low_vol, vol_multiplier=1.0
        )

        # High vol: barriers are scaled to 3% (vol=3.0 * base 1%)
        high_vol = pd.Series(3.0, index=uptrend_prices.index)
        result_high = labeler.label_with_volatility(
            uptrend_prices, high_vol, vol_multiplier=1.0
        )

        # With wider barriers (3%), fewer should hit TP (more expiry)
        n_tp_low = (result_low["touch_type"] == "tp").sum()
        n_tp_high = (result_high["touch_type"] == "tp").sum()
        # Wider barriers -> fewer or equal TP touches
        assert n_tp_high <= n_tp_low

    def test_vol_multiplier_effect(self, uptrend_prices):
        """Higher vol_multiplier should widen barriers further."""
        labeler = TripleBarrierLabeler(
            tp_pct=0.01, sl_pct=0.01, max_holding_days=5, label_mode="ternary"
        )
        vol = pd.Series(1.0, index=uptrend_prices.index)

        result_m1 = labeler.label_with_volatility(
            uptrend_prices, vol, vol_multiplier=1.0
        )
        result_m5 = labeler.label_with_volatility(
            uptrend_prices, vol, vol_multiplier=5.0
        )

        n_tp_m1 = (result_m1["touch_type"] == "tp").sum()
        n_tp_m5 = (result_m5["touch_type"] == "tp").sum()
        # With 5x multiplier, barriers are 5%, much fewer TPs
        assert n_tp_m5 <= n_tp_m1

    def test_vol_scaled_result_has_correct_columns(self, uptrend_prices):
        """Output of label_with_volatility should have same columns as label()."""
        labeler = TripleBarrierLabeler(
            tp_pct=0.01, sl_pct=0.01, max_holding_days=5, label_mode="binary"
        )
        vol = pd.Series(1.0, index=uptrend_prices.index)
        result = labeler.label_with_volatility(uptrend_prices, vol)
        expected_cols = {"label", "touch_type", "touch_idx",
                         "return_at_touch", "entry_price", "touch_price"}
        assert set(result.columns) == expected_cols


# ===========================================================================
# 13. TestEdgeCases
# ===========================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_prices(self):
        labeler = TripleBarrierLabeler()
        result = labeler.label(pd.Series([], dtype=float))
        assert len(result) == 0

    def test_single_price(self):
        """Single price point has no forward data -> empty result."""
        labeler = TripleBarrierLabeler()
        result = labeler.label(pd.Series([100.0]))
        assert len(result) == 0

    def test_two_prices(self):
        """Two prices: one entry, one forward day."""
        labeler = TripleBarrierLabeler(
            tp_pct=0.01, sl_pct=0.01, max_holding_days=5, label_mode="ternary"
        )
        prices = pd.Series([100.0, 105.0])  # 5% jump -> TP
        result = labeler.label(prices)
        assert len(result) == 1
        assert result["touch_type"].iloc[0] == "tp"

    def test_nan_prices_handled(self):
        """NaN prices are forward-filled and then dropped."""
        labeler = TripleBarrierLabeler(
            tp_pct=0.01, sl_pct=0.01, max_holding_days=5, label_mode="binary"
        )
        dates = pd.bdate_range("2025-01-02", periods=20)
        prices = pd.Series(
            [100.0] * 5 + [np.nan] * 3 + [102.0] * 12,
            index=dates,
        )
        result = labeler.label(prices)
        # Should handle without error
        assert len(result) > 0

    def test_max_holding_days_one(self):
        """With max_holding_days=1, every event looks only 1 day forward."""
        labeler = TripleBarrierLabeler(
            tp_pct=0.10, sl_pct=0.10, max_holding_days=1, label_mode="ternary"
        )
        # Small daily moves won't hit 10% TP/SL in 1 day -> all expiry
        dates = pd.bdate_range("2025-01-02", periods=20)
        prices = pd.Series(
            100.0 * np.cumprod(np.full(20, 1.001)),
            index=dates,
        )
        result = labeler.label(prices)
        assert (result["touch_type"] == "expiry").all()
        # touch_idx should all be 1 (one forward day)
        assert (result["touch_idx"] == 1).all()


# ===========================================================================
# 14. TestConstructorValidation
# ===========================================================================

class TestConstructorValidation:
    """Test constructor parameter validation."""

    def test_negative_tp_raises(self):
        with pytest.raises(ValueError, match="tp_pct"):
            TripleBarrierLabeler(tp_pct=-0.01)

    def test_negative_sl_raises(self):
        with pytest.raises(ValueError, match="sl_pct"):
            TripleBarrierLabeler(sl_pct=-0.01)

    def test_zero_holding_days_raises(self):
        with pytest.raises(ValueError, match="max_holding_days"):
            TripleBarrierLabeler(max_holding_days=0)

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="label_mode"):
            TripleBarrierLabeler(label_mode="invalid")


# ===========================================================================
# 15. TestRepr
# ===========================================================================

class TestRepr:
    """Test string representation."""

    def test_repr_contains_parameters(self):
        labeler = TripleBarrierLabeler(
            tp_pct=0.02, sl_pct=0.015, max_holding_days=10, label_mode="ternary"
        )
        r = repr(labeler)
        assert "tp_pct=0.02" in r
        assert "sl_pct=0.015" in r
        assert "max_holding_days=10" in r
        assert "ternary" in r
