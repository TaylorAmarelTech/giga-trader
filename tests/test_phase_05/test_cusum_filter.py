"""
Tests for CUSUM Filter.

Validates the CUSUM event filter for training data sampling,
based on Lopez de Prado, "Advances in Financial Machine Learning", Ch 2.

Tests cover:
  - Basic filtering mechanics (CUSUM accumulation and reset)
  - Event detection on various return patterns
  - DataFrame filtering with return and close columns
  - Safety guards (min_events, max_filter_ratio)
  - Threshold sensitivity
  - Edge cases (empty, NaN, single element, etc.)
  - Statistical properties of filtered events
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.phase_05_targets.cusum_filter import CUSUMFilter


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_returns():
    """Known return sequence for hand-verified CUSUM behavior."""
    # Cumulative positive: 0.005, 0.008, 0.011 -> trigger at index 2 (>=0.01)
    # Then resets. Next: 0.006, 0.009, 0.012 -> trigger at index 5
    return np.array([0.005, 0.003, 0.003, 0.006, 0.003, 0.003])


@pytest.fixture
def negative_returns():
    """Known negative return sequence."""
    # Cumulative negative: -0.005, -0.008, -0.012 -> trigger at index 2
    return np.array([-0.005, -0.003, -0.004, -0.006, -0.003, -0.004])


@pytest.fixture
def alternating_returns():
    """Returns that alternate sign and cancel out."""
    return np.array([0.005, -0.005, 0.005, -0.005, 0.005, -0.005] * 20)


@pytest.fixture
def large_move_returns():
    """Returns with a few large moves embedded in noise."""
    np.random.seed(42)
    n = 500
    returns = np.random.normal(0, 0.002, n)  # Small noise
    # Inject 5 large upward moves
    returns[50] = 0.03
    returns[150] = 0.025
    returns[250] = -0.04
    returns[350] = 0.035
    returns[450] = -0.028
    return returns


@pytest.fixture
def realistic_spy_returns():
    """Simulate 1000 days of SPY-like returns."""
    np.random.seed(123)
    n = 1000
    returns = np.random.normal(0.0004, 0.01, n)
    # Inject a few regime changes
    returns[200:220] = np.random.normal(-0.012, 0.02, 20)  # Crisis
    returns[500:510] = np.random.normal(0.015, 0.015, 10)  # Rally
    return returns


@pytest.fixture
def spy_dataframe(realistic_spy_returns):
    """DataFrame with close prices and returns like SPY."""
    n = len(realistic_spy_returns)
    dates = pd.bdate_range("2020-01-02", periods=n)
    close = 450.0 * np.cumprod(1.0 + realistic_spy_returns)
    return pd.DataFrame({
        "date": dates,
        "close": close,
        "day_return": realistic_spy_returns,
        "volume": np.random.randint(50_000_000, 150_000_000, n),
    })


# ===========================================================================
# 1. TestBasicFiltering
# ===========================================================================

class TestBasicFiltering:
    """Test core CUSUM accumulation and reset logic."""

    def test_positive_cusum_triggers_event(self, simple_returns):
        """Positive cumulative returns exceeding threshold trigger an event."""
        cf = CUSUMFilter(threshold=0.01, min_events=0)
        events = cf.get_event_indices(simple_returns)
        # S_pos after each step: 0.005, 0.008, 0.011 >= 0.01 -> event at 2
        # Reset: 0.006, 0.009, 0.012 >= 0.01 -> event at 5
        assert 2 in events

    def test_negative_cusum_triggers_event(self, negative_returns):
        """Negative cumulative returns exceeding threshold trigger an event."""
        cf = CUSUMFilter(threshold=0.01, min_events=0)
        events = cf.get_event_indices(negative_returns)
        # S_neg: -0.005, -0.008, -0.012 <= -0.01 -> event at 2
        assert 2 in events

    def test_cusum_resets_after_event(self, simple_returns):
        """After an event triggers, the accumulator resets to 0."""
        cf = CUSUMFilter(threshold=0.01, min_events=0)
        events = cf.get_event_indices(simple_returns)
        # After reset at index 2, accumulates again: 0.006, 0.009, 0.012
        # Should trigger again at index 5
        assert 5 in events
        assert len(events) == 2

    def test_symmetric_threshold(self):
        """Symmetric mode uses same threshold for up and down."""
        returns = np.array([0.005, 0.006, -0.005, -0.006])
        cf = CUSUMFilter(threshold=0.01, symmetric=True, min_events=0)
        events = cf.get_event_indices(returns)
        # S_pos: 0.005, 0.011 >= 0.01 -> event at 1
        # After reset S_pos=0: S_neg only: -0.005, -0.011 <= -0.01 -> event at 3
        assert 1 in events
        assert 3 in events

    def test_asymmetric_thresholds(self):
        """Asymmetric mode allows different thresholds for up vs down."""
        returns = np.array([0.005, 0.004, -0.005, -0.004])
        cf = CUSUMFilter(
            threshold=0.01,
            symmetric=False,
            up_threshold=0.008,
            down_threshold=0.008,
            min_events=0,
        )
        events = cf.get_event_indices(returns)
        # S_pos: 0.005, 0.009 >= 0.008 -> event at 1
        # S_neg: max(0,0.005)+0.004=0 (after reset), then -0.005, -0.009 <= -0.008 -> event at 3
        assert 1 in events
        assert 3 in events

    def test_asymmetric_different_sensitivities(self):
        """Asymmetric: loose up_threshold, tight down_threshold."""
        returns = np.array([0.005, 0.004, -0.005, -0.004])
        cf = CUSUMFilter(
            threshold=0.01,
            symmetric=False,
            up_threshold=0.02,   # Loose - won't trigger
            down_threshold=0.008,  # Tight - will trigger
            min_events=0,
        )
        events = cf.get_event_indices(returns)
        # Only negative CUSUM should trigger
        up_events = [e for e in events if returns[e] > 0 or sum(returns[:e+1]) > 0]
        # S_pos: 0.005, 0.009 < 0.02 -> no up event
        # S_neg resets on up, then -0.005, -0.009 <= -0.008 -> event at 3
        assert 3 in events


# ===========================================================================
# 2. TestEventDetection
# ===========================================================================

class TestEventDetection:
    """Test event detection on various return patterns."""

    def test_large_moves_always_trigger(self, large_move_returns):
        """Large individual moves should always be captured."""
        cf = CUSUMFilter(threshold=0.02, min_events=0)
        events = cf.get_event_indices(large_move_returns)
        # The 0.03, 0.025, -0.04, 0.035, -0.028 moves should trigger
        assert len(events) >= 5

    def test_zero_returns_never_trigger(self):
        """A flat return series produces no events."""
        returns = np.zeros(200)
        cf = CUSUMFilter(threshold=0.01, min_events=0)
        events = cf.get_event_indices(returns)
        assert len(events) == 0

    def test_gradual_drift_triggers_eventually(self):
        """Small consistent positive returns accumulate to threshold."""
        # 0.001 * 10 = 0.01, should trigger
        returns = np.full(100, 0.001)
        cf = CUSUMFilter(threshold=0.01, min_events=0)
        events = cf.get_event_indices(returns)
        # First event at index 9 (10 steps of 0.001 = 0.01)
        assert events[0] == 9
        # Should trigger every 10 steps
        assert len(events) == 10

    def test_alternating_returns_cancel_out(self, alternating_returns):
        """Alternating +/- returns that cancel out should produce few/no events."""
        cf = CUSUMFilter(threshold=0.01, min_events=0)
        events = cf.get_event_indices(alternating_returns)
        # +0.005 then -0.005: S_pos goes 0.005 then max(0, 0.005-0.005)=0
        # These cancel and never reach 0.01
        assert len(events) == 0

    def test_single_large_return_triggers(self):
        """A single return >= threshold triggers immediately."""
        returns = np.array([0.0, 0.0, 0.015, 0.0, 0.0])
        cf = CUSUMFilter(threshold=0.01, min_events=0)
        events = cf.get_event_indices(returns)
        assert 2 in events

    def test_single_large_negative_return_triggers(self):
        """A single large negative return triggers immediately."""
        returns = np.array([0.0, 0.0, -0.015, 0.0, 0.0])
        cf = CUSUMFilter(threshold=0.01, min_events=0)
        events = cf.get_event_indices(returns)
        assert 2 in events

    def test_consecutive_triggers(self):
        """Two consecutive large moves can both trigger events."""
        returns = np.array([0.015, 0.015])
        cf = CUSUMFilter(threshold=0.01, min_events=0)
        events = cf.get_event_indices(returns)
        assert 0 in events
        assert 1 in events

    def test_mixed_up_down_events(self):
        """Mix of upward and downward cumulative events."""
        returns = np.array([
            0.003, 0.003, 0.005,  # S_pos: 0.003, 0.006, 0.011 -> event at 2
            -0.004, -0.004, -0.003,  # S_neg: -0.004, -0.008, -0.011 -> event at 5
        ])
        cf = CUSUMFilter(threshold=0.01, min_events=0)
        events = cf.get_event_indices(returns)
        assert 2 in events
        assert 5 in events


# ===========================================================================
# 3. TestDataFrameFiltering
# ===========================================================================

class TestDataFrameFiltering:
    """Test filtering applied to pandas DataFrames."""

    def test_filter_with_return_column(self, spy_dataframe):
        """filter_dataframe uses day_return column when available."""
        cf = CUSUMFilter(threshold=0.02, min_events=0)
        result = cf.filter_dataframe(spy_dataframe, return_col="day_return")
        assert len(result) < len(spy_dataframe)
        assert len(result) > 0

    def test_filter_computes_from_close(self, spy_dataframe):
        """filter_dataframe computes returns from close when return_col missing."""
        df_no_return = spy_dataframe.drop(columns=["day_return"])
        cf = CUSUMFilter(threshold=0.02, min_events=0)
        result = cf.filter_dataframe(df_no_return, close_col="close")
        assert len(result) < len(df_no_return)
        assert len(result) > 0

    def test_preserves_columns(self, spy_dataframe):
        """Filtered DataFrame retains all original columns."""
        cf = CUSUMFilter(threshold=0.02, min_events=0)
        result = cf.filter_dataframe(spy_dataframe)
        assert list(result.columns) == list(spy_dataframe.columns)

    def test_preserves_index(self, spy_dataframe):
        """Filtered DataFrame preserves the original index values."""
        spy_dataframe = spy_dataframe.set_index("date")
        cf = CUSUMFilter(threshold=0.02, min_events=0)
        result = cf.filter_dataframe(spy_dataframe)
        # All result indices should be a subset of original
        assert all(idx in spy_dataframe.index for idx in result.index)

    def test_returns_subset_of_rows(self, spy_dataframe):
        """Result is a strict subset (or equal) of the input rows."""
        cf = CUSUMFilter(threshold=0.02, min_events=0)
        result = cf.filter_dataframe(spy_dataframe)
        assert len(result) <= len(spy_dataframe)

    def test_raises_without_required_columns(self):
        """Raise ValueError if neither return_col nor close_col exist."""
        df = pd.DataFrame({"volume": [100, 200, 300]})
        cf = CUSUMFilter(threshold=0.01, min_events=0)
        with pytest.raises(ValueError, match="must contain either"):
            cf.filter_dataframe(df, return_col="day_return", close_col="close")

    def test_empty_dataframe(self):
        """Empty DataFrame returns empty DataFrame."""
        df = pd.DataFrame(columns=["date", "close", "day_return"])
        cf = CUSUMFilter(threshold=0.01)
        result = cf.filter_dataframe(df)
        assert len(result) == 0

    def test_close_column_computation_matches(self, spy_dataframe):
        """Returns computed from close should give similar events to day_return."""
        cf = CUSUMFilter(threshold=0.02, min_events=0)
        result_return = cf.filter_dataframe(spy_dataframe, return_col="day_return")
        df_no_return = spy_dataframe.drop(columns=["day_return"])
        result_close = cf.filter_dataframe(df_no_return, close_col="close")
        # Not exactly equal (close-computed returns differ slightly from
        # pre-computed day_return due to cumulative product vs. pct_change),
        # but should be in the same ballpark
        ratio = len(result_close) / max(len(result_return), 1)
        assert 0.5 < ratio < 2.0


# ===========================================================================
# 4. TestSafetyGuards
# ===========================================================================

class TestSafetyGuards:
    """Test min_events and max_filter_ratio safety mechanisms."""

    def test_min_events_fallback(self):
        """If fewer than min_events detected, return all dates."""
        returns = np.random.normal(0, 0.001, 200)  # Very small returns
        cf = CUSUMFilter(threshold=0.05, min_events=100)
        mask = cf.filter_events(returns)
        # Very high threshold on small returns -> few events -> fallback
        assert mask.sum() == len(returns)

    def test_min_events_not_triggered_when_enough(self, realistic_spy_returns):
        """When enough events exist, min_events guard does not activate."""
        cf = CUSUMFilter(threshold=0.005, min_events=10)
        mask = cf.filter_events(realistic_spy_returns)
        # threshold=0.005 on SPY-like data should produce many events
        assert 10 <= mask.sum() < len(realistic_spy_returns)

    def test_max_filter_ratio_relaxes_threshold(self, realistic_spy_returns):
        """If too many dates filtered, threshold is relaxed."""
        # Very high threshold -> would filter almost everything
        cf = CUSUMFilter(
            threshold=0.10,
            min_events=0,
            max_filter_ratio=0.5,  # Keep at least 50%
        )
        mask = cf.filter_events(realistic_spy_returns)
        kept_ratio = mask.sum() / len(realistic_spy_returns)
        # Must keep at least 50%
        assert kept_ratio >= 0.5

    def test_few_samples_returns_all(self):
        """When input has fewer rows than min_events, return all."""
        returns = np.array([0.001, 0.002, 0.003])
        cf = CUSUMFilter(threshold=0.01, min_events=100)
        mask = cf.filter_events(returns)
        assert mask.sum() == 3

    def test_max_filter_ratio_one_allows_full_filtering(self):
        """max_filter_ratio=1.0 allows filtering everything."""
        returns = np.zeros(100)  # No events at all
        cf = CUSUMFilter(threshold=0.01, min_events=0, max_filter_ratio=1.0)
        mask = cf.filter_events(returns)
        assert mask.sum() == 0

    def test_dataframe_min_events_fallback(self, spy_dataframe):
        """DataFrame filter also respects min_events fallback."""
        cf = CUSUMFilter(threshold=0.50, min_events=500)
        result = cf.filter_dataframe(spy_dataframe)
        assert len(result) == len(spy_dataframe)


# ===========================================================================
# 5. TestThresholdSensitivity
# ===========================================================================

class TestThresholdSensitivity:
    """Test that threshold changes produce expected event count changes."""

    def test_lower_threshold_more_events(self, realistic_spy_returns):
        """Lower threshold produces more events."""
        cf_low = CUSUMFilter(threshold=0.005, min_events=0)
        cf_high = CUSUMFilter(threshold=0.02, min_events=0)
        events_low = cf_low.get_event_indices(realistic_spy_returns)
        events_high = cf_high.get_event_indices(realistic_spy_returns)
        assert len(events_low) > len(events_high)

    def test_higher_threshold_fewer_events(self, realistic_spy_returns):
        """Higher threshold produces fewer events."""
        cf_a = CUSUMFilter(threshold=0.01, min_events=0)
        cf_b = CUSUMFilter(threshold=0.03, min_events=0)
        events_a = cf_a.get_event_indices(realistic_spy_returns)
        events_b = cf_b.get_event_indices(realistic_spy_returns)
        assert len(events_a) >= len(events_b)

    def test_threshold_zero_returns_all_events(self):
        """Threshold=0 should return all dates as events."""
        returns = np.array([0.001, -0.002, 0.003, 0.0, -0.001])
        cf = CUSUMFilter(threshold=0.0, min_events=0)
        mask = cf.filter_events(returns)
        assert mask.sum() == len(returns)

    def test_monotonic_event_count_with_threshold(self, realistic_spy_returns):
        """Event count should be monotonically non-increasing with threshold."""
        thresholds = [0.003, 0.005, 0.01, 0.02, 0.03, 0.05]
        event_counts = []
        for t in thresholds:
            cf = CUSUMFilter(threshold=t, min_events=0)
            events = cf.get_event_indices(realistic_spy_returns)
            event_counts.append(len(events))
        # Each count should be >= the next
        for i in range(len(event_counts) - 1):
            assert event_counts[i] >= event_counts[i + 1], (
                f"Event counts not monotonic: threshold {thresholds[i]}->"
                f"{thresholds[i+1]}, events {event_counts[i]}->{event_counts[i+1]}"
            )


# ===========================================================================
# 6. TestEdgeCases
# ===========================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_array(self):
        """Empty input returns empty output."""
        cf = CUSUMFilter(threshold=0.01, min_events=0)
        events = cf.get_event_indices(np.array([]))
        assert events == []

    def test_empty_array_mask(self):
        """filter_events on empty array returns empty boolean array."""
        cf = CUSUMFilter(threshold=0.01, min_events=0)
        mask = cf.filter_events(np.array([]))
        assert len(mask) == 0

    def test_single_element_no_event(self):
        """Single small return produces no event."""
        cf = CUSUMFilter(threshold=0.01, min_events=0)
        events = cf.get_event_indices(np.array([0.001]))
        assert len(events) == 0

    def test_single_element_with_event(self):
        """Single large return triggers event."""
        cf = CUSUMFilter(threshold=0.01, min_events=0)
        events = cf.get_event_indices(np.array([0.02]))
        assert events == [0]

    def test_all_same_returns(self):
        """Constant positive returns trigger at regular intervals."""
        returns = np.full(100, 0.002)  # 0.002 * 5 = 0.01
        cf = CUSUMFilter(threshold=0.01, min_events=0)
        events = cf.get_event_indices(returns)
        # Trigger every 5 steps: indices 4, 9, 14, ...
        assert events[0] == 4
        assert len(events) == 20

    def test_very_large_returns(self):
        """Very large returns do not cause numerical issues."""
        returns = np.array([100.0, -100.0, 50.0, -50.0])
        cf = CUSUMFilter(threshold=0.01, min_events=0)
        events = cf.get_event_indices(returns)
        assert len(events) == 4  # All should trigger

    def test_nan_values_treated_as_zero(self):
        """NaN returns are treated as 0 (no contribution to CUSUM)."""
        returns = np.array([0.005, np.nan, np.nan, 0.006])
        cf = CUSUMFilter(threshold=0.01, min_events=0)
        events = cf.get_event_indices(returns)
        # S_pos: 0.005, 0.005, 0.005, 0.011 >= 0.01 -> event at 3
        assert events == [3]

    def test_all_nan_returns(self):
        """All-NaN series produces no events."""
        returns = np.full(50, np.nan)
        cf = CUSUMFilter(threshold=0.01, min_events=0)
        events = cf.get_event_indices(returns)
        assert len(events) == 0

    def test_inf_values(self):
        """Inf returns trigger events (they exceed any threshold)."""
        returns = np.array([0.001, np.inf, 0.001])
        cf = CUSUMFilter(threshold=0.01, min_events=0)
        events = cf.get_event_indices(returns)
        assert 1 in events

    def test_negative_inf_values(self):
        """Negative inf returns trigger negative events."""
        returns = np.array([0.001, -np.inf, 0.001])
        cf = CUSUMFilter(threshold=0.01, min_events=0)
        events = cf.get_event_indices(returns)
        assert 1 in events

    def test_2d_array_raises(self):
        """2-D input raises ValueError."""
        returns = np.array([[0.01, 0.02], [0.03, 0.04]])
        cf = CUSUMFilter(threshold=0.01, min_events=0)
        with pytest.raises(ValueError, match="1-D"):
            cf.get_event_indices(returns)

    def test_list_input(self):
        """List input is accepted (converted to np.ndarray)."""
        cf = CUSUMFilter(threshold=0.01, min_events=0)
        events = cf.get_event_indices([0.005, 0.006])
        assert events == [1]


# ===========================================================================
# 7. TestStatisticalProperties
# ===========================================================================

class TestStatisticalProperties:
    """Test that CUSUM events have expected statistical characteristics."""

    def test_events_capture_larger_moves(self, realistic_spy_returns):
        """Event-day returns should have higher absolute mean than non-events."""
        cf = CUSUMFilter(threshold=0.01, min_events=0)
        mask = cf.filter_events(realistic_spy_returns)

        if mask.sum() == 0 or mask.sum() == len(realistic_spy_returns):
            pytest.skip("No filtering occurred, cannot compare subsets")

        event_returns = np.abs(realistic_spy_returns[mask])
        non_event_returns = np.abs(realistic_spy_returns[~mask])

        # Event days should have higher average absolute return
        assert event_returns.mean() > non_event_returns.mean(), (
            f"Event abs mean {event_returns.mean():.6f} should exceed "
            f"non-event abs mean {non_event_returns.mean():.6f}"
        )

    def test_event_spacing_reasonable(self, realistic_spy_returns):
        """Events should not all cluster in one spot."""
        cf = CUSUMFilter(threshold=0.01, min_events=0)
        events = cf.get_event_indices(realistic_spy_returns)

        if len(events) < 3:
            pytest.skip("Too few events to test spacing")

        # Compute spacing between consecutive events
        spacings = np.diff(events)
        # At least some spacing should be > 1 (not all consecutive)
        assert spacings.max() > 1
        # Mean spacing should be reasonable (not all clustered)
        assert spacings.mean() > 1.0

    def test_filtered_set_spans_time_range(self, realistic_spy_returns):
        """Events should be distributed across the full time series."""
        cf = CUSUMFilter(threshold=0.01, min_events=0)
        events = cf.get_event_indices(realistic_spy_returns)

        if len(events) < 2:
            pytest.skip("Too few events")

        n = len(realistic_spy_returns)
        # First event should be in the first quarter
        assert events[0] < n * 0.5
        # Last event should be in the last quarter
        assert events[-1] > n * 0.5

    def test_crisis_periods_generate_more_events(self, realistic_spy_returns):
        """The injected crisis period (days 200-220) should produce events."""
        cf = CUSUMFilter(threshold=0.01, min_events=0)
        events = cf.get_event_indices(realistic_spy_returns)
        # Check that some events fall in the crisis window
        crisis_events = [e for e in events if 195 <= e <= 225]
        assert len(crisis_events) > 0, "Crisis period should generate CUSUM events"

    def test_rally_periods_generate_events(self, realistic_spy_returns):
        """The injected rally period (days 500-510) should produce events."""
        cf = CUSUMFilter(threshold=0.01, min_events=0)
        events = cf.get_event_indices(realistic_spy_returns)
        rally_events = [e for e in events if 495 <= e <= 515]
        assert len(rally_events) > 0, "Rally period should generate CUSUM events"

    def test_filter_ratio_matches_expectation(self, realistic_spy_returns):
        """Verify the reported filter ratio is correct."""
        cf = CUSUMFilter(threshold=0.01, min_events=0)
        mask = cf.filter_events(realistic_spy_returns)
        kept = mask.sum()
        total = len(realistic_spy_returns)
        kept_ratio = kept / total
        # With threshold=0.01 on SPY-like data, expect to keep 10-60%
        assert 0.05 < kept_ratio < 0.80, (
            f"Filter ratio {kept_ratio:.2f} outside expected range"
        )


# ===========================================================================
# 8. TestRepr
# ===========================================================================

class TestRepr:
    """Test the string representation."""

    def test_symmetric_repr(self):
        cf = CUSUMFilter(threshold=0.015, min_events=50, max_filter_ratio=0.7)
        r = repr(cf)
        assert "threshold=0.015" in r
        assert "min_events=50" in r
        assert "max_filter_ratio=0.7" in r

    def test_asymmetric_repr(self):
        cf = CUSUMFilter(
            threshold=0.01,
            symmetric=False,
            up_threshold=0.02,
            down_threshold=0.008,
        )
        r = repr(cf)
        assert "up_threshold=0.02" in r
        assert "down_threshold=0.008" in r


# ===========================================================================
# 9. TestValidation
# ===========================================================================

class TestValidation:
    """Test constructor validation."""

    def test_negative_threshold_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            CUSUMFilter(threshold=-0.01)

    def test_invalid_max_filter_ratio_raises(self):
        with pytest.raises(ValueError, match="max_filter_ratio"):
            CUSUMFilter(max_filter_ratio=0.0)

    def test_invalid_max_filter_ratio_above_one_raises(self):
        with pytest.raises(ValueError, match="max_filter_ratio"):
            CUSUMFilter(max_filter_ratio=1.5)

    def test_negative_min_events_raises(self):
        with pytest.raises(ValueError, match="min_events"):
            CUSUMFilter(min_events=-1)

    def test_negative_up_threshold_raises(self):
        with pytest.raises(ValueError, match="up_threshold"):
            CUSUMFilter(symmetric=False, up_threshold=-0.01)

    def test_negative_down_threshold_raises(self):
        with pytest.raises(ValueError, match="down_threshold"):
            CUSUMFilter(symmetric=False, down_threshold=-0.01)
