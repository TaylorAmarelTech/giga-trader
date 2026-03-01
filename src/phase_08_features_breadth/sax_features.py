"""
GIGA TRADER - SAX (Symbolic Aggregate approXimation) Pattern Features
=======================================================================
Convert price return series into symbolic strings using SAX encoding,
then derive pattern-recognition features.

SAX Overview:
  1. Take the last N days of returns.
  2. Z-normalize the window (zero mean, unit variance).
  3. Apply PAA (Piecewise Aggregate Approximation): reduce to W segments
     by averaging each consecutive block of N/W values.
  4. Discretize each PAA value into an alphabet symbol using breakpoints
     derived from a standard normal distribution (equal-probability bins).
  5. The resulting word (e.g. "bcdba") encodes the recent price path.

Features (3, prefix sax_):
  sax_pattern_20d   — Rolling pattern hash: hash(5-letter SAX word) % 10000
  sax_pattern_match — Directional pattern flag (+1 bullish, -1 bearish, 0 neutral)
  sax_novelty       — Rolling 50-day KL-like divergence of letter frequency
                      distributions (recent 20d vs historical 50d).

Breakpoints for 4-symbol alphabet (a/b/c/d) derived from N(0,1) quartiles:
  a : z < -0.6745   (below 25th percentile)
  b : -0.6745 <= z < 0
  c : 0 <= z < 0.6745
  d : z >= 0.6745   (above 75th percentile)

Bullish SAX endings (last 2 symbols): 'cd', 'dd'
Bearish SAX endings:                  'ab', 'aa'

Pure numpy/pandas — no external dependencies.
"""

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Breakpoints for standard normal equal-probability bins (alphabet_size = 4)
# These are the z-score thresholds that divide N(0,1) into 4 equal areas.
_BREAKPOINTS_4 = np.array([-0.6745, 0.0, 0.6745])  # 3 boundaries → 4 bins

# Breakpoints for other common alphabet sizes (3 and 5), precomputed.
_BREAKPOINTS = {
    3: np.array([-0.4307, 0.4307]),
    4: _BREAKPOINTS_4,
    5: np.array([-0.8416, -0.2533, 0.2533, 0.8416]),
}


class SAXFeatures:
    """
    Compute SAX (Symbolic Aggregate approXimation) pattern features.

    All computation is pure numpy/pandas; no external packages required.

    Parameters
    ----------
    pattern_window : int
        Number of days used for the rolling SAX pattern (default 20).
    n_segments : int
        Number of PAA segments (default 5).  pattern_window must be
        divisible by n_segments; if not, the window is trimmed to the
        nearest multiple.
    alphabet_size : int
        Number of SAX symbols (default 4 → letters a/b/c/d).
        Supported values: 3, 4, 5.
    """

    REQUIRED_COLS = {"close"}

    def __init__(
        self,
        pattern_window: int = 20,
        n_segments: int = 5,
        alphabet_size: int = 4,
    ) -> None:
        if alphabet_size not in _BREAKPOINTS:
            raise ValueError(
                f"alphabet_size must be one of {list(_BREAKPOINTS.keys())}, "
                f"got {alphabet_size}"
            )
        self.pattern_window = pattern_window
        self.n_segments = n_segments
        self.alphabet_size = alphabet_size
        self._breakpoints = _BREAKPOINTS[alphabet_size]
        self._letters = [chr(ord("a") + i) for i in range(alphabet_size)]
        # Effective window length (largest multiple of n_segments <= pattern_window)
        self._effective_window = (pattern_window // n_segments) * n_segments
        if self._effective_window == 0:
            self._effective_window = n_segments  # fallback

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def download_sax_data(
        self,
        start_date,
        end_date,
    ) -> pd.DataFrame:
        """
        SAX features are computed purely from price data; no download needed.

        Returns an empty DataFrame (interface compatibility with other feature classes).
        """
        return pd.DataFrame()

    def create_sax_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add SAX pattern features to df.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain a 'close' column.  All other columns are preserved.

        Returns
        -------
        pd.DataFrame
            Original df with 3 new sax_ columns appended:
              sax_pattern_20d, sax_pattern_match, sax_novelty
        """
        if "close" not in df.columns:
            logger.warning("SAXFeatures: 'close' column missing — skipping")
            return df

        out = df.copy()
        close = out["close"].values.astype(float)
        n = len(close)

        # Daily log-returns (handles price→return conversion cleanly)
        with np.errstate(divide="ignore", invalid="ignore"):
            rets = np.diff(np.log(np.where(close > 0, close, np.nan)))
        rets = np.concatenate([[0.0], rets])  # length stays n

        w = self._effective_window

        # Pre-allocate output arrays
        pattern_hash = np.zeros(n, dtype=np.float64)
        pattern_match = np.zeros(n, dtype=np.float64)
        novelty = np.zeros(n, dtype=np.float64)

        # Letter frequency history for novelty: shape (n, alphabet_size)
        letter_freqs = np.zeros((n, self.alphabet_size), dtype=np.float64)

        for i in range(w - 1, n):
            window = rets[i - w + 1 : i + 1]  # length = w

            # ---------- SAX encoding ----------
            word = self._encode_window(window)
            if word is None:
                continue

            # sax_pattern_20d: compact integer hash
            pattern_hash[i] = hash(word) % 10_000

            # sax_pattern_match: directional bias
            tail = word[-2:] if len(word) >= 2 else word
            if tail in ("cd", "dd"):
                pattern_match[i] = 1.0
            elif tail in ("ab", "aa"):
                pattern_match[i] = -1.0
            else:
                pattern_match[i] = 0.0

            # letter frequencies for this 20d window (for novelty computation)
            freq_20 = self._letter_freq(word)
            letter_freqs[i] = freq_20

        # ---------- sax_novelty ----------
        # Use a 50-day rolling historical distribution vs most-recent 20d.
        # For each position i: p = letter_freqs[i], q = mean(letter_freqs[i-49:i])
        hist_window = 50
        for i in range(hist_window - 1, n):
            p = letter_freqs[i]  # recent 20d distribution
            # Historical: mean letter freq over the past 50 positions
            slice_50 = letter_freqs[max(0, i - hist_window + 1) : i + 1]
            q = slice_50.mean(axis=0)

            novelty[i] = self._kl_divergence(p, q)

        # Fill early rows (< hist_window) with 0; already zero-initialised.

        # Assign to df (fill any remaining NaN → 0)
        out["sax_pattern_20d"] = pattern_hash
        out["sax_pattern_match"] = pattern_match
        out["sax_novelty"] = novelty

        # Safety: no NaN or Inf
        for col in ("sax_pattern_20d", "sax_pattern_match", "sax_novelty"):
            out[col] = out[col].replace([np.inf, -np.inf], 0.0).fillna(0.0)

        return out

    def analyze_current_pattern(self, df: pd.DataFrame) -> Optional[Dict]:
        """
        Summarise the latest SAX regime and novelty level.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain sax_ columns (run create_sax_features first) OR at
            minimum a 'close' column (features will be computed on-the-fly).

        Returns
        -------
        dict with keys:
          pattern_regime : "BULLISH" | "BEARISH" | "NEUTRAL"
          novelty_level  : "HIGH" | "LOW" | "NORMAL"
          sax_word       : str  — current 5-letter SAX word (empty if unavailable)
          sax_hash       : int
          novelty_score  : float
        """
        # Ensure features are available
        if "sax_pattern_match" not in df.columns:
            if "close" not in df.columns:
                return None
            df = self.create_sax_features(df)
            if "sax_pattern_match" not in df.columns:
                return None

        if df.empty:
            return None

        latest = df.iloc[-1]
        pm = latest.get("sax_pattern_match", 0.0)
        nov = latest.get("sax_novelty", 0.0)
        ph = int(latest.get("sax_pattern_20d", 0))

        # Pattern regime
        if pm == 1.0:
            regime = "BULLISH"
        elif pm == -1.0:
            regime = "BEARISH"
        else:
            regime = "NEUTRAL"

        # Novelty level: use rolling 50-day 75th/25th percentile if enough rows,
        # otherwise use fixed thresholds.
        nov_series = df["sax_novelty"].dropna()
        if len(nov_series) >= 50:
            q75 = nov_series.quantile(0.75)
            q25 = nov_series.quantile(0.25)
            if nov > q75:
                nov_level = "HIGH"
            elif nov < q25:
                nov_level = "LOW"
            else:
                nov_level = "NORMAL"
        else:
            if nov > 0.05:
                nov_level = "HIGH"
            elif nov < 0.001:
                nov_level = "LOW"
            else:
                nov_level = "NORMAL"

        # Reconstruct the SAX word for the last window (best-effort)
        close = df["close"].values.astype(float) if "close" in df.columns else None
        sax_word = ""
        if close is not None and len(close) >= self._effective_window:
            w = self._effective_window
            window = np.log(np.where(close[-w:] > 0, close[-w:], np.nan))
            window = np.diff(np.concatenate([[window[0]], window]))
            word = self._encode_window(window)
            if word:
                sax_word = word

        return {
            "pattern_regime": regime,
            "novelty_level": nov_level,
            "sax_word": sax_word,
            "sax_hash": ph,
            "novelty_score": float(nov),
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _encode_window(self, window: np.ndarray) -> Optional[str]:
        """
        Z-normalise → PAA → discretise → return SAX word string.

        Returns None if the window has zero variance (constant series).
        """
        w = len(window)
        if w == 0:
            return None

        # Z-normalise
        mu = np.nanmean(window)
        sigma = np.nanstd(window)
        if sigma < 1e-10:
            # Constant window → all mid-alphabet symbol
            mid = self._letters[self.alphabet_size // 2]
            return mid * self.n_segments

        z = (window - mu) / sigma

        # PAA: split into n_segments equal (or as-equal-as-possible) chunks
        seg_size = w / self.n_segments
        paa = np.empty(self.n_segments, dtype=np.float64)
        for s in range(self.n_segments):
            start = int(round(s * seg_size))
            end = int(round((s + 1) * seg_size))
            end = max(end, start + 1)
            end = min(end, w)
            paa[s] = np.nanmean(z[start:end])

        # Discretise PAA values to alphabet symbols
        word = "".join(self._discretise(v) for v in paa)
        return word

    def _discretise(self, value: float) -> str:
        """Map a single z-score to a SAX letter using the breakpoint table."""
        idx = np.searchsorted(self._breakpoints, value, side="right")
        return self._letters[idx]

    def _letter_freq(self, word: str) -> np.ndarray:
        """
        Compute normalised frequency vector for each letter in the word.

        Returns an array of length alphabet_size summing to 1.0.
        If the word is empty, returns a uniform distribution.
        """
        counts = np.zeros(self.alphabet_size, dtype=np.float64)
        for ch in word:
            idx = ord(ch) - ord("a")
            if 0 <= idx < self.alphabet_size:
                counts[idx] += 1.0
        total = counts.sum()
        if total == 0:
            return np.full(self.alphabet_size, 1.0 / self.alphabet_size)
        return counts / total

    @staticmethod
    def _kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
        """
        KL-like divergence sum(p * log(p/q)).

        Both p and q are probability vectors (non-negative, sum ~= 1).
        Undefined entries (q=0) are handled by using a small epsilon.
        Returns 0.0 if p is all-zero.
        """
        eps = 1e-12
        p = np.where(p < eps, eps, p)
        q = np.where(q < eps, eps, q)
        # Re-normalise to avoid drift from epsilon adjustments
        p = p / p.sum()
        q = q / q.sum()
        divergence = float(np.sum(p * np.log(p / q)))
        return max(0.0, divergence)  # numerical guard
