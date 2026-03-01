"""
Recurrence Quantification Analysis (RQA) Features — phase-space dynamics of price returns.

RQA characterizes the hidden structure of nonlinear dynamical systems by analysing the
recurrence plot (RP) of a time series embedded in phase space.  Five complementary
quantities are extracted, each capturing a different facet of market micro-structure:

  rqa_recurrence_rate   — Fraction of phase-space points that recur (predictability proxy)
  rqa_determinism       — Fraction of recurrence points on diagonal lines (deterministic structure)
  rqa_laminarity        — Fraction of recurrence points on vertical lines (market "stickiness")
  rqa_entropy           — Shannon entropy of diagonal line-length distribution (complexity)
  rqa_trapping_time     — Mean vertical line length (average time system stays in a state)

All features use a rolling 50-day window, embedding dimension 3, and time delay 1.
Pure-numpy implementation — no PyRQA dependency.

Reference:
  Zbilut & Webber (1992) "Embeddings and delays as derived from quantification of
  recurrence plots". Physics Letters A.
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ─── Module-level helpers ─────────────────────────────────────────────────────


def _phase_space_embed(series: np.ndarray, dim: int, delay: int) -> np.ndarray:
    """
    Embed a 1-D series into a *dim*-dimensional phase-space matrix.

    Parameters
    ----------
    series : np.ndarray
        1-D input of length N.
    dim : int
        Embedding dimension (number of columns in the output).
    delay : int
        Time delay between successive dimensions.

    Returns
    -------
    np.ndarray
        Shape ``(M, dim)`` where ``M = N - (dim - 1) * delay``.
    """
    n = len(series)
    m = n - (dim - 1) * delay
    if m <= 0:
        return np.empty((0, dim))
    embedded = np.empty((m, dim))
    for d in range(dim):
        embedded[:, d] = series[d * delay : d * delay + m]
    return embedded


def _recurrence_matrix(embedded: np.ndarray, epsilon: float) -> np.ndarray:
    """
    Compute the binary recurrence matrix R where R[i,j]=1 iff ||x_i - x_j|| < epsilon.

    Uses vectorised broadcasting — O(M^2) memory and computation.

    Parameters
    ----------
    embedded : np.ndarray
        Shape ``(M, dim)`` from :func:`_phase_space_embed`.
    epsilon : float
        Threshold distance.

    Returns
    -------
    np.ndarray
        Boolean matrix of shape ``(M, M)``.
    """
    # Squared Euclidean distances via broadcasting
    diff = embedded[:, np.newaxis, :] - embedded[np.newaxis, :, :]  # (M,M,dim)
    sq_dist = np.sum(diff ** 2, axis=2)  # (M,M)
    return sq_dist < epsilon ** 2


def _rqa_measures(R: np.ndarray, min_line: int = 2) -> Dict[str, float]:
    """
    Extract RQA scalar measures from a binary recurrence matrix.

    Parameters
    ----------
    R : np.ndarray
        Boolean square matrix.
    min_line : int
        Minimum line length to count (default 2).

    Returns
    -------
    dict with keys:
        recurrence_rate, determinism, laminarity, entropy, trapping_time
    """
    m = R.shape[0]
    if m == 0:
        return {
            "recurrence_rate": 0.0,
            "determinism": 0.0,
            "laminarity": 0.0,
            "entropy": 0.0,
            "trapping_time": 0.0,
        }

    total_points = m * m  # Include the main diagonal

    # ── Recurrence rate ──────────────────────────────────────────────────
    rr = float(np.sum(R)) / total_points

    # ── Diagonal lines (determinism + entropy) ──────────────────────────
    # Iterate over all diagonals (offset k from -m+1 to m-1)
    diag_line_lengths: List[int] = []
    diag_recurrence_count = 0  # Points that belong to a line of length >= min_line

    for k in range(-(m - 1), m):
        diag = np.diagonal(R, offset=k)
        # Extract run lengths of consecutive True values
        runs = _run_lengths(diag)
        for length in runs:
            diag_recurrence_count += length
            if length >= min_line:
                diag_line_lengths.append(length)

    # DET = (points in lines of length >= min_line) / (all recurrence points)
    all_recurrent = int(np.sum(R))
    if all_recurrent > 0:
        det_numerator = sum(
            l for l in diag_line_lengths
        )
        det = det_numerator / all_recurrent
    else:
        det = 0.0

    # ENT = Shannon entropy of diagonal line-length distribution
    if diag_line_lengths:
        lengths_arr = np.array(diag_line_lengths, dtype=float)
        unique_lengths, counts = np.unique(lengths_arr, return_counts=True)
        probs = counts / counts.sum()
        ent = float(-np.sum(probs * np.log(probs + 1e-15)))
    else:
        ent = 0.0

    # ── Vertical lines (laminarity + trapping time) ──────────────────────
    vert_line_lengths: List[int] = []
    for col_idx in range(m):
        col = R[:, col_idx]
        runs = _run_lengths(col)
        for length in runs:
            if length >= min_line:
                vert_line_lengths.append(length)

    if all_recurrent > 0:
        lam_numerator = sum(vert_line_lengths)
        lam = lam_numerator / all_recurrent
    else:
        lam = 0.0

    if vert_line_lengths:
        tt = float(np.mean(vert_line_lengths))
    else:
        tt = 0.0

    return {
        "recurrence_rate": float(np.clip(rr, 0.0, 1.0)),
        "determinism": float(np.clip(det, 0.0, 1.0)),
        "laminarity": float(np.clip(lam, 0.0, 1.0)),
        "entropy": float(np.clip(ent, 0.0, None)),
        "trapping_time": float(np.clip(tt, 0.0, None)),
    }


def _run_lengths(arr: np.ndarray) -> List[int]:
    """
    Return list of run lengths of consecutive True values in a boolean array.

    Parameters
    ----------
    arr : np.ndarray
        1-D boolean array.

    Returns
    -------
    List[int]
        E.g. ``[T,T,F,T,T,T]`` → ``[2, 3]``.
    """
    if len(arr) == 0:
        return []
    runs: List[int] = []
    current = 0
    for val in arr:
        if val:
            current += 1
        else:
            if current > 0:
                runs.append(current)
            current = 0
    if current > 0:
        runs.append(current)
    return runs


# ─── Main class ──────────────────────────────────────────────────────────────


class RQAFeatures:
    """
    Compute Recurrence Quantification Analysis features from daily OHLCV data.

    All five features use a rolling window with Takens-style phase-space embedding.
    The computation is pure-numpy with O(n * window^2) complexity — acceptable for
    daily data (a 50-day window produces a 48×48 recurrence matrix per step).

    Parameters
    ----------
    window : int
        Rolling lookback in trading days (default 50).
    embedding_dim : int
        Phase-space embedding dimension (default 3).
    delay : int
        Time delay for the embedding (default 1).
    epsilon_factor : float
        Threshold = ``epsilon_factor * std(returns_in_window)`` (default 0.5).
    min_line : int
        Minimum diagonal/vertical line length for DET/LAM/TT (default 2).
    """

    REQUIRED_COLS = {"close"}

    def __init__(
        self,
        window: int = 50,
        embedding_dim: int = 3,
        delay: int = 1,
        epsilon_factor: float = 0.5,
        min_line: int = 2,
    ) -> None:
        self.window = window
        self.embedding_dim = embedding_dim
        self.delay = delay
        self.epsilon_factor = epsilon_factor
        self.min_line = min_line

    # ── Public API ───────────────────────────────────────────────────────────

    def download_rqa_data(self, start_date, end_date) -> pd.DataFrame:
        """
        RQA features are computed entirely from the SPY close price.
        No external data download is needed — return an empty DataFrame.

        Parameters
        ----------
        start_date : str or datetime
            Start of the desired date range (unused).
        end_date : str or datetime
            End of the desired date range (unused).

        Returns
        -------
        pd.DataFrame
            Always empty — signals that no additional download is required.
        """
        logger.debug("RQAFeatures.download_rqa_data: no external data required")
        return pd.DataFrame()

    def create_rqa_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add RQA features to a daily OHLCV DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain a ``close`` column.  All other columns are preserved.

        Returns
        -------
        pd.DataFrame
            Original ``df`` plus 5 new ``rqa_`` columns.  The first
            ``window`` rows will contain zeros (warm-up period).
        """
        result = df.copy()

        if "close" not in result.columns:
            logger.warning("RQAFeatures: 'close' column missing — skipping")
            return result

        close = result["close"].values.astype(float)
        n = len(close)

        # Compute log returns; replace NaN at index 0 with 0
        with np.errstate(divide="ignore", invalid="ignore"):
            log_rets = np.where(
                close[:-1] > 0,
                np.log(close[1:] / close[:-1]),
                0.0,
            )
        returns = np.concatenate([[0.0], log_rets])
        # Replace any remaining NaN / inf
        returns = np.where(np.isfinite(returns), returns, 0.0)

        # Pre-allocate output arrays with zeros (warm-up rows stay 0)
        rr_arr = np.zeros(n)
        det_arr = np.zeros(n)
        lam_arr = np.zeros(n)
        ent_arr = np.zeros(n)
        tt_arr = np.zeros(n)

        min_data = self.embedding_dim + self.window
        for i in range(self.window, n):
            window_returns = returns[i - self.window + 1 : i + 1]  # length = window

            # Skip windows with too little data or near-zero variance
            valid = window_returns[np.isfinite(window_returns)]
            if len(valid) < self.embedding_dim + self.delay + 1:
                continue

            std_val = float(np.std(valid))
            if std_val < 1e-15:
                # All identical values → 100% recurrence, but no structure
                rr_arr[i] = 1.0
                det_arr[i] = 1.0
                lam_arr[i] = 1.0
                continue

            epsilon = self.epsilon_factor * std_val

            # Phase-space embedding
            embedded = _phase_space_embed(valid, self.embedding_dim, self.delay)
            if len(embedded) < 2:
                continue

            # Recurrence matrix
            R = _recurrence_matrix(embedded, epsilon)

            # RQA measures
            measures = _rqa_measures(R, min_line=self.min_line)
            rr_arr[i] = measures["recurrence_rate"]
            det_arr[i] = measures["determinism"]
            lam_arr[i] = measures["laminarity"]
            ent_arr[i] = measures["entropy"]
            tt_arr[i] = measures["trapping_time"]

        result["rqa_recurrence_rate"] = rr_arr
        result["rqa_determinism"] = det_arr
        result["rqa_laminarity"] = lam_arr
        result["rqa_entropy"] = ent_arr
        result["rqa_trapping_time"] = tt_arr

        # Final cleanup — no NaN / inf allowed
        for col in self._all_feature_names():
            result[col] = (
                result[col]
                .fillna(0.0)
                .replace([np.inf, -np.inf], 0.0)
            )

        n_features = sum(1 for c in result.columns if c.startswith("rqa_"))
        logger.info(f"RQAFeatures: added {n_features} features (window={self.window})")
        return result

    def analyze_current_rqa(self, df: pd.DataFrame) -> Optional[Dict]:
        """
        Classify the most recent RQA observation into a qualitative regime.

        Regimes:
          PERIODIC    — High recurrence rate AND high determinism (cyclical market)
          LAMINAR     — High laminarity (market stuck / trending in one state)
          CHAOTIC     — Low determinism AND low laminarity (turbulent market)
          STOCHASTIC  — Low recurrence rate (essentially random / no structure)

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame that already has ``rqa_`` columns from
            :meth:`create_rqa_features`.

        Returns
        -------
        dict or None
            ``None`` if the required columns are absent.
        """
        required = {"rqa_recurrence_rate", "rqa_determinism", "rqa_laminarity"}
        if not required.issubset(df.columns) or len(df) < 2:
            return None

        last = df.iloc[-1]
        rr = float(last.get("rqa_recurrence_rate", 0.0))
        det = float(last.get("rqa_determinism", 0.0))
        lam = float(last.get("rqa_laminarity", 0.0))
        ent = float(last.get("rqa_entropy", 0.0))
        tt = float(last.get("rqa_trapping_time", 0.0))

        # Classify regime
        if rr < 0.05:
            rqa_regime = "STOCHASTIC"
        elif lam > 0.6:
            rqa_regime = "LAMINAR"
        elif rr > 0.2 and det > 0.5:
            rqa_regime = "PERIODIC"
        else:
            rqa_regime = "CHAOTIC"

        return {
            "rqa_regime": rqa_regime,
            "rqa_recurrence_rate": round(rr, 4),
            "rqa_determinism": round(det, 4),
            "rqa_laminarity": round(lam, 4),
            "rqa_entropy": round(ent, 4),
            "rqa_trapping_time": round(tt, 4),
        }

    # ── Internal helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _all_feature_names() -> List[str]:
        return [
            "rqa_recurrence_rate",
            "rqa_determinism",
            "rqa_laminarity",
            "rqa_entropy",
            "rqa_trapping_time",
        ]
