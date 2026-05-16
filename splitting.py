"""
splitting.py — Train / validation / test split utilities (student-implementable).

``split_data`` receives the label array ``y`` and, optionally, the full
DataFrame ``df`` (for group-aware splits).  It must return a list of
``(idx_train, idx_val, idx_test)`` tuples of integer index arrays.

Contract
--------
* ``idx_train``, ``idx_val``, ``idx_test`` are 1-D NumPy arrays of integer
  indices into the full dataset.
* ``idx_val`` may be ``None`` if no separate validation fold is needed.
* All indices must be non-overlapping; together they must cover every sample.
* Return a **list** — one element for a single split, K elements for k-fold.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split


def split_data(
    y: np.ndarray,
    df: pd.DataFrame | None = None,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
) -> list[tuple[np.ndarray, np.ndarray | None, np.ndarray]]:
    """Split dataset indices into train, validation, and test subsets.

    The default strategy first creates one stratified held-out test set using
    ``test_size``. It then applies stratified K-fold splitting to the remaining
    train/validation pool, with ``val_size`` used to choose the number of folds.

    Args:
        y:            Label array of shape ``(N,)`` with values in ``{0, 1}``.
                      Used for stratification.
        df:           Optional full DataFrame (same row order as ``y``).
                      Required for group-aware splits.
        test_size:    Fraction of samples reserved for the held-out test set.
        val_size:     Fraction of samples reserved for validation.
        random_state: Random seed for reproducible splits.

    Returns:
        A list of ``(idx_train, idx_val, idx_test)`` tuples. The test indices
        are fixed across folds, while train and validation rotate inside the
        non-test pool.

    Student task:
        Replace or extend the skeleton below.  The only contract is that the
        function returns the list described above.
    """

    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be between 0 and 1.")
    if not 0.0 < val_size < 1.0:
        raise ValueError("val_size must be between 0 and 1.")
    if test_size + val_size >= 1.0:
        raise ValueError("test_size + val_size must be less than 1.")

    idx = np.arange(len(y))
    y_int = y.astype(int)

    idx_train_val, idx_test = train_test_split(
        idx,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
        stratify=y_int,
    )

    y_train_val = y_int[idx_train_val]
    requested_splits = int(round((1.0 - test_size) / val_size))
    min_class_count = int(np.bincount(y_train_val, minlength=2).min())
    n_splits = max(2, min(requested_splits, min_class_count))

    splitter = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    splits: list[tuple[np.ndarray, np.ndarray | None, np.ndarray]] = []
    train_val_positions = np.arange(len(idx_train_val))
    for train_pos, val_pos in splitter.split(train_val_positions, y_train_val):
        idx_train = idx_train_val[train_pos]
        idx_val = idx_train_val[val_pos]
        splits.append((idx_train, idx_val, idx_test))

    return splits
