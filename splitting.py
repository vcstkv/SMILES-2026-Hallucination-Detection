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

    The default strategy performs shuffled stratified K-fold evaluation and
    carves a non-empty stratified validation set out of each training fold.

    Args:
        y:            Label array of shape ``(N,)`` with values in ``{0, 1}``.
                      Used for stratification.
        df:           Optional full DataFrame (same row order as ``y``).
                      Required for group-aware splits.
        test_size:    Fraction of samples reserved for the held-out test set.
        val_size:     Fraction of samples reserved for validation.
        random_state: Random seed for reproducible splits.

    Returns:
        A list of ``(idx_train, idx_val, idx_test)`` tuples of integer index
        arrays.  ``idx_val`` is populated whenever there are enough labelled
        samples to stratify a validation split.

    Student task:
        Replace or extend the skeleton below.  The only contract is that the
        function returns the list described above.
    """

    idx = np.arange(len(y))
    min_class_count = int(np.bincount(y.astype(int), minlength=2).min())
    n_splits = min(5, min_class_count)

    if n_splits < 2:
        return [(idx, None, np.array([], dtype=int))]

    splitter = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    splits: list[tuple[np.ndarray, np.ndarray | None, np.ndarray]] = []
    for fold_idx, (idx_train_val, idx_test) in enumerate(splitter.split(idx, y)):
        y_train_val = y[idx_train_val]
        val_fraction = val_size / (1.0 - (1.0 / n_splits))
        val_fraction = min(max(val_fraction, 1.0 / len(idx_train_val)), 0.5)

        idx_train, idx_val = train_test_split(
            idx_train_val,
            test_size=val_fraction,
            random_state=random_state + fold_idx,
            shuffle=True,
            stratify=y_train_val,
        )
        splits.append((idx_train, idx_val, idx_test))

    return splits
