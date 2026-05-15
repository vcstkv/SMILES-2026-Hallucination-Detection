"""
probe.py — Hallucination probe classifier (student-implemented).

Implements ``HallucinationProbe``, a binary MLP that classifies feature
vectors as truthful (0) or hallucinated (1).  Called from ``solution.py``
via ``evaluate.run_evaluation``.  All four public methods (``fit``,
``fit_hyperparameters``, ``predict``, ``predict_proba``) must be implemented
and their signatures must not change.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler


class HallucinationProbe(nn.Module):
    """Binary classifier that detects hallucinations from hidden-state features.

    Extends ``torch.nn.Module``; the default architecture is a single
    hidden-layer MLP with ``StandardScaler`` pre-processing.  The network is
    built lazily in ``fit()`` once the feature dimension is known.
    """

    def __init__(self) -> None:
        super().__init__()
        self._net: nn.Sequential | None = None  # built lazily in fit()
        self._scaler = StandardScaler()
        self._pca: PCA | None = None
        self._linear_probe: LogisticRegression | None = None
        self._ensemble: list[tuple[PCA | None, LogisticRegression, float]] = []
        self._train_prior: float = 0.5
        self._threshold: float = 0.5  # tuned by fit_hyperparameters()

    # ------------------------------------------------------------------
    # STUDENT: Replace or extend the network definition below.
    # ------------------------------------------------------------------
    def _build_network(self, input_dim: int) -> None:
        """Instantiate the network layers.

        Called once at the start of ``fit()`` when ``input_dim`` is known.

        Args:
            input_dim: Feature vector dimensionality.
        """
        self._net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(64, 1),
        )

    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass — returns raw logits of shape ``(n_samples,)``.

        Args:
            x: Float tensor of shape ``(n_samples, feature_dim)``.

        Returns:
            1-D tensor of raw (pre-sigmoid) logits.
        """
        if self._net is None:
            raise RuntimeError(
                "Network has not been built yet. Call fit() before forward()."
            )
        return self._net(x).squeeze(-1)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HallucinationProbe":
        """Train the probe on labelled feature vectors.

        Scales features with ``StandardScaler``, builds the network if needed,
        and optimises with Adam + ``BCEWithLogitsLoss``.

        Args:
            X: Feature matrix of shape ``(n_samples, feature_dim)``.
            y: Integer label vector of shape ``(n_samples,)``; 0 = truthful,
               1 = hallucinated.

        Returns:
            ``self`` (for method chaining).
        """
        np.random.seed(42)
        torch.manual_seed(42)

        y_int = y.astype(int)
        self._train_prior = float(np.mean(y_int)) if len(y_int) else 0.5
        X_scaled = self._scaler.fit_transform(X)

        # ------------------------------------------------------------------
        # STUDENT: Replace or extend the training loop below.
        # ------------------------------------------------------------------
        self._ensemble = self._fit_logistic_ensemble(X_scaled, y_int)
        self._linear_probe = self._ensemble[0][1] if self._ensemble else None
        train_probs = self._predict_scaled_proba(X_scaled)
        self._threshold = self._prior_threshold(train_probs, self._train_prior)
        # ------------------------------------------------------------------

        self.eval()
        return self

    def _criterion_for(self, y_train: np.ndarray) -> nn.BCEWithLogitsLoss:
        n_pos = int(y_train.sum())
        n_neg = len(y_train) - n_pos
        pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32)
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def _train_for_epochs(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        epochs: int,
    ) -> None:
        X_t = torch.from_numpy(X_train.astype(np.float32)).float()
        y_t = torch.from_numpy(y_train.astype(np.float32))
        criterion = self._criterion_for(y_train)
        optimizer = torch.optim.AdamW(self.parameters(), lr=8e-4, weight_decay=5e-3)

        self.train()
        for _ in range(max(1, epochs)):
            optimizer.zero_grad()
            loss = criterion(self(X_t), y_t)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            optimizer.step()

    def _select_epoch_count(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> int:
        X_t = torch.from_numpy(X_train.astype(np.float32)).float()
        y_t = torch.from_numpy(y_train.astype(np.float32))
        X_val_t = torch.from_numpy(X_val.astype(np.float32)).float()
        y_val_t = torch.from_numpy(y_val.astype(np.float32))
        criterion = self._criterion_for(y_train)
        optimizer = torch.optim.AdamW(self.parameters(), lr=8e-4, weight_decay=5e-3)

        best_epoch = 80
        best_val_loss = float("inf")
        patience = 20
        stale_epochs = 0
        self.train()
        for epoch in range(1, 181):
            optimizer.zero_grad()
            loss = criterion(self(X_t), y_t)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            optimizer.step()

            self.eval()
            with torch.no_grad():
                val_loss = nn.BCEWithLogitsLoss()(self(X_val_t), y_val_t).item()
            self.train()

            if val_loss < best_val_loss - 1e-4:
                best_val_loss = val_loss
                best_epoch = epoch
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= patience:
                    break

        return best_epoch

    def _fit_linear_probe(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        C: float = 0.25,
    ) -> LogisticRegression | None:
        if len(np.unique(y_train)) < 2:
            return None
        model = LogisticRegression(
            C=C,
            class_weight="balanced",
            max_iter=2000,
            random_state=42,
            solver="liblinear",
        )
        model.fit(X_train, y_train)
        return model

    def _fit_logistic_ensemble(
        self,
        X_scaled: np.ndarray,
        y_train: np.ndarray,
    ) -> list[tuple[PCA | None, LogisticRegression, float]]:
        """Fit several regularized logistic probes on stable low-rank views."""
        if len(np.unique(y_train)) < 2:
            return []

        models: list[tuple[PCA | None, LogisticRegression, float]] = []
        max_components = min(X_scaled.shape[0] - 1, X_scaled.shape[1])
        component_grid = [32, 64, 96, 128]

        for n_components in component_grid:
            if n_components < 2 or n_components > max_components:
                continue
            pca = PCA(
                n_components=n_components,
                whiten=False,
                random_state=42 + n_components,
                svd_solver="randomized",
            )
            X_pca = pca.fit_transform(X_scaled)
            for C in (0.08, 0.2, 0.5):
                model = self._fit_linear_probe(X_pca, y_train, C=C)
                if model is not None:
                    # Lower-dimensional views are less brittle on this small
                    # dataset, so give them a modestly higher vote.
                    weight = 1.0 / np.sqrt(n_components / component_grid[0])
                    models.append((pca, model, float(weight)))

        raw_model = self._fit_linear_probe(X_scaled, y_train, C=0.05)
        if raw_model is not None:
            models.append((None, raw_model, 0.35))

        return models

    def _predict_transformed_proba(
        self,
        X_model: np.ndarray,
        linear_probe: LogisticRegression | None = None,
    ) -> np.ndarray:
        if linear_probe is not None:
            return linear_probe.predict_proba(X_model)[:, 1]
        return self._predict_scaled_proba(X_model)

    def _predict_scaled_proba(self, X_scaled: np.ndarray) -> np.ndarray:
        if not self._ensemble:
            return np.full(X_scaled.shape[0], self._train_prior, dtype=float)

        weighted_probs = np.zeros(X_scaled.shape[0], dtype=float)
        total_weight = 0.0
        for pca, model, weight in self._ensemble:
            X_view = pca.transform(X_scaled) if pca is not None else X_scaled
            weighted_probs += weight * model.predict_proba(X_view)[:, 1]
            total_weight += weight

        return weighted_probs / max(total_weight, 1e-12)

    def _prior_threshold(self, probs: np.ndarray, prior: float) -> float:
        if probs.size == 0:
            return 0.5
        prior = float(np.clip(prior, 0.01, 0.99))
        return float(np.quantile(probs, 1.0 - prior))

    def _best_threshold(self, probs: np.ndarray, y_true: np.ndarray) -> float:
        if len(y_true) < 8 or len(np.unique(y_true)) < 2:
            return self._prior_threshold(probs, self._train_prior)

        candidates = np.unique(np.concatenate([probs, np.linspace(0.02, 0.98, 97)]))

        best_accuracy = -1.0
        scores: list[tuple[float, float, float]] = []
        for t in candidates:
            y_pred_t = (probs >= t).astype(int)
            acc = accuracy_score(y_true, y_pred_t)
            f1 = f1_score(y_true, y_pred_t, zero_division=0)
            best_accuracy = max(best_accuracy, acc)
            scores.append((float(t), acc, f1))

        # Accuracy is the competition metric, but F1 should not be needlessly
        # sacrificed when several thresholds are effectively tied on accuracy.
        viable = [item for item in scores if item[1] >= best_accuracy - 0.01]
        best_threshold, _, _ = max(
            viable,
            key=lambda item: (
                item[2],
                item[1],
                -abs(item[0] - self._prior_threshold(probs, float(np.mean(y_true)))),
            ),
        )
        return float(best_threshold)

    def _augment_minority_class(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        """No-op retained for compatibility with earlier probe versions."""
        return X_train, y_train

    def fit_hyperparameters(
        self, X_val: np.ndarray, y_val: np.ndarray
    ) -> "HallucinationProbe":
        """Tune the decision threshold on a validation set to maximise accuracy.

        The chosen threshold is stored in ``self._threshold`` and used by
        subsequent ``predict`` calls.  Call this after ``fit`` and before
        ``predict``.

        Args:
            X_val: Validation feature matrix of shape
                   ``(n_val_samples, feature_dim)``.
            y_val: Integer label vector of shape ``(n_val_samples,)``;
                   0 = truthful, 1 = hallucinated.

        Returns:
            ``self`` (for method chaining).
        """
        probs = self.predict_proba(X_val)[:, 1]

        self._threshold = self._best_threshold(probs, y_val.astype(int))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary labels for feature vectors.

        Uses the decision threshold in ``self._threshold`` (default ``0.5``;
        updated by ``fit_hyperparameters``).

        Args:
            X: Feature matrix of shape ``(n_samples, feature_dim)``.

        Returns:
            Integer array of shape ``(n_samples,)`` with values in ``{0, 1}``.
        """
        return (self.predict_proba(X)[:, 1] >= self._threshold).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probability estimates.

        Args:
            X: Feature matrix of shape ``(n_samples, feature_dim)``.

        Returns:
            Array of shape ``(n_samples, 2)`` where column 1 contains the
            estimated probability of the hallucinated class (label 1).
            Used to compute AUROC.
        """
        X_scaled = self._scaler.transform(X)
        prob_pos = self._predict_scaled_proba(X_scaled)
        return np.stack([1.0 - prob_pos, prob_pos], axis=1)
