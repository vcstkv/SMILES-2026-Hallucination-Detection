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
import torch.nn.functional as F


class HallucinationProbe(nn.Module):
    """Binary classifier that detects hallucinations from hidden-state features.

    Extends ``torch.nn.Module`` and keeps the public probe API expected by
    ``evaluate.py``.  The active classifier is a torch-native ensemble of
    regularized linear probes over scaled PCA and raw feature views.
    """

    def __init__(self) -> None:
        super().__init__()
        self._net: nn.Sequential | None = None  # built lazily in fit()
        self._mean: torch.Tensor | None = None
        self._std: torch.Tensor | None = None
        self._pca_basis: torch.Tensor | None = None
        self._ensemble: list[tuple[torch.Tensor | None, torch.Tensor, torch.Tensor, float]] = []
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

        Scales features, builds PCA views, and fits a deterministic ensemble of
        regularized torch linear probes.

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
        X_t = torch.from_numpy(X.astype(np.float64))
        self._mean = X_t.mean(dim=0, keepdim=True)
        self._std = X_t.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
        X_scaled = (X_t - self._mean) / self._std

        # ------------------------------------------------------------------
        # STUDENT: Replace or extend the training loop below.
        # ------------------------------------------------------------------
        self._ensemble = self._fit_logistic_ensemble(X_scaled, y_int)
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
        X_train: torch.Tensor,
        y_train: np.ndarray,
        C: float = 0.25,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        if len(np.unique(y_train)) < 2:
            return None

        torch.manual_seed(42)
        X_view = X_train.double()
        y_t = torch.from_numpy(y_train.astype(np.float64))
        n_pos = float(y_t.sum().item())
        n_neg = float(len(y_t) - n_pos)
        if n_pos == 0.0 or n_neg == 0.0:
            return None

        sample_weights = torch.where(
            y_t > 0.5,
            torch.full_like(y_t, len(y_t) / (2.0 * n_pos)),
            torch.full_like(y_t, len(y_t) / (2.0 * n_neg)),
        )

        weight = torch.zeros(X_view.shape[1], dtype=torch.float64, requires_grad=True)
        bias = torch.zeros((), dtype=torch.float64, requires_grad=True)
        l2_strength = 1.0 / max(C, 1e-6)

        def objective() -> torch.Tensor:
            logits = X_view @ weight + bias
            loss = F.binary_cross_entropy_with_logits(
                logits,
                y_t,
                weight=sample_weights,
                reduction="mean",
            )
            return loss + 0.5 * l2_strength * weight.square().sum() / len(y_t)

        optimizer = torch.optim.LBFGS(
            [weight, bias],
            lr=1.0,
            max_iter=300,
            history_size=20,
            line_search_fn="strong_wolfe",
        )

        try:
            def closure() -> torch.Tensor:
                optimizer.zero_grad()
                loss = objective()
                loss.backward()
                return loss

            optimizer.step(closure)
        except RuntimeError:
            optimizer = torch.optim.AdamW([weight, bias], lr=0.05, weight_decay=0.0)
            for _ in range(400):
                optimizer.zero_grad()
                loss = objective()
                loss.backward()
                optimizer.step()

        return weight.detach(), bias.detach()

    def _fit_logistic_ensemble(
        self,
        X_scaled: torch.Tensor,
        y_train: np.ndarray,
    ) -> list[tuple[torch.Tensor | None, torch.Tensor, torch.Tensor, float]]:
        """Fit several regularized logistic probes on stable low-rank views."""
        if len(np.unique(y_train)) < 2:
            return []

        models: list[tuple[torch.Tensor | None, torch.Tensor, torch.Tensor, float]] = []
        max_components = min(X_scaled.shape[0] - 1, X_scaled.shape[1])
        component_grid = [32, 64, 96, 128]
        max_pca_components = min(max(component_grid), max_components)

        if max_pca_components >= 2:
            self._pca_basis = self._compute_pca_basis(X_scaled, max_pca_components)
        else:
            self._pca_basis = None

        for n_components in component_grid:
            if n_components < 2 or n_components > max_components:
                continue
            if self._pca_basis is None:
                continue
            basis = self._pca_basis[:, :n_components].contiguous()
            X_pca = X_scaled @ basis
            for C in (0.08, 0.2, 0.5):
                fitted = self._fit_linear_probe(X_pca, y_train, C=C)
                if fitted is not None:
                    # Lower-dimensional views are less brittle on this small
                    # dataset, so give them a modestly higher vote.
                    vote_weight = 1.0 / np.sqrt(n_components / component_grid[0])
                    models.append((basis, fitted[0], fitted[1], float(vote_weight)))

        fitted_raw = self._fit_linear_probe(X_scaled, y_train, C=0.05)
        if fitted_raw is not None:
            models.append((None, fitted_raw[0], fitted_raw[1], 0.35))

        return models

    def _compute_pca_basis(self, X_scaled: torch.Tensor, n_components: int) -> torch.Tensor:
        X_centered = X_scaled - X_scaled.mean(dim=0, keepdim=True)
        _, _, vh = torch.linalg.svd(X_centered, full_matrices=False)
        return vh[:n_components].T.contiguous()

    def _predict_transformed_proba(
        self,
        X_model: torch.Tensor,
        linear_probe: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> np.ndarray:
        if linear_probe is not None:
            weight, bias = linear_probe
            return torch.sigmoid(X_model @ weight + bias).numpy()
        return self._predict_scaled_proba(X_model)

    def _predict_scaled_proba(self, X_scaled: torch.Tensor) -> np.ndarray:
        if not self._ensemble:
            return np.full(X_scaled.shape[0], self._train_prior, dtype=float)

        weighted_probs = torch.zeros(X_scaled.shape[0], dtype=torch.float64)
        total_weight = 0.0
        for basis, linear_weight, bias, vote_weight in self._ensemble:
            X_view = X_scaled @ basis if basis is not None else X_scaled
            weighted_probs += vote_weight * torch.sigmoid(X_view @ linear_weight + bias)
            total_weight += vote_weight

        return (weighted_probs / max(total_weight, 1e-12)).numpy()

    def _prior_threshold(self, probs: np.ndarray, prior: float) -> float:
        if probs.size == 0:
            return 0.5
        prior = float(np.clip(prior, 0.01, 0.99))
        return float(np.quantile(probs, 1.0 - prior))

    def _accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.mean(y_true.astype(int) == y_pred.astype(int)))

    def _f1(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_true = y_true.astype(int)
        y_pred = y_pred.astype(int)
        tp = int(np.sum((y_true == 1) & (y_pred == 1)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))
        denom = (2 * tp) + fp + fn
        return float((2 * tp) / denom) if denom else 0.0

    def _best_threshold(self, probs: np.ndarray, y_true: np.ndarray) -> float:
        if len(y_true) < 8 or len(np.unique(y_true)) < 2:
            return self._prior_threshold(probs, self._train_prior)

        candidates = np.unique(np.concatenate([probs, np.linspace(0.02, 0.98, 97)]))

        best_accuracy = -1.0
        scores: list[tuple[float, float, float]] = []
        for t in candidates:
            y_pred_t = (probs >= t).astype(int)
            acc = self._accuracy(y_true, y_pred_t)
            f1 = self._f1(y_true, y_pred_t)
            best_accuracy = max(best_accuracy, acc)
            scores.append((float(t), acc, f1))

        # Accuracy is the competition metric, but F1 should not be needlessly
        # sacrificed when several thresholds are effectively tied on accuracy.
        # Validation folds are small, so exact max-accuracy thresholds can
        # overfit by predicting too few positives.  Keep near-optimal accuracy
        # thresholds and prefer the one that preserves positive-class recall.
        viable = [item for item in scores if item[1] >= best_accuracy - 0.03]
        prior_threshold = self._prior_threshold(
            probs,
            min(self._train_prior + 0.05, 0.95),
        )
        best_threshold, _, _ = max(
            viable,
            key=lambda item: (
                item[2],
                item[1],
                -abs(item[0] - prior_threshold),
            ),
        )
        return float(min(best_threshold, prior_threshold))

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
        if self._mean is None or self._std is None:
            raise RuntimeError("Probe has not been fitted yet.")
        X_t = torch.from_numpy(X.astype(np.float64))
        X_scaled = (X_t - self._mean) / self._std
        prob_pos = self._predict_scaled_proba(X_scaled)
        return np.stack([1.0 - prob_pos, prob_pos], axis=1)
