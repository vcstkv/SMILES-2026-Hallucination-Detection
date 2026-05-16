"""
probe.py — Hallucination probe classifier (student-implemented).

Implements ``HallucinationProbe``, a binary classifier that classifies feature
vectors as truthful (0) or hallucinated (1).  Called from ``solution.py``
via ``evaluate.run_evaluation``.  All four public methods (``fit``,
``fit_hyperparameters``, ``predict``, ``predict_proba``) must be implemented
and their signatures must not change.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression


@dataclass
class ProbeModel:
    kind: str
    selected_idx: torch.Tensor
    vote_weight: float
    linear_weight: torch.Tensor | None = None
    bias: torch.Tensor | None = None


class HallucinationProbe(nn.Module):
    """Binary classifier that detects hallucinations from hidden-state features.

    Extends ``torch.nn.Module`` and keeps the public probe API expected by
    ``evaluate.py``.  The active classifier is a small ensemble of regularized
    logistic probes over compact supervised feature subsets.
    """

    TOPK_SIZES = (32,)
    TOPK_C_VALUES = (0.03,)
    THRESHOLD = 0.42

    def __init__(self) -> None:
        super().__init__()
        self._mean: torch.Tensor | None = None
        self._std: torch.Tensor | None = None
        self._ensemble: list[ProbeModel] = []
        self._train_prior: float = 0.5
        self._threshold: float = self.THRESHOLD

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass — returns raw logits of shape ``(n_samples,)``.

        Args:
            x: Float tensor of shape ``(n_samples, feature_dim)``.

        Returns:
            1-D tensor of raw logits.
        """
        if self._mean is None or self._std is None:
            raise RuntimeError(
                "Probe has not been fitted yet. Call fit() before forward()."
            )
        device = x.device
        dtype = x.dtype
        x_cpu = x.detach().cpu().to(torch.float64)
        x_scaled = (x_cpu - self._mean) / self._std
        probs = np.clip(self._predict_scaled_proba(x_scaled), 1e-12, 1.0 - 1e-12)
        logits = np.log(probs / (1.0 - probs))
        return torch.from_numpy(logits).to(device=device, dtype=dtype)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HallucinationProbe":
        """Train the probe on labelled feature vectors.

        Scales features, chooses compact feature subsets, and fits a
        deterministic ensemble of regularized linear probes.

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

        self._threshold = self.THRESHOLD
        self._ensemble = self._fit_logistic_ensemble(X_scaled, y_int)
        self.eval()
        return self

    def _fit_linear_probe(
        self,
        X_train: torch.Tensor,
        y_train: np.ndarray,
        C: float,
        balance_classes: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        if len(np.unique(y_train)) < 2:
            return None

        class_weight = "balanced" if balance_classes else None
        classifier = LogisticRegression(
            C=C,
            solver="liblinear",
            max_iter=1000,
            class_weight=class_weight,
            random_state=42,
        )
        classifier.fit(X_train.numpy(), y_train.astype(int))
        weight = torch.from_numpy(classifier.coef_[0].astype(np.float64))
        bias = torch.tensor(float(classifier.intercept_[0]), dtype=torch.float64)
        return weight, bias

    def _fit_logistic_ensemble(
        self,
        X_scaled: torch.Tensor,
        y_train: np.ndarray,
    ) -> list[ProbeModel]:
        if len(np.unique(y_train)) < 2:
            return []

        models: list[ProbeModel] = []
        for selected_idx in self._supervised_feature_rankings(X_scaled, y_train):
            X_topk = X_scaled[:, selected_idx]
            for C in self.TOPK_C_VALUES:
                fitted = self._fit_linear_probe(
                    X_topk,
                    y_train,
                    C=C,
                    balance_classes=False,
                )
                if fitted is None:
                    continue
                models.append(
                    ProbeModel(
                        kind="linear",
                        selected_idx=selected_idx,
                        vote_weight=1.0,
                        linear_weight=fitted[0],
                        bias=fitted[1],
                    )
                )

        return models

    def _supervised_feature_rankings(
        self,
        X_scaled: torch.Tensor,
        y_train: np.ndarray,
    ) -> list[torch.Tensor]:
        y_t = torch.from_numpy(y_train.astype(bool))
        if y_t.sum() == 0 or (~y_t).sum() == 0:
            return []

        X_pos = X_scaled[y_t]
        X_neg = X_scaled[~y_t]
        n_pos = float(X_pos.shape[0])
        n_neg = float(X_neg.shape[0])
        mean_pos = X_pos.mean(dim=0)
        mean_neg = X_neg.mean(dim=0)
        grand_mean = X_scaled.mean(dim=0)
        between = n_pos * (mean_pos - grand_mean).square()
        between += n_neg * (mean_neg - grand_mean).square()
        within = X_pos.var(dim=0, unbiased=True) * max(n_pos - 1.0, 1.0)
        within += X_neg.var(dim=0, unbiased=True) * max(n_neg - 1.0, 1.0)
        f_scores = between / (within / max(n_pos + n_neg - 2.0, 1.0)).clamp_min(1e-8)
        f_scores = torch.nan_to_num(f_scores, nan=0.0, posinf=0.0, neginf=0.0)
        mean_diff_scores = (mean_pos - mean_neg).abs()

        rankings: list[torch.Tensor] = []
        seen_sizes: set[int] = set()
        seen_rankings: set[bytes] = set()
        max_features = int(X_scaled.shape[1])
        for scores in (f_scores, mean_diff_scores):
            for k in self.TOPK_SIZES:
                k_eff = min(k, max_features)
                if k_eff < 8 or (scores is f_scores and k_eff in seen_sizes):
                    continue
                selected = torch.topk(scores, k=k_eff, largest=True).indices
                selected = selected.sort().values.contiguous()
                ranking_key = selected.cpu().numpy().tobytes()
                if ranking_key in seen_rankings:
                    continue
                rankings.append(selected)
                seen_rankings.add(ranking_key)
                if scores is f_scores:
                    seen_sizes.add(k_eff)
        return rankings

    def _predict_scaled_proba(self, X_scaled: torch.Tensor) -> np.ndarray:
        if not self._ensemble:
            return np.full(X_scaled.shape[0], self._train_prior, dtype=float)

        weighted_probs = torch.zeros(X_scaled.shape[0], dtype=torch.float64)
        total_weight = 0.0
        for model in self._ensemble:
            weighted_probs += model.vote_weight * self._predict_single_model_proba(
                X_scaled,
                model,
            )
            total_weight += model.vote_weight

        return (weighted_probs / max(total_weight, 1e-12)).numpy()

    def _predict_single_model_proba(
        self,
        X_scaled: torch.Tensor,
        model: ProbeModel,
    ) -> torch.Tensor:
        X_view = X_scaled[:, model.selected_idx]
        if model.kind == "linear":
            if model.linear_weight is None or model.bias is None:
                raise RuntimeError("Linear probe is missing parameters.")
            logits = X_view @ model.linear_weight + model.bias
            return torch.sigmoid(logits)

        raise RuntimeError(f"Unknown probe model kind: {model.kind}")

    def fit_hyperparameters(
        self, X_val: np.ndarray, y_val: np.ndarray
    ) -> "HallucinationProbe":
        """Keep the fixed decision threshold selected during local experiments.

        Args:
            X_val: Validation feature matrix of shape
                   ``(n_val_samples, feature_dim)``.
            y_val: Integer label vector of shape ``(n_val_samples,)``;
                   0 = truthful, 1 = hallucinated.

        Returns:
            ``self`` (for method chaining).
        """
        del X_val, y_val
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary labels for feature vectors.

        Uses the decision threshold in ``self._threshold``.

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
