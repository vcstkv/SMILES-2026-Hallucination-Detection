"""
probe.py — Hallucination probe classifier (student-implemented).

Implements ``HallucinationProbe``, a binary classifier that classifies feature
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

ProbeModel = tuple[str, torch.Tensor, torch.Tensor, torch.Tensor, float]


class HallucinationProbe(nn.Module):
    """Binary classifier that detects hallucinations from hidden-state features.

    Extends ``torch.nn.Module`` and keeps the public probe API expected by
    ``evaluate.py``.  The active classifier is a torch-native ensemble of
    regularized linear probes over scaled PCA and selected feature views.
    """

    PCA_COMPONENTS = (16, 32, 64, 96, 128)
    PCA_C_VALUES = (0.02, 0.04, 0.08, 0.16, 0.32)
    TOPK_SIZES = (64, 128, 256, 512, 1024)
    TOPK_C_VALUES = (0.005, 0.01)
    THRESHOLD = 0.45

    def __init__(self) -> None:
        super().__init__()
        self._mean: torch.Tensor | None = None
        self._std: torch.Tensor | None = None
        self._pca_basis: torch.Tensor | None = None
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

        torch.manual_seed(42)
        X_view = X_train.double()
        y_t = torch.from_numpy(y_train.astype(np.float64))
        n_pos = float(y_t.sum().item())
        n_neg = float(len(y_t) - n_pos)
        if n_pos == 0.0 or n_neg == 0.0:
            return None

        if balance_classes:
            sample_weights = torch.where(
                y_t > 0.5,
                torch.full_like(y_t, len(y_t) / (2.0 * n_pos)),
                torch.full_like(y_t, len(y_t) / (2.0 * n_neg)),
            )
        else:
            sample_weights = torch.ones_like(y_t)

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
    ) -> list[ProbeModel]:
        if len(np.unique(y_train)) < 2:
            return []

        models: list[ProbeModel] = []
        max_components = min(X_scaled.shape[0] - 1, X_scaled.shape[1])
        max_pca_components = min(max(self.PCA_COMPONENTS), max_components)

        if max_pca_components >= 2:
            self._pca_basis = self._compute_pca_basis(X_scaled, max_pca_components)
        else:
            self._pca_basis = None

        if self._pca_basis is not None:
            for n_components in self.PCA_COMPONENTS:
                if n_components < 2 or n_components > max_components:
                    continue
                basis = self._pca_basis[:, :n_components].contiguous()
                X_pca = X_scaled @ basis
                for C in self.PCA_C_VALUES:
                    fitted = self._fit_linear_probe(X_pca, y_train, C=C)
                    if fitted is None:
                        continue
                    vote_weight = 0.05 / np.sqrt(n_components / self.PCA_COMPONENTS[0])
                    models.append(
                        ("pca", basis, fitted[0], fitted[1], float(vote_weight))
                    )

        for selected_idx in self._supervised_feature_rankings(X_scaled, y_train):
            n_selected = int(selected_idx.numel())
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
                size_weight = min(max(n_selected, 1) / 1024.0, 1.0)
                vote_weight = 4.0 * size_weight * size_weight
                models.append(
                    ("select", selected_idx, fitted[0], fitted[1], float(vote_weight))
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
        scores = between / (within / max(n_pos + n_neg - 2.0, 1.0)).clamp_min(1e-8)
        scores = torch.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)

        rankings: list[torch.Tensor] = []
        seen_sizes: set[int] = set()
        max_features = int(X_scaled.shape[1])
        for k in self.TOPK_SIZES:
            k_eff = min(k, max_features)
            if k_eff < 8 or k_eff in seen_sizes:
                continue
            selected = torch.topk(scores, k=k_eff, largest=True).indices
            rankings.append(selected.sort().values.contiguous())
            seen_sizes.add(k_eff)
        return rankings

    def _compute_pca_basis(self, X_scaled: torch.Tensor, n_components: int) -> torch.Tensor:
        X_centered = X_scaled - X_scaled.mean(dim=0, keepdim=True)
        _, _, vh = torch.linalg.svd(X_centered, full_matrices=False)
        return vh[:n_components].T.contiguous()

    def _predict_scaled_proba(self, X_scaled: torch.Tensor) -> np.ndarray:
        if not self._ensemble:
            return np.full(X_scaled.shape[0], self._train_prior, dtype=float)

        weighted_probs = torch.zeros(X_scaled.shape[0], dtype=torch.float64)
        total_weight = 0.0
        for model in self._ensemble:
            vote_weight = model[4]
            weighted_probs += vote_weight * torch.from_numpy(
                self._predict_single_model_proba(X_scaled, model)
            )
            total_weight += vote_weight

        return (weighted_probs / max(total_weight, 1e-12)).numpy()

    def _predict_single_model_proba(
        self,
        X_scaled: torch.Tensor,
        model: ProbeModel,
    ) -> np.ndarray:
        view_kind, transform, linear_weight, bias, _ = model
        if view_kind == "pca":
            X_view = X_scaled @ transform
        elif view_kind == "select":
            X_view = X_scaled[:, transform.long()]
        else:
            raise RuntimeError(f"Unknown probe view kind: {view_kind}")
        return torch.sigmoid(X_view @ linear_weight + bias).numpy()

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
