"""
aggregation.py — Token aggregation strategy and feature extraction
               (student-implemented).

Converts per-token, per-layer hidden states from the extraction loop in
``solution.py`` into flat feature vectors for the probe classifier.

Two stages can be customised independently:

  1. ``aggregate`` — select layers and token positions, pool into a vector.
  2. ``extract_geometric_features`` — optional hand-crafted features
     (enabled by setting ``USE_GEOMETRIC = True`` in ``solution.py``).

Both stages are combined by ``aggregation_and_feature_extraction``, the
single entry point called from the notebook.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def selected_layer_indices(n_layers: int) -> list[int]:
    """Return stable cross-depth layer indices for a hidden-state stack."""
    candidates = [4, 8, 12, 16, 20, n_layers - 1]
    indices: list[int] = []
    for idx in candidates:
        idx = max(0, min(n_layers - 1, idx))
        if idx not in indices:
            indices.append(idx)
    return indices


def real_token_slice(attention_mask: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Return real-token positions and the final real-token index."""
    real_positions = attention_mask.nonzero(as_tuple=False).flatten()
    if real_positions.numel() == 0:
        real_positions = torch.tensor([0], device=attention_mask.device)
    return real_positions, int(real_positions[-1].item())


def recency_weighted_mean(layer_tokens: torch.Tensor) -> torch.Tensor:
    """Pool tokens with a smooth bias toward the answer tail."""
    if layer_tokens.size(0) == 1:
        return layer_tokens[0]
    weights = torch.linspace(
        0.25,
        1.0,
        steps=layer_tokens.size(0),
        dtype=layer_tokens.dtype,
        device=layer_tokens.device,
    )
    weights = weights / weights.sum().clamp_min(1e-12)
    return (layer_tokens * weights[:, None]).sum(dim=0)


def aggregate(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Convert per-token hidden states into a single feature vector.

    Args:
        hidden_states:  Tensor of shape ``(n_layers, seq_len, hidden_dim)``.
                        Layer index 0 is the token embedding; index -1 is the
                        final transformer layer.
        attention_mask: 1-D tensor of shape ``(seq_len,)`` with 1 for real
                        tokens and 0 for padding.

    Returns:
        A 1-D feature tensor of shape ``(hidden_dim,)`` or
        ``(k * hidden_dim,)`` if multiple layers are concatenated.

    Student task:
        Replace or extend the skeleton below with alternative layer selection,
        token pooling (mean, max, weighted), or multi-layer fusion strategies.
    """
    # ------------------------------------------------------------------
    # STUDENT: Replace or extend the aggregation below.
    # ------------------------------------------------------------------

    real_positions, last_pos = real_token_slice(attention_mask)
    real_positions = real_positions.to(hidden_states.device)
    tail4_positions = real_positions[-4:]
    tail8_positions = real_positions[-8:]
    tail16_positions = real_positions[-16:]
    tail32_positions = real_positions[-32:]

    features = []
    for layer_idx in selected_layer_indices(hidden_states.size(0)):
        layer = hidden_states[layer_idx].float()
        real_tokens = layer[real_positions]
        last_token = layer[last_pos]
        tail4_mean = layer[tail4_positions].mean(dim=0)
        tail8_mean = layer[tail8_positions].mean(dim=0)
        tail16_mean = layer[tail16_positions].mean(dim=0)
        tail32_mean = layer[tail32_positions].mean(dim=0)
        full_mean = real_tokens.mean(dim=0)
        recency_mean = recency_weighted_mean(real_tokens)
        features.extend(
            [
                last_token,
                tail4_mean,
                tail8_mean,
                tail16_mean,
                tail32_mean,
                full_mean,
                recency_mean,
                last_token - tail32_mean,
                recency_mean - full_mean,
            ]
        )

    return torch.cat(features, dim=0)
    # ------------------------------------------------------------------


def extract_geometric_features(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Extract hand-crafted geometric / statistical features from hidden states.

    Called only when ``USE_GEOMETRIC = True`` in ``solution.ipynb``.  The
    returned tensor is concatenated with the output of ``aggregate``.

    Args:
        hidden_states:  Tensor of shape ``(n_layers, seq_len, hidden_dim)``.
        attention_mask: 1-D tensor of shape ``(seq_len,)`` with 1 for real
                        tokens and 0 for padding.

    Returns:
        A 1-D float tensor of shape ``(n_geometric_features,)``.  The length
        must be the same for every sample.

    Student task:
        Replace the stub below.  Possible features: layer-wise activation
        norms, inter-layer cosine similarity (representation drift), or
        sequence length.
    """
    # ------------------------------------------------------------------
    # STUDENT: Replace or extend the geometric feature extraction below.
    # ------------------------------------------------------------------

    real_positions, last_pos = real_token_slice(attention_mask)
    real_positions = real_positions.to(hidden_states.device)
    tail4_positions = real_positions[-4:]
    tail8_positions = real_positions[-8:]
    tail16_positions = real_positions[-16:]
    tail32_positions = real_positions[-32:]
    selected = selected_layer_indices(hidden_states.size(0))

    real_len = float(real_positions.numel())
    max_model_len = 512.0
    is_truncated = float(real_len >= max_model_len)
    features = [
        torch.tensor(
            real_len / max_model_len,
            dtype=torch.float32,
            device=hidden_states.device,
        ),
        torch.tensor(
            is_truncated,
            dtype=torch.float32,
            device=hidden_states.device,
        ),
        torch.log1p(
            torch.tensor(real_len, dtype=torch.float32, device=hidden_states.device)
        ),
        torch.tensor(
            min(real_len, 8.0) / max(real_len, 1.0),
            dtype=torch.float32,
            device=hidden_states.device,
        ),
        torch.tensor(
            min(real_len, 32.0) / max(real_len, 1.0),
            dtype=torch.float32,
            device=hidden_states.device,
        ),
    ]

    last_vectors = []
    all_last_vectors = []
    all_tail_means = []
    all_full_means = []
    for layer_idx in selected:
        layer = hidden_states[layer_idx, real_positions].float()
        full_layer = hidden_states[layer_idx].float()
        last_token = full_layer[last_pos]
        tail4 = full_layer[tail4_positions]
        tail8 = full_layer[tail8_positions]
        tail16 = full_layer[tail16_positions]
        tail32 = full_layer[tail32_positions]
        tail4_mean = tail4.mean(dim=0)
        tail8_mean = tail8.mean(dim=0)
        tail16_mean = tail16.mean(dim=0)
        tail32_mean = tail32.mean(dim=0)
        full_mean = layer.mean(dim=0)
        recency_mean = recency_weighted_mean(layer)
        token_norms = layer.norm(dim=1)
        tail4_norms = tail4.norm(dim=1)
        tail8_norms = tail8.norm(dim=1)
        tail16_norms = tail16.norm(dim=1)
        tail32_norms = tail32.norm(dim=1)

        features.append(token_norms.mean())
        features.append(token_norms.std(unbiased=False))
        features.append(last_token.norm())
        features.append(tail4_norms.mean())
        features.append(tail4_norms.std(unbiased=False))
        features.append(tail8_norms.mean())
        features.append(tail8_norms.std(unbiased=False))
        features.append(tail16_norms.mean())
        features.append(tail16_norms.std(unbiased=False))
        features.append(tail32_norms.mean())
        features.append(tail32_norms.std(unbiased=False))
        features.append(F.cosine_similarity(last_token, tail4_mean, dim=0))
        features.append(F.cosine_similarity(last_token, tail8_mean, dim=0))
        features.append(F.cosine_similarity(last_token, tail16_mean, dim=0))
        features.append(F.cosine_similarity(last_token, tail32_mean, dim=0))
        features.append(F.cosine_similarity(last_token, recency_mean, dim=0))
        features.append(F.cosine_similarity(tail4_mean, tail8_mean, dim=0))
        features.append(F.cosine_similarity(tail8_mean, tail32_mean, dim=0))
        features.append(F.cosine_similarity(tail16_mean, tail32_mean, dim=0))
        features.append(F.cosine_similarity(tail32_mean, full_mean, dim=0))
        features.append(F.cosine_similarity(recency_mean, full_mean, dim=0))
        features.append((tail4_mean - full_mean).norm())
        features.append((tail8_mean - full_mean).norm())
        features.append((tail16_mean - full_mean).norm())
        features.append((tail32_mean - full_mean).norm())
        features.append((recency_mean - full_mean).norm())
        features.append((tail4_mean - tail32_mean).norm())
        features.append((last_token - tail4_mean).norm())
        last_vectors.append(last_token)

    for left, right in zip(last_vectors, last_vectors[1:]):
        features.append(F.cosine_similarity(left, right, dim=0))

    for layer_idx in range(hidden_states.size(0)):
        layer = hidden_states[layer_idx].float()
        real_tokens = layer[real_positions]
        tail32_mean = layer[tail32_positions].mean(dim=0)
        full_mean = real_tokens.mean(dim=0)
        last_token = layer[last_pos]

        features.append(last_token.norm())
        features.append(tail32_mean.norm())
        features.append(full_mean.norm())
        features.append(F.cosine_similarity(last_token, tail32_mean, dim=0))
        features.append((last_token - tail32_mean).norm())

        all_last_vectors.append(last_token)
        all_tail_means.append(tail32_mean)
        all_full_means.append(full_mean)

    middle_idx = len(all_last_vectors) // 2
    final_last = all_last_vectors[-1]
    middle_last = all_last_vectors[middle_idx]
    final_tail = all_tail_means[-1]
    middle_tail = all_tail_means[middle_idx]
    final_full = all_full_means[-1]
    middle_full = all_full_means[middle_idx]
    features.append(F.cosine_similarity(final_last, middle_last, dim=0))
    features.append(F.cosine_similarity(final_tail, middle_tail, dim=0))
    features.append(F.cosine_similarity(final_full, middle_full, dim=0))
    features.append((final_last - middle_last).norm())
    features.append((final_tail - middle_tail).norm())
    features.append((final_full - middle_full).norm())

    for left, right in zip(all_last_vectors, all_last_vectors[1:]):
        features.append(F.cosine_similarity(left, right, dim=0))
        features.append((right - left).norm())

    for left, right in zip(all_tail_means, all_tail_means[1:]):
        features.append(F.cosine_similarity(left, right, dim=0))
        features.append((right - left).norm())

    return torch.stack(features).float()


def aggregation_and_feature_extraction(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    use_geometric: bool = False,
) -> torch.Tensor:
    """Aggregate hidden states and optionally append geometric features.

    Main entry point called from ``solution.ipynb`` for each sample.
    Concatenates the output of ``aggregate`` with that of
    ``extract_geometric_features`` when ``use_geometric=True``.

    Args:
        hidden_states:  Tensor of shape ``(n_layers, seq_len, hidden_dim)``
                        for a single sample.
        attention_mask: 1-D tensor of shape ``(seq_len,)`` with 1 for real
                        tokens and 0 for padding.
        use_geometric:  Whether to append geometric features.  Controlled by
                        the ``USE_GEOMETRIC`` flag in ``solution.ipynb``.

    Returns:
        A 1-D float tensor of shape ``(feature_dim,)`` where
        ``feature_dim = hidden_dim`` (or larger for multi-layer or geometric
        concatenations).
    """
    agg_features = aggregate(hidden_states, attention_mask)  # (feature_dim,)

    if use_geometric:
        geo_features = extract_geometric_features(hidden_states, attention_mask)
        return torch.cat([agg_features, geo_features], dim=0)

    return agg_features
