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
    """Return stable early, mid, and late layer indices for a hidden-state stack."""
    candidates = [n_layers - 1, n_layers - 2, n_layers // 2, n_layers // 4]
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
    tail_positions = real_positions[-16:]

    features = []
    for layer_idx in selected_layer_indices(hidden_states.size(0)):
        layer = hidden_states[layer_idx].float()
        last_token = layer[last_pos]
        tail_mean = layer[tail_positions].mean(dim=0)
        features.extend([last_token, tail_mean])

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
    tail_positions = real_positions[-16:]
    selected = selected_layer_indices(hidden_states.size(0))

    real_len = float(real_positions.numel())
    seq_len = float(attention_mask.numel())
    features = [
        torch.tensor(
            real_len / max(seq_len, 1.0),
            dtype=torch.float32,
            device=hidden_states.device,
        )
    ]

    last_vectors = []
    for layer_idx in selected:
        layer = hidden_states[layer_idx, real_positions].float()
        full_layer = hidden_states[layer_idx].float()
        last_token = full_layer[last_pos]
        tail_mean = full_layer[tail_positions].mean(dim=0)

        features.append(layer.norm(dim=1).mean())
        features.append(last_token.norm())
        features.append(F.cosine_similarity(last_token, tail_mean, dim=0))
        last_vectors.append(last_token)

    for left, right in zip(last_vectors, last_vectors[1:]):
        features.append(F.cosine_similarity(left, right, dim=0))

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
