from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from pyhealth.datasets import SampleDataset
from pyhealth.models import BaseModel


class AdaptiveTransferModel(BaseModel):
    """Adaptive transfer model for multi-source time-series classification.

    This model is a practical PyHealth-style implementation inspired by
    "Daily Physical Activity Monitoring: Adaptive Learning from Multi-Source
    Motion Sensor Data".

    The model itself is a simple LSTM classifier over one dense time-series
    feature. To keep the implementation feasible for a
    course project and compatible with PyHealth's model API, this class
    provides:

    1. a standard forward pass for supervised training/evaluation
    2. utilities to compute source-target similarity using paired samples.
    3. utilities for similarity-weighted transfer in example scripts.

    Expected input format
    ---------------------
    The selected feature_key should correspond to a dense time-series input.
    The processor schema should expose at least a "value" tensor, optionally
    a "mask" tensor.

    Supported shapes for value:
    - [batch, seq_len] -> interpreted as 1 feature per timestep
    - [batch, seq_len, input_dim]

    Supported shapes for mask:
    - [batch, seq_len]
    - [batch, seq_len, input_dim] (collapsed across feature dimension)

    Args:
        dataset: PyHealth sample dataset.
        feature_key: The dense time-series feature to use. If None, the first
            feature in dataset.input_schema is used.
        hidden_dim: Hidden size of the LSTM.
        num_layers: Number of LSTM layers.
        dropout: Dropout used in the encoder/classifier.
        bidirectional: Whether to use a bidirectional LSTM.
        use_similarity_weighting: Whether transfer learning rates should be
            scaled by source-target similarity.
        use_kde_smoothing: Whether to smooth pairwise distances before forming
            a similarity score. This is a lightweight approximation for the
            smoothing idea in the paper.
        smoothing_std: Standard deviation used in the smoothing approximation.
        eps: Small constant for numerical stability.
    """

    def __init__(
        self,
        dataset: SampleDataset,
        feature_key: Optional[str] = None,
        hidden_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.2,
        bidirectional: bool = False,
        use_similarity_weighting: bool = True,
        use_kde_smoothing: bool = True,
        smoothing_std: float = 0.01,
        eps: float = 1e-8,
    ) -> None:
        super().__init__(dataset)

        if len(self.label_keys) != 1:
            raise ValueError("AdaptiveTransferModel supports exactly one label key.")

        self.label_key = self.label_keys[0]
        self.feature_key = feature_key or self.feature_keys[0]
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_prob = dropout
        self.bidirectional = bidirectional
        self.use_similarity_weighting = use_similarity_weighting
        self.use_kde_smoothing = use_kde_smoothing
        self.smoothing_std = smoothing_std
        self.eps = eps

        # Infer input dimension from dataset statistics, if possible.
        input_dim = self._infer_input_dim(self.feature_key)

        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
            bidirectional=bidirectional,
        )

        direction_factor = 2 if bidirectional else 1
        classifier_in_dim = hidden_dim * direction_factor

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(classifier_in_dim, self.get_output_size())

    def _infer_input_dim(self, feature_key: str) -> int:
        """Infer dense feature dimensionality from dataset statistics.

        Falls back to 1 when shape metadata is unavailable.
        """
        if self.dataset is None:
            return 1

        # Prefer explicit dimension statistics if they exist.
        try:
            stats = self.dataset.input_info[feature_key]
            if "len" in stats and isinstance(stats["len"], int):
                # For dense vectors represented at each timestep.
                return max(1, int(stats["len"]))
            if "dim" in stats and isinstance(stats["dim"], int):
                return max(1, int(stats["dim"]))
        except Exception:
            pass

        # Conservative fallback.
        return 1

    def _get_feature_value_and_mask(
        self, feature: torch.Tensor | Tuple[torch.Tensor, ...]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Extract value and mask tensors from a PyHealth feature tuple."""
        if isinstance(feature, torch.Tensor):
            value = feature
            mask = None
        else:
            schema = self.dataset.input_processors[self.feature_key].schema()
            if "value" not in schema:
                raise ValueError(
                    f"Feature '{self.feature_key}' must contain 'value' in its schema."
                )

            value = feature[schema.index("value")]
            mask = feature[schema.index("mask")] if "mask" in schema else None

            if mask is None and len(feature) == len(schema) + 1:
                mask = feature[-1]

        value = value.to(self.device).float()
        if mask is not None:
            mask = mask.to(self.device).float()

        # Normalize shapes:
        # [B, T] -> [B, T, 1]
        if value.dim() == 2:
            value = value.unsqueeze(-1)
        elif value.dim() != 3:
            raise ValueError(
                f"Unsupported input shape for '{self.feature_key}': {tuple(value.shape)}"
            )

        if mask is not None:
            if mask.dim() == 3:
                mask = mask.any(dim=-1).float()
            elif mask.dim() != 2:
                raise ValueError(
                    f"Unsupported mask shape for '{self.feature_key}': {tuple(mask.shape)}"
                )

        return value, mask

    def _encode_sequence(
        self, value: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Encode a dense time series into a fixed-size embedding."""
        if mask is not None:
            lengths = mask.sum(dim=1).long().clamp(min=1).cpu()
            packed = nn.utils.rnn.pack_padded_sequence(
                value, lengths, batch_first=True, enforce_sorted=False
            )
            _, (h_n, _) = self.encoder(packed)
        else:
            _, (h_n, _) = self.encoder(value)

        if self.bidirectional:
            # last layer forward/backward states
            h_last = torch.cat([h_n[-2], h_n[-1]], dim=-1)
        else:
            h_last = h_n[-1]

        return self.dropout(h_last)

    def forward(
        self, **kwargs: torch.Tensor | Tuple[torch.Tensor, ...]
    ) -> Dict[str, torch.Tensor]:
        """Forward pass following the PyHealth BaseModel convention."""
        if self.feature_key not in kwargs:
            raise ValueError(
                f"Expected feature key '{self.feature_key}' in model inputs."
            )

        value, mask = self._get_feature_value_and_mask(kwargs[self.feature_key])
        patient_emb = self._encode_sequence(value, mask)
        logits = self.classifier(patient_emb)
        y_prob = self.prepare_y_prob(logits)

        results: Dict[str, torch.Tensor] = {
            "logit": logits,
            "y_prob": y_prob,
        }

        if self.label_key in kwargs:
            y_true = cast(torch.Tensor, kwargs[self.label_key]).to(self.device)

            # Cross-entropy expects [B] labels for multiclass.
            if self.mode == "multiclass" and y_true.dim() > 1:
                y_true = y_true.squeeze(-1).long()
            elif self.mode == "binary":
                y_true = y_true.float()
                if y_true.dim() == 1:
                    y_true = y_true.unsqueeze(-1)

            loss_fn = self.get_loss_function()
            loss = loss_fn(logits, y_true)

            results["loss"] = loss
            results["y_true"] = y_true

        return results

    def forward_from_embedding(
        self, **kwargs: torch.Tensor | Tuple[torch.Tensor, ...]
    ) -> Dict[str, torch.Tensor]:
        """Compatibility hook for interpretability.

        Since this model already operates on dense values, we reuse forward().
        """
        return self.forward(**kwargs)

    @torch.no_grad()
    def extract_embedding(
        self, batch: Dict[str, torch.Tensor | Tuple[torch.Tensor, ...]]
    ) -> torch.Tensor:
        """Return the latent embedding for a batch."""
        value, mask = self._get_feature_value_and_mask(batch[self.feature_key])
        return self._encode_sequence(value, mask)

    @torch.no_grad()
    def compute_pairwise_distances(
        self,
        source_batch: Dict[str, torch.Tensor | Tuple[torch.Tensor, ...]],
        target_batch: Dict[str, torch.Tensor | Tuple[torch.Tensor, ...]],
    ) -> torch.Tensor:
        """Compute paired distances between source and target embeddings.

        The two batches are assumed to be paired by row index.
        """
        source_emb = self.extract_embedding(source_batch)
        target_emb = self.extract_embedding(target_batch)

        if source_emb.shape[0] != target_emb.shape[0]:
            raise ValueError(
                "Source and target batches must have the same batch size for paired IPD."
            )

        return torch.norm(source_emb - target_emb, p=2, dim=1)

    @torch.no_grad()
    def compute_ipd(
        self,
        source_batch: Dict[str, torch.Tensor | Tuple[torch.Tensor, ...]],
        target_batch: Dict[str, torch.Tensor | Tuple[torch.Tensor, ...]],
    ) -> float:
        """Compute an IPD-style distance between one source batch and target batch.

        This is a lightweight approximation:
        - compute paired embedding distances,
        - optionally smooth them with Gaussian perturbation,
        - average the result.
        """
        distances = self.compute_pairwise_distances(source_batch, target_batch)

        if self.use_kde_smoothing:
            noise = torch.randn_like(distances) * self.smoothing_std
            distances = (distances + noise).clamp_min(0.0)

        return float(distances.mean().item())

    @torch.no_grad()
    def compute_source_similarities(
        self,
        source_batches: Sequence[Dict[str, torch.Tensor | Tuple[torch.Tensor, ...]]],
        target_batch: Dict[str, torch.Tensor | Tuple[torch.Tensor, ...]],
    ) -> List[float]:
        """Compute source-target similarities for multiple source domains.

        Similarity is defined as inverse distance:
            similarity = 1 / (ipd + eps)
        """
        similarities: List[float] = []
        for source_batch in source_batches:
            ipd = self.compute_ipd(source_batch, target_batch)
            sim = 1.0 / (ipd + self.eps)
            similarities.append(sim)
        return similarities

    @torch.no_grad()
    def rank_source_domains(
        self,
        source_batches: Sequence[Dict[str, torch.Tensor | Tuple[torch.Tensor, ...]]],
        target_batch: Dict[str, torch.Tensor | Tuple[torch.Tensor, ...]],
    ) -> List[int]:
        """Return source domain indices sorted by descending similarity."""
        similarities = self.compute_source_similarities(source_batches, target_batch)
        ranked = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)
        return ranked

    def get_adaptive_lr(self, base_lr: float, similarity: float) -> float:
        """Scale the base learning rate by similarity."""
        if not self.use_similarity_weighting:
            return base_lr
        return base_lr * max(similarity, self.eps)
