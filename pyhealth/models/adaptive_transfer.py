from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from pyhealth.datasets import SampleDataset
from pyhealth.models import BaseModel
from pyhealth.models.ipd_dtw_kde import (
    batched_paired_dtw_distances,
    kde_smoothed_scalar,
)

DistanceFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


class _CallableBackbone(nn.Module):
    """Wraps a stateless callable ``fn(x) -> Tensor`` as an ``nn.Module``."""

    def __init__(
        self,
        fn: Callable[[torch.Tensor], torch.Tensor],
        output_dim: int,
    ) -> None:
        super().__init__()
        self.fn = fn
        self.output_dim = int(output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.fn(x)


class _PaperDSAFCNBackbone(nn.Module):
    """1D CNN + GAP, aligned with the reference ``build_fcn`` (Keras) stack."""

    def __init__(
        self,
        in_channels: int,
        conv_channels: int = 128,
        kernel_size: int = 7,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.output_dim = conv_channels
        self.bn = nn.BatchNorm1d(in_channels)
        pad = kernel_size // 2
        self.conv = nn.Conv1d(
            in_channels,
            conv_channels,
            kernel_size=kernel_size,
            padding=pad,
        )
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B, T, D] -> [B, D, T]
        x = x.transpose(1, 2)
        x = self.bn(x)
        x = self.conv(x)
        x = self.act(x)
        x = self.drop(x)
        return x.mean(dim=-1)


class AdaptiveTransferModel(BaseModel):
    """Adaptive transfer model for multi-source time-series classification.

    This model is inspired by
    "Daily Physical Activity Monitoring: Adaptive Learning from Multi-Source
    Motion Sensor Data".

    The model supports:
        1. standard supervised forward passes;
        2. paired source-target similarity computation;
        3. similarity-weighted transfer utilities for example scripts;
        4. dependency injection for both the backbone and distance function.

    Args:
        dataset: PyHealth sample dataset.
        feature_key: Dense time-series feature key. If None, uses the first
            available feature key.
        hidden_dim: Hidden size for built-in backbones.
        num_layers: Number of recurrent layers for built-in backbones.
        dropout: Dropout probability.
        bidirectional: Whether recurrent backbones are bidirectional.
        backbone: Backbone encoder specification. Supported string values are
            {"lstm", "gru", "mlp"}, or a custom ``nn.Module`` can be passed.
        backbone_output_dim: Output dimension of a custom backbone. Required
            when it cannot be inferred from the module itself.
        distance_fn: Used when ``ipd_backend="embedding"``: one of
            {"euclidean", "manhattan", "cosine"} or a callable on embeddings.
        use_similarity_weighting: Whether to scale learning rates by similarity.
        use_kde_smoothing: If ``ipd_backend="embedding"``, adds Gaussian noise to
            batch distances before averaging. If ``ipd_backend="dtw_kde"``,
            fits a Gaussian KDE on paired DTW distances and averages KDE
            samples (paper-style smooth bootstrap).
        smoothing_std: Standard deviation of Gaussian noise (embedding IPD only).
        eps: Small constant for numerical stability.
        input_dim: Optional per-time-step input width; inferred from the
            dataset when omitted.
        ipd_backend: ``"embedding"`` (encoder distances) or ``"dtw_kde"``
            (paired multivariate DTW + optional KDE).
        dtw_window_frac: Sakoe–Chiba band as a fraction of ``max(T1, T2)``;
            ``None`` means unconstrained DTW.
        kde_bandwidth: Gaussian KDE bandwidth for ``dtw_kde`` (default matches
            the authors' reference script).
        kde_n_draws: Number of KDE samples to average for ``dtw_kde``.
        kde_random_state: RNG seed for KDE sampling reproducibility.

    Raises:
        ValueError: If the dataset does not expose exactly one label key.

    Example:
        >>> model = AdaptiveTransferModel(dataset=dataset, feature_key="signal")
        >>> output = model(**batch)
        >>> output["logit"].shape
    """

    def __init__(
        self,
        dataset: SampleDataset,
        feature_key: Optional[str] = None,
        hidden_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.2,
        bidirectional: bool = False,
        backbone: Union[str, nn.Module, Callable[[torch.Tensor], torch.Tensor]] = "lstm",
        backbone_output_dim: Optional[int] = None,
        distance_fn: Union[str, DistanceFn] = "euclidean",
        use_similarity_weighting: bool = True,
        use_kde_smoothing: bool = True,
        smoothing_std: float = 0.01,
        eps: float = 1e-8,
        input_dim: Optional[int] = None,
        ipd_backend: str = "embedding",
        dtw_window_frac: Optional[float] = None,
        kde_bandwidth: float = 7.8,
        kde_n_draws: int = 10,
        kde_random_state: Optional[int] = None,
    ) -> None:
        """Initialize the adaptive transfer model.

        Args:
            dataset: PyHealth sample dataset.
            feature_key: Dense input feature key. If None, uses the first
                available feature key from the dataset.
            hidden_dim: Hidden size for built-in backbones.
            num_layers: Number of recurrent layers for built-in backbones.
            dropout: Dropout probability.
            bidirectional: Whether recurrent backbones are bidirectional.
            backbone: Backbone encoder specification. Supported string values
                are {"lstm", "gru", "mlp"}, or a custom ``nn.Module``.
            backbone_output_dim: Output dimension of a custom backbone.
            distance_fn: Distance function identifier or callable used for
                IPD-style similarity.
            use_similarity_weighting: Whether adaptive learning rates should be
                scaled by source-target similarity.
            use_kde_smoothing: Embedding: jitter distances; DTW+KDE: enable KDE.
            smoothing_std: Standard deviation of the Gaussian smoothing noise.
            eps: Small constant for numerical stability.
            input_dim: If set, per-time-step input size for built-in backbones
                (e.g. number of DSA channels). When ``None``, the model infers
                from ``dataset.input_info`` when present, otherwise from the
                first training sample.
            ipd_backend: ``embedding`` or ``dtw_kde``.
            dtw_window_frac: Optional Sakoe–Chiba window for DTW.
            kde_bandwidth: KDE bandwidth for ``dtw_kde``.
            kde_n_draws: KDE Monte Carlo draws for ``dtw_kde``.
            kde_random_state: Seed for KDE sampling.

        Raises:
            ValueError: If the dataset exposes more than one label key.
        """
        super().__init__(dataset)

        if len(self.label_keys) != 1:
            raise ValueError("AdaptiveTransferModel supports exactly one label key.")

        backend_norm = ipd_backend.lower().replace("-", "_")
        if backend_norm not in {"embedding", "dtw_kde"}:
            raise ValueError(
                f"ipd_backend must be 'embedding' or 'dtw_kde', got {ipd_backend!r}."
            )

        self.label_key = self.label_keys[0]
        self.feature_key = feature_key or self.feature_keys[0]
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.use_similarity_weighting = use_similarity_weighting
        self.use_kde_smoothing = use_kde_smoothing
        self.smoothing_std = smoothing_std
        self.eps = eps
        self.ipd_backend = backend_norm
        self.dtw_window_frac = dtw_window_frac
        self.kde_bandwidth = kde_bandwidth
        self.kde_n_draws = kde_n_draws
        self.kde_random_state = kde_random_state

        encoder_input_dim = (
            max(1, int(input_dim))
            if input_dim is not None
            else self._infer_input_dim(self.feature_key)
        )
        self.encoder, encoder_output_dim = self._build_encoder(
            input_dim=encoder_input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            backbone=backbone,
            backbone_output_dim=backbone_output_dim,
        )

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(encoder_output_dim, self.get_output_size())
        self.distance_fn = self._resolve_distance_fn(distance_fn)

    def _build_encoder(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
        backbone: Union[str, nn.Module, Callable[[torch.Tensor], torch.Tensor]],
        backbone_output_dim: Optional[int],
    ) -> Tuple[nn.Module, int]:
        """Build the encoder and infer its output size."""
        if isinstance(backbone, nn.Module):
            if isinstance(backbone, nn.LSTM) and backbone.input_size != input_dim:
                raise ValueError(
                    f"LSTM input_size={backbone.input_size} does not match "
                    f"dataset feature width {input_dim}."
                )
            if isinstance(backbone, nn.GRU) and backbone.input_size != input_dim:
                raise ValueError(
                    f"GRU input_size={backbone.input_size} does not match "
                    f"dataset feature width {input_dim}."
                )
            output_dim = self._infer_backbone_output_dim(
                backbone, backbone_output_dim
            )
            return backbone, output_dim

        if callable(backbone) and not isinstance(backbone, type):
            if backbone_output_dim is None:
                raise ValueError(
                    "backbone_output_dim is required when backbone is a callable."
                )
            enc = _CallableBackbone(backbone, backbone_output_dim)
            return enc, backbone_output_dim

        if not isinstance(backbone, str):
            raise ValueError(
                f"Unsupported backbone type: {type(backbone)}. "
                "Expected str, nn.Module, or a callable tensor->tensor."
            )

        backbone_name = backbone.lower()

        if backbone_name == "lstm":
            encoder = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
                bidirectional=bidirectional,
            )
            output_dim = hidden_dim * (2 if bidirectional else 1)
            return encoder, output_dim

        if backbone_name == "gru":
            encoder = nn.GRU(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
                bidirectional=bidirectional,
            )
            output_dim = hidden_dim * (2 if bidirectional else 1)
            return encoder, output_dim

        if backbone_name == "mlp":
            encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
            )
            return encoder, hidden_dim

        if backbone_name == "fcn":
            encoder = _PaperDSAFCNBackbone(
                in_channels=input_dim,
                conv_channels=128,
                kernel_size=7,
                dropout=dropout,
            )
            return encoder, encoder.output_dim

        raise ValueError(
            f"Unsupported backbone: {backbone}. Expected one of "
            "{'lstm', 'gru', 'mlp', 'fcn'} or a custom backbone module."
        )

    def _infer_backbone_output_dim(
        self,
        backbone: nn.Module,
        backbone_output_dim: Optional[int],
    ) -> int:
        """Infer output size for a custom backbone."""
        inferred: Optional[int] = None
        for attr in ["output_dim", "hidden_dim", "hidden_size", "embedding_dim"]:
            if hasattr(backbone, attr):
                value = getattr(backbone, attr)
                if isinstance(value, int) and value > 0:
                    inferred = value
                    break

        if backbone_output_dim is not None:
            if inferred is not None and inferred != backbone_output_dim:
                raise ValueError(
                    f"backbone_output_dim={backbone_output_dim} conflicts with "
                    f"inferred output size {inferred} from the backbone module."
                )
            return backbone_output_dim

        if inferred is not None:
            return inferred

        raise ValueError(
            "Could not infer backbone output dimension. Please provide "
            "backbone_output_dim for a custom backbone."
        )

    def _resolve_distance_fn(
        self,
        distance_fn: Union[str, DistanceFn],
    ) -> DistanceFn:
        """Resolve a string or callable distance function."""
        if callable(distance_fn):
            return distance_fn

        name = distance_fn.lower()
        if name == "euclidean":
            return lambda x, y: torch.norm(x - y, p=2, dim=1)
        if name == "manhattan":
            return lambda x, y: torch.norm(x - y, p=1, dim=1)
        if name == "cosine":
            return lambda x, y: 1.0 - F.cosine_similarity(x, y, dim=1)
        if name in {"dtw_kde", "dtw"}:
            raise ValueError(
                "distance_fn='dtw_kde' is invalid; set ipd_backend='dtw_kde' instead."
            )

        raise ValueError(
            f"Unsupported distance_fn: {distance_fn}. Expected one of "
            "{'euclidean', 'manhattan', 'cosine'} or a callable."
        )

    def _infer_input_dim(self, feature_key: str) -> int:
        """Infer per-time-step feature width from metadata or one sample."""
        if self.dataset is None:
            return 1

        info = getattr(self.dataset, "input_info", None)
        if isinstance(info, dict) and feature_key in info:
            try:
                stats = info[feature_key]
                if isinstance(stats, dict):
                    if "len" in stats and isinstance(stats["len"], int):
                        return max(1, int(stats["len"]))
                    if "dim" in stats and isinstance(stats["dim"], int):
                        return max(1, int(stats["dim"]))
            except (KeyError, TypeError):
                pass

        try:
            if len(self.dataset) == 0:
                return 1
            sample = self.dataset[0]
            if feature_key not in sample:
                return 1
            feat = sample[feature_key]
            if isinstance(feat, torch.Tensor):
                if feat.dim() >= 2:
                    return max(1, int(feat.shape[-1]))
                return 1
            if isinstance(feat, (list, tuple)):
                proc = self.dataset.input_processors.get(feature_key)
                if proc is not None:
                    schema = proc.schema()
                    if "value" in schema:
                        t = feat[schema.index("value")]
                        if isinstance(t, torch.Tensor) and t.dim() >= 2:
                            return max(1, int(t.shape[-1]))
        except Exception:
            pass

        return 1

    def _get_feature_value_and_mask(
        self,
        feature: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Extract normalized value and mask tensors from a feature input."""
        if isinstance(feature, torch.Tensor):
            value = feature
            mask = None
        else:
            schema = self.dataset.input_processors[self.feature_key].schema()
            if "value" not in schema:
                raise ValueError(
                    f"Feature '{self.feature_key}' must contain 'value'."
                )

            value = feature[schema.index("value")]
            mask = feature[schema.index("mask")] if "mask" in schema else None

            if mask is None and len(feature) == len(schema) + 1:
                mask = feature[-1]

        value = value.to(self.device).float()
        if mask is not None:
            mask = mask.to(self.device).float()

        # [B, T] -> [B, T, 1]
        if value.dim() == 2:
            value = value.unsqueeze(-1)
        elif value.dim() != 3:
            raise ValueError(
                f"Unsupported input shape for '{self.feature_key}': "
                f"{tuple(value.shape)}"
            )

        if mask is not None:
            if mask.dim() == 3:
                mask = mask.any(dim=-1).float()
            elif mask.dim() != 2:
                raise ValueError(
                    f"Unsupported mask shape for '{self.feature_key}': "
                    f"{tuple(mask.shape)}"
                )

        return value, mask

    def _masked_mean_pool(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Apply masked mean pooling over the time dimension."""
        if mask is None:
            return x.mean(dim=1)

        weights = mask.unsqueeze(-1).float()
        denom = weights.sum(dim=1).clamp_min(1.0)
        return (x * weights).sum(dim=1) / denom

    def _encode_sequence(
        self,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode a dense time series into a fixed-size embedding."""
        if isinstance(self.encoder, nn.LSTM):
            if mask is not None:
                lengths = mask.sum(dim=1).long().clamp(min=1).cpu()
                packed = nn.utils.rnn.pack_padded_sequence(
                    value,
                    lengths,
                    batch_first=True,
                    enforce_sorted=False,
                )
                _, (h_n, _) = self.encoder(packed)
            else:
                _, (h_n, _) = self.encoder(value)

            if self.bidirectional:
                emb = torch.cat([h_n[-2], h_n[-1]], dim=-1)
            else:
                emb = h_n[-1]
            return self.dropout(emb)

        if isinstance(self.encoder, nn.GRU):
            if mask is not None:
                lengths = mask.sum(dim=1).long().clamp(min=1).cpu()
                packed = nn.utils.rnn.pack_padded_sequence(
                    value,
                    lengths,
                    batch_first=True,
                    enforce_sorted=False,
                )
                _, h_n = self.encoder(packed)
            else:
                _, h_n = self.encoder(value)

            if self.bidirectional:
                emb = torch.cat([h_n[-2], h_n[-1]], dim=-1)
            else:
                emb = h_n[-1]
            return self.dropout(emb)

        # Non-recurrent encoders may return [B, D] or [B, T, D].
        encoder_out = self.encoder(value)

        if encoder_out.dim() == 3:
            emb = self._masked_mean_pool(encoder_out, mask)
        elif encoder_out.dim() == 2:
            emb = encoder_out
        else:
            raise ValueError(
                "Custom backbone must return a tensor of shape [B, D] "
                "or [B, T, D]."
            )

        return self.dropout(emb)

    def forward(
        self,
        **kwargs: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
    ) -> Dict[str, torch.Tensor]:
        """Run the forward pass and optionally compute loss.

        Args:
            **kwargs: Keyword inputs expected by PyHealth. Must contain the
                configured feature key. May also contain the label key.

        Returns:
            A dictionary containing model outputs and, when labels are
            provided, the loss and ground-truth labels.

        Raises:
            ValueError: If the configured feature key is missing from inputs.
        """
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

            if self.mode == "multiclass" and y_true.dim() > 1:
                y_true = y_true.squeeze(-1).long()
            elif self.mode == "binary":
                y_true = y_true.float()
                if y_true.dim() == 1:
                    y_true = y_true.unsqueeze(-1)

            loss = self.get_loss_function()(logits, y_true)
            results["loss"] = loss
            results["y_true"] = y_true

        return results

    def forward_from_embedding(
        self,
        **kwargs: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
    ) -> Dict[str, torch.Tensor]:
        """Forward hook kept for compatibility with PyHealth interfaces."""
        return self.forward(**kwargs)

    @torch.no_grad()
    def extract_embedding(
        self,
        batch: Dict[str, Union[torch.Tensor, Tuple[torch.Tensor, ...]]],
    ) -> torch.Tensor:
        """Extract latent embeddings for a batch.

        Args:
            batch: Batch dictionary containing the configured feature key.

        Returns:
            Batch embedding tensor of shape [B, H].
        """
        value, mask = self._get_feature_value_and_mask(batch[self.feature_key])
        return self._encode_sequence(value, mask)

    @torch.no_grad()
    def compute_pairwise_distances(
        self,
        source_batch: Dict[str, Union[torch.Tensor, Tuple[torch.Tensor, ...]]],
        target_batch: Dict[str, Union[torch.Tensor, Tuple[torch.Tensor, ...]]],
    ) -> torch.Tensor:
        """Compute paired distances between source and target embeddings.

        Args:
            source_batch: Source-domain batch dictionary.
            target_batch: Target-domain batch dictionary.

        Returns:
            Distance tensor of shape [B].

        Raises:
            ValueError: If the source and target batch sizes differ.
        """
        if self.ipd_backend == "dtw_kde":
            return self._compute_paired_dtw_distances(source_batch, target_batch)

        source_emb = self.extract_embedding(source_batch)
        target_emb = self.extract_embedding(target_batch)

        if source_emb.shape[0] != target_emb.shape[0]:
            raise ValueError(
                "Source and target batches must have the same batch size "
                "for paired IPD."
            )

        return self.distance_fn(source_emb, target_emb)

    @torch.no_grad()
    def _compute_paired_dtw_distances(
        self,
        source_batch: Dict[str, Union[torch.Tensor, Tuple[torch.Tensor, ...]]],
        target_batch: Dict[str, Union[torch.Tensor, Tuple[torch.Tensor, ...]]],
    ) -> torch.Tensor:
        src_val, _ = self._get_feature_value_and_mask(source_batch[self.feature_key])
        tgt_val, _ = self._get_feature_value_and_mask(target_batch[self.feature_key])
        if src_val.shape[0] != tgt_val.shape[0]:
            raise ValueError(
                "Source and target batches must have the same batch size "
                "for paired DTW IPD."
            )
        x = src_val.detach().cpu().float().numpy()
        y = tgt_val.detach().cpu().float().numpy()
        dists = batched_paired_dtw_distances(x, y, window_frac=self.dtw_window_frac)
        return torch.as_tensor(dists, device=self.device, dtype=torch.float32)

    @torch.no_grad()
    def compute_ipd(
        self,
        source_batch: Dict[str, Union[torch.Tensor, Tuple[torch.Tensor, ...]]],
        target_batch: Dict[str, Union[torch.Tensor, Tuple[torch.Tensor, ...]]],
    ) -> float:
        """Compute an IPD-style distance between one source and target batch.

        Args:
            source_batch: Source-domain batch dictionary.
            target_batch: Target-domain batch dictionary.

        Returns:
            Scalar IPD summary for this batch (mean distance, or KDE smooth).
        """
        if self.ipd_backend == "dtw_kde":
            distances = self._compute_paired_dtw_distances(source_batch, target_batch)
            if self.use_kde_smoothing:
                return kde_smoothed_scalar(
                    distances.detach().cpu().float().numpy(),
                    bandwidth=self.kde_bandwidth,
                    n_draws=self.kde_n_draws,
                    random_state=self.kde_random_state,
                )
            return float(torch.clamp(distances.mean(), min=0.0).item())

        distances = self.compute_pairwise_distances(source_batch, target_batch)

        if self.use_kde_smoothing:
            noise = torch.randn_like(distances) * self.smoothing_std
            distances = (distances + noise).clamp_min(0.0)

        return float(distances.mean().item())

    @torch.no_grad()
    def compute_source_similarities(
        self,
        source_batches: Sequence[
            Dict[str, Union[torch.Tensor, Tuple[torch.Tensor, ...]]]
        ],
        target_batch: Dict[str, Union[torch.Tensor, Tuple[torch.Tensor, ...]]],
    ) -> List[float]:
        """Compute inverse-distance similarities for multiple source batches.

        Args:
            source_batches: Sequence of source-domain batches.
            target_batch: Target-domain batch dictionary.

        Returns:
            List of similarity scores, one per source batch.
        """
        similarities: List[float] = []
        for source_batch in source_batches:
            ipd = self.compute_ipd(source_batch, target_batch)
            similarities.append(1.0 / (ipd + self.eps))
        return similarities

    @torch.no_grad()
    def rank_source_domains(
        self,
        source_batches: Sequence[
            Dict[str, Union[torch.Tensor, Tuple[torch.Tensor, ...]]]
        ],
        target_batch: Dict[str, Union[torch.Tensor, Tuple[torch.Tensor, ...]]],
    ) -> List[int]:
        """Rank source domains by descending similarity to the target.

        Args:
            source_batches: Sequence of source-domain batches.
            target_batch: Target-domain batch dictionary.

        Returns:
            List of source-domain indices sorted from most similar to least
            similar.
        """
        similarities = self.compute_source_similarities(source_batches, target_batch)
        return sorted(
            range(len(similarities)),
            key=lambda i: similarities[i],
            reverse=True,
        )

    def get_adaptive_lr(self, base_lr: float, similarity: float) -> float:
        """Scale the base learning rate by similarity if enabled.

        Args:
            base_lr: Base learning rate before adaptation.
            similarity: Source-target similarity score.

        Returns:
            Adapted learning rate.
        """
        if not self.use_similarity_weighting:
            return base_lr
        return base_lr * max(similarity, self.eps)
