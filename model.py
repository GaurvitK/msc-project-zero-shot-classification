from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


@dataclass
class BaseClassifierConfig:
    """Architecture values stored inside checkpoints."""

    document_encoder_name: str
    label_embedding_size: int
    num_seen_labels: int
    label_encoder_name: str = "precomputed"
    max_document_length: int = 512
    dropout: float = 0.1


class BaseClassifier(nn.Module):
    """Base seen-label classifier with externally supplied frozen label embeddings."""

    def __init__(self, config: BaseClassifierConfig) -> None:
        super().__init__()

        if config.label_embedding_size <= 0:
            raise ValueError("label_embedding_size must be positive")
        if config.num_seen_labels < 0:
            raise ValueError("num_seen_labels cannot be negative")

        self.config = config
        self.document_encoder = AutoModel.from_pretrained(config.document_encoder_name)

        self.hidden_size = int(self.document_encoder.config.hidden_size)
        self.label_hidden_size = int(config.label_embedding_size)

        self.attention_projection = nn.Linear(self.label_hidden_size, self.hidden_size)
        self.label_projection = nn.Linear(self.label_hidden_size, self.hidden_size)
        self.dropout = nn.Dropout(config.dropout)

        self.logit_scale = nn.Parameter(torch.tensor(math.log(10.0)))
        self.seen_label_bias = nn.Parameter(torch.zeros(config.num_seen_labels, dtype=torch.float32))

    def _validate_label_embeddings(self, label_embeddings: torch.Tensor) -> None:
        if label_embeddings.ndim != 2:
            raise ValueError("label_embeddings must have shape [num_labels, label_dim]")
        if label_embeddings.size(1) != self.label_hidden_size:
            raise ValueError(f"Expected label embedding width {self.label_hidden_size}, received {label_embeddings.size(1)}")
        if not torch.is_floating_point(label_embeddings):
            raise TypeError("label_embeddings must use a floating dtype")
        if not torch.isfinite(label_embeddings).all():
            raise ValueError("label_embeddings contains NaN or Inf")

    def encode_documents(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.document_encoder(input_ids=input_ids, attention_mask=attention_mask,)
        return outputs.last_hidden_state

    def project_attention_queries(self, label_embeddings: torch.Tensor) -> torch.Tensor:
        self._validate_label_embeddings(label_embeddings)
        return self.attention_projection(label_embeddings)

    def project_label_features(self, label_embeddings: torch.Tensor) -> torch.Tensor:
        self._validate_label_embeddings(label_embeddings)
        return self.label_projection(label_embeddings)

    def attend_documents(self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor, attention_queries: torch.Tensor) -> torch.Tensor:
        """Return label-specific document representations [B, L, H]."""
        attention_scores = torch.einsum(
            "bth,lh->blt",
            token_embeddings,
            attention_queries,
        )
        attention_scores = attention_scores / math.sqrt(self.hidden_size)

        valid_tokens = attention_mask[:, None, :].bool()
        attention_scores = attention_scores.masked_fill(~valid_tokens, torch.finfo(attention_scores.dtype).min)

        attention_weights = torch.softmax(attention_scores, dim=-1)
        attended_documents = torch.einsum(
            "blt,bth->blh",
            attention_weights,
            token_embeddings,
        )
        return self.dropout(attended_documents)

    def cosine_score(self,
        document_features: torch.Tensor,
        label_features: torch.Tensor,
        label_bias: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return scaled cosine logits and normalized representations."""

        normalized_documents = F.normalize(document_features, dim=-1)
        normalized_labels = F.normalize(label_features, dim=-1)
        cosine_scores = torch.einsum(
            "blh,lh->bl",
            normalized_documents,
            normalized_labels,
        )
        logits = cosine_scores * self.logit_scale.exp().clamp(max=100.0)
        if label_bias is not None:
            if label_bias.size(0) != label_features.size(0):
                raise ValueError("Label bias count does not match label count")
            logits = logits + label_bias.unsqueeze(0)
        return logits, normalized_documents, normalized_labels

    def score_from_hidden(
        self,
        token_embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
        label_embeddings: torch.Tensor,
        label_bias: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        attention_queries = self.project_attention_queries(label_embeddings)
        label_features = self.project_label_features(label_embeddings)
        document_features = self.attend_documents(
            token_embeddings=token_embeddings,
            attention_mask=attention_mask,
            attention_queries=attention_queries,
        )
        return self.cosine_score(
            document_features=document_features,
            label_features=label_features,
            label_bias=label_bias,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        label_embeddings: torch.Tensor,
        return_features: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if label_embeddings.size(0) != self.seen_label_bias.size(0):
            raise ValueError("Seen-label embedding count does not match seen-label bias count")

        token_embeddings = self.encode_documents(input_ids, attention_mask)
        outputs = self.score_from_hidden(
            token_embeddings=token_embeddings,
            attention_mask=attention_mask,
            label_embeddings=label_embeddings,
            label_bias=self.seen_label_bias,
        )

        return outputs if return_features else outputs[0]


class ResidualAdapter(nn.Module):
    """Shared residual MLP with zero-initialized output."""

    def __init__(self, hidden_size: int, adapter_hidden_size: int, dropout: float) -> None:
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(hidden_size, adapter_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(adapter_hidden_size, hidden_size),
        )

        final_layer = self.network[-1]
        nn.init.zeros_(final_layer.weight)
        nn.init.zeros_(final_layer.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


class UnseenRepresentationAdapter(nn.Module):
    """Residual adaptation for a fixed unseen-label set."""

    def __init__(self, base_model: BaseClassifier, num_unseen_labels: int, adapter_hidden_size: int | None = None, adapter_dropout: float = 0.1) -> None:
        super().__init__()

        if num_unseen_labels <= 0:
            raise ValueError("num_unseen_labels must be positive")

        self.base_model = base_model
        self.num_unseen_labels = int(num_unseen_labels)
        self.adapter_hidden_size = int(
            adapter_hidden_size
            if adapter_hidden_size is not None
            else max(64, base_model.hidden_size // 4)
        )
        self.adapter_dropout = float(adapter_dropout)
        self.attention_adapter = ResidualAdapter(
            base_model.hidden_size,
            self.adapter_hidden_size,
            self.adapter_dropout,
        )
        self.projection_adapter = ResidualAdapter(
            base_model.hidden_size,
            self.adapter_hidden_size,
            self.adapter_dropout,
        )
        self.unseen_label_bias = nn.Parameter(
            torch.zeros(self.num_unseen_labels, dtype=torch.float32)
        )
        self.freeze_base_model()

    def freeze_base_model(self) -> None:
        self.base_model.requires_grad_(False)
        self.base_model.eval()

    def train(self, mode: bool = True) -> "UnseenRepresentationAdapter":
        super().train(mode)
        self.base_model.eval()
        return self

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [parameter for parameter in self.parameters() if parameter.requires_grad]

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        unseen_label_embeddings: torch.Tensor,
        return_features: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if unseen_label_embeddings.size(0) != self.num_unseen_labels:
            raise ValueError(
                "Unseen-label embedding count does not match adapter label count"
            )

        self.base_model._validate_label_embeddings(unseen_label_embeddings)

        with torch.no_grad():
            token_embeddings = self.base_model.encode_documents(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            base_attention_queries = self.base_model.project_attention_queries(
                unseen_label_embeddings
            )
            base_label_features = self.base_model.project_label_features(
                unseen_label_embeddings
            )

        attention_delta = self.attention_adapter(base_attention_queries)
        projection_delta = self.projection_adapter(base_label_features)

        adapted_attention_queries = base_attention_queries + attention_delta
        adapted_label_features = base_label_features + projection_delta

        adapted_document_features = self.base_model.attend_documents(
            token_embeddings=token_embeddings,
            attention_mask=attention_mask,
            attention_queries=adapted_attention_queries,
        )

        logits, normalized_documents, normalized_labels = self.base_model.cosine_score(
            document_features=adapted_document_features,
            label_features=adapted_label_features,
            label_bias=self.unseen_label_bias,
        )

        if not return_features:
            return logits

        return logits, {
            "base_attention_queries": base_attention_queries,
            "adapted_attention_queries": adapted_attention_queries,
            "attention_delta": attention_delta,
            "base_label_features": base_label_features,
            "adapted_label_features": adapted_label_features,
            "projection_delta": projection_delta,
            "document_features": normalized_documents,
            "label_features": normalized_labels,
        }


def validate_label_ids(label_ids: Sequence[int], *, num_labels: int | None = None, field_name: str = "label_ids") -> list[int]:
    """Validate label IDs while preserving order."""
    ids = [int(label_id) for label_id in label_ids]

    if len(ids) != len(set(ids)):
        raise ValueError(f"{field_name} contains duplicate label IDs")
    if any(label_id < 0 for label_id in ids):
        raise ValueError(f"{field_name} contains negative label IDs")
    if num_labels is not None and any(label_id >= num_labels for label_id in ids):
        raise ValueError(
            f"{field_name} contains IDs outside [0, {num_labels - 1}]"
        )

    return ids


def select_label_embeddings(all_label_embeddings: torch.Tensor, label_ids: Sequence[int]) -> torch.Tensor:
    """Select embedding rows in the supplied label-ID order."""
    ids = validate_label_ids(
        label_ids,
        num_labels=all_label_embeddings.size(0),
    )
    if not ids:
        return all_label_embeddings.new_empty(
            (0, all_label_embeddings.size(1))
        )
    indices = torch.tensor(
        ids,
        dtype=torch.long,
        device=all_label_embeddings.device,
    )
    return all_label_embeddings.index_select(0, indices)


def load_label_embeddings(
    path: Path | str,
    *,
    label_ids: Sequence[int] | None = None,
    expected_num_labels: int | None = None,
    expected_dimension: int | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Load a UID-aligned frozen label-embedding matrix."""
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(f"Label embedding file not found: {path.resolve()}")

    matrix = np.load(path, allow_pickle=False)

    if matrix.ndim != 2:
        raise ValueError("Label embedding matrix must be two-dimensional")
    if expected_num_labels is not None and matrix.shape[0] != expected_num_labels:
        raise ValueError(
            f"Expected {expected_num_labels} label embeddings, "
            f"received {matrix.shape[0]}"
        )
    if expected_dimension is not None and matrix.shape[1] != expected_dimension:
        raise ValueError(
            f"Expected embedding dimension {expected_dimension}, "
            f"received {matrix.shape[1]}"
        )
    if not np.issubdtype(matrix.dtype, np.floating):
        raise TypeError("Label embedding matrix must use a floating dtype")
    if not np.isfinite(matrix).all():
        raise ValueError("Label embedding matrix contains NaN or Inf")

    embeddings = torch.from_numpy(
        np.asarray(matrix, dtype=np.float32).copy()
    )
    if label_ids is not None:
        embeddings = select_label_embeddings(embeddings, label_ids)
    if device is not None:
        embeddings = embeddings.to(device)

    return embeddings


def _config_from_checkpoint(config: dict[str, Any]) -> BaseClassifierConfig:
    """Load both current and existing EURLEX/AmazonCat checkpoint configs."""
    values = dict(config)

    # Old EURLEX checkpoints contain this unused field.
    values.pop("max_label_length", None)

    return BaseClassifierConfig(**values)


def save_seen_checkpoint(
    path: Path,
    model: BaseClassifier,
    seen_label_embeddings: torch.Tensor,
    seen_label_ids: Sequence[int],
    seen_label_texts: Sequence[str],
) -> None:
    """Save the stage-one model and exact seen-label ordering."""
    expected = model.config.num_seen_labels

    if len(seen_label_ids) != expected:
        raise ValueError("Seen label ID count does not match model configuration")
    if len(seen_label_texts) != expected:
        raise ValueError("Seen label text count does not match model configuration")
    if seen_label_embeddings.size(0) != expected:
        raise ValueError("Seen embedding count does not match model configuration")

    model._validate_label_embeddings(seen_label_embeddings)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "checkpoint_type": "base_seen",
            "config": asdict(model.config),
            "model_state_dict": model.state_dict(),
            "seen_label_embeddings": seen_label_embeddings.cpu(),
            "seen_label_ids": [int(x) for x in seen_label_ids],
            "seen_label_texts": [str(x) for x in seen_label_texts],
        },
        path,
    )


def load_seen_checkpoint(
    path: Path,
    device: torch.device,
) -> tuple[BaseClassifier, torch.Tensor, list[int], list[str]]:
    """Load a stage-one checkpoint."""
    checkpoint: dict[str, Any] = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )
    model = BaseClassifier(_config_from_checkpoint(checkpoint["config"]))
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)

    embeddings = checkpoint["seen_label_embeddings"].to(device)
    ids = [int(x) for x in checkpoint["seen_label_ids"]]
    texts = [str(x) for x in checkpoint["seen_label_texts"]]

    model._validate_label_embeddings(embeddings)

    if embeddings.size(0) != model.seen_label_bias.size(0):
        raise ValueError("Checkpoint seen embedding count does not match bias count")

    return model, embeddings, ids, texts


def _adapter_only_state_dict(model: UnseenRepresentationAdapter) -> dict[str, torch.Tensor]:
    return {key: value for key, value in model.state_dict().items() if not key.startswith("base_model.")}


def save_combined_checkpoint(
    path: Path,
    unseen_model: UnseenRepresentationAdapter,
    seen_label_embeddings: torch.Tensor,
    seen_label_ids: Sequence[int],
    seen_label_texts: Sequence[str],
    unseen_label_embeddings: torch.Tensor,
    unseen_label_ids: Sequence[int],
    unseen_label_texts: Sequence[str],
) -> None:
    """Save the frozen seen model and learned unseen adapters."""
    num_unseen = unseen_model.num_unseen_labels

    if len(unseen_label_ids) != num_unseen:
        raise ValueError("Unseen label ID count does not match adapter configuration")
    if len(unseen_label_texts) != num_unseen:
        raise ValueError("Unseen label text count does not match adapter configuration")
    if unseen_label_embeddings.size(0) != num_unseen:
        raise ValueError("Unseen embedding count does not match adapter configuration")
    if set(seen_label_ids) & set(unseen_label_ids):
        raise ValueError("Seen and unseen label sets overlap")

    unseen_model.base_model._validate_label_embeddings(seen_label_embeddings)
    unseen_model.base_model._validate_label_embeddings(unseen_label_embeddings)

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "checkpoint_type": "base_seen_plus_unseen_representation_adapter",
            "config": asdict(unseen_model.base_model.config),
            "base_model_state_dict": unseen_model.base_model.state_dict(),
            "unseen_adapter_config": {
                "num_unseen_labels": unseen_model.num_unseen_labels,
                "adapter_hidden_size": unseen_model.adapter_hidden_size,
                "adapter_dropout": unseen_model.adapter_dropout,
            },
            "unseen_adapter_state_dict": _adapter_only_state_dict(unseen_model),
            "seen_label_embeddings": seen_label_embeddings.cpu(),
            "seen_label_ids": [int(x) for x in seen_label_ids],
            "seen_label_texts": [str(x) for x in seen_label_texts],
            "unseen_label_embeddings": unseen_label_embeddings.cpu(),
            "unseen_label_ids": [int(x) for x in unseen_label_ids],
            "unseen_label_texts": [str(x) for x in unseen_label_texts],
        },
        path,
    )


def load_combined_checkpoint(
    path: Path,
    device: torch.device,
) -> tuple[
    UnseenRepresentationAdapter,
    torch.Tensor,
    list[int],
    list[str],
    torch.Tensor,
    list[int],
    list[str],
]:
    """Load a final seen + unseen-adapter checkpoint."""
    checkpoint: dict[str, Any] = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    base_model = BaseClassifier(_config_from_checkpoint(checkpoint["config"]))
    base_model.load_state_dict(
        checkpoint["base_model_state_dict"],
        strict=True,
    )

    adapter_config = checkpoint["unseen_adapter_config"]
    model = UnseenRepresentationAdapter(
        base_model=base_model,
        num_unseen_labels=int(adapter_config["num_unseen_labels"]),
        adapter_hidden_size=int(adapter_config["adapter_hidden_size"]),
        adapter_dropout=float(adapter_config["adapter_dropout"]),
    )

    missing, unexpected = model.load_state_dict(
        checkpoint["unseen_adapter_state_dict"],
        strict=False,
    )
    missing_non_base = [
        key
        for key in missing
        if not key.startswith("base_model.")
    ]

    if missing_non_base or unexpected:
        raise RuntimeError(
            "Could not load unseen adapters. "
            f"Missing non-base keys: {missing_non_base}; "
            f"unexpected keys: {unexpected}"
        )

    model.to(device)
    model.base_model.eval()

    seen_embeddings = checkpoint["seen_label_embeddings"].to(device)
    unseen_embeddings = checkpoint["unseen_label_embeddings"].to(device)
    seen_ids = [int(x) for x in checkpoint["seen_label_ids"]]
    unseen_ids = [int(x) for x in checkpoint["unseen_label_ids"]]
    seen_texts = [str(x) for x in checkpoint["seen_label_texts"]]
    unseen_texts = [str(x) for x in checkpoint["unseen_label_texts"]]

    model.base_model._validate_label_embeddings(seen_embeddings)
    model.base_model._validate_label_embeddings(unseen_embeddings)

    return (
        model,
        seen_embeddings,
        seen_ids,
        seen_texts,
        unseen_embeddings,
        unseen_ids,
        unseen_texts,
    )
