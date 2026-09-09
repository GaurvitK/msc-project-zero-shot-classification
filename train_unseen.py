from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from model import (
    UnseenRepresentationAdapter,
    load_label_embeddings,
    load_seen_checkpoint,
    save_combined_checkpoint,
    select_label_embeddings,
    validate_label_ids,
)


class DataError(RuntimeError):
    pass


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_json_records(path: Path) -> list[dict[str, Any]]:
    """Read a JSON array, JSONL file, or consecutive JSON objects."""
    text = path.read_text(encoding="utf-8-sig")
    if not text.strip():
        return []

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = None

    if parsed is not None:
        if isinstance(parsed, dict):
            return [parsed]
        if isinstance(parsed, list):
            return parsed
        raise DataError(f"{path} must contain JSON objects")

    decoder = json.JSONDecoder()
    rows: list[dict[str, Any]] = []
    position = 0

    while position < len(text):
        while position < len(text) and text[position].isspace():
            position += 1
        if position >= len(text):
            break

        row, position = decoder.raw_decode(text, position)
        if not isinstance(row, dict):
            raise DataError(f"{path} contains a non-object record")
        rows.append(row)

    return rows


def load_labels(path: Path, expected_count: int) -> dict[int, dict[str, Any]]:
    rows = read_json_records(path)
    if len(rows) != expected_count:
        raise DataError(f"Expected {expected_count:,} labels, found {len(rows):,}")

    labels: dict[int, dict[str, Any]] = {}
    for position, row in enumerate(rows):
        if "uid" not in row:
            raise DataError(f"Label record {position} is missing uid")

        label_id = int(row["uid"])
        if label_id != position:
            raise DataError(f"Labels must be in canonical UID order: expected {position}, found {label_id}")
        labels[label_id] = row

    return labels


def label_text(row: dict[str, Any], fields: Sequence[str]) -> str:
    values: list[str] = []
    seen: set[str] = set()

    for field in fields:
        value = str(row.get(field, "")).strip()
        key = value.casefold()
        if value and key not in seen:
            values.append(value)
            seen.add(key)

    return " | ".join(values) if values else str(row["uid"])


def derive_unseen_ids(
    seen_ids: Sequence[int],
    *,
    num_labels: int,
    expected_seen: int,
    expected_unseen: int,
) -> list[int]:
    seen_ids = validate_label_ids(
        seen_ids,
        num_labels=num_labels,
        field_name="seen checkpoint label IDs",
    )

    if len(seen_ids) != expected_seen:
        raise DataError(
            f"Expected {expected_seen:,} seen labels in the checkpoint, found {len(seen_ids):,}")

    unseen_ids = sorted(set(range(num_labels)) - set(seen_ids))

    if len(unseen_ids) != expected_unseen:
        raise DataError(f"Expected {expected_unseen:,} unseen labels, found {len(unseen_ids):,}")

    return unseen_ids


def validate_and_split_synthetic(
    rows: list[dict[str, Any]],
    unseen_ids: Sequence[int],
    *,
    examples_per_label: int,
    train_per_label: int,
    val_per_label: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Counter[int]]:
    """Validate synthetic coverage and optionally create a per-label holdout."""
    if train_per_label + val_per_label != examples_per_label:
        raise ValueError(
            "synthetic_train_per_label + synthetic_val_per_label must equal examples_per_label")

    expected_total = len(unseen_ids) * examples_per_label
    if len(rows) != expected_total:
        raise DataError(f"Expected {expected_total:,} synthetic rows ({examples_per_label} per unseen label), found {len(rows):,}")

    unseen_set = set(unseen_ids)
    rows_by_label: dict[int, list[dict[str, Any]]] = defaultdict(list)
    document_uids: set[str] = set()

    for index, row in enumerate(rows):
        missing = {"uid", "content", "target_ind"} - row.keys()
        if missing:
            raise DataError(f"Synthetic record {index}: missing fields {sorted(missing)}")

        uid = str(row["uid"]).strip()
        content = str(row["content"]).strip()
        targets = row["target_ind"]

        if not uid or not content:
            raise DataError(f"Synthetic record {index} has empty uid/content")
        if uid in document_uids:
            raise DataError(f"Duplicate synthetic uid {uid!r}")
        document_uids.add(uid)

        if not isinstance(targets, list) or len(targets) != 1:
            raise DataError(f"Synthetic record {index} must target exactly one label")

        target_id = int(targets[0])
        if target_id not in unseen_set:
            raise DataError(f"Synthetic record {index} targets non-unseen label {target_id}")

        rows_by_label[target_id].append(row)

    train_rows: list[dict[str, Any]] = []
    val_rows: list[dict[str, Any]] = []
    train_counts: Counter[int] = Counter()

    for label_id in unseen_ids:
        label_rows = rows_by_label[label_id]

        if len(label_rows) != examples_per_label:
            raise DataError(
                f"Unseen label {label_id} has {len(label_rows)} examples; "
                f"expected {examples_per_label}"
            )

        if val_per_label > 0:
            label_rows = list(label_rows)
            random.Random(seed + label_id * 1_000_003).shuffle(label_rows)

        label_train = label_rows[:train_per_label]
        label_val = label_rows[train_per_label:]

        train_rows.extend(label_train)
        val_rows.extend(label_val)
        train_counts[label_id] = len(label_train)

    if val_per_label > 0:
        random.Random(seed + 11).shuffle(train_rows)
        random.Random(seed + 29).shuffle(val_rows)

    return train_rows, val_rows, train_counts


class SyntheticDataset(Dataset):
    def __init__(
        self,
        rows: Sequence[dict[str, Any]],
        label_to_position: dict[int, int],
    ) -> None:
        self.rows = rows
        self.label_to_position = label_to_position

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[str, int]:
        row = self.rows[index]
        label_id = int(row["target_ind"][0])
        return str(row["content"]), self.label_to_position[label_id]


class EvaluationDataset(Dataset):
    """Documents with targets intersected with the unseen label set."""

    def __init__(
        self,
        rows: Sequence[dict[str, Any]],
        label_to_position: dict[int, int],
    ) -> None:
        self.rows = rows
        self.label_to_position = label_to_position

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[str, list[int]]:
        row = self.rows[index]
        content = str(row["content"]).strip()
        if not content:
            raise DataError(f"Validation row {index} has empty content")

        positions = [
            self.label_to_position[int(label_id)]
            for label_id in row["target_ind"]
            if int(label_id) in self.label_to_position
        ]
        return content, positions


def build_train_collate(tokenizer, num_labels: int, max_length: int):
    def collate(examples):
        texts, positive_positions = zip(*examples)
        encoded = tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        targets = torch.zeros(len(examples), num_labels, dtype=torch.float32)
        rows = torch.arange(len(examples))
        columns = torch.tensor(positive_positions, dtype=torch.long)
        targets[rows, columns] = 1.0
        return encoded, targets

    return collate


def build_eval_collate(tokenizer, num_labels: int, max_length: int):
    def collate(examples):
        texts, positive_positions = zip(*examples)
        encoded = tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        targets = torch.zeros(len(examples), num_labels, dtype=torch.float32)
        for row, positions in enumerate(positive_positions):
            if positions:
                targets[row, positions] = 1.0
        return encoded, targets

    return collate


def positive_weights(
    counts: Counter[int],
    unseen_ids: Sequence[int],
    *,
    total_examples: int,
    cap: float,
) -> torch.Tensor:
    positives = torch.tensor(
        [counts[label_id] for label_id in unseen_ids],
        dtype=torch.float32,
    )

    if torch.any(positives <= 0):
        raise DataError("Every unseen label must have a synthetic training positive")

    return ((float(total_examples) - positives) / positives).clamp(max=cap)


@torch.no_grad()
def verify_checkpoint_embeddings(
    checkpoint_embeddings: torch.Tensor,
    canonical_embeddings: torch.Tensor,
) -> None:
    checkpoint = checkpoint_embeddings.detach().float().cpu()
    canonical = canonical_embeddings.detach().float().cpu()

    if checkpoint.shape != canonical.shape:
        raise DataError("Seen checkpoint embedding shape differs from the configured label matrix")

    if not torch.equal(checkpoint, canonical):
        max_diff = float((checkpoint - canonical).abs().max())
        raise DataError(f"Seen checkpoint embeddings differ from the configured label matrix; max difference={max_diff:.8g}")


def forward_with_base(model: UnseenRepresentationAdapter, encoded: dict[str, torch.Tensor], unseen_embeddings: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute frozen-base and adapted unseen logits from one encoding."""
    with torch.no_grad():
        token_embeddings = model.base_model.encode_documents(
            encoded["input_ids"], encoded["attention_mask"]
        )
        base_queries = model.base_model.project_attention_queries(
            unseen_embeddings
        )
        base_labels = model.base_model.project_label_features(unseen_embeddings)
        base_documents = model.base_model.attend_documents(
            token_embeddings=token_embeddings,
            attention_mask=encoded["attention_mask"],
            attention_queries=base_queries,
        )
        base_scores = model.base_model.cosine_score(
            base_documents,
            base_labels,
            label_bias=None,
        )[0]

    adapted_queries = base_queries + model.attention_adapter(base_queries)
    adapted_labels = base_labels + model.projection_adapter(base_labels)

    adapted_documents = model.base_model.attend_documents(
        token_embeddings=token_embeddings,
        attention_mask=encoded["attention_mask"],
        attention_queries=adapted_queries,
    )
    adapted_scores = model.base_model.cosine_score(
        adapted_documents,
        adapted_labels,
        label_bias=model.unseen_label_bias,
    )[0]

    return adapted_scores, base_scores


@torch.no_grad()
def verify_zero_initialization(
    model: UnseenRepresentationAdapter,
    encoded: dict[str, torch.Tensor],
    unseen_embeddings: torch.Tensor,
) -> float:
    model.eval()
    adapted, base = forward_with_base(model, encoded, unseen_embeddings)
    difference = float((adapted - base).abs().max())

    if difference > 1e-7:
        raise RuntimeError(f"Zero-initialized adapters do not reproduce the frozen base scorer; max difference={difference:.8g}")

    return difference


@torch.no_grad()
def evaluate(
    model: UnseenRepresentationAdapter,
    loader: DataLoader,
    unseen_embeddings: torch.Tensor,
    pos_weight: torch.Tensor,
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype,
) -> dict[str, float]:
    """Compare frozen-base and adapted ranking on validation data."""
    model.eval()

    base_loss_sum = 0.0
    adapted_loss_sum = 0.0
    decision_count = 0
    eligible = 0
    sums = {
        "base_p1": 0.0,
        "adapted_p1": 0.0,
        "base_ndcg5": 0.0,
        "adapted_ndcg5": 0.0,
        "base_r100": 0.0,
        "adapted_r100": 0.0,
    }

    for encoded, targets in loader:
        encoded = {
            key: value.to(device, non_blocking=True)
            for key, value in encoded.items()
        }
        targets = targets.to(device, non_blocking=True)

        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=use_amp,
        ):
            adapted_scores, base_scores = forward_with_base(model, encoded, unseen_embeddings)

            base_loss_sum += float(
                F.binary_cross_entropy_with_logits(
                    base_scores,
                    targets,
                    pos_weight=pos_weight,
                    reduction="sum",
                )
            )
            adapted_loss_sum += float(
                F.binary_cross_entropy_with_logits(
                    adapted_scores,
                    targets,
                    pos_weight=pos_weight,
                    reduction="sum",
                )
            )

        decision_count += targets.numel()
        targets_cpu = targets.bool().cpu()
        true_counts = targets_cpu.sum(dim=1)
        mask = true_counts > 0

        if not mask.any():
            continue

        targets_cpu = targets_cpu[mask]
        true_counts = true_counts[mask]
        eligible += int(mask.sum())

        for prefix, scores in (
            ("base", base_scores.float().cpu()[mask]),
            ("adapted", adapted_scores.float().cpu()[mask]),
        ):
            top1 = torch.topk(scores, k=1, dim=1).indices
            sums[f"{prefix}_p1"] += float(
                targets_cpu.gather(1, top1).float().sum()
            )

            k5 = min(5, scores.size(1))
            top5 = torch.topk(scores, k=k5, dim=1).indices
            relevance = targets_cpu.gather(1, top5).float()
            discounts = 1.0 / torch.log2(
                torch.arange(2, k5 + 2, dtype=torch.float32)
            )
            dcg = (relevance * discounts).sum(dim=1)
            ideal_lengths = torch.minimum(true_counts, torch.tensor(k5))
            idcg = torch.cumsum(discounts, dim=0)[ideal_lengths - 1]
            sums[f"{prefix}_ndcg5"] += float((dcg / idcg).sum())

            k100 = min(100, scores.size(1))
            top100 = torch.topk(scores, k=k100, dim=1).indices
            hits = targets_cpu.gather(1, top100).sum(dim=1)
            sums[f"{prefix}_r100"] += float(
                (hits.float() / true_counts.float()).sum()
            )

    denominator = max(eligible, 1)
    return {
        "base_loss": base_loss_sum / max(decision_count, 1),
        "adapted_loss": adapted_loss_sum / max(decision_count, 1),
        "base_precision_at_1": sums["base_p1"] / denominator,
        "adapted_precision_at_1": sums["adapted_p1"] / denominator,
        "base_ndcg_at_5": sums["base_ndcg5"] / denominator,
        "adapted_ndcg_at_5": sums["adapted_ndcg5"] / denominator,
        "base_recall_at_100": sums["base_r100"] / denominator,
        "adapted_recall_at_100": sums["adapted_r100"] / denominator,
        "eligible_documents": eligible,
    }


def save_history(history: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(history, indent=2), encoding="utf-8")


def save_checkpoint_atomically(
    path: Path,
    model: UnseenRepresentationAdapter,
    seen_embeddings: torch.Tensor,
    seen_ids: Sequence[int],
    seen_texts: Sequence[str],
    unseen_embeddings: torch.Tensor,
    unseen_ids: Sequence[int],
    unseen_texts: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.unlink(missing_ok=True)

    save_combined_checkpoint(
        temporary,
        model,
        seen_embeddings,
        seen_ids,
        seen_texts,
        unseen_embeddings,
        unseen_ids,
        unseen_texts,
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    dataset_cfg = config["dataset"]
    paths = config["paths"]
    model_cfg = config["model"]
    train_cfg = config["unseen_training"]
    outputs = config["outputs"]

    seed = int(train_cfg["seed"])
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True

    use_amp = bool(train_cfg.get("use_amp", True)) and device.type == "cuda"
    amp_dtype = (
        torch.bfloat16
        if use_amp and torch.cuda.is_bf16_supported()
        else torch.float16
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_amp and amp_dtype == torch.float16,
    )

    num_labels = int(dataset_cfg["num_labels"])
    expected_seen = int(dataset_cfg["num_seen_labels"])
    expected_unseen = int(dataset_cfg["num_unseen_labels"])

    labels = load_labels(Path(paths["labels"]), num_labels)

    (
        base_model,
        checkpoint_seen_embeddings,
        seen_ids,
        seen_texts,
    ) = load_seen_checkpoint(
        Path(outputs["seen_checkpoint"]),
        torch.device("cpu"),
    )

    unseen_ids = derive_unseen_ids(
        seen_ids,
        num_labels=num_labels,
        expected_seen=expected_seen,
        expected_unseen=expected_unseen,
    )

    text_fields = dataset_cfg.get("label_text_fields", ["title", "label", "content"])

    unseen_texts = [label_text(labels[label_id], text_fields) for label_id in unseen_ids]

    if base_model.config.document_encoder_name != model_cfg["document_encoder"]:
        raise DataError("Seen checkpoint document encoder does not match config")

    configured_source = model_cfg.get("label_embedding_source")
    if (
        configured_source is not None
        and base_model.config.label_encoder_name != configured_source
    ):
        raise DataError("Seen checkpoint label embedding source does not match config")

    all_embeddings = load_label_embeddings(
        paths["label_embeddings"],
        expected_num_labels=num_labels,
        expected_dimension=model_cfg.get("label_embedding_dimension"),
    )

    canonical_seen_embeddings = select_label_embeddings(all_embeddings, seen_ids)
    verify_checkpoint_embeddings(checkpoint_seen_embeddings, canonical_seen_embeddings)

    unseen_embeddings = select_label_embeddings(all_embeddings, unseen_ids).to(device)
    del all_embeddings
    unseen_embeddings.requires_grad_(False)

    synthetic_rows = read_json_records(Path(paths["synthetic"]))

    examples_per_label = int(train_cfg["examples_per_label"])
    train_per_label = int(train_cfg["synthetic_train_per_label"])
    val_per_label = int(train_cfg["synthetic_val_per_label"])

    train_rows, synthetic_val_rows, train_counts = validate_and_split_synthetic(
        synthetic_rows,
        unseen_ids,
        examples_per_label=examples_per_label,
        train_per_label=train_per_label,
        val_per_label=val_per_label,
        seed=seed,
    )

    unseen_to_position = {label_id: position for position, label_id in enumerate(unseen_ids)}

    base_model.to(device)
    model = UnseenRepresentationAdapter(
        base_model=base_model,
        num_unseen_labels=len(unseen_ids),
        adapter_hidden_size=int(train_cfg["adapter_hidden_size"]),
        adapter_dropout=float(train_cfg["adapter_dropout"]),
    ).to(device)

    unfrozen_base = [
        name
        for name, parameter in model.base_model.named_parameters()
        if parameter.requires_grad
    ]
    if unfrozen_base:
        raise RuntimeError(f"Base classifier parameters remain trainable: {unfrozen_base[:20]}")

    allowed_roots = {
        "attention_adapter",
        "projection_adapter",
        "unseen_label_bias",
    }
    invalid_trainable = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
        and not any(
            name == root or name.startswith(root + ".")
            for root in allowed_roots
        )
    ]
    if invalid_trainable:
        raise RuntimeError(f"Unexpected stage-two trainable parameters: {invalid_trainable[:20]}")

    tokenizer = AutoTokenizer.from_pretrained(
        base_model.config.document_encoder_name
    )

    workers = int(train_cfg["num_workers"])
    train_loader = DataLoader(
        SyntheticDataset(train_rows, unseen_to_position),
        batch_size=int(train_cfg["batch_size"]),
        shuffle=True,
        collate_fn=build_train_collate(
            tokenizer,
            len(unseen_ids),
            base_model.config.max_document_length,
        ),
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
    )

    validation_mode = str(train_cfg.get("validation", "none")).lower()
    val_loader = None

    if validation_mode in ["synthetic", "none"]:
        if not synthetic_val_rows:
            raise DataError("Synthetic validation requested but synthetic_val_per_label is zero")
        val_dataset = EvaluationDataset(
            synthetic_val_rows, unseen_to_position
        )
    elif validation_mode == "real_dev":
        val_dataset = EvaluationDataset(
            read_json_records(Path(paths["dev"])),
            unseen_to_position,
        )
    # elif validation_mode == "none":
    #     val_dataset = None
    else:
        raise ValueError("unseen_training.validation must be synthetic, real_dev, or none")

    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=int(train_cfg["val_batch_size"]),
            shuffle=False,
            collate_fn=build_eval_collate(
                tokenizer,
                len(unseen_ids),
                base_model.config.max_document_length,
            ),
            num_workers=workers,
            pin_memory=device.type == "cuda",
            persistent_workers=workers > 0,
        )

    first_encoded, _ = next(iter(train_loader))
    first_encoded = {
        key: value.to(device)
        for key, value in first_encoded.items()
    }
    initial_difference = verify_zero_initialization(
        model, first_encoded, unseen_embeddings
    )

    pos_weight = positive_weights(
        train_counts,
        unseen_ids,
        total_examples=len(train_rows),
        cap=float(train_cfg["max_pos_weight"]),
    ).to(device)

    trainable_parameters = model.trainable_parameters()
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )

    epochs = int(train_cfg["epochs"])
    grad_clip = float(train_cfg["gradient_clip_norm"])
    print_every = int(train_cfg.get("print_every_steps", 50))
    skip_nonfinite = bool(
        train_cfg.get("skip_nonfinite_gradients", False)
    )

    output_path = Path(outputs["unseen_checkpoint"])
    history_path = Path(outputs["unseen_history"])
    history: list[dict[str, Any]] = []

    print("=" * 70)
    print(f"UNSEEN ADAPTER TRAINING: {dataset_cfg['name']}")
    print("=" * 70)
    print(f"Device:                {device}")
    print(f"Seen checkpoint:       {outputs['seen_checkpoint']}")
    print(f"Unseen labels:         {len(unseen_ids):,}")
    print(f"Synthetic examples:    {len(synthetic_rows):,}")
    print(
        f"Synthetic split/label: {train_per_label} train / "
        f"{val_per_label} validation"
    )
    print(f"Validation:            {validation_mode}")
    print(f"Batch size:            {train_cfg['batch_size']}")
    print(f"Epochs:                {epochs}")
    print(f"Learning rate:         {train_cfg['learning_rate']}")
    print(f"Positive weight cap:   {train_cfg['max_pos_weight']}")
    print(f"Adapter hidden size:   {train_cfg['adapter_hidden_size']}")
    print(f"Initial base diff:     {initial_difference:.3e}")
    print(f"Output:                {output_path}")
    print()

    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        completed_steps = 0
        skipped_steps = 0
        gradient_sum = 0.0
        gradient_max = 0.0

        for step, (encoded, targets) in enumerate(train_loader, start=1):
            encoded = {key: value.to(device, non_blocking=True) for key, value in encoded.items()}
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=use_amp,
            ):
                scores = model(
                    input_ids=encoded["input_ids"],
                    attention_mask=encoded["attention_mask"],
                    unseen_label_embeddings=unseen_embeddings,
                )
                loss = F.binary_cross_entropy_with_logits(
                    scores,
                    targets,
                    pos_weight=pos_weight,
                )

            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at epoch {epoch}, step {step}")

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                trainable_parameters,
                max_norm=grad_clip,
            )
            gradient_value = float(gradient_norm)

            if not math.isfinite(gradient_value):
                if skip_nonfinite:
                    if scaler.is_enabled():
                        scaler.step(optimizer)
                        scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    skipped_steps += 1
                    print(f"WARNING: skipped non-finite gradient at epoch {epoch}, step {step}")
                    continue
                raise RuntimeError(f"Non-finite gradient at epoch {epoch}, step {step}")

            scaler.step(optimizer)
            scaler.update()

            loss_sum += float(loss.detach())
            completed_steps += 1
            gradient_sum += gradient_value
            gradient_max = max(gradient_max, gradient_value)

            if step % print_every == 0 or step == len(train_loader):
                print(
                    f"Epoch {epoch:02d}/{epochs} | "
                    f"step {step:,}/{len(train_loader):,} | "
                    f"loss {loss_sum / max(completed_steps, 1):.6f}"
                )

        record: dict[str, Any] = {
            "epoch": epoch,
            "train_loss": loss_sum / max(completed_steps, 1),
            "learning_rate": optimizer.param_groups[0]["lr"],
            "mean_gradient_norm": gradient_sum / max(completed_steps, 1),
            "max_gradient_norm": gradient_max,
            "skipped_steps": skipped_steps,
            "logit_scale": float(model.base_model.logit_scale.detach().exp()),
            "mean_unseen_label_bias": float(
                model.unseen_label_bias.detach().mean()
            ),
            "mean_absolute_unseen_label_bias": float(
                model.unseen_label_bias.detach().abs().mean()
            ),
        }

        if val_loader is not None:
            val_metrics = evaluate(
                model,
                val_loader,
                unseen_embeddings,
                pos_weight,
                device,
                use_amp,
                amp_dtype,
            )
            record.update({f"val_{key}": value for key, value in val_metrics.items()})

            print(
                f"Epoch {epoch:02d} | "
                f"train={record['train_loss']:.6f} | "
                f"base/adapted P@1="
                f"{val_metrics['base_precision_at_1']:.4f}/"
                f"{val_metrics['adapted_precision_at_1']:.4f} | "
                f"base/adapted nDCG@5="
                f"{val_metrics['base_ndcg_at_5']:.4f}/"
                f"{val_metrics['adapted_ndcg_at_5']:.4f} | "
                f"base/adapted R@100="
                f"{val_metrics['base_recall_at_100']:.4f}/"
                f"{val_metrics['adapted_recall_at_100']:.4f}"
            )
        else:
            print(
                f"Epoch {epoch:02d} | train={record['train_loss']:.6f}"
            )

        history.append(record)
        save_history(history, history_path)
        save_checkpoint_atomically(output_path, model, checkpoint_seen_embeddings, seen_ids, seen_texts, unseen_embeddings, unseen_ids, unseen_texts)

    print(f"\nTraining complete: {output_path.resolve()}")


if __name__ == "__main__":
    main()
