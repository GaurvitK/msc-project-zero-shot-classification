from __future__ import annotations

import argparse
import json
import math
import os
import random
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset, Subset
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from model import (
    BaseClassifier,
    BaseClassifierConfig,
    load_label_embeddings,
    load_seen_checkpoint,
    save_seen_checkpoint,
    select_label_embeddings,
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
    rows = []
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
        raise DataError(
            f"Expected {expected_count:,} labels, found {len(rows):,}"
        )

    labels = {}
    for position, row in enumerate(rows):
        label_id = int(row["uid"])
        if label_id != position:
            raise DataError(
                f"Labels must be in UID order: expected {position}, "
                f"found {label_id}"
            )
        labels[label_id] = row

    return labels


def label_text(row: dict[str, Any], fields: Sequence[str]) -> str:
    values = []
    seen = set()

    for field in fields:
        value = str(row.get(field, "")).strip()
        key = value.casefold()
        if value and key not in seen:
            values.append(value)
            seen.add(key)

    return " | ".join(values) if values else str(row["uid"])


def validate_row(row: dict[str, Any], labels: dict[int, dict[str, Any]], source: str) -> list[int]:
    missing = {"uid", "content", "target_ind"} - row.keys()
    if missing:
        raise DataError(f"{source}: missing fields {sorted(missing)}")

    if not str(row["uid"]).strip() or not str(row["content"]).strip():
        raise DataError(f"{source}: empty uid/content")

    targets = row["target_ind"]
    if not isinstance(targets, list) or not targets:
        raise DataError(f"{source}: target_ind must be a non-empty list")

    target_ids = [int(value) for value in targets]
    if len(target_ids) != len(set(target_ids)):
        raise DataError(f"{source}: duplicate target IDs")

    unknown = [label_id for label_id in target_ids if label_id not in labels]
    if unknown:
        raise DataError(f"{source}: unknown label IDs {unknown[:20]}")

    return target_ids


def scan_rows(rows: Sequence[dict[str, Any]], labels: dict[int, dict[str, Any]], count_positives: bool) -> tuple[set[int], Counter[int]]:
    supported = set()
    counts = Counter()

    for index, row in enumerate(rows):
        target_ids = validate_row(row, labels, f"record {index}")
        supported.update(target_ids)
        if count_positives:
            counts.update(target_ids)

    return supported, counts


def scan_jsonl(path: Path, labels: dict[int, dict[str, Any]], count_positives: bool) -> tuple[list[int], set[int], Counter[int]]:
    """Scan large JSONL files without storing document text in memory."""
    offsets = []
    supported = set()
    counts = Counter()

    with path.open("rb") as file:
        while True:
            offset = file.tell()
            line = file.readline()
            if not line:
                break
            if not line.strip():
                continue

            row = json.loads(line)
            target_ids = validate_row(
                row, labels, f"{path.name} record {len(offsets)}"
            )
            offsets.append(offset)
            supported.update(target_ids)

            if count_positives:
                counts.update(target_ids)

            if len(offsets) % 100_000 == 0:
                print(f"  scanned {len(offsets):,} rows from {path.name}")

    return offsets, supported, counts


class MemoryDataset(Dataset):
    def __init__(self, rows: Sequence[dict[str, Any]], label_to_position: dict[int, int], allow_unseen: bool) -> None:
        self.rows = rows
        self.label_to_position = label_to_position
        self.allow_unseen = allow_unseen

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[str, list[int]]:
        row = self.rows[index]
        targets = [int(value) for value in row["target_ind"]]

        if not self.allow_unseen:
            outside = [
                label_id
                for label_id in targets
                if label_id not in self.label_to_position
            ]
            if outside:
                raise DataError(f"Training row has unseen labels: {outside[:20]}")

        positions = [
            self.label_to_position[label_id]
            for label_id in targets
            if label_id in self.label_to_position
        ]
        return str(row["content"]), positions


class JsonlDataset(Dataset):
    def __init__(self, path: Path, offsets: Sequence[int], label_to_position: dict[int, int], allow_unseen: bool) -> None:
        self.path = path
        self.offsets = list(offsets)
        self.label_to_position = label_to_position
        self.allow_unseen = allow_unseen
        self._file = None
        self._pid = None

    def __len__(self) -> int:
        return len(self.offsets)

    def _handle(self):
        pid = os.getpid()
        if self._file is None or self._pid != pid:
            if self._file is not None:
                self._file.close()
            self._file = self.path.open("rb")
            self._pid = pid
        return self._file

    def __getitem__(self, index: int) -> tuple[str, list[int]]:
        file = self._handle()
        file.seek(self.offsets[index])
        row = json.loads(file.readline())

        targets = [int(value) for value in row["target_ind"]]
        if not self.allow_unseen:
            outside = [
                label_id
                for label_id in targets
                if label_id not in self.label_to_position
            ]
            if outside:
                raise DataError(f"Training row has unseen labels: {outside[:20]}")

        positions = [
            self.label_to_position[label_id]
            for label_id in targets
            if label_id in self.label_to_position
        ]
        return str(row["content"]), positions


def build_datasets(mode: str, train_path: Path, dev_path: Path, labels: dict[int, dict[str, Any]]):
    if mode == "in_memory":
        train_rows = read_json_records(train_path)
        dev_rows = read_json_records(dev_path)

        seen_set, counts = scan_rows(train_rows, labels, True)
        dev_labels, _ = scan_rows(dev_rows, labels, False)

        seen_ids = sorted(seen_set)
        label_to_position = {
            label_id: position for position, label_id in enumerate(seen_ids)
        }

        return (
            MemoryDataset(train_rows, label_to_position, False),
            MemoryDataset(dev_rows, label_to_position, True),
            seen_ids,
            counts,
            dev_labels,
        )

    if mode == "jsonl_offsets":
        train_offsets, seen_set, counts = scan_jsonl(train_path, labels, True)
        dev_offsets, dev_labels, _ = scan_jsonl(dev_path, labels, False)

        seen_ids = sorted(seen_set)
        label_to_position = {
            label_id: position for position, label_id in enumerate(seen_ids)
        }

        return (
            JsonlDataset(train_path, train_offsets, label_to_position, False),
            JsonlDataset(dev_path, dev_offsets, label_to_position, True),
            seen_ids,
            counts,
            dev_labels,
        )

    raise ValueError("dataset.data_mode must be in_memory or jsonl_offsets")


def build_collate(tokenizer, num_labels: int, max_length: int):
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


def positive_weights(counts: Counter[int], seen_ids: Sequence[int], num_documents: int, cap: float) -> torch.Tensor:
    positives = torch.tensor(
        [counts[label_id] for label_id in seen_ids],
        dtype=torch.float32,
    )

    if torch.any(positives <= 0):
        raise DataError("Every seen label must have a training positive")

    return (
        (float(num_documents) - positives) / positives
    ).clamp(max=cap)


def score_chunks(model: BaseClassifier, encoded: dict[str, torch.Tensor], label_embeddings: torch.Tensor, chunk_size: int) -> torch.Tensor:
    token_embeddings = model.encode_documents(
        encoded["input_ids"], encoded["attention_mask"]
    )
    scores = []

    for start in range(0, label_embeddings.size(0), chunk_size):
        end = min(start + chunk_size, label_embeddings.size(0))
        scores.append(
            model.score_from_hidden(
                token_embeddings=token_embeddings,
                attention_mask=encoded["attention_mask"],
                label_embeddings=label_embeddings[start:end],
                label_bias=model.seen_label_bias[start:end],
            )[0]
        )

    return torch.cat(scores, dim=1)


def train_batch(
    model: BaseClassifier,
    encoded: dict[str, torch.Tensor],
    targets: torch.Tensor,
    label_embeddings: torch.Tensor,
    pos_weight: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    use_amp: bool,
    amp_dtype: torch.dtype,
    chunk_size: int | None,
) -> float:
    optimizer.zero_grad(set_to_none=True)

    if chunk_size is None:
        with torch.autocast(
            device_type=encoded["input_ids"].device.type,
            dtype=amp_dtype,
            enabled=use_amp,
        ):
            logits = model(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
                label_embeddings=label_embeddings,
            )
            loss = F.binary_cross_entropy_with_logits(
                logits, targets, pos_weight=pos_weight
            )

        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite training loss")

        scaler.scale(loss).backward()
        return float(loss.detach())

    # Exact full-label BCE, split only to reduce [B, L, T] attention memory.
    token_embeddings = None
    with torch.autocast(
        device_type=encoded["input_ids"].device.type,
        dtype=amp_dtype,
        enabled=use_amp,
    ):
        token_embeddings = model.encode_documents(
            encoded["input_ids"], encoded["attention_mask"]
        )

    total_labels = label_embeddings.size(0)
    total_decisions = targets.size(0) * total_labels
    total_loss = 0.0

    for start in range(0, total_labels, chunk_size):
        end = min(start + chunk_size, total_labels)

        with torch.autocast(
            device_type=encoded["input_ids"].device.type,
            dtype=amp_dtype,
            enabled=use_amp,
        ):
            logits = model.score_from_hidden(
                token_embeddings=token_embeddings,
                attention_mask=encoded["attention_mask"],
                label_embeddings=label_embeddings[start:end],
                label_bias=model.seen_label_bias[start:end],
            )[0]
            loss = F.binary_cross_entropy_with_logits(
                logits,
                targets[:, start:end],
                pos_weight=pos_weight[start:end],
                reduction="sum",
            ) / total_decisions

        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss for labels [{start}:{end})")

        total_loss += float(loss.detach())
        scaler.scale(loss).backward(retain_graph=end < total_labels)

    return total_loss


@torch.no_grad()
def evaluate(
    model: BaseClassifier,
    loader: DataLoader,
    label_embeddings: torch.Tensor,
    pos_weight: torch.Tensor,
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype,
    chunk_size: int,
) -> dict[str, float]:
    model.eval()

    loss_sum = 0.0
    decision_count = 0
    eligible = 0
    p1_sum = p5_sum = ndcg5_sum = r100_sum = 0.0

    for encoded, targets in loader:
        encoded = {key: value.to(device) for key, value in encoded.items()}
        targets = targets.to(device)

        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=use_amp,
        ):
            scores = score_chunks(model, encoded, label_embeddings, chunk_size)
            loss_sum += float(
                F.binary_cross_entropy_with_logits(
                    scores,
                    targets,
                    pos_weight=pos_weight,
                    reduction="sum",
                )
            )
            decision_count += targets.numel()

        scores = scores.float().cpu()
        targets = targets.bool().cpu()
        true_counts = targets.sum(dim=1)
        mask = true_counts > 0

        if not mask.any():
            continue

        scores = scores[mask]
        targets = targets[mask]
        true_counts = true_counts[mask]
        eligible += int(mask.sum())

        top1 = torch.topk(scores, k=1, dim=1).indices
        p1_sum += float(targets.gather(1, top1).float().sum())

        k5 = min(5, scores.size(1))
        top5 = torch.topk(scores, k=k5, dim=1).indices
        rel5 = targets.gather(1, top5)
        p5_sum += float((rel5.sum(dim=1).float() / k5).sum())

        discounts = 1.0 / torch.log2(
            torch.arange(2, k5 + 2, dtype=torch.float32)
        )
        dcg = (rel5.float() * discounts).sum(dim=1)
        ideal_len = torch.minimum(true_counts, torch.tensor(k5))
        idcg = torch.cumsum(discounts, dim=0)[ideal_len - 1]
        ndcg5_sum += float((dcg / idcg).sum())

        k100 = min(100, scores.size(1))
        top100 = torch.topk(scores, k=k100, dim=1).indices
        hits100 = targets.gather(1, top100).sum(dim=1)
        r100_sum += float(
            (hits100.float() / true_counts.float()).sum()
        )

    denominator = max(eligible, 1)
    return {
        "loss": loss_sum / max(decision_count, 1),
        "precision_at_1": p1_sum / denominator,
        "precision_at_5": p5_sum / denominator,
        "ndcg_at_5": ndcg5_sum / denominator,
        "recall_at_100": r100_sum / denominator,
    }


def save_history(history: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(history, indent=2), encoding="utf-8")


def save_training_state(
    path: Path,
    model: BaseClassifier,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: torch.amp.GradScaler,
    epoch: int,
    history: list[dict[str, Any]],
    seen_ids: Sequence[int],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "completed_epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict":
                scheduler.state_dict() if scheduler is not None else None,
            "scaler_state_dict": scaler.state_dict(),
            "history": history,
            "seen_label_ids": list(seen_ids),
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    dataset_cfg = config["dataset"]
    paths = config["paths"]
    model_cfg = config["model"]
    train_cfg = config["seen_training"]
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
        train_dataset,
        dev_dataset,
        seen_ids,
        counts,
        dev_labels,
    ) = build_datasets(
        dataset_cfg["data_mode"],
        Path(paths["train"]),
        Path(paths["dev"]),
        labels,
    )

    # unseen ids fetched by checking labels not present in training, but in test or in lbl.json
    unseen_ids = sorted(set(labels) - set(seen_ids))
    if len(seen_ids) != expected_seen or len(unseen_ids) != expected_unseen:
        raise DataError(f"Configured seen/unseen counts do not match the training split: got {len(seen_ids):,}/{len(unseen_ids):,}")

    if dataset_cfg.get("require_dev_no_unseen", False):
        leaked = set(unseen_ids) & dev_labels
        if leaked:
            raise DataError(f"{len(leaked)} unseen labels occur in dev")

    text_fields = dataset_cfg.get(
        "label_text_fields", ["title", "label", "content"]
    )
    seen_texts = [label_text(labels[label_id], text_fields) for label_id in seen_ids]

    all_embeddings = load_label_embeddings(
        paths["label_embeddings"],
        expected_num_labels=num_labels,
        expected_dimension=model_cfg.get("label_embedding_dimension"),
    )
    seen_embeddings = select_label_embeddings(all_embeddings, seen_ids).to(device)
    del all_embeddings
    seen_embeddings.requires_grad_(False)

    resume_checkpoint = train_cfg.get("resume_checkpoint")
    resume_epoch = int(train_cfg.get("resume_epoch", 0))

    if resume_checkpoint:
        model, checkpoint_embeddings, checkpoint_ids, checkpoint_texts = (
            load_seen_checkpoint(Path(resume_checkpoint), device)
        )
        if checkpoint_ids != seen_ids or checkpoint_texts != seen_texts:
            raise DataError("Resume checkpoint does not match current labels")
        if not torch.equal(
            checkpoint_embeddings.cpu(), seen_embeddings.cpu()
        ):
            raise DataError("Resume checkpoint uses different label embeddings")
    else:
        model = BaseClassifier(
            BaseClassifierConfig(
                document_encoder_name=model_cfg["document_encoder"],
                label_embedding_size=seen_embeddings.size(1),
                num_seen_labels=len(seen_ids),
                label_encoder_name=model_cfg.get(
                    "label_embedding_source", "precomputed"
                ),
                max_document_length=int(model_cfg["max_document_length"]),
                dropout=float(model_cfg["dropout"]),
            )
        ).to(device)

    if model.config.document_encoder_name != model_cfg["document_encoder"]:
        raise DataError("Document encoder differs between checkpoint and config")

    train_chunk = train_cfg.get("label_chunk_size")
    train_chunk = int(train_chunk) if train_chunk is not None else None

    if model_cfg.get("gradient_checkpointing", False):
        if train_chunk is not None:
            raise ValueError(
                "Disable gradient checkpointing when label chunking is enabled"
            )
        if hasattr(model.document_encoder, "gradient_checkpointing_enable"):
            model.document_encoder.gradient_checkpointing_enable()
        if hasattr(model.document_encoder.config, "use_cache"):
            model.document_encoder.config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(model.config.document_encoder_name)
    collate = build_collate(
        tokenizer,
        len(seen_ids),
        int(model_cfg["max_document_length"]),
    )

    monitor_docs = train_cfg.get("dev_monitor_documents")
    if monitor_docs is not None and int(monitor_docs) < len(dev_dataset):
        indices = sorted(
            random.Random(seed).sample(
                range(len(dev_dataset)), int(monitor_docs)
            )
        )
        dev_dataset = Subset(dev_dataset, indices)

    workers = int(train_cfg["num_workers"])
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=True,
        collate_fn=collate,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=int(train_cfg["dev_batch_size"]),
        shuffle=False,
        collate_fn=collate,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
    )

    pos_weight = positive_weights(
        counts,
        seen_ids,
        len(train_dataset),
        float(train_cfg["max_pos_weight"]),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )

    total_epochs = int(train_cfg["epochs"])
    start_epoch = resume_epoch + 1

    scheduler_name = str(train_cfg.get("scheduler", "none")).lower()
    scheduler = None
    if scheduler_name == "linear":
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(train_cfg.get("warmup_steps", 0)),
            num_training_steps=(total_epochs - resume_epoch) * len(train_loader),
        )
    elif scheduler_name != "none":
        raise ValueError("scheduler must be 'none' or 'linear'")

    history = []
    state_path_value = outputs.get("seen_training_state")
    state_path = Path(state_path_value) if state_path_value else None

    # Exact resume for runs created by this unified script.
    if (
        train_cfg.get("resume_if_available", False)
        and state_path is not None
        and state_path.is_file()
    ):
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        if list(state["seen_label_ids"]) != seen_ids:
            raise DataError("Training-state label ordering mismatch")

        model.load_state_dict(state["model_state_dict"], strict=True)
        optimizer.load_state_dict(state["optimizer_state_dict"])

        if scheduler is not None and state.get("scheduler_state_dict"):
            scheduler.load_state_dict(state["scheduler_state_dict"])
        if state.get("scaler_state_dict"):
            scaler.load_state_dict(state["scaler_state_dict"])

        history = list(state.get("history", []))
        start_epoch = int(state["completed_epoch"]) + 1

    eval_chunk = int(
        train_cfg.get("eval_label_chunk_size")
        or train_chunk
        or len(seen_ids)
    )

    output_path = Path(outputs["seen_checkpoint"])
    history_path = Path(outputs["seen_history"])

    print("=" * 70)
    print(f"SEEN TRAINING: {dataset_cfg['name']}")
    print("=" * 70)
    print(f"Device:             {device}")
    print(f"Train/dev docs:     {len(train_dataset):,}/{len(dev_dataset):,}")
    print(f"Total/seen/unseen:  {num_labels:,}/{len(seen_ids):,}/{len(unseen_ids):,}")
    print(f"Label embeddings:   {tuple(seen_embeddings.shape)}")
    print(f"Epochs:             {start_epoch}-{total_epochs}")
    print(f"Learning rate:      {train_cfg['learning_rate']}")
    print(f"Train label chunk:  {train_chunk}")
    print(f"Scheduler:          {scheduler_name}")
    print(f"Output:             {output_path}")
    print()

    if start_epoch > total_epochs:
        print(f"Already trained through epoch {start_epoch - 1}.")
        return

    print_every = int(train_cfg.get("print_every_batches", 100))
    grad_clip = float(train_cfg["gradient_clip_norm"])
    skip_nonfinite = bool(
        train_cfg.get("skip_nonfinite_gradients", False)
    )

    for epoch in range(start_epoch, total_epochs + 1):
        model.train()
        loss_sum = 0.0
        completed_batches = 0
        skipped_batches = 0

        for batch_index, (encoded, targets) in enumerate(train_loader, start=1):
            encoded = {key: value.to(device) for key, value in encoded.items()}
            targets = targets.to(device)

            batch_loss = train_batch(
                model,
                encoded,
                targets,
                seen_embeddings,
                pos_weight,
                optimizer,
                scaler,
                use_amp,
                amp_dtype,
                train_chunk,
            )

            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), grad_clip
            )

            if not math.isfinite(float(grad_norm)):
                if skip_nonfinite:
                    if scaler.is_enabled():
                        scaler.step(optimizer)  # skipped internally after inf detection
                        scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    skipped_batches += 1
                    print(
                        f"WARNING: skipped non-finite gradient at "
                        f"epoch {epoch}, batch {batch_index}"
                    )
                    continue

                raise RuntimeError(
                    f"Non-finite gradient at epoch {epoch}, batch {batch_index}"
                )

            scaler.step(optimizer)
            scaler.update()

            if scheduler is not None:
                scheduler.step()

            loss_sum += batch_loss
            completed_batches += 1

            if batch_index % print_every == 0 or batch_index == len(train_loader):
                print(
                    f"Epoch {epoch:02d}/{total_epochs} | "
                    f"batch {batch_index:,}/{len(train_loader):,} | "
                    f"loss {loss_sum / max(completed_batches, 1):.6f}"
                )

        dev = evaluate(model, dev_loader, seen_embeddings, pos_weight, device, use_amp, amp_dtype, eval_chunk)

        record = {
            "epoch": epoch,
            "train_loss": loss_sum / max(completed_batches, 1),
            "learning_rate": optimizer.param_groups[0]["lr"],
            "skipped_batches": skipped_batches,
            "logit_scale": float(model.logit_scale.detach().exp()),
            **{f"dev_{key}": value for key, value in dev.items()},
        }
        history.append(record)

        print(
            f"Epoch {epoch:02d} | "
            f"train={record['train_loss']:.6f} | "
            f"dev={dev['loss']:.6f} | "
            f"P@1={dev['precision_at_1']:.4f} | "
            f"P@5={dev['precision_at_5']:.4f} | "
            f"nDCG@5={dev['ndcg_at_5']:.4f} | "
            f"R@100={dev['recall_at_100']:.4f}"
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_seen_checkpoint(output_path, model, seen_embeddings, seen_ids, seen_texts)
        save_history(history, history_path)

        if state_path is not None:
            save_training_state(state_path, model, optimizer, scheduler, scaler, epoch, history, seen_ids)

    print(f"\nTraining complete: {output_path.resolve()}")


if __name__ == "__main__":
    main()
