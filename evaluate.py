from __future__ import annotations

import argparse
import json
import math
import os
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterator, Sequence

import torch
import yaml
from torch.utils.data import DataLoader, Dataset, Subset
from transformers import AutoTokenizer

from model import (
    UnseenRepresentationAdapter,
    load_combined_checkpoint,
    load_label_embeddings,
    select_label_embeddings,
)


class EvaluationDataError(RuntimeError):
    pass


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("Config must contain a YAML mapping")

    return config


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_json_records(path: Path) -> list[dict[str, Any]]:
    """Read a JSON array, JSONL file, or consecutive JSON objects."""
    if not path.is_file():
        raise FileNotFoundError(path.resolve())

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
        raise EvaluationDataError(f"{path} must contain JSON objects")

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
            raise EvaluationDataError(f"{path} contains a non-object record")

        rows.append(row)

    return rows


def iter_records(path: Path, data_mode: str,) -> Iterator[dict[str, Any]]:
    """Iterate records without loading large AmazonCat splits into memory."""
    if data_mode == "jsonl_offsets":
        with path.open("r", encoding="utf-8-sig") as file:
            for line in file:
                if line.strip():
                    yield json.loads(line)
        return

    if data_mode == "in_memory":
        yield from read_json_records(path)
        return

    raise ValueError("dataset.data_mode must be 'in_memory' or 'jsonl_offsets'")


def scan_offsets(path: Path) -> list[int]:
    offsets: list[int] = []

    with path.open("rb") as file:
        while True:
            offset = file.tell()
            line = file.readline()

            if not line:
                break
            if not line.strip():
                continue

            offsets.append(offset)

    return offsets


class MemoryDataset(Dataset):
    def __init__(self, rows: Sequence[dict[str, Any]]) -> None:
        self.rows = list(rows)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int,) -> tuple[str, list[int]]:
        row = self.rows[index]
        return (str(row["content"]), [int(x) for x in row["target_ind"]],)


class JsonlDataset(Dataset):
    def __init__(self, path: Path, offsets: Sequence[int],) -> None:
        self.path = path
        self.offsets = list(offsets)
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

    def __getitem__(self, index: int,) -> tuple[str, list[int]]:
        file = self._handle()
        file.seek(self.offsets[index])
        row = json.loads(file.readline())

        return (str(row["content"]), [int(x) for x in row["target_ind"]],)


def make_dataset(path: Path, data_mode: str,) -> Dataset:
    if data_mode == "in_memory":
        return MemoryDataset(read_json_records(path))

    if data_mode == "jsonl_offsets":
        return JsonlDataset(path, scan_offsets(path),)

    raise ValueError("dataset.data_mode must be 'in_memory' or 'jsonl_offsets'")


def collate_examples(examples: Sequence[tuple[str, list[int]]],) -> tuple[list[str], list[list[int]]]:
    texts, targets = zip(*examples)
    return list(texts), list(targets)


def make_loader(
    dataset: Dataset,
    *,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_examples,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )


def make_targets(
    target_ids: Sequence[Sequence[int]],
    positions: dict[int, int],
    number_of_labels: int,
    device: torch.device,
) -> torch.Tensor:
    targets = torch.zeros(len(target_ids), number_of_labels, dtype=torch.bool, device=device,)

    for row_index, labels in enumerate(target_ids):
        local = [positions[label_id] for label_id in labels if label_id in positions]

        if local:
            targets[row_index, local] = True

    return targets


def training_frequencies(
    path: Path,
    data_mode: str,
    label_ids: Sequence[int],
) -> tuple[Counter[int], int]:
    allowed = set(int(x) for x in label_ids)
    counts: Counter[int] = Counter()
    document_count = 0

    for row in iter_records(path, data_mode):
        document_count += 1

        for label_id in set(int(x) for x in row.get("target_ind", [])):
            if label_id in allowed:
                counts[label_id] += 1

    return counts, document_count


def inverse_propensity(
    counts: Counter[int],
    label_ids: Sequence[int],
    document_count: int,
    *,
    a: float,
    b: float,
) -> torch.Tensor:
    frequencies = torch.tensor([counts[label_id] for label_id in label_ids], dtype=torch.float64,)

    constant = (math.log(float(document_count)) - 1.0) * (b + 1.0) ** a

    return (1.0 + constant * torch.pow(frequencies + b, -a)).float()


def safe_divide(numerator: torch.Tensor, denominator: torch.Tensor,) -> torch.Tensor:
    return torch.where(
        denominator > 0,
        numerator.double() / denominator.double(),
        torch.zeros_like(denominator, dtype=torch.float64,),
    )


class RankingAccumulator:
    """P@K, R@K, nDCG@K and normalized dataset-level PSP@K."""

    def __init__(
        self,
        name: str,
        number_of_labels: int,
        k_values: Sequence[int],
        inverse_propensity_weights: torch.Tensor | None,
    ) -> None:
        self.name = name
        self.number_of_labels = int(number_of_labels)
        self.k_values = tuple(sorted(set(int(k) for k in k_values)))
        self.inverse_propensity = (
            inverse_propensity_weights.detach().cpu()
            if inverse_propensity_weights is not None
            else None
        )

        self.total_documents = 0
        self.eligible_documents = 0
        self.support = torch.zeros(self.number_of_labels, dtype=torch.int64,)

        self.precision_sum = {k: 0.0 for k in self.k_values}
        self.recall_sum = {k: 0.0 for k in self.k_values}
        self.ndcg_sum = {k: 0.0 for k in self.k_values}
        self.psp_numerator = {k: 0.0 for k in self.k_values}
        self.psp_denominator = {k: 0.0 for k in self.k_values}

    @torch.no_grad()
    def update(self, scores: torch.Tensor, targets: torch.Tensor,) -> None:
        if scores.shape != targets.shape:
            raise ValueError("scores and targets must have the same shape")

        self.total_documents += targets.size(0)
        self.support += targets.sum(dim=0).cpu()

        true_count = targets.sum(dim=1)
        eligible = true_count > 0
        count = int(eligible.sum().item())
        self.eligible_documents += count

        if count == 0:
            return

        scores = scores[eligible]
        targets = targets[eligible]
        true_count = true_count[eligible]

        max_k = min(max(self.k_values), self.number_of_labels,)

        top_positions = torch.topk(scores, k=max_k, dim=1,).indices

        relevance = targets.gather(1, top_positions,)

        if self.inverse_propensity is not None:
            weights = self.inverse_propensity.to(scores.device, dtype=torch.float64,)
            true_weights = (targets.to(torch.float64) * weights.unsqueeze(0))

        for k in self.k_values:
            kk = min(k, self.number_of_labels)
            positions_k = top_positions[:, :kk]
            relevance_k = relevance[:, :kk]
            hits = relevance_k.sum(dim=1)

            self.precision_sum[k] += float((hits.to(torch.float64) / float(kk)).sum().item())

            self.recall_sum[k] += float(
                (hits.to(torch.float64) / true_count.to(torch.float64)).sum().item()
            )

            discounts = 1.0 / torch.log2(
                torch.arange(2, kk + 2, dtype=torch.float64, device=scores.device,)
            )

            dcg = (relevance_k.to(torch.float64) * discounts.unsqueeze(0)).sum(dim=1)

            ideal_lengths = torch.minimum(
                true_count,
                torch.tensor(kk, dtype=true_count.dtype, device=true_count.device,),
            )

            idcg = torch.cumsum(discounts, dim=0,)[ideal_lengths - 1]

            self.ndcg_sum[k] += float((dcg / idcg).sum().item())

            if self.inverse_propensity is not None:
                predicted_weights = weights[positions_k]

                weighted_hits = (relevance_k.to(torch.float64) * predicted_weights).sum(dim=1)

                ideal_weighted_hits = torch.topk(true_weights, k=kk, dim=1,).values.sum(dim=1)

                self.psp_numerator[k] += float(weighted_hits.sum().item())
                self.psp_denominator[k] += float(ideal_weighted_hits.sum().item())

    def compute(self) -> dict[str, Any]:
        eligible = max(self.eligible_documents, 1,)

        result: dict[str, Any] = {
            "name": self.name,
            "number_of_labels":
                self.number_of_labels,
            "supported_labels_in_test":
                int((self.support > 0).sum().item()),
            "total_documents":
                self.total_documents,
            "ranking_eligible_documents":
                self.eligible_documents,
        }

        for k in self.k_values:
            result[f"precision_at_{k}"] = (self.precision_sum[k] / eligible)
            result[f"recall_at_{k}"] = (self.recall_sum[k] / eligible)
            result[f"ndcg_at_{k}"] = (self.ndcg_sum[k] / eligible)

            if self.inverse_propensity is not None:
                denominator = self.psp_denominator[k]
                result[f"psp_at_{k}"] = (self.psp_numerator[k] / denominator if denominator > 0 else 0.0)

        return result


class ZeroShotRecallAccumulator:
    """Recall of true unseen labels in the complete label ranking."""

    def __init__(
        self,
        k_values: Sequence[int],
        number_of_seen_labels: int,
        number_of_unseen_labels: int,
    ) -> None:
        self.k_values = tuple(sorted(set(int(k) for k in k_values)))
        self.number_of_seen_labels = int(number_of_seen_labels)
        self.number_of_unseen_labels = int(number_of_unseen_labels)

        self.recall_sum = {k: 0.0 for k in self.k_values}
        self.eligible_documents = 0
        self.true_unseen_assignments = 0

    @torch.no_grad()
    def update(self, overall_scores: torch.Tensor, unseen_targets: torch.Tensor,) -> None:
        unseen_count = unseen_targets.sum(dim=1)
        eligible = unseen_count > 0

        if not bool(eligible.any()):
            return

        scores = overall_scores[eligible]
        targets = unseen_targets[eligible]
        true_count = unseen_count[eligible]

        self.eligible_documents += int(eligible.sum().item())
        self.true_unseen_assignments += int(true_count.sum().item())

        max_k = min(max(self.k_values), scores.size(1),)

        top_positions = torch.topk(scores, k=max_k, dim=1,).indices

        unseen_positions = (top_positions - self.number_of_seen_labels)

        valid = ((unseen_positions >= 0) & (unseen_positions < self.number_of_unseen_labels))

        safe_positions = unseen_positions.clamp(min=0, max=self.number_of_unseen_labels - 1,)

        hits_at_rank = (targets.gather(1, safe_positions) & valid)

        cumulative_hits = (hits_at_rank.to(torch.int64) .cumsum(dim=1))

        for k in self.k_values:
            kk = min(k, scores.size(1))
            hits = cumulative_hits[:, kk - 1]

            self.recall_sum[k] += float(
                (hits.to(torch.float64) / true_count.to(torch.float64)).sum().item()
            )

    def compute(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "eligible_documents":
                self.eligible_documents,
            "true_unseen_assignments":
                self.true_unseen_assignments,
        }

        denominator = max(self.eligible_documents, 1,)

        for k in self.k_values:
            result[f"zsr_at_{k}"] = (self.recall_sum[k] / denominator)

        return result


class ClassificationAccumulator:
    """Micro, macro, sample, Hamming and exact-match metrics."""

    def __init__(self, number_of_labels: int, device: torch.device,) -> None:
        self.number_of_labels = int(number_of_labels)
        self.device = device

        self.tp = torch.zeros(number_of_labels, dtype=torch.int64, device=device,)
        self.fp = torch.zeros_like(self.tp)
        self.fn = torch.zeros_like(self.tp)
        self.support = torch.zeros_like(self.tp)

        self.total_documents = 0
        self.hamming_errors = 0
        self.exact_matches = 0
        self.true_count = 0
        self.predicted_count = 0

        self.sample_precision_sum = 0.0
        self.sample_recall_sum = 0.0
        self.sample_f1_sum = 0.0

    @torch.no_grad()
    def update(self, predictions: torch.Tensor, targets: torch.Tensor,) -> None:
        self.tp += (predictions & targets).sum(dim=0)
        self.fp += (predictions & ~targets).sum(dim=0)
        self.fn += (~predictions & targets).sum(dim=0)
        self.support += targets.sum(dim=0)

        self.total_documents += targets.size(0)
        self.hamming_errors += int((predictions != targets).sum().item())
        self.exact_matches += int((predictions == targets) .all(dim=1) .sum() .item())

        true_per_doc = targets.sum(dim=1)
        pred_per_doc = predictions.sum(dim=1)
        self.true_count += int(true_per_doc.sum().item())
        self.predicted_count += int(pred_per_doc.sum().item())

        sample_tp = (predictions & targets).sum(dim=1)

        sample_precision = safe_divide(sample_tp, pred_per_doc,)
        sample_recall = safe_divide(sample_tp, true_per_doc,)
        sample_f1 = safe_divide(2 * sample_precision * sample_recall, sample_precision + sample_recall,)

        self.sample_precision_sum += float(sample_precision.sum().item())
        self.sample_recall_sum += float(sample_recall.sum().item())
        self.sample_f1_sum += float(sample_f1.sum().item())

    def compute(self) -> dict[str, Any]:
        total_tp = self.tp.sum()
        total_fp = self.fp.sum()
        total_fn = self.fn.sum()

        micro_p = safe_divide(total_tp, total_tp + total_fp,)
        micro_r = safe_divide(total_tp, total_tp + total_fn,)
        micro_f1 = safe_divide(2 * micro_p * micro_r, micro_p + micro_r,)

        label_p = safe_divide(self.tp, self.tp + self.fp,)
        label_r = safe_divide(self.tp, self.tp + self.fn,)
        label_f1 = safe_divide(2 * label_p * label_r, label_p + label_r,)

        supported = self.support > 0

        if bool(supported.any()):
            macro_p = float(label_p[supported].mean().item())
            macro_r = float(label_r[supported].mean().item())
            macro_f1 = float(label_f1[supported].mean().item())
        else:
            macro_p = macro_r = macro_f1 = 0.0

        docs = max(self.total_documents, 1,)
        decisions = max(self.total_documents * self.number_of_labels, 1,)

        return {
            "micro_precision":
                float(micro_p.item()),
            "micro_recall":
                float(micro_r.item()),
            "micro_f1":
                float(micro_f1.item()),
            "macro_precision":
                macro_p,
            "macro_recall":
                macro_r,
            "macro_f1":
                macro_f1,
            "sample_precision":
                self.sample_precision_sum / docs,
            "sample_recall":
                self.sample_recall_sum / docs,
            "sample_f1":
                self.sample_f1_sum / docs,
            "hamming_loss":
                self.hamming_errors / decisions,
            "exact_match_ratio":
                self.exact_matches / docs,
            "average_true_labels_per_document":
                self.true_count / docs,
            "average_predicted_labels_per_document":
                self.predicted_count / docs,
            "supported_labels":
                int(supported.sum().item()),
        }


class ThresholdSearch:
    """Choose one threshold by micro-F1."""

    def __init__(self, values: Sequence[float], device: torch.device,) -> None:
        self.thresholds = torch.tensor([float(x) for x in values], dtype=torch.float32, device=device,)

        size = len(values)
        self.tp = torch.zeros(size, dtype=torch.int64, device=device,)
        self.fp = torch.zeros_like(self.tp)
        self.fn = torch.zeros_like(self.tp)

    @torch.no_grad()
    def update(self, probabilities: torch.Tensor, targets: torch.Tensor,) -> None:
        predictions = (probabilities.unsqueeze(0) >= self.thresholds[:, None, None])
        truth = targets.unsqueeze(0)

        self.tp += (predictions & truth).sum(dim=(1, 2))
        self.fp += (predictions & ~truth).sum(dim=(1, 2))
        self.fn += (~predictions & truth).sum(dim=(1, 2))

    def result(self) -> dict[str, float]:
        precision = safe_divide(self.tp, self.tp + self.fp,)
        recall = safe_divide(self.tp, self.tp + self.fn,)
        f1 = safe_divide(2 * precision * recall, precision + recall,)

        index = int(torch.argmax(f1).item())

        return {
            "threshold":
                float(self.thresholds[index].item()),
            "micro_precision":
                float(precision[index].item()),
            "micro_recall":
                float(recall[index].item()),
            "micro_f1":
                float(f1[index].item()),
        }


@torch.no_grad()
def build_static_features(
    model: UnseenRepresentationAdapter,
    seen_embeddings: torch.Tensor,
    unseen_embeddings: torch.Tensor,
) -> dict[str, torch.Tensor]:
    model.eval()

    base = model.base_model

    seen_queries = (base.project_attention_queries(seen_embeddings))
    seen_labels = (base.project_label_features(seen_embeddings))

    unseen_base_queries = (base.project_attention_queries(unseen_embeddings))
    unseen_base_labels = (base.project_label_features(unseen_embeddings))

    unseen_queries = (unseen_base_queries + model.attention_adapter(unseen_base_queries))
    unseen_labels = (unseen_base_labels + model.projection_adapter(unseen_base_labels))

    return {
        "seen_queries": seen_queries,
        "seen_labels": seen_labels,
        "unseen_queries": unseen_queries,
        "unseen_labels": unseen_labels,
    }


@torch.no_grad()
def score_group_in_chunks(
    model: UnseenRepresentationAdapter,
    token_embeddings: torch.Tensor,
    attention_mask: torch.Tensor,
    queries: torch.Tensor,
    label_features: torch.Tensor,
    label_bias: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    scores: list[torch.Tensor] = []

    for start in range(0, queries.size(0), chunk_size,):
        end = min(start + chunk_size, queries.size(0),)

        documents = (
            model.base_model.attend_documents(
                token_embeddings=token_embeddings,
                attention_mask=attention_mask,
                attention_queries=queries[start:end],
            )
        )

        logits = model.base_model.cosine_score(
            document_features=documents,
            label_features=label_features[start:end],
            label_bias=label_bias[start:end],
        )[0]

        scores.append(logits)

    return torch.cat(scores, dim=1,)


@torch.no_grad()
def score_batch(
    model: UnseenRepresentationAdapter,
    tokenizer: Any,
    texts: Sequence[str],
    static: dict[str, torch.Tensor],
    *,
    device: torch.device,
    max_length: int,
    label_chunk_size: int,
    use_amp: bool,
    amp_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    encoded = tokenizer(
        list(texts),
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    encoded = {key: value.to(device, non_blocking=True,) for key, value in encoded.items()}

    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp,):
        token_embeddings = (
            model.base_model.encode_documents(encoded["input_ids"], encoded["attention_mask"],)
        )

        seen_scores = score_group_in_chunks(
            model,
            token_embeddings,
            encoded["attention_mask"],
            static["seen_queries"],
            static["seen_labels"],
            model.base_model.seen_label_bias,
            label_chunk_size,
        )

        unseen_scores = score_group_in_chunks(
            model,
            token_embeddings,
            encoded["attention_mask"],
            static["unseen_queries"],
            static["unseen_labels"],
            model.unseen_label_bias,
            label_chunk_size,
        )

    return seen_scores, unseen_scores


def synthetic_validation_rows(
    rows: Sequence[dict[str, Any]],
    unseen_ids: Sequence[int],
    *,
    examples_per_label: int,
    train_per_label: int,
    val_per_label: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Recreate the same deterministic per-label split used by train_unseen.py."""
    if (train_per_label + val_per_label != examples_per_label):
        raise ValueError("Synthetic split does not sum to examples_per_label")

    rows_by_label: dict[int, list[dict[str, Any]],] = defaultdict(list)

    for row in rows:
        targets = row["target_ind"]

        if (not isinstance(targets, list) or len(targets) != 1):
            raise EvaluationDataError("Synthetic rows must target exactly one unseen label")

        rows_by_label[int(targets[0])].append(row)

    validation: list[dict[str, Any]] = []

    for label_id in unseen_ids:
        label_rows = list(rows_by_label[label_id])

        if len(label_rows) != examples_per_label:
            raise EvaluationDataError(
                f"Unseen label {label_id} has "
                f"{len(label_rows)} synthetic rows; "
                f"expected {examples_per_label}"
            )

        random.Random(seed + label_id * 1_000_003).shuffle(label_rows)

        validation.extend(label_rows[train_per_label: train_per_label + val_per_label])

    random.Random(seed + 29).shuffle(validation)

    return validation


@torch.no_grad()
def calibrate_thresholds(
    *,
    config: dict[str, Any],
    model: UnseenRepresentationAdapter,
    tokenizer: Any,
    static: dict[str, torch.Tensor],
    seen_ids: list[int],
    unseen_ids: list[int],
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype,
) -> dict[str, Any]:
    evaluation = config["evaluation"]
    classification = evaluation["classification"]

    mode = str(classification.get("threshold_mode", "fixed",)).lower()

    if mode == "fixed":
        return {
            "mode": "fixed",
            "seen": float(classification["seen_threshold"]),
            "unseen": float(classification["unseen_threshold"]),
        }

    if mode != "calibrated":
        raise ValueError("evaluation.classification.threshold_mode " "must be 'fixed' or 'calibrated'")

    threshold_values = classification.get(
        "threshold_values",
        [value / 100 for value in range(5, 100, 5)],
    )

    seen_search = ThresholdSearch(threshold_values, device,)
    unseen_search = ThresholdSearch(threshold_values, device,)

    seen_positions = {label_id: position for position, label_id in enumerate(seen_ids)}
    unseen_positions = {label_id: position for position, label_id in enumerate(unseen_ids)}

    dataset_cfg = config["dataset"]
    paths = config["paths"]

    dev_dataset = make_dataset(Path(paths["dev"]), dataset_cfg["data_mode"],)

    calibration_docs = classification.get("seen_calibration_documents")

    if (calibration_docs is not None and int(calibration_docs) < len(dev_dataset)):
        indices = sorted(
            random.Random(int(evaluation["seed"])).sample(
                range(len(dev_dataset)),
                int(calibration_docs),
            )
        )
        dev_dataset = Subset(dev_dataset, indices,)

    dev_loader = make_loader(
        dev_dataset,
        batch_size=int(evaluation["batch_size"]),
        num_workers=int(evaluation["num_workers"]),
        device=device,
    )

    for texts, targets in dev_loader:
        seen_scores, _ = score_batch(
            model,
            tokenizer,
            texts,
            static,
            device=device,
            max_length=int(config["model"]["max_document_length"]),
            label_chunk_size=int(evaluation["label_chunk_size"]),
            use_amp=use_amp,
            amp_dtype=amp_dtype,
        )

        seen_targets = make_targets(targets, seen_positions, len(seen_ids), device,)

        seen_search.update(torch.sigmoid(seen_scores), seen_targets,)

    synthetic_rows = read_json_records(Path(paths["synthetic"]))
    unseen_training = config["unseen_training"]

    validation_rows = synthetic_validation_rows(
        synthetic_rows,
        unseen_ids,
        examples_per_label=int(unseen_training["examples_per_label"]),
        train_per_label=int(unseen_training["synthetic_train_per_label"]),
        val_per_label=int(unseen_training["synthetic_val_per_label"]),
        seed=int(unseen_training["seed"]),
    )

    if not validation_rows:
        raise EvaluationDataError(
            "Unseen threshold calibration requested but "
            "no held-out synthetic validation rows exist"
        )

    synthetic_loader = make_loader(
        MemoryDataset(validation_rows),
        batch_size=int(evaluation["batch_size"]),
        num_workers=0,
        device=device,
    )

    for texts, targets in synthetic_loader:
        _, unseen_scores = score_batch(
            model,
            tokenizer,
            texts,
            static,
            device=device,
            max_length=int(config["model"]["max_document_length"]),
            label_chunk_size=int(evaluation["label_chunk_size"]),
            use_amp=use_amp,
            amp_dtype=amp_dtype,
        )

        unseen_targets = make_targets(targets, unseen_positions, len(unseen_ids), device,)

        unseen_search.update(torch.sigmoid(unseen_scores), unseen_targets,)

    seen_result = seen_search.result()
    unseen_result = unseen_search.result()

    return {
        "mode": "calibrated",
        "seen": seen_result["threshold"],
        "unseen": unseen_result["threshold"],
        "seen_calibration": seen_result,
        "unseen_calibration": unseen_result,
        "uses_test_labels": False,
    }


@torch.no_grad()
def evaluate_test(
    *,
    config: dict[str, Any],
    model: UnseenRepresentationAdapter,
    tokenizer: Any,
    static: dict[str, torch.Tensor],
    seen_ids: list[int],
    unseen_ids: list[int],
    seen_inverse_propensity: torch.Tensor | None,
    unseen_inverse_propensity: torch.Tensor | None,
    overall_inverse_propensity: torch.Tensor | None,
    thresholds: dict[str, Any],
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype,
) -> dict[str, Any]:
    evaluation = config["evaluation"]
    dataset_cfg = config["dataset"]

    seen_positions = {label_id: position for position, label_id in enumerate(seen_ids)}
    unseen_positions = {label_id: position for position, label_id in enumerate(unseen_ids)}

    ranking_k = [int(k) for k in evaluation["ranking_k"]]
    zsr_k = [int(k) for k in evaluation["zsr_k"]]

    seen_metrics = RankingAccumulator("seen", len(seen_ids), ranking_k, seen_inverse_propensity,)
    unseen_metrics = RankingAccumulator(
        "adapted_unseen",
        len(unseen_ids),
        ranking_k,
        unseen_inverse_propensity,
    )
    overall_metrics = RankingAccumulator(
        "overall",
        len(seen_ids) + len(unseen_ids),
        ranking_k,
        overall_inverse_propensity,
    )
    zsr_metrics = ZeroShotRecallAccumulator(zsr_k, len(seen_ids), len(unseen_ids),)

    classification_enabled = bool(evaluation["classification"].get("enabled", True,))
    classification_metrics = (
        ClassificationAccumulator(len(seen_ids) + len(unseen_ids), device,)
        if classification_enabled
        else None
    )

    test_dataset = make_dataset(Path(config["paths"]["test"]), dataset_cfg["data_mode"],)

    max_test_documents = evaluation.get("max_test_documents")
    if (max_test_documents is not None and int(max_test_documents) < len(test_dataset)):
        test_dataset = Subset(test_dataset, range(int(max_test_documents)),)

    loader = make_loader(
        test_dataset,
        batch_size=int(evaluation["batch_size"]),
        num_workers=int(evaluation["num_workers"]),
        device=device,
    )

    print_every = int(evaluation.get("print_every_batches", 100,))

    for step, (texts, target_ids) in enumerate(loader, start=1,):
        seen_scores, unseen_scores = score_batch(
            model,
            tokenizer,
            texts,
            static,
            device=device,
            max_length=int(config["model"]["max_document_length"]),
            label_chunk_size=int(evaluation["label_chunk_size"]),
            use_amp=use_amp,
            amp_dtype=amp_dtype,
        )

        seen_targets = make_targets(target_ids, seen_positions, len(seen_ids), device,)
        unseen_targets = make_targets(target_ids, unseen_positions, len(unseen_ids), device,)

        overall_scores = torch.cat([seen_scores, unseen_scores], dim=1,)
        overall_targets = torch.cat([seen_targets, unseen_targets], dim=1,)

        seen_metrics.update(seen_scores, seen_targets,)
        unseen_metrics.update(unseen_scores, unseen_targets,)
        overall_metrics.update(overall_scores, overall_targets,)
        zsr_metrics.update(overall_scores, unseen_targets,)

        if classification_metrics is not None:
            seen_predictions = (torch.sigmoid(seen_scores) >= float(thresholds["seen"]))
            unseen_predictions = (torch.sigmoid(unseen_scores) >= float(thresholds["unseen"]))

            classification_metrics.update(
                torch.cat([seen_predictions, unseen_predictions,], dim=1,),
                overall_targets,
            )

        if (step % print_every == 0 or step == len(loader)):
            print(f"  test batch {step:,}/{len(loader):,}")

    result = {
        "seen": seen_metrics.compute(),
        "adapted_unseen":
            unseen_metrics.compute(),
        "overall":
            overall_metrics.compute(),
        "zsr":
            zsr_metrics.compute(),
    }

    if classification_metrics is not None:
        result["classification"] = (classification_metrics.compute())

    return result


def print_results(metrics: dict[str, Any], ranking_k: Sequence[int], zsr_k: Sequence[int],) -> None:
    overall = metrics["overall"]

    print("\nOVERALL RANKING")
    print("-" * 68)

    for k in ranking_k:
        line = (
            f"K={k:<3} "
            f"P={100 * overall[f'precision_at_{k}']:.2f}  "
            f"R={100 * overall[f'recall_at_{k}']:.2f}  "
            f"nDCG={100 * overall[f'ndcg_at_{k}']:.2f}"
        )

        psp_key = f"psp_at_{k}"
        if psp_key in overall:
            line += (f"  PSP=" f"{100 * overall[psp_key]:.2f}")

        print(line)

    print("\nZERO-SHOT RECALL")
    print("-" * 68)

    for k in zsr_k:
        print(f"ZSR@{k:<3} " f"{100 * metrics['zsr'][f'zsr_at_{k}']:.2f}")

    if "classification" in metrics:
        classification = metrics["classification"]

        print("\nCLASSIFICATION")
        print("-" * 68)
        print(
            "Micro P/R/F1:  "
            f"{100 * classification['micro_precision']:.2f} / "
            f"{100 * classification['micro_recall']:.2f} / "
            f"{100 * classification['micro_f1']:.2f}"
        )
        print(
            "Macro P/R/F1:  "
            f"{100 * classification['macro_precision']:.2f} / "
            f"{100 * classification['macro_recall']:.2f} / "
            f"{100 * classification['macro_f1']:.2f}"
        )
        print(
            "Sample P/R/F1: "
            f"{100 * classification['sample_precision']:.2f} / "
            f"{100 * classification['sample_recall']:.2f} / "
            f"{100 * classification['sample_f1']:.2f}"
        )
        print(f"Hamming loss:   " f"{classification['hamming_loss']:.6f}")
        print(f"Exact match:    " f"{100 * classification['exact_match_ratio']:.2f}%")
        print(
            "Pred/true labels per document: "
            f"{classification['average_predicted_labels_per_document']:.2f}/"
            f"{classification['average_true_labels_per_document']:.2f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True,)
    args = parser.parse_args()

    config = load_config(args.config)
    dataset_cfg = config["dataset"]
    model_cfg = config["model"]
    evaluation = config["evaluation"]
    outputs = config["outputs"]

    seed = int(evaluation["seed"])
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    use_tf32 = bool(evaluation.get("allow_tf32", False,))
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = (use_tf32)
        torch.backends.cudnn.allow_tf32 = (use_tf32)

    use_amp = bool(evaluation.get("use_amp", True,)) and device.type == "cuda"

    amp_dtype = (torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16)

    (
        model,
        seen_embeddings,
        seen_ids,
        seen_texts,
        unseen_embeddings,
        unseen_ids,
        unseen_texts,
    ) = load_combined_checkpoint(Path(outputs["unseen_checkpoint"]), device,)

    expected_seen = int(dataset_cfg["num_seen_labels"])
    expected_unseen = int(dataset_cfg["num_unseen_labels"])
    expected_total = int(dataset_cfg["num_labels"])

    if (
        len(seen_ids) != expected_seen
        or len(unseen_ids) != expected_unseen
        or len(seen_ids) + len(unseen_ids)
        != expected_total
    ):
        raise EvaluationDataError("Checkpoint label counts do not match config")

    if set(seen_ids) & set(unseen_ids):
        raise EvaluationDataError("Checkpoint seen and unseen label sets overlap")

    if (model.base_model.config.document_encoder_name != model_cfg["document_encoder"]):
        raise EvaluationDataError("Checkpoint document encoder does not match config")

    configured_source = model_cfg.get("label_embedding_source")
    if (
        configured_source is not None
        and model.base_model.config.label_encoder_name
        != configured_source
    ):
        raise EvaluationDataError("Checkpoint label embedding source does not match config")

    canonical = load_label_embeddings(
        config["paths"]["label_embeddings"],
        expected_num_labels=expected_total,
        expected_dimension=model_cfg.get("label_embedding_dimension"),
    )

    canonical_seen = select_label_embeddings(canonical, seen_ids,)
    canonical_unseen = select_label_embeddings(canonical, unseen_ids,)

    if not torch.equal(canonical_seen, seen_embeddings.cpu(),):
        raise EvaluationDataError("Checkpoint seen embeddings differ from configured matrix")

    if not torch.equal(canonical_unseen, unseen_embeddings.cpu(),):
        raise EvaluationDataError("Checkpoint unseen embeddings differ from configured matrix")

    del canonical
    del canonical_seen
    del canonical_unseen

    tokenizer = AutoTokenizer.from_pretrained(model.base_model.config.document_encoder_name)

    static = build_static_features(model, seen_embeddings, unseen_embeddings,)

    psp_cfg = evaluation.get("psp", {"enabled": False},)
    use_psp = bool(psp_cfg.get("enabled", False,))

    seen_inverse = None
    unseen_inverse = None
    overall_inverse = None

    if use_psp:
        all_ids = seen_ids + unseen_ids
        counts, train_document_count = (
            training_frequencies(Path(config["paths"]["train"]), dataset_cfg["data_mode"], all_ids,)
        )

        psp_a = float(psp_cfg["a"])
        psp_b = float(psp_cfg["b"])

        seen_inverse = inverse_propensity(
            counts,
            seen_ids,
            train_document_count,
            a=psp_a,
            b=psp_b,
        ).to(device)

        unseen_inverse = inverse_propensity(
            counts,
            unseen_ids,
            train_document_count,
            a=psp_a,
            b=psp_b,
        ).to(device)

        overall_inverse = inverse_propensity(
            counts,
            all_ids,
            train_document_count,
            a=psp_a,
            b=psp_b,
        ).to(device)

    thresholds = calibrate_thresholds(
        config=config,
        model=model,
        tokenizer=tokenizer,
        static=static,
        seen_ids=seen_ids,
        unseen_ids=unseen_ids,
        device=device,
        use_amp=use_amp,
        amp_dtype=amp_dtype,
    )

    print("=" * 72)
    print(f"EVALUATION: {dataset_cfg['name']}")
    print("=" * 72)
    print(f"Device:               {device}")
    print(f"Checkpoint:           {outputs['unseen_checkpoint']}")
    print(f"Total/seen/unseen:    " f"{expected_total:,}/" f"{expected_seen:,}/" f"{expected_unseen:,}")
    print(f"Ranking K:            " f"{evaluation['ranking_k']}")
    print(f"ZSR K:                " f"{evaluation['zsr_k']}")
    print(f"Classification mode:  " f"{thresholds['mode']}")
    print(f"Seen/unseen threshold:" f" {thresholds['seen']:.2f}/" f"{thresholds['unseen']:.2f}")
    print("Test threshold tuning: no")
    print()

    metrics = evaluate_test(
        config=config,
        model=model,
        tokenizer=tokenizer,
        static=static,
        seen_ids=seen_ids,
        unseen_ids=unseen_ids,
        seen_inverse_propensity=seen_inverse,
        unseen_inverse_propensity=unseen_inverse,
        overall_inverse_propensity=overall_inverse,
        thresholds=thresholds,
        device=device,
        use_amp=use_amp,
        amp_dtype=amp_dtype,
    )

    print_results(metrics, evaluation["ranking_k"], evaluation["zsr_k"],)

    results = {
        "dataset": dataset_cfg["name"],
        "checkpoint":
            outputs["unseen_checkpoint"],
        "test_path":
            config["paths"]["test"],
        "thresholds":
            thresholds,
        "metrics":
            metrics,
        "evaluation_contract": {
            "uses_test_labels_for_threshold_calibration":
                False,
            "scores_frozen_base_on_unseen_labels":
                False,
            "seen_scorer":
                "frozen base classifier",
            "unseen_scorer":
                "adapted attention and label representations "
                "+ unseen-label bias",
            "overall_scorer":
                "concatenated seen and adapted-unseen logits",
            "ranking_k_values":
                list(evaluation["ranking_k"]),
            "zsr_k_values":
                list(evaluation["zsr_k"]),
            "zsr_eligibility":
                "test documents with at least one true unseen label",
            "psp_enabled":
                use_psp,
        },
    }

    output_path = Path(outputs["evaluation_results"])
    output_path.parent.mkdir(parents=True, exist_ok=True,)

    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(results, indent=2,), encoding="utf-8",)
    temporary.replace(output_path)

    print(f"\nSaved metrics: " f"{output_path.resolve()}")


if __name__ == "__main__":
    main()