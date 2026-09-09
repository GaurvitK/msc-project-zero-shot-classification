from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterator, Sequence

import matplotlib.pyplot as plt
import yaml
from transformers import AutoTokenizer


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("Config must contain a YAML mapping")

    return config


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
        return parsed if isinstance(parsed, list) else [parsed]

    decoder = json.JSONDecoder()
    rows = []
    position = 0

    while position < len(text):
        while position < len(text) and text[position].isspace():
            position += 1

        if position >= len(text):
            break

        row, position = decoder.raw_decode(text, position)
        rows.append(row)

    return rows


def iter_records(path: Path, data_mode: str) -> Iterator[dict[str, Any]]:
    """Stream JSONL for large datasets, otherwise use the generic JSON reader."""
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


def percentile(values: Sequence[int], q: float) -> float:
    if not values:
        return 0.0

    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)

    if lower == upper:
        return float(ordered[lower])

    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def summary(values: Sequence[int]) -> dict[str, float | int]:
    if not values:
        return {"mean": 0.0, "median": 0.0, "min": 0, "max": 0}

    return {
        "mean": float(mean(values)),
        "median": float(median(values)),
        "min": int(min(values)),
        "max": int(max(values)),
    }


def frequency_summary(values: Sequence[int]) -> dict[str, float | int]:
    if not values:
        return {}

    return {
        "min": int(min(values)),
        "p25": percentile(values, 0.25),
        "median": percentile(values, 0.50),
        "p75": percentile(values, 0.75),
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": int(max(values)),
        "mean": float(mean(values)),
        "frequency_1_labels": int(sum(value == 1 for value in values)),
        "frequency_2_to_5_labels": int(sum(2 <= value <= 5 for value in values)),
        "frequency_at_most_5_labels": int(sum(value <= 5 for value in values)),
    }


def collect_split(
    path: Path,
    data_mode: str,
    tokenizer: Any | None,
    token_batch_size: int,
) -> tuple[dict[str, Any], Counter[int]]:
    documents = 0
    assignments = 0
    labels_per_document: list[int] = []
    words_per_document: list[int] = []
    tokens_per_document: list[int] = []
    frequencies: Counter[int] = Counter()
    token_texts: list[str] = []

    def flush_tokens() -> None:
        if tokenizer is None or not token_texts:
            return

        encoded = tokenizer(
            token_texts,
            add_special_tokens=True,
            truncation=False,
            padding=False,
        )
        tokens_per_document.extend(len(ids) for ids in encoded["input_ids"])
        token_texts.clear()

    for row in iter_records(path, data_mode):
        content = str(row["content"])
        targets = [int(value) for value in row["target_ind"]]

        documents += 1
        assignments += len(targets)
        labels_per_document.append(len(targets))
        words_per_document.append(len(content.split()))
        frequencies.update(set(targets))

        if tokenizer is not None:
            token_texts.append(content)

            if len(token_texts) >= token_batch_size:
                flush_tokens()

    flush_tokens()

    result = {
        "documents": documents,
        "assignments": assignments,
        "unique_labels": len(frequencies),
        "labels_per_document": summary(labels_per_document),
        "words_per_document": summary(words_per_document),
    }

    if tokenizer is not None:
        result["tokens_per_document"] = summary(tokens_per_document)

    return result, frequencies


def unseen_support(
    path: Path,
    data_mode: str,
    unseen_ids: set[int],
) -> dict[str, Any]:
    frequencies: Counter[int] = Counter()
    documents = 0
    assignments = 0

    for row in iter_records(path, data_mode):
        unseen_targets = [
            int(value)
            for value in row["target_ind"]
            if int(value) in unseen_ids
        ]

        if unseen_targets:
            documents += 1
            assignments += len(unseen_targets)
            frequencies.update(unseen_targets)

    frequency_distribution = Counter(frequencies.values())

    return {
        "supported_unseen_labels": len(frequencies),
        "documents_with_unseen_labels": documents,
        "unseen_label_assignments": assignments,
        "unseen_label_frequency_distribution": {
            str(frequency): count
            for frequency, count in sorted(frequency_distribution.items())
        },
    }


def label_text(row: dict[str, Any]) -> str:
    for field in ("eurovoc_title", "title", "label", "content"):
        value = str(row.get(field, "")).strip()

        if value:
            return value

    return str(row["uid"])


def save_frequency_outputs(
    train_frequencies: Counter[int],
    label_path: Path,
    output_dir: Path,
    dataset_name: str,
) -> tuple[Path, Path]:
    labels = read_json_records(label_path)
    labels_by_id = {int(row["uid"]): row for row in labels}

    csv_path = output_dir / "seen_label_train_frequency.csv"

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["label_id", "label", "train_frequency"])

        for label_id in sorted(train_frequencies):
            writer.writerow(
                [
                    label_id,
                    label_text(labels_by_id[label_id]),
                    train_frequencies[label_id],
                ]
            )

    ordered_frequencies = sorted(train_frequencies.values(), reverse=True)
    figure_path = output_dir / "seen_label_frequency_distribution.png"

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(ordered_frequencies) + 1), ordered_frequencies)
    plt.yscale("log")
    plt.xlabel("Seen labels ordered by training frequency")
    plt.ylabel("Training frequency (log scale)")
    plt.title(f"{dataset_name} seen-label training frequency")
    plt.tight_layout()
    plt.savefig(figure_path, dpi=300)
    plt.close()

    return csv_path, figure_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--no-token-lengths", action="store_true")
    parser.add_argument("--token-batch-size", type=int, default=128)
    args = parser.parse_args()

    config = load_config(args.config)
    dataset = config["dataset"]
    paths = config["paths"]
    model = config["model"]

    data_mode = dataset["data_mode"]
    train_path = Path(paths["train"])
    dev_path = Path(paths["dev"])
    test_path = Path(paths["test"])
    label_path = Path(paths["labels"])

    output_dir = args.output_dir or train_path.parent / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = None

    if not args.no_token_lengths:
        tokenizer = AutoTokenizer.from_pretrained(model["document_encoder"])

    print("Analyzing training split...")
    train_stats, train_frequencies = collect_split(
        train_path,
        data_mode,
        tokenizer,
        args.token_batch_size,
    )

    print("Analyzing development split...")
    dev_stats, _ = collect_split(
        dev_path,
        data_mode,
        tokenizer,
        args.token_batch_size,
    )

    print("Analyzing test split...")
    test_stats, _ = collect_split(
        test_path,
        data_mode,
        tokenizer,
        args.token_batch_size,
    )

    total_labels = int(dataset["num_labels"])
    seen_ids = sorted(train_frequencies)
    unseen_ids = sorted(set(range(total_labels)) - set(seen_ids))

    seen_frequency_values = [
        train_frequencies[label_id]
        for label_id in seen_ids
    ]

    result = {
        "dataset": dataset["name"],
        "labels": {
            "total": total_labels,
            "seen": len(seen_ids),
            "unseen": len(unseen_ids),
        },
        "splits": {
            "train": train_stats,
            "dev": dev_stats,
            "test": test_stats,
        },
        "seen_label_train_frequency": frequency_summary(seen_frequency_values),
        "unseen_support": {
            "dev": unseen_support(dev_path, data_mode, set(unseen_ids)),
            "test": unseen_support(test_path, data_mode, set(unseen_ids)),
        },
    }

    statistics_path = output_dir / "dataset_statistics.json"
    statistics_path.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    csv_path, figure_path = save_frequency_outputs(
        train_frequencies,
        label_path,
        output_dir,
        dataset["name"],
    )

    print("=" * 72)
    print(f"DATASET ANALYSIS: {dataset['name']}")
    print("=" * 72)
    print(
        f"Train/dev/test documents: "
        f"{train_stats['documents']:,}/"
        f"{dev_stats['documents']:,}/"
        f"{test_stats['documents']:,}"
    )
    print(
        f"Labels total/seen/unseen: "
        f"{total_labels:,}/{len(seen_ids):,}/{len(unseen_ids):,}"
    )
    print(
        f"Average train labels/document: "
        f"{train_stats['labels_per_document']['mean']:.4f}"
    )

    if "tokens_per_document" in train_stats:
        print(
            f"Average train tokens/document: "
            f"{train_stats['tokens_per_document']['mean']:.2f}"
        )

    test_unseen = result["unseen_support"]["test"]
    print(
        f"Test unseen support: "
        f"{test_unseen['supported_unseen_labels']:,} labels, "
        f"{test_unseen['documents_with_unseen_labels']:,} documents, "
        f"{test_unseen['unseen_label_assignments']:,} assignments"
    )

    print()
    print(f"Statistics:    {statistics_path.resolve()}")
    print(f"Frequency CSV: {csv_path.resolve()}")
    print(f"Frequency plot:{figure_path.resolve()}")


if __name__ == "__main__":
    main()
