from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import yaml


class SamplingError(RuntimeError):
    pass


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
        if isinstance(parsed, dict):
            return [parsed]
        if isinstance(parsed, list):
            return parsed
        raise SamplingError(f"{path} must contain JSON objects")

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
            raise SamplingError(f"{path} contains a non-object record")

        rows.append(row)

    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_labels(path: Path) -> dict[int, dict[str, Any]]:
    rows = read_json_records(path)
    labels: dict[int, dict[str, Any]] = {}

    for position, row in enumerate(rows):
        if "uid" not in row:
            raise SamplingError(f"Label record {position} is missing uid")

        label_id = int(row["uid"])

        if label_id != position:
            raise SamplingError(
                f"lbl.json must be in UID order: expected {position}, found {label_id}"
            )

        labels[label_id] = row

    return labels


def validate_documents(
    rows: Sequence[dict[str, Any]],
    labels: dict[int, dict[str, Any]],
    split_name: str,
) -> None:
    for index, row in enumerate(rows):
        missing = {"uid", "content", "target_ind"} - row.keys()

        if missing:
            raise SamplingError(
                f"{split_name} record {index}: missing fields {sorted(missing)}"
            )

        if not str(row["uid"]).strip() or not str(row["content"]).strip():
            raise SamplingError(f"{split_name} record {index}: empty uid/content")

        targets = row["target_ind"]

        if not isinstance(targets, list) or not targets:
            raise SamplingError(
                f"{split_name} record {index}: target_ind must be a non-empty list"
            )

        target_ids = [int(value) for value in targets]

        if len(target_ids) != len(set(target_ids)):
            raise SamplingError(f"{split_name} record {index}: duplicate target IDs")

        unknown = [label_id for label_id in target_ids if label_id not in labels]

        if unknown:
            raise SamplingError(
                f"{split_name} record {index}: unknown labels {unknown[:20]}"
            )


def label_frequencies(rows: Sequence[dict[str, Any]]) -> Counter[int]:
    frequencies: Counter[int] = Counter()

    for row in rows:
        frequencies.update(set(int(value) for value in row["target_ind"]))

    return frequencies


def select_labels(
    train_frequency: Counter[int],
    test_frequency: Counter[int],
    *,
    num_seen_labels: int,
    num_unseen_labels: int,
    seed: int,
) -> tuple[list[int], list[int]]:
    """Select the final seen and unseen label sets.

    Unseen labels follow the project split definition: they occur exactly once in
    the original training split and have at least one test example.

    Seen labels are sampled uniformly from the remaining training-supported labels.
    The fixed seed makes the selection deterministic.
    """
    unseen_ids = sorted(
        label_id
        for label_id, frequency in train_frequency.items()
        if frequency == 1 and test_frequency[label_id] > 0
    )

    if len(unseen_ids) != num_unseen_labels:
        raise SamplingError(
            f"Expected {num_unseen_labels:,} unseen labels from "
            f"train_frequency == 1 and test_frequency > 0, found {len(unseen_ids):,}"
        )

    unseen_set = set(unseen_ids)
    seen_candidates = sorted(
        label_id
        for label_id in train_frequency
        if label_id not in unseen_set
    )

    if len(seen_candidates) < num_seen_labels:
        raise SamplingError(
            f"Only {len(seen_candidates):,} seen-label candidates are available, "
            f"cannot select {num_seen_labels:,}"
        )

    selected_seen = sorted(
        random.Random(seed).sample(seen_candidates, num_seen_labels)
    )

    return selected_seen, unseen_ids


def filter_rows(
    rows: Sequence[dict[str, Any]],
    retained_labels: set[int],
) -> list[dict[str, Any]]:
    """Keep only retained targets and drop documents with no remaining labels."""
    filtered: list[dict[str, Any]] = []

    for row in rows:
        targets = [
            int(value)
            for value in row["target_ind"]
            if int(value) in retained_labels
        ]

        if not targets:
            continue

        new_row = dict(row)
        new_row["target_ind"] = targets
        filtered.append(new_row)

    return filtered


def protect_seen_coverage(
    rows: Sequence[dict[str, Any]],
    seen_ids: Sequence[int],
    seed: int,
) -> set[int]:
    """Choose rows so every selected seen label has at least one positive."""
    seen_set = set(seen_ids)
    rows_by_label: dict[int, list[int]] = defaultdict(list)

    for row_index, row in enumerate(rows):
        for label_id in row["target_ind"]:
            label_id = int(label_id)

            if label_id in seen_set:
                rows_by_label[label_id].append(row_index)

    missing = [label_id for label_id in seen_ids if not rows_by_label[label_id]]

    if missing:
        raise SamplingError(
            f"Selected seen labels missing from filtered train: {missing[:20]}"
        )

    label_order = list(seen_ids)
    random.Random(seed).shuffle(label_order)

    protected: set[int] = set()
    covered: set[int] = set()

    for label_id in label_order:
        if label_id in covered:
            continue

        # Prefer a row that covers as many currently uncovered selected labels
        # as possible. This reduces the number of protected rows.
        best_row = max(
            rows_by_label[label_id],
            key=lambda row_index: sum(
                int(target) in seen_set and int(target) not in covered
                for target in rows[row_index]["target_ind"]
            ),
        )

        protected.add(best_row)
        covered.update(
            int(target)
            for target in rows[best_row]["target_ind"]
            if int(target) in seen_set
        )

    if covered != seen_set:
        raise SamplingError("Failed to protect every selected seen label")

    return protected


def sample_training_documents(
    rows: Sequence[dict[str, Any]],
    seen_ids: Sequence[int],
    *,
    sample_size: int,
    seed: int,
) -> list[dict[str, Any]]:
    if sample_size > len(rows):
        raise SamplingError(
            f"Only {len(rows):,} filtered training rows are available; "
            f"cannot sample {sample_size:,}"
        )

    protected = protect_seen_coverage(rows, seen_ids, seed)

    if len(protected) > sample_size:
        raise SamplingError(
            f"Coverage protection requires {len(protected):,} rows, "
            f"larger than requested sample size {sample_size:,}"
        )

    remaining = [
        row_index
        for row_index in range(len(rows))
        if row_index not in protected
    ]

    additional = random.Random(seed + 1).sample(
        remaining,
        sample_size - len(protected),
    )

    selected_indices = sorted(protected | set(additional))

    return [rows[index] for index in selected_indices]


def remap_rows(
    rows: Sequence[dict[str, Any]],
    old_to_new: dict[int, int],
) -> list[dict[str, Any]]:
    remapped: list[dict[str, Any]] = []

    for row in rows:
        targets = sorted(
            {
                old_to_new[int(value)]
                for value in row["target_ind"]
                if int(value) in old_to_new
            }
        )

        if not targets:
            continue

        new_row = dict(row)
        new_row["target_ind"] = targets
        remapped.append(new_row)

    return remapped


def remap_labels(
    labels: dict[int, dict[str, Any]],
    ordered_original_ids: Sequence[int],
) -> list[dict[str, Any]]:
    remapped: list[dict[str, Any]] = []

    for new_id, original_id in enumerate(ordered_original_ids):
        row = dict(labels[original_id])
        row["original_uid"] = original_id
        row["uid"] = new_id
        remapped.append(row)

    return remapped


def verify_subset(
    train_rows: Sequence[dict[str, Any]],
    dev_rows: Sequence[dict[str, Any]],
    test_rows: Sequence[dict[str, Any]],
    *,
    num_seen_labels: int,
    num_unseen_labels: int,
    train_documents: int,
) -> None:
    seen_ids = set(range(num_seen_labels))
    unseen_ids = set(
        range(num_seen_labels, num_seen_labels + num_unseen_labels)
    )

    train_frequency = label_frequencies(train_rows)
    dev_frequency = label_frequencies(dev_rows)
    test_frequency = label_frequencies(test_rows)

    if len(train_rows) != train_documents:
        raise SamplingError(
            f"Expected {train_documents:,} training documents, found {len(train_rows):,}"
        )

    if set(train_frequency) != seen_ids:
        missing = sorted(seen_ids - set(train_frequency))
        extra = sorted(set(train_frequency) - seen_ids)
        raise SamplingError(
            f"Final train label space is incorrect. Missing={missing[:20]}, extra={extra[:20]}"
        )

    if set(dev_frequency) & unseen_ids:
        raise SamplingError("Final development split contains unseen-label leakage")

    if not unseen_ids.issubset(test_frequency):
        missing = sorted(unseen_ids - set(test_frequency))
        raise SamplingError(
            f"Final test split is missing unseen labels: {missing[:20]}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    sampling = config["amazoncat_sampling"]

    source_dir = Path(sampling["source_dir"])
    output_dir = Path(sampling["output_dir"])

    train_path = source_dir / sampling.get("train_file", "trn.json")
    dev_path = source_dir / sampling.get("dev_file", "dev.json")
    test_path = source_dir / sampling.get("test_file", "tst.json")
    label_path = source_dir / sampling.get("label_file", "lbl.json")

    num_seen_labels = int(sampling["num_seen_labels"])
    num_unseen_labels = int(sampling["num_unseen_labels"])
    train_documents = int(sampling["train_documents"])
    seed = int(sampling["seed"])

    print("Loading original AmazonCat data...")
    labels = load_labels(label_path)
    train_rows = read_json_records(train_path)
    dev_rows = read_json_records(dev_path)
    test_rows = read_json_records(test_path)

    validate_documents(train_rows, labels, "train")
    validate_documents(dev_rows, labels, "dev")
    validate_documents(test_rows, labels, "test")

    print("Computing original label frequencies...")
    train_frequency = label_frequencies(train_rows)
    test_frequency = label_frequencies(test_rows)

    selected_seen, selected_unseen = select_labels(
        train_frequency,
        test_frequency,
        num_seen_labels=num_seen_labels,
        num_unseen_labels=num_unseen_labels,
        seed=seed,
    )

    print(
        f"Selected {len(selected_seen):,} seen labels and "
        f"{len(selected_unseen):,} unseen labels"
    )

    # Unseen labels are deliberately excluded from train/dev supervision.
    filtered_train = filter_rows(train_rows, set(selected_seen))
    filtered_dev = filter_rows(dev_rows, set(selected_seen))
    filtered_test = filter_rows(
        test_rows,
        set(selected_seen) | set(selected_unseen),
    )

    print(f"Filtered train documents: {len(filtered_train):,}")
    print("Sampling training documents while preserving seen-label coverage...")

    sampled_train = sample_training_documents(
        filtered_train,
        selected_seen,
        sample_size=train_documents,
        seed=seed,
    )

    # Seen labels occupy the first block; unseen labels occupy the second.
    ordered_original_ids = selected_seen + selected_unseen
    old_to_new = {
        original_id: new_id
        for new_id, original_id in enumerate(ordered_original_ids)
    }

    remapped_train = remap_rows(sampled_train, old_to_new)
    remapped_dev = remap_rows(filtered_dev, old_to_new)
    remapped_test = remap_rows(filtered_test, old_to_new)
    remapped_labels = remap_labels(labels, ordered_original_ids)

    verify_subset(
        remapped_train,
        remapped_dev,
        remapped_test,
        num_seen_labels=num_seen_labels,
        num_unseen_labels=num_unseen_labels,
        train_documents=train_documents,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    write_jsonl(output_dir / "trn.json", remapped_train)
    write_jsonl(output_dir / "dev.json", remapped_dev)
    write_jsonl(output_dir / "tst.json", remapped_test)
    write_jsonl(output_dir / "lbl.json", remapped_labels)

    manifest = {
        "seed": seed,
        "unseen_selection": "original train frequency == 1 and test frequency > 0",
        "seen_selection": "uniform random sample from remaining train-supported labels",
        "num_labels": num_seen_labels + num_unseen_labels,
        "num_seen_labels": num_seen_labels,
        "num_unseen_labels": num_unseen_labels,
        "selected_seen_original_ids": selected_seen,
        "selected_unseen_original_ids": selected_unseen,
        "old_to_new_label_id": {
            str(original_id): new_id
            for original_id, new_id in old_to_new.items()
        },
        "splits": {
            "train_documents": len(remapped_train),
            "dev_documents": len(remapped_dev),
            "test_documents": len(remapped_test),
        },
    }

    (output_dir / "subset_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print("=" * 72)
    print("AMAZONCAT SUBSET COMPLETE")
    print("=" * 72)
    print(
        f"Labels total/seen/unseen: "
        f"{num_seen_labels + num_unseen_labels:,}/"
        f"{num_seen_labels:,}/{num_unseen_labels:,}"
    )
    print(
        f"Train/dev/test documents: "
        f"{len(remapped_train):,}/"
        f"{len(remapped_dev):,}/"
        f"{len(remapped_test):,}"
    )
    print(f"Output directory: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
