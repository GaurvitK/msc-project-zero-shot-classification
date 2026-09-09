from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator, Sequence

import torch
import torch.nn.functional as F
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from model import load_label_embeddings, select_label_embeddings


class GenerationDataError(RuntimeError):
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
        raise GenerationDataError(f"{path} must contain JSON objects")

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
            raise GenerationDataError(f"{path} contains a non-object record")

        rows.append(row)

    return rows


def iter_records(path: Path, data_mode: str) -> Iterator[dict[str, Any]]:
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


def load_labels(path: Path, expected_count: int) -> dict[int, dict[str, Any]]:
    rows = read_json_records(path)

    if len(rows) != expected_count:
        raise GenerationDataError(
            f"Expected {expected_count:,} labels, found {len(rows):,}"
        )

    labels: dict[int, dict[str, Any]] = {}

    for position, row in enumerate(rows):
        if "uid" not in row:
            raise GenerationDataError(f"Label record {position} is missing uid")

        label_id = int(row["uid"])

        if label_id != position:
            raise GenerationDataError(
                f"Labels must be in UID order: expected {position}, found {label_id}"
            )

        labels[label_id] = row

    return labels


def get_label_text(row: dict[str, Any], fields: Sequence[str]) -> str:
    """Return compact, non-duplicated label text for prompting."""
    parts: list[str] = []
    seen: set[str] = set()

    for field in fields:
        value = str(row.get(field, "")).strip()
        folded = value.casefold()

        if value and folded not in seen:
            parts.append(value)
            seen.add(folded)

    return " | ".join(parts) if parts else str(row["uid"])


def build_training_index(
    train_path: Path,
    data_mode: str,
    labels: dict[int, dict[str, Any]],
) -> tuple[list[int], dict[int, list[str]]]:
    """Return seen label IDs and the real training documents attached to each label."""
    seen_ids: set[int] = set()
    documents_by_label: dict[int, list[str]] = defaultdict(list)

    document_count = 0

    for row in iter_records(train_path, data_mode):
        document_count += 1

        missing = {"uid", "content", "target_ind"} - row.keys()
        if missing:
            raise GenerationDataError(
                f"Training record {document_count}: missing fields {sorted(missing)}"
            )

        content = str(row["content"]).strip()
        targets = row["target_ind"]

        if not content:
            raise GenerationDataError(f"Training record {document_count} has empty content")

        if not isinstance(targets, list) or not targets:
            raise GenerationDataError(
                f"Training record {document_count}: target_ind must be a non-empty list"
            )

        target_ids = [int(value) for value in targets]

        if len(target_ids) != len(set(target_ids)):
            raise GenerationDataError(
                f"Training record {document_count} contains duplicate target IDs"
            )

        unknown = [label_id for label_id in target_ids if label_id not in labels]

        if unknown:
            raise GenerationDataError(
                f"Training record {document_count} contains unknown labels {unknown[:20]}"
            )

        for label_id in target_ids:
            seen_ids.add(label_id)
            documents_by_label[label_id].append(content)

        if document_count % 100_000 == 0:
            print(f"  indexed {document_count:,} training documents")

    if not seen_ids:
        raise GenerationDataError("No seen labels were found in the training split")

    return sorted(seen_ids), documents_by_label


def retrieve_neighbours(
    seen_ids: Sequence[int],
    unseen_ids: Sequence[int],
    all_embeddings: torch.Tensor,
    neighbours_per_label: int,
) -> dict[int, list[int]]:
    """Retrieve nearest seen labels by cosine similarity."""
    if neighbours_per_label <= 0:
        raise ValueError("neighbours_per_label must be positive")

    seen_embeddings = select_label_embeddings(all_embeddings, seen_ids)
    unseen_embeddings = select_label_embeddings(all_embeddings, unseen_ids)

    seen_embeddings = F.normalize(seen_embeddings.float(), dim=-1)
    unseen_embeddings = F.normalize(unseen_embeddings.float(), dim=-1)

    k = min(neighbours_per_label, len(seen_ids))
    similarity = unseen_embeddings @ seen_embeddings.T
    top_positions = torch.topk(similarity, k=k, dim=1).indices.cpu()

    neighbours: dict[int, list[int]] = {}

    for row, unseen_id in enumerate(unseen_ids):
        neighbours[unseen_id] = [
            int(seen_ids[position])
            for position in top_positions[row].tolist()
        ]

    return neighbours


def sample_demonstrations(
    neighbour_pool: Sequence[int],
    documents_by_label: dict[int, list[str]],
    demonstrations_per_generation: int,
    rng: random.Random,
) -> list[tuple[int, str]]:
    if demonstrations_per_generation > len(neighbour_pool):
        raise ValueError(
            "demonstrations_per_generation cannot exceed the neighbour pool size"
        )

    selected_labels = rng.sample(list(neighbour_pool), demonstrations_per_generation)
    demonstrations: list[tuple[int, str]] = []

    for label_id in selected_labels:
        documents = documents_by_label.get(label_id)

        if not documents:
            raise GenerationDataError(
                f"Retrieved seen label {label_id} has no real training documents"
            )

        demonstrations.append((label_id, rng.choice(documents)))

    return demonstrations


def build_messages(
    unseen_id: int,
    demonstrations: Sequence[tuple[int, str]],
    labels: dict[int, dict[str, Any]],
    *,
    label_text_fields: Sequence[str],
    document_type: str,
    max_demonstration_chars: int,
) -> list[dict[str, str]]:
    """Build the deliberately simple generation prompt."""
    formatted: list[str] = []

    for index, (label_id, document) in enumerate(demonstrations, start=1):
        formatted.append(
            f"Demonstration {index}:\n"
            f"Label: {get_label_text(labels[label_id], label_text_fields)}\n"
            f"Document: {document[:max_demonstration_chars].strip()}"
        )

    prompt = (
        "Target unseen label:\n"
        f"{get_label_text(labels[unseen_id], label_text_fields)}\n\n"
        f"Generate exactly 1 realistic {document_type} that should be assigned the target label.\n"
        "Return only a JSON array containing exactly 1 string.\n\n"
        "Real demonstrations from related labels:\n"
        + "\n\n".join(formatted)
    )

    return [
        {
            "role": "system",
            "content": "Generate one synthetic training document. Return only valid JSON.",
        },
        {"role": "user", "content": prompt},
    ]


def load_generator(model_name: str) -> tuple[Any, Any]:
    print(f"Loading generator: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=(
            torch.bfloat16
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else "auto"
        ),
        device_map="auto" if torch.cuda.is_available() else "cpu",
    )
    model.eval()

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer, model


def generate_document(
    tokenizer: Any,
    model: Any,
    messages: list[dict[str, str]],
    generation_config: dict[str, Any],
) -> str:
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )

    device = next(model.parameters()).device
    encoded = {key: value.to(device) for key, value in encoded.items()}
    prompt_length = encoded["input_ids"].shape[1]

    with torch.inference_mode():
        output = model.generate(
            **encoded,
            max_new_tokens=int(generation_config["max_new_tokens"]),
            do_sample=True,
            temperature=float(generation_config["temperature"]),
            top_p=float(generation_config["top_p"]),
            repetition_penalty=float(generation_config["repetition_penalty"]),
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    text = tokenizer.decode(
        output[0, prompt_length:],
        skip_special_tokens=True,
    ).strip()

    try:
        values = json.loads(text)
    except json.JSONDecodeError:
        # Small recovery for models that wrap otherwise-valid JSON in extra text.
        match = re.search(r"\[[\s\S]*\]", text)

        if match is None:
            raise ValueError(f"Generator did not return a JSON array:\n{text}")

        values = json.loads(match.group(0))

    if not isinstance(values, list):
        raise ValueError(
            f"Generator returned {type(values).__name__}; expected a JSON array"
        )

    if len(values) != 1:
        raise ValueError(f"Expected exactly one generated document, got {len(values)}")

    document = str(values[0]).strip()

    if not document:
        raise ValueError("Generator returned an empty document")

    return document


def synthetic_uid(unseen_id: int, example_index: int) -> str:
    return f"synthetic_{unseen_id}_{example_index:04d}"


def generation_seed(
    base_seed: int,
    unseen_id: int,
    example_index: int,
    attempt_index: int,
) -> int:
    return (
        base_seed
        + unseen_id * 1_000_003
        + example_index * 10_007
        + attempt_index * 101
    )


def set_generation_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_existing_rows(
    output_path: Path,
    unseen_ids: set[int],
    examples_per_label: int,
) -> tuple[list[dict[str, Any]], dict[int, set[int]], dict[int, set[str]]]:
    """Load a partial output so generation can resume after interruption."""
    generated_indices: dict[int, set[int]] = defaultdict(set)
    generated_contents: dict[int, set[str]] = defaultdict(set)

    if not output_path.exists():
        return [], generated_indices, generated_contents

    rows = read_json_records(output_path)
    uid_pattern = re.compile(r"^synthetic_(\d+)_(\d{4})$")
    seen_uids: set[str] = set()

    for index, row in enumerate(rows):
        missing = {"uid", "content", "target_ind"} - row.keys()

        if missing:
            raise GenerationDataError(
                f"Existing synthetic row {index}: missing fields {sorted(missing)}"
            )

        uid = str(row["uid"]).strip()
        content = str(row["content"]).strip()
        targets = row["target_ind"]

        if uid in seen_uids:
            raise GenerationDataError(f"Duplicate synthetic UID {uid!r}")
        seen_uids.add(uid)

        match = uid_pattern.fullmatch(uid)

        if match is None:
            raise GenerationDataError(f"Unexpected synthetic UID format: {uid!r}")

        unseen_id = int(match.group(1))
        example_index = int(match.group(2))

        if unseen_id not in unseen_ids:
            raise GenerationDataError(
                f"Existing synthetic row targets unknown unseen label {unseen_id}"
            )

        if example_index >= examples_per_label:
            raise GenerationDataError(
                f"Existing example index {example_index} exceeds configured count"
            )

        if not isinstance(targets, list) or targets != [unseen_id]:
            raise GenerationDataError(
                f"Existing synthetic row {uid!r} has inconsistent target_ind"
            )

        if not content:
            raise GenerationDataError(f"Existing synthetic row {uid!r} is empty")

        if example_index in generated_indices[unseen_id]:
            raise GenerationDataError(
                f"Duplicate example index {example_index} for unseen label {unseen_id}"
            )

        if content in generated_contents[unseen_id]:
            raise GenerationDataError(
                f"Duplicate generated content for unseen label {unseen_id}"
            )

        generated_indices[unseen_id].add(example_index)
        generated_contents[unseen_id].add(content)

    return rows, generated_indices, generated_contents


def write_output_rows(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    """Atomically save after each successful generation call."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")

    with temporary.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")

    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    dataset_cfg = config["dataset"]
    paths = config["paths"]
    model_cfg = config["model"]
    generation_cfg = config["generation"]

    num_labels = int(dataset_cfg["num_labels"])
    expected_seen = int(dataset_cfg["num_seen_labels"])
    expected_unseen = int(dataset_cfg["num_unseen_labels"])

    examples_per_label = int(generation_cfg["examples_per_label"])
    neighbours_per_label = int(generation_cfg["neighbours_per_label"])
    demonstrations_per_generation = int(
        generation_cfg["demonstrations_per_generation"]
    )
    max_attempts = int(generation_cfg["max_generation_attempts"])
    base_seed = int(generation_cfg["seed"])

    if examples_per_label <= 0:
        raise ValueError("generation.examples_per_label must be positive")

    if demonstrations_per_generation > neighbours_per_label:
        raise ValueError(
            "demonstrations_per_generation cannot exceed neighbours_per_label"
        )

    unseen_training_cfg = config.get("unseen_training", {})

    if (
        "examples_per_label" in unseen_training_cfg
        and int(unseen_training_cfg["examples_per_label"]) != examples_per_label
    ):
        raise ValueError(
            "generation.examples_per_label must match unseen_training.examples_per_label"
        )

    train_path = Path(paths["train"])
    label_path = Path(paths["labels"])
    embedding_path = Path(paths["label_embeddings"])
    output_path = Path(paths["synthetic"])

    labels = load_labels(label_path, num_labels)

    print("Indexing real training documents...")
    seen_ids, documents_by_label = build_training_index(
        train_path,
        dataset_cfg["data_mode"],
        labels,
    )
    unseen_ids = sorted(set(labels) - set(seen_ids))

    if len(seen_ids) != expected_seen or len(unseen_ids) != expected_unseen:
        raise GenerationDataError(
            "Configured seen/unseen counts do not match the training split: "
            f"got {len(seen_ids):,}/{len(unseen_ids):,}"
        )

    all_embeddings = load_label_embeddings(
        embedding_path,
        expected_num_labels=num_labels,
        expected_dimension=model_cfg.get("label_embedding_dimension"),
    )

    print("Retrieving semantic neighbours...")
    neighbours = retrieve_neighbours(
        seen_ids,
        unseen_ids,
        all_embeddings,
        neighbours_per_label,
    )
    del all_embeddings

    existing_rows, generated_indices, generated_contents = load_existing_rows(
        output_path,
        set(unseen_ids),
        examples_per_label,
    )

    completed = sum(len(indices) for indices in generated_indices.values())
    total_required = len(unseen_ids) * examples_per_label

    print("=" * 72)
    print(f"SYNTHETIC GENERATION: {dataset_cfg['name']}")
    print("=" * 72)
    print(f"Seen/unseen labels:        {len(seen_ids):,}/{len(unseen_ids):,}")
    print(f"Examples per unseen label: {examples_per_label}")
    print(f"Neighbour pool:            {neighbours_per_label}")
    print(f"Demonstrations per call:   {demonstrations_per_generation}")
    print(f"Generator:                 {generation_cfg['generator_model']}")
    print(f"Already generated:         {completed:,}/{total_required:,}")
    print(f"Output:                    {output_path}")
    print()

    if completed == total_required:
        print("All synthetic examples are already generated.")
        return

    tokenizer, generator = load_generator(generation_cfg["generator_model"])
    output_rows = list(existing_rows)

    for label_position, unseen_id in enumerate(unseen_ids, start=1):
        missing_indices = [
            index
            for index in range(examples_per_label)
            if index not in generated_indices[unseen_id]
        ]

        for example_index in missing_indices:
            document: str | None = None
            last_error: Exception | None = None
            selected_neighbours: list[int] = []

            for attempt_index in range(max_attempts):
                call_seed = generation_seed(
                    base_seed,
                    unseen_id,
                    example_index,
                    attempt_index,
                )
                set_generation_seed(call_seed)
                rng = random.Random(call_seed)

                demonstrations = sample_demonstrations(
                    neighbours[unseen_id],
                    documents_by_label,
                    demonstrations_per_generation,
                    rng,
                )
                selected_neighbours = [label_id for label_id, _ in demonstrations]

                messages = build_messages(
                    unseen_id,
                    demonstrations,
                    labels,
                    label_text_fields=dataset_cfg.get(
                        "label_text_fields",
                        ["title", "label", "content"],
                    ),
                    document_type=generation_cfg["document_type"],
                    max_demonstration_chars=int(
                        generation_cfg["max_demonstration_chars"]
                    ),
                )

                try:
                    candidate = generate_document(
                        tokenizer,
                        generator,
                        messages,
                        generation_cfg,
                    )

                    if candidate in generated_contents[unseen_id]:
                        raise ValueError(
                            f"Duplicate generated document for unseen label {unseen_id}"
                        )

                    document = candidate
                    break

                except (ValueError, json.JSONDecodeError) as error:
                    last_error = error
                    print(
                        f"Retry {attempt_index + 1}/{max_attempts} for unseen "
                        f"label {unseen_id}, example {example_index + 1}: {error}"
                    )

            if document is None:
                raise RuntimeError(
                    f"Failed to generate unseen label {unseen_id}, "
                    f"example {example_index + 1}"
                ) from last_error

            output_rows.append(
                {
                    "uid": synthetic_uid(unseen_id, example_index),
                    "content": document,
                    "target_ind": [unseen_id],
                }
            )
            generated_indices[unseen_id].add(example_index)
            generated_contents[unseen_id].add(document)

            write_output_rows(output_path, output_rows)

            completed += 1
            print(
                f"Generated {example_index + 1}/{examples_per_label} for unseen "
                f"label {unseen_id} | label {label_position}/{len(unseen_ids)} | "
                f"neighbours={selected_neighbours} | total {completed}/{total_required}"
            )

    if completed != total_required:
        raise GenerationDataError(
            f"Expected {total_required:,} synthetic examples, found {completed:,}"
        )

    print(f"\nSaved {completed:,} synthetic examples to {output_path.resolve()}")


if __name__ == "__main__":
    main()