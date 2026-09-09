# msc-project-zero-shot-classification

## Datasets

### EURLEX57K

Multi-label legal document classification dataset based on EU legislation and EUROVOC labels.

The project split contains:
- 4,271 labels
- 4,108 seen labels
- 163 unseen labels

Expected data directory:

```text
data/hf_EURLEX57K/
├── trn.json
├── dev.json
├── tst.json
├── lbl.json
└── lbl_embs_sfr_eurlex57k.npy
```

### AmazonCat-13K

Multi-label product classification dataset using product descriptions and category labels.

The reduced subset used in this project contains:
- 5,753 labels
- 5,000 seen labels
- 753 unseen labels
- 300,000 training documents

To build the reduced subset from the full converted AmazonCat dataset:

```bash
python sample_amazoncat.py --config configs/amazoncat_sampling.yaml
```

Expected output directory:

```text
data/hf_AmazonCat-13K_sub/
├── trn.json
├── dev.json
├── tst.json
├── lbl.json
└── lbl_embs_qwen_amzn.npy
```

## Python packages

Main dependencies:

```text
torch
transformers
numpy
PyYAML
matplotlib
accelerate
```

Install them with:

```bash
pip install torch transformers numpy pyyaml matplotlib accelerate
```

## 1. Analyze a dataset

```bash
python dataset_analysis.py --config configs/eurlex.yaml
python dataset_analysis.py --config configs/amazoncat.yaml
```

To skip tokenizer-based document-length statistics:

```bash
python dataset_analysis.py --config configs/amazoncat.yaml --no-token-lengths
```

## 2. Generate synthetic examples

```bash
python generate_examples.py --config configs/eurlex.yaml
python generate_examples.py --config configs/amazoncat.yaml
```

## 3. Train the seen-label classifier

```bash
python train_seen.py --config configs/eurlex.yaml
python train_seen.py --config configs/amazoncat.yaml
```

## 4. Train the unseen-label adapters

```bash
python train_unseen.py --config configs/eurlex.yaml
python train_unseen.py --config configs/amazoncat.yaml
```

## 5. Evaluate

```bash
python evaluate.py --config configs/eurlex.yaml
python evaluate.py --config configs/amazoncat.yaml
```

Dataset paths, model settings, training parameters, generation settings, and output paths are defined in the YAML config files.

matplotlib
accelerate

Install them with:

pip install torch transformers numpy pyyaml matplotlib accelerate

1. Analyze a dataset

python dataset_analysis.py --config configs/eurlex.yaml
python dataset_analysis.py --config configs/amazoncat.yaml

To skip tokenizer-based document-length statistics:

python dataset_analysis.py --config configs/amazoncat.yaml --no-token-lengths

2. Generate synthetic examples

python generate_examples.py --config configs/eurlex.yaml
python generate_examples.py --config configs/amazoncat.yaml

3. Train the seen-label classifier

python train_seen.py --config configs/eurlex.yaml
python train_seen.py --config configs/amazoncat.yaml

4. Train the unseen-label adapters

python train_unseen.py --config configs/eurlex.yaml
python train_unseen.py --config configs/amazoncat.yaml

5. Evaluate

python evaluate.py --config configs/eurlex.yaml
python evaluate.py --config configs/amazoncat.yaml

Dataset paths, model settings, training parameters, generation settings, and output paths are defined in the YAML config files.