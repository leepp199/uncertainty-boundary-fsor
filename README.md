# Confusability-Guided Boundary Learning for FOAC

Official implementation of **Few-Shot Open-Set Audio Classification with
Confusability-Guided Boundary Learning**.

The method has two trainable stages. A ResNet-18 audio encoder is first trained
on the meta-training classes and then frozen. Confusability-guided Episode
Scheduling (CES) orders meta-training episodes using class uncertainty and
class-mean cosine similarity. Task-conditioned Boundary Generation (TBG) learns
one negative prototype for each support-derived positive prototype through a
single-head attention conditioner and residual generator. At test
time, the nearest positive prototype determines the candidate class, and its
positive-to-negative similarity margin is used for open-set rejection.

## Protocol

Experiments use class-disjoint meta-training, validation, and test partitions.
Each task contains five known classes and five unknown classes. A known class
provides either one or five support examples and 15 queries; an unknown class
provides 15 queries and no support examples. The reported test result averages
1,000 independently sampled tasks.

| Dataset | Meta-train classes | Validation classes | Test classes |
|---|---:|---:|---:|
| LS-100 | 0-59 | 60-79 | 80-99 |
| NS-100 | 0-54 | 55-79 | 80-99 |
| FSC-89 | 0-48 | 49-68 | 69-88 |

Known-class accuracy is computed before rejection. AUROC uses the candidate
margin \(m_{c^*}(x)\), with larger values indicating known-class membership.
For the binary decision, a query is accepted when the margin is positive.

## Installation

Python 3.8 and CUDA-compatible PyTorch are recommended. For a network-isolated
machine, prepare a local wheel directory and run:

```bash
python -m pip install --no-index --find-links /path/to/wheels -r requirements.txt
python -m pip install --no-deps -e .
bash scripts/offline_preflight.sh
```

No executable path downloads data or model weights. Set the local dataset and
checkpoint paths before running an experiment:

```bash
export BOUNDARY_FSOR_LS100_DATA=/path/to/LS100
export BOUNDARY_FSOR_LS100_CHECKPOINT=/path/to/base_checkpoint.pth

export BOUNDARY_FSOR_NS100_DATA=/path/to/nsynth
export BOUNDARY_FSOR_NS100_METADATA=/path/to/ns100_csv
export BOUNDARY_FSOR_NS100_CHECKPOINT=/path/to/base_checkpoint.pth

export BOUNDARY_FSOR_FSC89_DATA=/path/to/FSC89
export BOUNDARY_FSOR_FSC89_METADATA=/path/to/fsc89_csv
export BOUNDARY_FSOR_FSC89_CHECKPOINT=/path/to/base_checkpoint.pth
```

## Reproduction

The development command trains the base encoder, extracts frozen features,
trains the TBG controls, and evaluates validation tasks. It does not open the
test partition.

```bash
python scripts/run_full_experiment.py \
  --config configs/ls100.yaml --device cuda:0 --shots 1 5
```

After the configuration is fixed, add `--frozen-test` to generate the 1-shot
and 5-shot test ledgers, paired comparison, tables, and the NS-100 openness
inputs used in the paper:

```bash
python scripts/run_full_experiment.py \
  --config configs/ls100.yaml --device cuda:0 --shots 1 5 --frozen-test
```

The three principal commands are also installed as console entry points:

```bash
foac-pretrain --config configs/ls100.yaml --device cuda:0
foac-meta-train --config configs/ls100.yaml --shots 5 --device cuda:0
foac-evaluate --config configs/ls100.yaml --shots 5 --method ces_tbg --device cuda:0
```

## Experimental settings

- 16-kHz audio, 128-bin log-Mel spectrogram, 25-ms Hann window, 10-ms hop;
- 30 pre-training epochs with SGD, learning rate `5e-3`, weight decay `5e-4`;
- 6,000 TBG episodes with AdamW, learning rate `3e-4`, weight decay `1e-4`;
- one attention head with dropout probability `0.1` in TBG;
- 10 feature-dropout predictions with dropout probability `0.3`;
- min-max normalization with stabilizer `1e-8`;
- CES beta scheduled linearly from `-2` to `2` over 1,500 episodes;
- uniform mixture coefficient `epsilon=0.1`;
- uncertainty weight `lambda_u=1`, margin `delta=0.1`, and unknown loss weight `lambda=1`.

Dataset-specific settings live in `configs/`. Checkpoints, feature caches,
manifests, raw scores, and tables are written below `artifacts/` and are ignored
by Git.

## Repository layout

```text
boundary_fsor/
  data/              audio metadata and deterministic episode sampling
  models/            audio encoder and task-conditioned boundary generator
  training/          uncertainty, CES, pre-training, and meta-training
  evaluation/        episodic scoring and open-set metrics
  reproducibility/   split, file, checkpoint, and environment manifests
configs/              LS-100, NS-100, and FSC-89 protocols
scripts/              training, evaluation, auditing, and table generation
tests/                protocol and equation-level unit tests
```
