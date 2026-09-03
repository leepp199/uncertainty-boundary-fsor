# Not All Unknowns Are Equally Hard

## Uncertainty-Ordered Boundary Learning for Few-Shot Open-Set Audio Recognition

This repository studies one question: **how should pseudo-unknown
episodes be ordered when the difficulty of class boundaries is uneven?**

The method estimates class-level uncertainty from stochastic audio views and
pairwise boundary overlap from base-class prototype similarity. Early episodes
emphasize stable, well-separated class pairs; later episodes emphasize uncertain
classes and their closest held-out neighbors. For each known episode class, a
task-conditioned companion represents its local open boundary. Validation-only
selection combines positive-prototype confidence with the corresponding
positive-to-boundary margin for unknown rejection.

The repository contains no LSRB, category discovery, or class-incremental
classifier expansion. Those components belong to the separate discovery paper.

## Protocol

Each final episode contains a 5-way 5-shot support set, 15 known queries per
class, and 15 queries from five label-disjoint unknown classes. Meta-training,
validation, and final-test classes are disjoint. Hyperparameters and the
rejection threshold are selected on validation episodes and then frozen.

Primary measures are known-class accuracy and AUROC. AUPR, FPR95, and OSCR are
reported as complementary open-set measures.

## Offline workflow

The default command performs protocol auditing, base training, feature
extraction, all matched boundary controls, and validation-only model selection.
It does not open the final-test classes:

```bash
python scripts/run_full_experiment.py --config configs/ls100.yaml --device cuda:0
```

After the curriculum choice is frozen, the same entry point produces the
test ledger, paired significance test, difficulty analysis, and paper table:

```bash
python scripts/run_full_experiment.py --config configs/ls100.yaml \
  --device cuda:0 --frozen-test
```

The individual stages remain callable for controlled experiments:

```bash
bash scripts/offline_preflight.sh
python scripts/extract_features.py --config configs/ls100.yaml --split train
python scripts/extract_features.py --config configs/ls100.yaml --split val
python scripts/extract_features.py --config configs/ls100.yaml --split test
python scripts/train_boundary.py --config configs/ls100.yaml --sampling uniform
python scripts/train_boundary.py --config configs/ls100.yaml \
  --sampling uncertainty --curriculum-components class --tag uncertainty_class
python scripts/train_boundary.py --config configs/ls100.yaml \
  --sampling uncertainty --curriculum-components pair --tag uncertainty_pair
python scripts/train_boundary.py --config configs/ls100.yaml \
  --sampling uncertainty --curriculum-components joint --tag uncertainty
bash scripts/run_comparisons.sh configs/ls100.yaml
python scripts/paired_significance.py --dataset ls100 \
  --baseline prototype --method uncertainty_boundary
```

Dataset and checkpoint paths are supplied through the environment variables in
the configuration files. All executable paths are guarded for offline use and
do not download datasets or model weights.

## Matched controls

- maximum positive-prototype similarity;
- energy score;
- one shared global boundary companion;
- class-conditional companions with uniform episode sampling;
- class-conditional companions with uncertainty-ordered sampling.

External methods are included only after they are rerun with the same class
splits and episode seeds.

## Implementation map

- `boundary_fsor/models/`: ResNet-18 audio encoder and class-conditional
  boundary-companion network;
- `boundary_fsor/data/`: audio metadata, wave batching, and local-seed
  class-disjoint 5-way 5-shot episodes;
- `boundary_fsor/training/uncertainty.py`: entropy and mutual-information
  uncertainty;
- `boundary_fsor/training/curriculum.py`: class instability, pairwise intrusion risk, and the
  easy-to-uniform-to-hard curriculum;
- `boundary_fsor/training/base_trainer.py` and `boundary_trainer.py`: complete
  optimization and checkpoint-selection loops;
- `boundary_fsor/evaluation/`: AUROC, AUPR, FPR95, OSCR, and known accuracy;
- `boundary_fsor/reproducibility/`: split, row-order, checkpoint, and environment
  manifests;
- `scripts/analyze_boundary_difficulty.py`: AUROC gains stratified by unknown
  proximity to the support prototypes;
- `scripts/build_results_table.py`: CSV and Markdown tables built only from raw
  frozen ledgers.

The tests cover exact support counts, split disjointness, seed isolation,
curriculum direction, boundary correspondence, manifest hashing, and stochastic
uncertainty behavior.
