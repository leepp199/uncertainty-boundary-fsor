# Experiment records

Every reported run is represented by local artifacts rather than handwritten
table entries. A complete record contains:

- the repository revision and configuration SHA-256;
- the encoder checkpoint SHA-256;
- ordered sample manifests for meta-training, validation, and test partitions;
- the shot count, episode seed, known classes, and unknown classes for each task;
- raw known and unknown decision scores;
- known-class correctness before rejection;
- the metric summary generated from the raw scores.

Development runs stop after validation. Test audio and test metadata are opened
only by a command containing `--frozen-test`.

## Main experiments

```bash
python scripts/run_full_experiment.py --config configs/ls100.yaml --shots 1 5
python scripts/run_full_experiment.py --config configs/ns100.yaml --shots 1 5
python scripts/run_full_experiment.py --config configs/fsc89.yaml --shots 1 5
```

Repeat the same commands with `--frozen-test` after the method and configuration
are fixed. Result tables are generated from `raw_scores.npz` and `summary.json`;
no result value is embedded in source code.

## Component controls

The pipeline evaluates maximum positive-prototype similarity, energy scoring, a
shared global negative prototype, TBG with uniform episode sampling, CES using
only class uncertainty, CES using only class similarity, and the full CES+TBG
method. All controls use the same feature cache and episode seeds.
