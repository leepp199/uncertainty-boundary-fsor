# Experiment ledger contract

An experiment can enter the paper only when its record contains:

- repository commit and configuration SHA-256;
- local checkpoint SHA-256 and split-specific feature manifest;
- disjoint meta-train, validation, and final-test class ranges;
- validation-selected threshold with no final-test refitting;
- raw known and unknown scores for every shared episode seed;
- matched prototype, energy, global-boundary, uniform class-boundary, and
  uncertainty-ordered class-boundary results.

The previous global-local anchor experiment is retired because it addresses a
different mechanism. Its numbers are not evidence for the present paper.

## Current status

LS-100 is complete under the isolated evaluator. Over 1,000 paired final
episodes, uncertainty-ordered class-conditional boundary learning preserves
96.39% known accuracy and reaches 95.43% AUROC, compared with 95.26% for maximum
prototype confidence and 95.40% for the same boundary model with uniform episode
sampling. The paired AUROC gain over prototype confidence is 0.165 points with a
95% bootstrap interval of [0.138, 0.193]. NS-100 and FSC-89 are being rerun with
the same code before the cross-domain table is frozen.
