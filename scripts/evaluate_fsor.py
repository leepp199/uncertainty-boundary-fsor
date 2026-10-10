#!/usr/bin/env python3
"""Evaluate known-class accuracy and open-set discrimination."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from boundary_fsor.evaluation.evaluator import main


if __name__ == "__main__":
    main()
