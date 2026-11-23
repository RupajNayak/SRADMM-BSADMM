#!/usr/bin/env bash
# Example: everything in ./data, 4 worker processes, HiGHS benchmark only for n <= 100.
set -e
python3 bqpgub_bench.py data/*.txt data/*.npz \
    --workers 4 --time-limit 900 --madmm-starts 5 \
    --mip highs --mip-time 600 --mip-max-n 100 \
    --out results_all
