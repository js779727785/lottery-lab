#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
END_DATE="${END_DATE:-$(date +%F)}"
PYTHONPATH=src python3 -m lottery_lab.collect20 --start 2006-01-01 --end "$END_DATE"
PYTHONPATH=src python3 -m lottery_lab.pipeline
PYTHONPATH=src python3 -m lottery_lab.holdout --seed 20261120 --count 10
PYTHONPATH=src python3 -m lottery_lab.extreme --start 2008-01-01 --target 0.99
PYTHONPATH=src python3 -m unittest discover -s tests -v
