PYTHON ?= python3
PYTHONPATH := src
START ?= 2026-01-01
END ?= 2026-01-31
VALIDATION_START ?= 2024-01-01
VALIDATION_END ?= 2025-12-31

.PHONY: collect backtest holdout extreme signal review test serve all

collect:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m lottery_lab.collect20 --start 2006-01-01 --end "$$(date +%F)"

backtest:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m lottery_lab.pipeline

holdout:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m lottery_lab.holdout --seed 20261120 --count 10

extreme:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m lottery_lab.extreme --start 2008-01-01 --target 0.99

signal:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m lottery_lab.signal_audit --start 2008-01-01

review:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m lottery_lab.period_review --start "$(START)" --end "$(END)" --validation-start "$(VALIDATION_START)" --validation-end "$(VALIDATION_END)"

test:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m unittest discover -s tests -v

serve:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m lottery_lab.server --port 8765

all: collect backtest holdout extreme signal test
