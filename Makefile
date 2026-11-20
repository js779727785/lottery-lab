PYTHON ?= python3
PYTHONPATH := src

.PHONY: collect backtest holdout extreme test serve all

collect:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m lottery_lab.collect20 --start 2006-01-01 --end "$$(date +%F)"

backtest:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m lottery_lab.pipeline

holdout:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m lottery_lab.holdout --seed 20261120 --count 10

extreme:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m lottery_lab.extreme --start 2008-01-01 --target 0.99

test:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m unittest discover -s tests -v

serve:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m lottery_lab.server --port 8765

all: collect backtest holdout extreme test
