PY ?= python

.PHONY: all table figures baselines models test clean

all: table figures baselines models

table:
	$(PY) scripts/build_table.py

figures: table
	$(PY) scripts/make_figures.py

baselines: table
	$(PY) scripts/run_baselines.py

models: table
	$(PY) scripts/run_models.py

test:
	$(PY) -m pytest -q

clean:
	rm -rf outputs/figures outputs/results outputs/forecast_examples data/processed/per_cycle_table.csv
