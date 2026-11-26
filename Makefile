PY ?= python

.PHONY: all table figures baselines models bootstrap ridge-figure tune test clean

all: table figures baselines models bootstrap ridge-figure

table:
	$(PY) scripts/build_table.py

figures: table
	$(PY) scripts/make_figures.py

baselines: table
	$(PY) scripts/run_baselines.py

models: table
	$(PY) scripts/run_models.py

bootstrap: models
	$(PY) scripts/bootstrap_comparisons.py

ridge-figure: models
	$(PY) scripts/make_ridge_figure.py

tune: table
	$(PY) scripts/tune_ridge.py

test:
	$(PY) -m pytest -q

clean:
	rm -rf outputs/figures outputs/results outputs/forecast_examples data/processed/per_cycle_table.csv
