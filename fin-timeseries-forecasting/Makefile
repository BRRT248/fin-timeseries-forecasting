
PY?=python

setup:
	$(PY) -m venv .venv && . .venv/bin/activate && pip install -U pip && pip install -r requirements.txt

train:
	$(PY) -m src.models.train --ticker SPY --start 2015-01-01 --end 2024-12-31 --outdir artifacts

eval:
	$(PY) -m src.models.evaluate --artifacts artifacts --outdir figures
