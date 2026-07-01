.PHONY: install data eval serve

PYTHON = .venv/bin/python

install:
	@if [ ! -d .venv ]; then python3 -m venv .venv && .venv/bin/pip install -r requirements.txt; fi

data:
	$(PYTHON) src/generate_logs.py

eval:
	$(PYTHON) evals/run_eval.py

serve:
	$(PYTHON) -m uvicorn serve:app --reload
