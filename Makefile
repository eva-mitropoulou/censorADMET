PYTHON ?= python3
MODULE_PATH := src/censoradmet

.PHONY: test verify

test:
	PYTHONPATH=$(MODULE_PATH) $(PYTHON) -m pytest -q

verify:
	PYTHONPATH=$(MODULE_PATH) $(PYTHON) scripts/check_method_numbers.py
