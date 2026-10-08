.PHONY: help scaffold test test-00 clean env

help:
	@echo "Available commands:"
	@echo "  make scaffold   - Generate student notebooks by stripping instructor solutions"
	@echo "  make test       - Run all autograding unit tests via pytest"
	@echo "  make test-00    - Run Workshop 00 unit tests"
	@echo "  make clean      - Remove build artifacts, bytecode, and cache folders"

scaffold:
	python3 scripts/generate_scaffold.py

test:
	pytest workshop_*/tests/

test-00:
	pytest workshop_00_revising/tests/

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} +
	rm -rf build dist *.egg-info
