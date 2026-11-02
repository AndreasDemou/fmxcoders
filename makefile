# Developer shortcuts. The tools are not installed by `pip install -e .`; install them with:
#   pip install ruff pyright==1.1.365 pytest pytest-cov docstr-coverage mkdocs mkdocs-material \
#       mkdocs-autorefs mkdocs-section-index mkdocstrings mkdocstrings-python

format:
	ruff format .
	ruff check --fix-only .

check-format:
	ruff check .
	ruff format --check .

check-type:
	pyright .

test:
	pytest -v --cov=sae_lens/ --cov-report=term-missing --cov-branch tests

check-ci:
	make check-format
	make check-type
	make test

docstring-coverage:
	docstr-coverage sae_lens --skip-file-doc

docs-serve:
	mkdocs serve