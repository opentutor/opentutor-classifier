LICENSE=LICENSE
LICENSE_HEADER?=LICENSE_HEADER
VENV=.venv
$(VENV):
	$(MAKE) install

.PHONY: clean
clean:
	rm -rf .venv

.PHONY: deps-show
deps-show:
	poetry show

.PHONY: deps-show
deps-show-outdated:
	poetry show --outdated

.PHONY: deps-update
deps-update:
	poetry update

.PHONY: format
format: $(VENV)
	poetry run black .

LICENSE:
	@echo "you must have a LICENSE file" 1>&2
	exit 1

LICENSE_HEADER:
	@echo "you must have a LICENSE_HEADER file" 1>&2
	exit 1

.PHONY: license
license: LICENSE LICENSE_HEADER $(VENV)
	poetry run python -m licenseheaders -t ${LICENSE_HEADER} -d serverless_modules $(args)
	poetry run python -m licenseheaders -t ${LICENSE_HEADER} -d serverless_modules/train_job $(args)
	poetry run python -m licenseheaders -t ${LICENSE_HEADER} -d serverless_modules/train_job/lr2 $(args)
	poetry run python -m licenseheaders -t ${LICENSE_HEADER} -d serverless_modules/evaluate $(args)
	poetry run python -m licenseheaders -t ${LICENSE_HEADER} -d tools $(args)
	poetry run python -m licenseheaders -t ${LICENSE_HEADER} -d src/functions $(args)
	poetry run python -m licenseheaders -t ${LICENSE_HEADER} -d word2vec $(args)

.PHONY: poetry-ensure-installed
poetry-ensure-installed:
	sh ./tools/poetry_ensure_installed.sh

.PHONY: test-format
test-format: $(VENV)
	poetry run black --check .

.PHONY: test-lint
test-lint: $(VENV)
	poetry run flake8 .

.PHONY: test-license
test-license: LICENSE LICENSE_HEADER
	args="--check" $(MAKE) license

.PHONY: test-types
test-types: $(VENV)
	poetry run mypy src
	poetry run mypy serverless_modules