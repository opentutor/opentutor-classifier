LICENSE=LICENSE
LICENSE_HEADER?=LICENSE_HEADER
VENV=.venv
$(VENV):
	$(MAKE) install

.PHONY: clean
clean:
	rm -rf .venv

.PHONY: docker-build
docker-build:
	cd opentutor_classifier && $(MAKE) docker-build
	cd opentutor_classifier_api && $(MAKE) docker-build

.PHONY: install
install:
	uv sync

.PHONY: format
format: $(VENV)
	uv run black .

LICENSE:
	@echo "you must have a LICENSE file" 1>&2
	exit 1

LICENSE_HEADER:
	@echo "you must have a LICENSE_HEADER file" 1>&2
	exit 1

.PHONY: license
license: LICENSE LICENSE_HEADER $(VENV)
	uv run python -m licenseheaders -t ${LICENSE_HEADER} -d opentutor_classifier/opentutor_classifier $(args)
	uv run python -m licenseheaders -t ${LICENSE_HEADER} -d opentutor_classifier/opentutor_classifier_tasks $(args)
	uv run python -m licenseheaders -t ${LICENSE_HEADER} -d opentutor_classifier/tests $(args) --exclude opentutor_classifier/tests/fixtures/models/**.yaml
	uv run python -m licenseheaders -t ${LICENSE_HEADER} -d opentutor_classifier_api/src $(args)
	uv run python -m licenseheaders -t ${LICENSE_HEADER} -d opentutor_classifier_api/tests $(args)
	uv run python -m licenseheaders -t ${LICENSE_HEADER} -d serverless/functions $(args)
	uv run python -m licenseheaders -t ${LICENSE_HEADER} -d serverless/src $(args)
	uv run python -m licenseheaders -t ${LICENSE_HEADER} -d cicd $(args)
	uv run python -m licenseheaders -t ${LICENSE_HEADER} -d tools $(args)
	uv run python -m licenseheaders -t ${LICENSE_HEADER} -d word2vec $(args)

.PHONY: build-requirements
build-requirements:
	cd serverless && $(MAKE) build-requirements

.PHONY: test
test:
	cd opentutor_classifier && $(MAKE) test
	cd opentutor_classifier_api && $(MAKE) test

.PHONY: test-not-slow
test-not-slow:
	cd opentutor_classifier && $(MAKE) test-not-slow
	cd opentutor_classifier_api && $(MAKE) test-not-slow

.PHONY: test-all
test-all:
	cd opentutor_classifier && $(MAKE) test-all
	cd opentutor_classifier_api && $(MAKE) test-all

.PHONY: test-all-not-slow
test-all-not-slow:
	cd opentutor_classifier && $(MAKE) test-all-not-slow
	cd opentutor_classifier_api && $(MAKE) test-all-not-slow

.PHONY: test-format
test-format: $(VENV)
	uv run black --check .

.PHONY: test-lint
test-lint: $(VENV)
	uv run flake8 .

.PHONY: test-license
test-license: LICENSE LICENSE_HEADER
	args="--check" $(MAKE) license

.PHONY: test-types
test-types: $(VENV)
	uv run mypy opentutor_classifier
	uv run mypy opentutor_classifier_api
	uv run mypy shared
