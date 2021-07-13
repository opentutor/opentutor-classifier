LICENSE=LICENSE
LICENSE_HEADER=LICENSE_HEADER
$(POETRY):
	$(MAKE) $(POETRY)-update

.PHONY: $(POETRY)-update
$(POETRY)-update:
	poetry env use python3.8
	poetry install

.PHONY: docker-build
docker-build:
	cd opentutor_classifier && $(MAKE) docker-build
	cd opentutor_classifier_api && $(MAKE) docker-build

.PHONY: format
format: $(POETRY)
	poetry run black .

LICENSE:
	@echo "you must have a LICENSE file" 1>&2
	exit 1

LICENSE_HEADER:
	@echo "you must have a LICENSE_HEADER file" 1>&2
	exit 1

.PHONY: license
license: LICENSE LICENSE_HEADER $(POETRY)
	poetry run python -m licenseheaders -t LICENSE_HEADER -d opentutor_classifier/src $(args)
	poetry run python -m licenseheaders -t LICENSE_HEADER -d opentutor_classifier/tests $(args)
	poetry run python -m licenseheaders -t LICENSE_HEADER -d opentutor_classifier_api/src $(args)
	poetry run python -m licenseheaders -t LICENSE_HEADER -d opentutor_classifier_api/tests $(args)
	poetry run python -m licenseheaders -t LICENSE_HEADER -d tools $(args)
	poetry run python -m licenseheaders -t LICENSE_HEADER -d word2vec $(args)

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
test-format: $(POETRY)
	poetry run black --check .

.PHONY: test-lint
test-lint: $(POETRY)
	poetry run flake8 .

.PHONY: test-license
test-license: LICENSE LICENSE_HEADER
	args="--check" $(MAKE) license

.PHONY: test-types
test-types: $(POETRY)
	poetry run mypy opentutor_classifier
	poetry run mypy opentutor_classifier_api
	poetry run mypy shared

.PHONY: update-deps
update-deps:
	poetry update
