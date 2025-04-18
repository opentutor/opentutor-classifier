DOCKER_IMAGE?=opentutor-classifier-api
ROOT=$(shell dirname ${PWD})
VENV=.venv

$(VENV):
	$(MAKE) install

.PHONY clean:
clean:
	rm -rf .pytest_cache .mypy_cache $(VENV)

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
	cd $(ROOT) && $(MAKE) format

.PHONY: install
install: poetry-ensure-installed
	cd $(ROOT)/opentutor_classifier \
		&& $(MAKE) poetry-build-classifier
	poetry config --local virtualenvs.in-project true
	poetry env use python3.11
	poetry install

.PHONY: license
license:
	cd $(ROOT) && make license

.PHONY: build-requirements
build-requirements:
	poetry export --without-hashes --only=evaluate --output=./requirements-evaluate.txt
	poetry export --without-hashes --only=train --output=./requirements-train_job.txt
	poetry export --without-hashes --only=extractconfig --output=./requirements-extract_config.txt
	poetry export --without-hashes --only=trainingconfigdata --output=./requirements-training_config.txt
	poetry export --without-hashes --only=trainingconfigdata --output=./requirements-training_data.txt
	poetry export --without-hashes --only=base --output=./requirements.txt

.PHONY: poetry-ensure-installed
poetry-ensure-installed:
	sh $(ROOT)/tools/poetry_ensure_installed.sh

.PHONY: test
test: $(VENV)
	cd $(ROOT)/shared/ \
		&& $(MAKE) installed/word2vec.bin
	poetry run	coverage run \
		--omit="$(PWD)/tests" \
		-m py.test -vv $(args)

PHONY: test
test-not-slow: $(VENV)
	args="-m '(not slow)'" $(MAKE) test

.PHONY: test-all
test-all: test-format test-lint test-license test-types test

.PHONY: test-all
test-all-not-slow:
	args="-m '(not slow)'" $(MAKE) test-all

.PHONY: test-format
test-format:
	cd $(ROOT) && $(MAKE) test-format

.PHONY: test-license
test-license:
	cd $(ROOT) && make test-license
	
.PHONY: test-lint
test-lint:
	cd $(ROOT) && $(MAKE) test-lint

.PHONY: test-types
test-types:
	cd $(ROOT) && $(MAKE) test-types
