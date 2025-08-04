DOCKER_IMAGE?=opentutor-classifier-api
ROOT=$(shell dirname ${PWD})
VENV=.venv

$(VENV):
	$(MAKE) install

.PHONY clean:
clean:
	rm -rf .pytest_cache .mypy_cache $(VENV)

.PHONY: format
format: $(VENV)
	cd $(ROOT) && $(MAKE) format

.PHONY: install
install:
	uv sync

.PHONY: license
license:
	cd $(ROOT) && make license

.PHONY: build-requirements
build-requirements:
	uv export --no-hashes --only-group=evaluate --output-file=./requirements-evaluate.txt
	uv export --no-hashes --only-group=train --output-file=./requirements-train_job.txt
	uv export --no-hashes --only-group=extractconfig --output-file=./requirements-extract_config.txt
	uv export --no-hashes --only-group=trainingconfigdata --output-file=./requirements-training_config.txt
	uv export --no-hashes --only-group=trainingconfigdata --output-file=./requirements-training_data.txt
	uv export --no-hashes --only-group=base --output-file=./requirements.txt

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
