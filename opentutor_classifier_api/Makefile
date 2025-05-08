DOCKER_IMAGE?=opentutor-classifier-api
ROOT=$(shell dirname ${PWD})
VENV=.venv

$(VENV):
	$(MAKE) install

.PHONY clean:
clean:
	rm -rf .pytest_cache .mypy_cache $(VENV)

.PHONY docker-build:
docker-build: clean
	cd .. \
		&& docker build -t $(DOCKER_IMAGE) -f ./opentutor_classifier_api/Dockerfile .

.PHONY docker-build:
docker-run:
	docker run \
		-it \
		--rm \
		-p 5000:5000 \
	$(DOCKER_IMAGE)

.PHONY: format
format: $(VENV)
	cd $(ROOT) && $(MAKE) format

.PHONY: install
install: 
	cd $(ROOT)/opentutor_classifier \
		&& $(MAKE) poetry-build-classifier
	poetry sync

.PHONY: license
license:
	cd $(ROOT) && make license

.PHONY: test
test: $(VENV)
	cd $(ROOT)/shared/ \
		&& $(MAKE) installed/word2vec.bin
	uv run	coverage run \
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
