DOCKER_IMAGE?=opentutor-classifier
ROOT=$(shell dirname ${PWD})
VENV=.venv

$(VENV):
	$(MAKE) install
	
.PHONY: install
install: poetry-ensure-installed
	poetry config --local virtualenvs.in-project true
	poetry env use python3.8
	poetry install
	
.PHONY: poetry-build-classifier
poetry-build-classifier: $(VENV)
	poetry build

.PHONY: poetry-ensure-installed
poetry-ensure-installed:
	sh $(ROOT)/tools/poetry_ensure_installed.sh

.PHONY clean:
clean:
	rm -rf htmlcov .coverage tests/fixtures/shared/word2vec.bin $(VENV)

.PHONY: deps-show
deps-show:
	poetry show

.PHONY: deps-show
deps-show-outdated:
	poetry show --outdated

.PHONY: deps-update
deps-update:
	poetry update

.PHONY docker-build:
docker-build:
	docker build -t $(DOCKER_IMAGE) .

.PHONY docker-run-shell:
docker-run-shell:
	docker run -it --rm  --entrypoint /bin/bash $(DOCKER_IMAGE)

# use to test dockerized training locally
.PHONY: docker-train
docker-train-%:
	docker run \
		-it \
		--rm \
		-v $(PWD)/tests/fixtures/data/$*:/data \
		-v $(PWD)/tests/fixtures/shared:/shared \
		-v $(PWD)/tests/fixtures/models/$*:/output \
	$(DOCKER_IMAGE) train --data /data/ --shared /shared --output /output 

.PHONY: docker-train-default
docker-train-default:
	docker run \
		-it \
		--rm \
		-v $(PWD)/tests/fixtures/data/:/data \
		-v $(PWD)/tests/fixtures/shared:/shared \
		-v $(PWD)/tests/fixtures/models/default:/output \
	$(DOCKER_IMAGE) traindefault --data /data/ --shared /shared --output /output 

.PHONY: format
format:
	cd $(ROOT) && $(MAKE) format

.PHONY: license
license:
	cd $(ROOT) && make license

PHONY: test
test: $(VENV)
	cd $(ROOT)/shared/ \
		&& $(MAKE) installed/word2vec.bin
	cd $(ROOT)/shared/ \
		&& $(MAKE) installed/spacy-model
	poetry run coverage run \
		--omit="$(PWD)/tests" \
		-m py.test -vv $(args)

PHONY: test
test-not-slow: $(VENV)
	args="-m '(not slow)'" $(MAKE) test


.PHONY: test-all
test-all: test-format test-lint test-types test-license test

.PHONY: test-all
test-all-not-slow:
	args="-m '(not slow)'" $(MAKE) test-all


.PHONY: test-format
test-format:
	cd $(ROOT) && $(MAKE) test-format

.PHONY: test-license
test-license: $(LICENSE) $(LICENSE_HEADER)
	cd $(ROOT) && $(MAKE) test-license

.PHONY: test-lint
test-lint:
	cd $(ROOT) && $(MAKE) test-lint

.PHONY: test-types
test-types:
	cd $(ROOT) && $(MAKE) test-types
