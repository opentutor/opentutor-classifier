DOCKER_IMAGE?=opentutor-classifier
ROOT=$(shell dirname ${PWD})
# VENV=.venv
$(POETRY):
	$(MAKE) $(POETRY)-update

.PHONY: $(POETRY)-update
$(POETRY)-update: virtualenv-installed
	poetry update
	poetry install
	poetry run python -m nltk.downloader punkt
	poetry run python -m nltk.downloader wordnet
	poetry run python -m nltk.downloader averaged_perceptron_tagger
	poetry run python -m nltk.downloader stopwords

.PHONY clean:
clean:
	rm -rf htmlcov .coverage tests/fixtures/shared/word2vec.bin

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
format: $(POETRY)
	cd $(ROOT) && $(MAKE) format

.PHONY: license
license:
	cd $(ROOT) && make license

PHONY: test
test: $(POETRY)
	cd $(ROOT)/word2vec/ \
		&& $(MAKE) word2vec.bin
	poetry run coverage run \
		--omit="$(PWD)/tests" \
		-m py.test -vv $(args)

PHONY: test
test-not-slow: $(POETRY)
	args="-m '(not slow)'" $(MAKE) test


.PHONY: test-all
test-all: test-format test-lint test-types test-license test

.PHONY: test-all
test-all-not-slow:
	args="-m '(not slow)'" $(MAKE) test-all


.PHONY: test-format
test-format: $(POETRY)
	cd $(ROOT) && $(MAKE) test-format

.PHONY: test-license
test-license: $(LICENSE) $(LICENSE_HEADER) $(POETRY)
	cd $(ROOT) && $(MAKE) test-license

.PHONY: test-lint
test-lint: $(POETRY)
	cd $(ROOT) && $(MAKE) test-lint

.PHONY: test-types
test-types: $(POETRY)
	cd $(ROOT) && $(MAKE) test-types

.PHONY: update-deps
update-deps: $(POETRY)
	poetry update

virtualenv-installed:
	$(ROOT)/tools/virtualenv_ensure_installed.sh
