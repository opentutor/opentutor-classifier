BLACK_EXCLUDES="/(\.venv|build)/"
DOCKER_IMAGE?=opentutor-classifier-api
DOCKER_IMAGE_ID=$(shell docker images -q ${DOCKER_IMAGE} 2> /dev/null)
ROOT=$(shell dirname ${PWD})
OPENTUTOR_CLASSIFIER_SRC=$(ROOT)/opentutor_classifier
OPENTUTOR_CLASSIFIER_DIST=$(ROOT)/opentutor_classifier/dist/opentutor_classifier-1.0.0.tar.gz
POETRY=install

.PHONY: $(POETRY)
$(POETRY):
	cd $(ROOT)/opentutor_classifier \
		&& $(MAKE) poetry-build-classifier
	poetry env use python3.8
	poetry install
	poetry run python -m nltk.downloader punkt
	poetry run python -m nltk.downloader wordnet
	poetry run python -m nltk.downloader averaged_perceptron_tagger
	poetry run python -m nltk.downloader stopwords

.PHONY clean:
clean:
	rm -rf .pytest_cache .mypy_cache build

build/opentutor_classifier:
	mkdir -p build
	rsync -rv \
			--exclude .venv \
			--exclude htmlcov \
			--exclude .pytest_cache \
			--exclude tests \
			--exclude __pycache__ \
		$(OPENTUTOR_CLASSIFIER_SRC) build


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
format: $(POETRY)
	cd $(ROOT) && $(MAKE) format

.PHONY: license
license:
	cd $(ROOT) && make license

.PHONY: test
test: $(POETRY)
	cd $(ROOT)/shared/ \
		&& $(MAKE) installed/word2vec.bin
	poetry run	coverage run \
		--omit="$(PWD)/tests" \
		-m py.test -vv $(args)

PHONY: test
test-not-slow: $(POETRY)
	args="-m '(not slow)'" $(MAKE) test

.PHONY: test-all
test-all: test-format test-lint test-license test-types test

.PHONY: test-all
test-all-not-slow:
	args="-m '(not slow)'" $(MAKE) test-all

.PHONY: test-format
test-format: $(POETRY)
	cd $(ROOT) && $(MAKE) test-format

.PHONY: test-license
test-license:
	cd $(ROOT) && make test-license
	
.PHONY: test-lint
test-lint: $(POETRY)
	cd $(ROOT) && $(MAKE) test-lint

.PHONY: test-types
test-types: $(POETRY)
	cd $(ROOT) && $(MAKE) test-types

.PHONY: update-deps
update-deps:
	poetry update
