BLACK_EXCLUDES="/(\.venv|build)/"
DOCKER_IMAGE?=opentutor-classifier-api
DOCKER_IMAGE_ID=$(shell docker images -q ${DOCKER_IMAGE} 2> /dev/null)
ROOT=$(shell dirname ${PWD})
OPENTUTOR_CLASSIFIER_SRC=$(ROOT)/opentutor_classifier
VENV=.venv
VENV_PIP=$(VENV)/bin/pip
$(VENV):
	$(MAKE) $(VENV)-update

.PHONY $(VENV)-installed:
$(VENV)-installed:
	$(ROOT)/tools/virtualenv_ensure_installed.sh

.PHONY: $(VENV)-update
$(VENV)-update: $(VENV)-installed
	[ -d $(VENV) ] || virtualenv -p python3.8 $(VENV)
	$(VENV_PIP) install --upgrade pip
	$(VENV_PIP) install -r requirements.test.txt
	$(VENV_PIP) install ../opentutor_classifier
	$(VENV)/bin/python3.8 -m nltk.downloader punkt
	$(VENV)/bin/python3.8 -m nltk.downloader wordnet
	$(VENV)/bin/python3.8 -m nltk.downloader averaged_perceptron_tagger
	$(VENV)/bin/python3.8 -m nltk.downloader stopwords

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
docker-build: clean build/opentutor_classifier
	docker build -t $(DOCKER_IMAGE) .

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

.PHONY: license
license:
	cd $(ROOT) && make license

.PHONY: test
test: $(VENV)
	cd $(ROOT)/word2vec/ \
		&& $(MAKE) word2vec.bin
	. $(VENV)/bin/activate \
		&& export PYTHONPATH=$${PYTHONPATH}:$(PWD)/src \
		&& coverage run \
			--omit="$(PWD)/tests $(VENV)" \
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
test-format: $(VENV)
	cd $(ROOT) && $(MAKE) test-format

.PHONY: test-license
test-license:
	cd $(ROOT) && make test-license
	
.PHONY: test-lint
test-lint: $(VENV)
	cd $(ROOT) && $(MAKE) test-lint

.PHONY: test-types
test-types: $(VENV)
	cd $(ROOT) && $(MAKE) test-types

.PHONY: update-deps
update-deps: $(VENV)
	. $(VENV)/bin/activate && pip-upgrade requirements*
