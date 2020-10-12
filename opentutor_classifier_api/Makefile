BLACK_EXCLUDES="/(\.venv|build)/"
DOCKER_IMAGE?=opentutor-classifier-api
DOCKER_IMAGE_ID=$(shell docker images -q ${DOCKER_IMAGE} 2> /dev/null)
ROOT=$(shell dirname ${PWD})
LICENSE=${ROOT}/LICENSE
LICENSE_HEADER=${ROOT}/LICENSE_HEADER
VENV=.venv
VENV_PIP=$(VENV)/bin/pip
$(VENV):
	$(MAKE) $(VENV)-update

.PHONY $(VENV)-installed:
$(VENV)-installed:
	tools/virtualenv_ensure_installed.sh

.PHONY: $(VENV)-update
$(VENV)-update: $(VENV)-installed
	[ -d $(VENV) ] || virtualenv -p python3.8 $(VENV)
	$(VENV_PIP) install --upgrade pip
	$(VENV_PIP) install -r requirements.txt
	$(VENV_PIP) install -r requirements.test.txt
	$(VENV)/bin/python3.8 -m nltk.downloader punkt
	$(VENV)/bin/python3.8 -m nltk.downloader wordnet
	$(VENV)/bin/python3.8 -m nltk.downloader averaged_perceptron_tagger
	$(VENV)/bin/python3.8 -m nltk.downloader stopwords

.PHONY clean:
clean:
	rm -rf .pytest_cache .mypy_cache

.PHONY docker-build:
docker-build:
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
	$(VENV)/bin/black --exclude $(BLACK_EXCLUDES) .

.PHONY: test
test: $(VENV)
	. $(VENV)/bin/activate \
		&& export PYTHONPATH=$${PYTHONPATH}:$(PWD)/src \
		&& py.test -vv $(args)

.PHONY: test-all
test-all:
	$(MAKE) test-format
	$(MAKE) test-lint
	$(MAKE) test-license
	$(MAKE) test-types
	$(MAKE) test

.PHONY: test-format
test-format: $(VENV)
	$(VENV)/bin/black --check --exclude $(BLACK_EXCLUDES) .

.PHONY: test-lint
test-lint: $(VENV)
	$(VENV)/bin/flake8 .

.PHONY: test-types
test-types: $(VENV)
	. $(VENV)/bin/activate && mypy src

.PHONY: update-deps
update-deps: $(VENV)
	. $(VENV)/bin/activate && pip-upgrade requirements*

$(LICENSE):
	@echo "you must have a LICENSE file" 1>&2
	exit 1

$(LICENSE_HEADER):
	@echo "you must have a LICENSE_HEADER file" 1>&2
	exit 1

.PHONY: license
license: $(LICENSE) $(LICENSE_HEADER) $(VENV)
	$(VENV)/bin/python3.8 -m licenseheaders -t $(LICENSE_HEADER) -d src
	$(VENV)/bin/python3.8 -m licenseheaders -t $(LICENSE_HEADER) -d tests

.PHONY: test-license
test-license: $(LICENSE) $(LICENSE_HEADER) $(VENV)
	$(VENV)/bin/python3.8 -m licenseheaders -t $(LICENSE_HEADER) -d src --check
	$(VENV)/bin/python3.8 -m licenseheaders -t $(LICENSE_HEADER) -d tests --check
