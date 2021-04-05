LICENSE=LICENSE
LICENSE_HEADER=LICENSE_HEADER
VENV=.venv
$(VENV):
	$(MAKE) $(VENV)-update

.PHONY: $(VENV)-update
$(VENV)-update: virtualenv-installed
	[ -d $(VENV) ] || virtualenv -p python3.8 $(VENV)
	$(VENV)/bin/pip install --upgrade pip
	$(VENV)/bin/pip install -r ./requirements.txt

.PHONY: docker-build
docker-build:
	cd opentutor_classifier && $(MAKE) docker-build
	cd opentutor_classifier_api && $(MAKE) docker-build

.PHONY: format
format: $(VENV)
	$(VENV)/bin/black .

LICENSE:
	@echo "you must have a LICENSE file" 1>&2
	exit 1

LICENSE_HEADER:
	@echo "you must have a LICENSE_HEADER file" 1>&2
	exit 1

.PHONY: license
license: LICENSE LICENSE_HEADER $(VENV)
	. $(VENV)/bin/activate \
		&& python -m licenseheaders -t LICENSE_HEADER -d opentutor_classifier/src $(args) \
		&& python -m licenseheaders -t LICENSE_HEADER -d opentutor_classifier/tests $(args) \
		&& python -m licenseheaders -t LICENSE_HEADER -d opentutor_classifier_api/src $(args) \
		&& python -m licenseheaders -t LICENSE_HEADER -d opentutor_classifier_api/tests $(args) \
		&& python -m licenseheaders -t LICENSE_HEADER -d tools $(args) \
		&& python -m licenseheaders -t LICENSE_HEADER -d word2vec $(args)

.PHONY: test
test:
	cd opentutor_classifier && $(MAKE) test
	cd opentutor_classifier_api && $(MAKE) test

.PHONY: test-all
test-all:
	$(MAKE) test-format
	$(MAKE) test-lint
	$(MAKE) test-license
	$(MAKE) test-types
	$(MAKE) test

.PHONY: test-format
test-format: $(VENV)
	$(VENV)/bin/black --check .

.PHONY: test-lint
test-lint: $(VENV)
	$(VENV)/bin/flake8 .

.PHONY: test-license
test-license: LICENSE LICENSE_HEADER
	args="--check" $(MAKE) license

.PHONY: test-types
test-types: $(VENV)
	. $(VENV)/bin/activate \
		&& mypy opentutor_classifier \
		&& mypy opentutor_classifier_api \
		&& mypy word2vec

virtualenv-installed:
	tools/virtualenv_ensure_installed.sh

.PHONY: update-deps
update-deps: $(VENV)
	. $(VENV)/bin/activate && pip-upgrade requirements*
