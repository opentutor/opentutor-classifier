.PHONY: docker-build
docker-build:
	cd opentutor_classifier && $(MAKE) docker-build
	cd opentutor_classifier_api && $(MAKE) docker-build

.PHONY: format
format:
	cd opentutor_classifier && $(MAKE) format
	cd opentutor_classifier_api && $(MAKE) format

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
test-format:
	cd opentutor_classifier && $(MAKE) test-format
	cd opentutor_classifier_api && $(MAKE) test-format

.PHONY: test-lint
test-lint:
	cd opentutor_classifier && $(MAKE) test-lint
	cd opentutor_classifier_api && $(MAKE) test-lint

.PHONY: test-types
test-types:
	cd opentutor_classifier && $(MAKE) test-types
	cd opentutor_classifier_api && $(MAKE) test-types

LICENSE:
	@echo "you must have a LICENSE file" 1>&2
	exit 1

LICENSE_HEADER:
	@echo "you must have a LICENSE_HEADER file" 1>&2
	exit 1

.PHONY: license
license: LICENSE LICENSE_HEADER $(VENV)
	$(VENV)/bin/python3.8 -m licenseheaders -t LICENSE_HEADER -d src
	$(VENV)/bin/python3.8 -m licenseheaders -t LICENSE_HEADER -d tests

.PHONY: test-license
test-license: LICENSE LICENSE_HEADER
	cd opentutor_classifier && $(MAKE) test-license
	cd opentutor_classifier_api && $(MAKE) test-license
