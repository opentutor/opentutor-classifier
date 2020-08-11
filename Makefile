DOCKER_IMAGE?=opentutor_classifier_docker:latest

# virtualenv used for pytest
VENV=.venv
$(VENV):
	$(MAKE) $(VENV)-update

.PHONY: $(VENV)-update
$(VENV)-update: virtualenv-installed
	[ -d $(VENV) ] || virtualenv -p python3.8 $(VENV)
	$(VENV)/bin/pip install --upgrade pip
	$(VENV)/bin/pip install -r ./requirements.test.txt
	$(VENV)/bin/python3.8 -m nltk.downloader punkt
	$(VENV)/bin/python3.8 -m nltk.downloader wordnet
	$(VENV)/bin/python3.8 -m nltk.downloader averaged_perceptron_tagger
	$(VENV)/bin/python3.8 -m nltk.downloader stopwords

.PHONY clean:
clean:
	rm -rf .venv htmlcov .coverage tests/fixtures/shared/word2vec.bin

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
format: $(VENV)
	$(VENV)/bin/black .

PHONY: test
test: $(VENV)
	$(VENV)/bin/py.test -vv $(args)

.PHONY: test-all
test-all: test-format test-lint test-types test-license test

.PHONY: test-format
test-format: $(VENV)
	$(VENV)/bin/black --check .

.PHONY: test-lint
test-lint: $(VENV)
	$(VENV)/bin/flake8 .

.PHONY: test-types
test-types: $(VENV)
	. $(VENV)/bin/activate && mypy opentutor_classifier

.PHONY: update-deps
update-deps: $(VENV)
	. $(VENV)/bin/activate && pip-upgrade requirements*

virtualenv-installed:
	tools/virtualenv_ensure_installed.sh

LICENSE:
	@echo "you must have a LICENSE file" 1>&2
	exit 1

LICENSE_HEADER:
	@echo "you must have a LICENSE_HEADER file" 1>&2
	exit 1

.PHONY: license
license: LICENSE LICENSE_HEADER $(VENV)
	$(VENV)/bin/python3.8 -m licenseheaders -t LICENSE_HEADER -d tests
	$(VENV)/bin/python3.8 -m licenseheaders -t LICENSE_HEADER -d opentutor_classifier

.PHONY: test-license
test-license: LICENSE LICENSE_HEADER $(VENV)
	$(VENV)/bin/python3.8 -m licenseheaders -t LICENSE_HEADER -d tests --check
	$(VENV)/bin/python3.8 -m licenseheaders -t LICENSE_HEADER -d opentutor_classifier --check
