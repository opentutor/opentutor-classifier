DOCKER_IMAGE?=opentutor_classifier

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
.PHONY: docker-test-train
docker-test-train:
	docker run \
		-it \
		--rm \
		-v $(PWD)/tests/fixtures/data:/data \
		-v $(PWD)/docker-test-train-out:/docker-test-train-out \
	$(DOCKER_IMAGE) train --data /data/lesson1_dataset.csv --output /output

		
.PHONY: format
format: $(VENV)
	$(VENV)/bin/black .

PHONY: test
test: $(VENV)
	$(VENV)/bin/py.test -vv $(args)

.PHONY: test-all
test-all: test-format test-lint test-types test

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
