# virtualenv used for pytest
VENV=.venv
$(VENV):
	$(MAKE) venv-create

.PHONY clean:
clean:
	rm -rf .venv htmlcov .coverage 

.PHONY: format
format: $(VENV)
	$(VENV)/bin/black opentutor_classifier tests

PHONY: test
test: $(VENV)
	$(VENV)/bin/py.test -vv $(args)

.PHONY: test-all
test-all: test-format test-lint test-types test

.PHONY: test-format
test-format: $(VENV)
	$(VENV)/bin/black --check opentutor_classifier tests

.PHONY: test-lint
test-lint: $(VENV)
	$(VENV)/bin/flake8 .

.PHONY: test-types
test-types: $(VENV)
	. $(VENV)/bin/activate && mypy opentutor_classifier

.PHONY: train
train: $(VENV)
	. $(VENV)/bin/activate \
	&& python3 classifier_train.py

.PHONY: update-deps
update-deps: $(VENV)
	. $(VENV)/bin/activate && pip-upgrade requirements*

.PHONY: venv-create
venv-create: virtualenv-installed
	[ -d $(VENV) ] || virtualenv -p python3.8 $(VENV)
	$(VENV)/bin/pip install --upgrade pip
	$(VENV)/bin/pip install -r ./requirements.test.txt
	$(VENV)/bin/python3.8 -m nltk.downloader punkt
	$(VENV)/bin/python3.8 -m nltk.downloader wordnet
	$(VENV)/bin/python3.8 -m nltk.downloader averaged_perceptron_tagger
	$(VENV)/bin/python3.8 -m nltk.downloader stopwords

virtualenv-installed:
	bin/virtualenv_ensure_installed.sh
