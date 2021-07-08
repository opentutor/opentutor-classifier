ROOT=$(shell dirname ${PWD})
VENV=.venv
$(VENV):
	$(MAKE) $(VENV)-update

.PHONY: $(VENV)-update
$(VENV)-update: virtualenv-installed
	[ -d $(VENV) ] || virtualenv -p python3.8 $(VENV)
	$(VENV)/bin/pip install --upgrade pip
	$(VENV)/bin/pip install -r ./requirements.txt

.PHONY clean:
clean:
	rm -rf .venv installed

installed:
	mkdir -p installed

installed/word2vec.bin: $(VENV)
	$(MAKE) installed
	. $(VENV)/bin/activate && python word2vec_download.py 

installed/sentence-transformer: $(VENV)
	$(MAKE) installed
	. $(VENV)/bin/activate && python sentence_transformer_download.py 

.PHONY: update-deps
update-deps: $(VENV)
	. $(VENV)/bin/activate && pip-upgrade requirements*

virtualenv-installed:
	$(ROOT)/tools/virtualenv_ensure_installed.sh