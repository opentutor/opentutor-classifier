ROOT=$(shell dirname ${PWD})
$(POETRY):
	$(MAKE) $(POETRY)-update

.PHONY: $(POETRY)-update
$(POETRY)-update: virtualenv-installed
	poetry update
	poetry install

.PHONY clean:
clean:
	rm -rf word2vec.bin word2vec.zip

.PHONY: update-deps
update-deps: $(POETRY)
	poetry update

virtualenv-installed:
	$(ROOT)/tools/virtualenv_ensure_installed.sh

word2vec.bin: $(POETRY)
	poetry run python word2vec_download.py 
