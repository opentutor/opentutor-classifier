ROOT=$(shell dirname ${PWD})
POETRY=install

.PHONY: $(POETRY)
$(POETRY):
	poetry install

.PHONY clean:
clean:
	rm -rf installed

installed:
	mkdir -p installed

installed/word2vec.bin: $(POETRY)
	$(MAKE) installed
	poetry run word2vec_download.py 

.PHONY: update-deps
update-deps:
	poetry update
