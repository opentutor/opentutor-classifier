$(POETRY):
	$(MAKE) $(POETRY)-update

.PHONY: $(POETRY)-update
$(POETRY)-update:
	poetry update
	poetry install

.PHONY clean:
clean:
	rm -rf installed

installed:
	mkdir -p installed

installed/word2vec.bin: $(POETRY)
	$(MAKE) installed
	poetry run python word2vec_download.py

.PHONY: update-deps
update-deps: $(POETRY)
	poetry update
