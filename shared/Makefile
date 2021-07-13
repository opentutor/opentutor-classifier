VENV=.venv

$(VENV):
	$(MAKE) install

.PHONY clean:
clean:
	rm -rf installed

.PHONY: install
install:
	poetry config --local virtualenvs.in-project true
	poetry env use python3.8
	poetry install

installed:
	mkdir -p installed

installed/word2vec.bin: $(VENV)
	$(MAKE) installed
	poetry run python word2vec_download.py 

.PHONY: update-deps
update-deps:
	poetry update
