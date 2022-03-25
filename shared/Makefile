ROOT=$(shell dirname ${PWD})
VENV=.venv

$(VENV):
	$(MAKE) install

.PHONY clean:
clean:
	rm -rf installed $(VENV)

.PHONY: deps-show
deps-show:
	poetry show

.PHONY: deps-show
deps-show-outdated:
	poetry show --outdated

.PHONY: deps-update
deps-update:
	poetry update

.PHONY: install
install: poetry-ensure-installed
	poetry config --local virtualenvs.in-project true
	poetry env use python3.8
	poetry install

installed:
	mkdir -p installed

installed/word2vec.bin: $(VENV)
	$(MAKE) installed
	poetry run python word2vec_download.py 
	poetry run python word2vec_slim_download.py 


.PHONY: poetry-ensure-installed
poetry-ensure-installed:
	sh $(ROOT)/tools/poetry_ensure_installed.sh
