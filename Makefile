.DEFAULT_GOAL := help
SHELL := /usr/bin/env bash

UNAME_S := $(shell uname -s)
PYENV_NAME := $(notdir $(CURDIR))
PYTHON_VERSION := 3.11.3
NOTEBOOK_VERSION := 7.0.0rc2

PYTHON_PACKAGES := \
	pipreqs black flake8 \
	pandas ipykernel matplotlib seaborn \
	jupyter notebook==$(NOTEBOOK_VERSION) \
	torch

deps:  ## Ensure OS Dependencies (Only works for MacOS)
ifeq ($(UNAME_S),Darwin)
	@# Check only for MacOS
	@for dep in brew pyenv pyenv-virtualenv; do \
	  if ! which $$dep 2>&1 > /dev/null; then echo "Please install $$dep"; fi; \
	done;
endif

pyenv: deps  ## Create the pyenv for Python development
	@if ! pyenv versions | grep $(PYTHON_VERSION) 2>&1 > /dev/null; then \
	  pyenv install $(PYTHON_VERSION); \
	fi
	@if ! pyenv virtualenvs | grep $(PYENV_NAME) 2>&1 > /dev/null; then \
	  pyenv virtualenv $(PYTHON_VERSION) $(PYENV_NAME); \
	fi
	@if ! pyenv local 2>&1 > /dev/null; then \
	  pyenv local $(PYENV_NAME); \
	fi
	@PIP_FREEZE_OUT=$$(pip freeze) && \
	for dep in $(PYTHON_PACKAGES); do \
	  if ! echo "$$PIP_FREEZE_OUT" | grep $$dep 2>&1 > /dev/null; then pip install $$dep; fi; \
	done

jupyter: pyenv  ## Run Jupyter Notebook
	if ! jupyter kernelspec list | awk '{print $$1}' | grep $(PYENV_NAME) 2>&1 > /dev/null; then \
	  python -m ipykernel install --user --name $(PYENV_NAME) --display-name $(PYENV_NAME); \
	fi
	jupyter notebook

mrclean: ## Clean everything
	jupyter kernelspec remove $(PYENV_NAME)
	pyenv local --unset
	pyenv virtualenv-delete -f $(PYENV_NAME)

help: ## Print list of Makefile targets
	@# Taken from https://github.com/spf13/hugo/blob/master/Makefile
	@grep --with-filename -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  cut -d ":" -f2- | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}' | sort

# References
# - Using Jupyter Notebook in Virtual Environments for Python Data Science Projects
#   https://towardsdatascience.com/jupyter-notebooks-i-getting-started-with-jupyter-notebooks-f529449797d2
