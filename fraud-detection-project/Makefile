# Makefile for Fraud Detection Project

# Define variables
PYTHON=python
NOTEBOOKS=notebooks
SCRIPTS=scripts
SRC=src
TESTS=tests

# Define tasks
.PHONY: all clean install run-notebooks run-scripts run-tests

all: install

install:
	$(PYTHON) -m pip install -r requirements.txt

run-notebooks:
	jupyter nbconvert --to notebook --execute $(NOTEBOOKS)/00-eda-fraud-data.ipynb
	jupyter nbconvert --to notebook --execute $(NOTEBOOKS)/01-data-preprocessing.ipynb
	jupyter nbconvert --to notebook --execute $(NOTEBOOKS)/02-feature-engineering.ipynb
	jupyter nbconvert --to notebook --execute $(NOTEBOOKS)/03-modeling.ipynb
	jupyter nbconvert --to notebook --execute $(NOTEBOOKS)/04-explainability.ipynb

run-scripts:
	$(PYTHON) $(SCRIPTS)/preprocess.py
	$(PYTHON) $(SCRIPTS)/train.py
	$(PYTHON) $(SCRIPTS)/evaluate.py
	$(PYTHON) $(SCRIPTS)/explain.py

run-tests:
	$(PYTHON) -m unittest discover -s $(TESTS)

clean:
	find . -type d -name '__pycache__' -exec rm -r {} +
	rm -f *.pyc
	rm -f $(NOTEBOOKS)/*.nbconvert.ipynb
	rm -f $(SRC)/**/*.pyc
	rm -f $(TESTS)/**/*.pyc