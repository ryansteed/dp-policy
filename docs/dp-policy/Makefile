.PHONY: dp_policy

VENV_NAME?=venv
PYTHON=${VENV_NAME}/bin/python

dp_policy: $(VENV_NAME)/bin/activate
	mkdir logs
	mkdir -p plots/bootstrap
	mkdir -p plots/geo
	mkdir -p plots/race
	mkdir -p plots/robustness
	mkdir -p plots/smooths
	mkdir -p plots/tables
	mkdir -p results/policy_experiments
	mkdir -p results/bootstrap
	mkdir -p results/regressions

$(VENV_NAME)/bin/activate: requirements.txt setup.py
	test -d $(VENV_NAME) || python3.9 -m venv $(VENV_NAME)
	${PYTHON} -m pip install -e .
	${PYTHON} -m pip install -r requirements.txt
	touch $(VENV_NAME)/bin/activate

clean:
	rm -rf venv
	rm -rf results

zip: software.zip data.zip
software.zip:
	zip -r software.zip \
		Makefile \
		R/*.R \
		R/*.Rmd \
		README.md \
		dp_policy \
		notebooks/*.ipynb \
		requirements.txt \
		scripts \
		setup.py \
		-x "dp_policy/titlei/__pycache__/*" \
		-x "*.DS_Store"
data.zip:
	zip -r data.zip \
		data \
		-x "*.DS_Store"
		# -x "data/discrimination/*" \
		# -x "data/shapefiles/*"
	# for i in data/discrimination/*.txt; do zip -r "$$i".zip "$$i"; done
