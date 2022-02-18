VENV_NAME?=venv
PYTHON=${VENV_NAME}/bin/python

dp_policy: $(VENV_NAME)/bin/activate
	mkdir -p results/policy_experiments

$(VENV_NAME)/bin/activate: requirements.txt setup.py
	test -d $(VENV_NAME) || python3 -m venv $(VENV_NAME)
	${PYTHON} -m pip install -U pip
	${PYTHON} -m pip install -e .
	${PYTHON} -m pip install -r requirements.txt
	touch $(VENV_NAME)/bin/activate

clean:
	rm -rf venv
	rm -rf results