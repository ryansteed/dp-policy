from dp_policy.experiments import titlei_grid as test_params
from dp_policy.titlei.mechanisms import Laplace
from dp_policy.titlei.utils import *
from dp_policy.titlei.thresholders import *
from dp_policy.titlei.evaluation import *
from dp_policy.experiments import *

import numpy as np
import pickle

saipe = get_inputs(2021)
sppe = get_sppe("data/sppe18.xlsx")
results = test_params(
  saipe, Laplace,
  eps=[0.01, 0.1, 1.0, 2.52, 10.0]+list(np.logspace(-3, 0.1, num=5)),
  trials=1000, print_results=False
)
pickle.dump(results, open("results/titlei_laplace.pkl", 'wb'))
