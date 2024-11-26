import sys

sys.path.append("../")

from autoray import numpy as anp
from autoray import to_backend_dtype

from integration.vegas_mul_map import VEGASMultiMap

from helper_functions import setup_test_for_backend


VEGASMultiMap()