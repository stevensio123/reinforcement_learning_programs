import fin_algo_util as utils
import numpy as np

RANDOM_SAMPLE = np.random.uniform(low=5, high=20, size=5000)
# print(f"autocorrelation: {utils.autocorrelation(RANDOM_SAMPLE.tolist(), lag_time=3)}")
print(f"autocorrelation_v2: {utils.autocorrelation_v2(RANDOM_SAMPLE.tolist(), lag=3)}")
