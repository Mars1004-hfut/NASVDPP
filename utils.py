from __future__ import print_function
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt

# shuffle training data and labels
def shuffle_data(data):
    rng_state = np.random.get_state()
    np.random.shuffle(data['user'])
    np.random.set_state(rng_state)
    np.random.shuffle(data['item'])
    np.random.set_state(rng_state)
    np.random.shuffle(data['rating'])
    return data
    
# evaluate the RMSE    
def rmse(y_actual, y_pred):
    num_example = len(y_pred)
    y_pred = np.reshape(y_pred, (num_example,))
    y_actual = np.reshape(y_actual, (num_example,))
    y_bounded = np.maximum(y_pred, np.ones(num_example) * 1)  # bound the lower values
    y_bounded = np.minimum(y_pred, np.ones(num_example) * 5)  # bound the higher values
    return sqrt(mean_squared_error(y_actual, y_bounded))

# whether to early stop
def early_stop(valid):
    if len(valid) >= 5:
        if valid[-1] > valid[-2] and valid[-2] > valid[-3] and valid[-3] > valid[-4] and valid[-4] > valid[-5]:
        	return True
    return False

