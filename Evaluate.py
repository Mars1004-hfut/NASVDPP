'''
Created on Apr 15, 2016
Evaluate the performance of Top-K recommendation:
    Protocol: leave-1-out evaluation
    Measures: Hit Ratio and NDCG
    (more details are in: Xiangnan He, et al. Fast Matrix Factorization for Online Recommendation with Implicit Feedback. SIGIR'16)
@author: hexiangnan
'''
import numpy as np
import tensorflow as tf
from math import sqrt 
from sklearn.metrics import mean_squared_error

# Global variables that are shared across processes
_model = None
_validRatings = None
_testRatings  = None
_sess = None

def init_evaluate_model(model, sess, validRatings, testRatings, trainList):
    """
    Evaluate the performance (Hit_Ratio, NDCG) of top-K recommendation
    Return: score of each test rating.
    """
    global _model
    global _testRatings
    global _validRatings
    global _trainList
    global _sess
    _sess = sess
    _model = model
    _validRatings = validRatings
    _testRatings  = testRatings
    _trainList = trainList
    return (load_valid_as_list(), load_test_as_list())

def eval(model, sess, ratings, dictList):
    
    global _model
    global _sess
    _model = model
    _sess = sess
    
    rmse = _eval_rating(ratings, dictList)
    return np.array(rmse).mean()

def load_valid_as_list():
    DictList = []

    for idx in range(len(_validRatings)):
        rating = _validRatings[idx]
        user_rated_items_num = _trainList[rating[0]-1]
        num_idx = len(user_rated_items_num)

        user_hist = []
        user_hist.append(user_rated_items_num)  # the items for one user
        user_input = rating[0]
        item_input = rating[1]
        label = rating[2]
        feed_dict = {_model.user_input: [[user_input]], _model.user_hist: user_hist, _model.num_idx: [[num_idx]],
                     _model.item_input: [[item_input]], _model.labels: [[label]]}  # , _model.is_train_phase: False

        DictList.append(feed_dict)
    print("already load the evaluate model...")
    return DictList

def load_test_as_list():
    DictList = []
    for idx in range(len(_testRatings)):
        rating = _testRatings[idx]
        # user_rated_items_num = _trainList[int(idx / 2)]   # fixme
        user_rated_items_num = _trainList[rating[0] - 1]
        num_idx = len(user_rated_items_num)

        user_hist = []
        user_hist.append(user_rated_items_num)  # the items for one user
        user_input = rating[0]
        item_input = rating[1]
        label = rating[2]

        feed_dict = {_model.user_input: [[user_input]], _model.user_hist: user_hist, _model.num_idx: [[num_idx]],
                     _model.item_input: [[item_input]], _model.labels: [[label]]}
        DictList.append(feed_dict)
    print("already load the evaluate model...")
    return DictList

def _eval_rating(ratings, dictList):
    prediction = []
    for idx1 in range(len(dictList)):
        prediction1 = _sess.run(_model.output, feed_dict = dictList[idx1])
        prediction.append(prediction1[0][0])

    # print(prediction)
    # print(len(prediction))
    rating = []
    for idx in range(len(ratings)):
        rating.append([ratings[idx][2]])

    num_example = len(prediction)
    y_pred = np.reshape(prediction, (num_example,))
    y_actual = np.reshape(np.array(rating), (num_example,))
    y_bounded = np.maximum(y_pred, np.ones(num_example) * 1)  # bound the lower values
    y_bounded = np.minimum(y_pred, np.ones(num_example) * 5)  # bound the higher values

    return sqrt(mean_squared_error(y_actual, y_bounded))

