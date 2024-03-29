import multiprocessing
import numpy as np

_Dataset = None 
_batch_size = None
_num_items = None
_user_input = None
_item_input = None
_labels = None
_index = None
_num_batch = None
_batch_length = None

def shuffle(dataset, batch_choice):   #negative sampling and shuffle the data

    global _Dataset
    global _batch_size
    global _num_users
    global _num_items
    global _user_input
    global _item_input
    global _labels
    global _index
    global _num_batch
    global _batch_length
    _Dataset = dataset

    if batch_choice == 'user':
        _num_users, _num_items, _user_input, _item_input, _labels, _batch_length = _get_train_data_user()
        _num_batch = len(_batch_length)
        return _preprocess(_get_train_batch_user)

    else:
        batch_choices = batch_choice.split(":")
        if batch_choices[0] == 'fixed':
            _batch_size = int(batch_choices[1])
            _num_users, _num_items, _user_input, _item_input, _labels = _get_train_data_fixed()
            iterations = len(_user_input)
            _index = np.arange(iterations)
            _num_batch = iterations / _batch_size
            return _preprocess(_get_train_batch_fixed)
        else:
            print ("invalid batch size !")

def batch_gen(batches, i):  
    return [(batches[r])[i] for r in range(5)]

def _preprocess(get_train_batch):    #generate the masked batch list
    user_input_list, user_hist_list, num_idx_list, item_input_list, labels_list = [], [], [], [], []
    # cpu_count = multiprocessing.cpu_count()
    # if cpu_count == 1:
    for i in range(_num_batch):
        ui, uh, ni, ii, l = get_train_batch(i)
        user_input_list.append(ui)
        user_hist_list.append(uh)
        num_idx_list.append(ni)
        item_input_list.append(ii)
        labels_list.append(l)
    # else:
    #     pool = multiprocessing.Pool(cpu_count)
    #     res = pool.map(get_train_batch, range(_num_batch))
    #     pool.close()
    #     pool.join()
    #     user_input_list = [r[0] for r in res]
    #     user_hist_list = [r[1] for r in res]
    #     num_idx_list = [r[2] for r in res]
    #     item_input_list = [r[3] for r in res]
    #     labels_list = [r[4] for r in res]
    return (user_input_list, user_hist_list, num_idx_list, item_input_list, labels_list)

def _get_train_data_user():
    user_input, item_input, labels, batch_length = [], [], [], []
    train = _Dataset.trainMatrix
    trainList = _Dataset.trainList # get all items rated by the corresponding user
    num_users = train.shape[0] # get the number of users
    num_items = train.shape[1] # get the number of items
    for u in range(0, num_users-1):
       if u == 0:
           batch_length.append(len(trainList[u]))
       else:
           batch_length.append(len(trainList[u]) + batch_length[u-1])
       for i in trainList[u]:
            user_input.append(u+1)
            item_input.append(i)
            labels.append(train[u+1, i])

    return num_users, num_items, user_input, item_input, labels, batch_length  # length for batch

def _get_train_batch_user(i):
    #represent the feature of users via items rated by him/her
    user_list, user_hist_list, num_list, item_list, labels_list = [], [], [], [], []
    trainList = _Dataset.trainList
    if i == 0:
        begin = 0
    else:
        begin = _batch_length[i-1]
    batch_index = list(range(begin, _batch_length[i]))
    np.random.shuffle(batch_index)
    for idx in batch_index:
        user_idx = _user_input[idx]
        item_idx = _item_input[idx]
        nonzero_row = []
        nonzero_row += trainList[user_idx-1]
        # num_list.append(_remove_item(_num_items, nonzero_row, item_idx))
        num_list.append(len(nonzero_row))
        user_hist_list.append(nonzero_row)
        user_list.append(user_idx)
        item_list.append(item_idx)
        labels_list.append(_labels[idx])
    # user_hist = np.array(_add_mask(_num_items, user_hist_list, max(num_list)))  # fixme 3953
    user_hist = np.array(user_hist_list)
    num_idx = np.array(num_list)
    user_input = np.array(user_list)
    item_input = np.array(item_list)
    labels = np.array(labels_list)
    return (user_input, user_hist, num_idx, item_input, labels)

def _get_train_data_fixed():
    user_input, item_input, labels = [], [], []
    train = _Dataset.trainMatrix
    num_users = train.shape[0]
    num_items = train.shape[1]
    for (u, i) in train.keys():
        # positive instance
        user_input.append(u)
        item_input.append(i)
        labels.append(train[u, i])
    return num_users, num_items, user_input, item_input, labels

def _get_train_batch_fixed(i):
    #represent the feature of users via items rated by him/her
    user_list, user_hist_list, num_list, item_list,labels_list = [], [], [], [], []
    trainList = _Dataset.trainList
    begin = i * _batch_size
    for idx in range(begin, begin+_batch_size):
        user_idx = _user_input[_index[idx]]
        item_idx = _item_input[_index[idx]]
        nonzero_row = []
        nonzero_row += trainList[user_idx-1]
        # num_list.append(_remove_item(_num_items, nonzero_row, item_idx))   #fixme
        num_list.append(len(nonzero_row))
        user_hist_list.append(nonzero_row)
        user_list.append(user_idx)
        item_list.append(item_idx)
        labels_list.append(_labels[_index[idx]])
    
    user_hist = np.array(_add_mask(_num_items, user_hist_list, max(num_list)))
    num_idx = np.array(num_list)
    user_input = np.array(user_list)
    item_input = np.array(item_list)
    labels = np.array(labels_list)
    return (user_input, user_hist, num_idx, item_input, labels)

# def _remove_item(feature_mask, users, item):
#     flag = 0
    # for i in range(len(users)):
    #     if users[i] == item:
    #         users[i]  = users[-1]
    #         users[-1] = feature_mask
    #         flag = 1
    #         break
    # return len(users) - flag

def _add_mask(feature_mask, features, num_max):
    #uniformalize the length of each batch
    for i in range(len(features)):
        features[i] = features[i] + [feature_mask] * (num_max + 1 - len(features[i]))
    return features
