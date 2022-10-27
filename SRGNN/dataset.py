import os
import pickle
import torch
from torch.utils.data import Dataset
import numpy as np

def load_data(root, valid_portion=0.1, maxlen=19, sort_by_len=True):
    '''Loads the dataset
    :root: The path to the dataset 
    :valid_portion: The proportion of the full train set used for the validation set.
    :maxlen(type: None or positive int): the max sequence length we use in the train/valid set.
    :sort_by_len(type : bool): Sort by the sequence lenght for the train,
        valid and test set. This allow faster execution as it cause
        less padding per minibatch. Another mechanism must be used to
        shuffle the train set at each epoch.
    '''
    
    # Load the dataset
    train_path_data=os.path.join(root,"train.txt")
    test_path_data=os.path.join(root,"test.txt")
    
    with open(train_path_data, 'rb') as f1:
        train_set = pickle.load(f1)

    with open(test_path_data, 'rb') as f2:
        test_set = pickle.load(f2)

    
    if maxlen:
        new_train_set_x = []
        new_train_set_y = []
        for x, y in zip(train_set[0], train_set[1]):
            if len(x) < maxlen:
                new_train_set_x.append(x)
                new_train_set_y.append(y)
            else:
                new_train_set_x.append(x[:maxlen])
                new_train_set_y.append(y)
        train_set = (new_train_set_x, new_train_set_y)
        del new_train_set_x, new_train_set_y

        new_test_set_x = []
        new_test_set_y = []
        for xx, yy in zip(test_set[0], test_set[1]):
            if len(xx) < maxlen:
                new_test_set_x.append(xx)
                new_test_set_y.append(yy)
            else:
                new_test_set_x.append(xx[:maxlen])
                new_test_set_y.append(yy)
        test_set = (new_test_set_x, new_test_set_y)
        del new_test_set_x, new_test_set_y

    # split training set into validation set
    train_set_x, train_set_y = train_set
    
    n_samples = len(train_set_x)
    sidx = np.arange(n_samples, dtype='int32')
    np.random.shuffle(sidx)
    n_train = int(np.round(n_samples * (1. - valid_portion)))
    valid_set_x = [train_set_x[s] for s in sidx[n_train:]]
    valid_set_y = [train_set_y[s] for s in sidx[n_train:]]
    train_set_x = [train_set_x[s] for s in sidx[:n_train]]
    train_set_y = [train_set_y[s] for s in sidx[:n_train]]

    (test_set_x, test_set_y) = test_set

    def len_argsort(seq):
        return sorted(range(len(seq)), key=lambda x: len(seq[x]))

    if sort_by_len:
        sorted_index = len_argsort(test_set_x)
        test_set_x = [test_set_x[i] for i in sorted_index]
        test_set_y = [test_set_y[i] for i in sorted_index]

        sorted_index = len_argsort(valid_set_x)
        valid_set_x = [valid_set_x[i] for i in sorted_index]
        valid_set_y = [valid_set_y[i] for i in sorted_index]
        
        sorted_index = len_argsort(train_set_x)
        train_set_x = [train_set_x[i] for i in sorted_index]
        train_set_y = [train_set_y[i] for i in sorted_index]

    train = (train_set_x, train_set_y)
    valid = (valid_set_x, valid_set_y)
    test = (test_set_x, test_set_y)
    
    return train, valid, test


class RecSysDataset(Dataset):
    """define the pytorch Dataset class for yoochoose and diginetica datasets.
    """
    def __init__(self, data):
        self.data = data
        print('-'*50)
        print('Dataset info:')
        print('Number of sessions: {}'.format(len(data[0])))
        print('-'*50)
        
    def __getitem__(self, index):
        session_items = self.data[0][index]
        try: 
            target_item = self.data[1][index]
            return session_items, target_item
        except: 
#             return session_items, ['pred_output']
            return  session_items, [['pred_output']]
        

    def __len__(self):
        return len(self.data[0])
    
