import os
import pickle
import torch
from torch.utils.data import Dataset
import numpy as np

def load_data(root, sequence_type="all",maxlen=None, sort_by_len=False):
    '''Loads the dataset
    :root: The path to the dataset 
    :sequence_type(type:str): all sequence or longer only sequence(>5) or short only sequence(<=5) 
    :maxlen(type: None or positive int): the max sequence length we use in the train/valid set.
    :sort_by_len(type : bool): Sort by the sequence lenght for the train,
        valid and test set. This allow faster execution as it cause
        less padding per minibatch. Another mechanism must be used to
        shuffle the train set at each epoch.
    '''
    
    # Load the dataset
    if sequence_type=="all":
        train_path_data=os.path.join(root,"train.txt")
        test_path_data=os.path.join(root,"test.txt")
        with open(train_path_data, 'rb') as f1:
            train_set = pickle.load(f1)
        with open(test_path_data, 'rb') as f2:
            test_set = pickle.load(f2)
    elif sequence_type=="long":
        data_dir=os.path.join(root,"long_short_seq")
        train_path_data=os.path.join(data_dir,"long_train.txt")
        test_path_data=os.path.join(data_dir,"long_test.txt")
        with open(train_path_data, 'rb') as f1:
            train_set = pickle.load(f1)
        with open(test_path_data, 'rb') as f2:
            test_set = pickle.load(f2)
    elif sequence_type=="short":
        data_dir=os.path.join(root,"long_short_seq")
        train_path_data=os.path.join(data_dir,"short_train.txt")
        test_path_data=os.path.join(data_dir,"short_test.txt")
        with open(train_path_data, 'rb') as f1:
            train_set = pickle.load(f1)
        with open(test_path_data, 'rb') as f2:
            test_set = pickle.load(f2)        
    else:
        raise ValueError("unknown sequence type")

    
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

#     ### split training set into validation set and shuffle them
#     train_set_x, train_set_y = train_set
    
#     n_samples = len(train_set_x)
#     sidx = np.arange(n_samples, dtype='int32')
#     np.random.seed(101)
#     np.random.shuffle(sidx)
#     n_train = int(np.round(n_samples * (1. - valid_portion)))
#     valid_set_x = [train_set_x[s] for s in sidx[n_train:]]
#     valid_set_y = [train_set_y[s] for s in sidx[n_train:]]
#     train_set_x = [train_set_x[s] for s in sidx[:n_train]]
#     train_set_y = [train_set_y[s] for s in sidx[:n_train]]

#     ### shuffle test set
#     (test_set_x, test_set_y) = test_set
#     n_samples_test = len(test_set_x)
#     sidx_test = np.arange(n_samples_test, dtype='int32')
#     np.random.seed(102)
#     np.random.shuffle(sidx_test)
#     test_set_x = [test_set_x[s] for s in sidx_test]
#     test_set_y = [test_set_y[s] for s in sidx_test]

    def len_argsort(seq):
        return sorted(range(len(seq)), key=lambda x: len(seq[x]))

    train_set_x, train_set_y = train_set
    test_set_x, test_set_y = test_set
    
    if sort_by_len:
        sorted_index = len_argsort(test_set_x)
        test_set_x = [test_set_x[i] for i in sorted_index]
        test_set_y = [test_set_y[i] for i in sorted_index]
 
        sorted_index = len_argsort(train_set_x)
        train_set_x = [train_set_x[i] for i in sorted_index]
        train_set_y = [train_set_y[i] for i in sorted_index]

    train = (train_set_x, train_set_y)
    test = (test_set_x, test_set_y)
    
    return train, test


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
    
