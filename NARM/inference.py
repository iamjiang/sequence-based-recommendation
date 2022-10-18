#!/usr/bin/env python37
# -*- coding: utf-8 -*-
"""
Created on 19 Sep, 2019

@author: wangshuo
"""

import os
import time
import random
import argparse
import pickle
import numpy as np
from tqdm import tqdm
from os.path import join

import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable
from torch.backends import cudnn

import metric
from utils import collate_fn
from narm import NARM
from dataset import load_data, RecSysDataset

from sklearn import preprocessing
import joblib

# parser = argparse.ArgumentParser()
# parser.add_argument('--dataset_path', default='datasets/diginetica/', help='dataset directory path: datasets/diginetica/yoochoose1_4/yoochoose1_64/amex_poi/amex_category/amex_context/amex_explorepoi/amex_explorepoi_category/amex_explorepoi_context')
# parser.add_argument('--batch_size', type=int, default=512, help='input batch size')
# parser.add_argument('--hidden_size', type=int, default=100, help='hidden state size of gru module')
# parser.add_argument('--embed_dim', type=int, default=50, help='the dimension of item embedding')
# parser.add_argument('--epoch', type=int, default=100, help='the number of epochs to train for')
# parser.add_argument('--lr', type=float, default=0.001, help='learning rate')  
# parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay rate')
# parser.add_argument('--lr_dc_step', type=int, default=80, help='the number of steps after which the learning rate decay') 
# parser.add_argument('--test', action='store_true', help='test')
# parser.add_argument('--topk', type=int, default=20, help='number of top score items selected for calculating recall and mrr metrics')
# parser.add_argument('--valid_portion', type=float, default=0.1, help='split the portion of training set as validation set')
# args = parser.parse_args()
# print(args)

# here = os.path.dirname(os.path.abspath(__file__))
device = torch.device('cuda' if torch.cuda.is_available() else 'cuda:0')

# Specify a path to model 
# PATH = 'latest_checkpoint.pth.tar'
PATH = 'amex_explorepoi-poi_context_latest_checkpoint.pth.tar'

checkpoint = torch.load(PATH)
#nitems, hidden size, embeded dim, batch size
#n items depends on the last model that was ran and saved as checkpoint
# model = NARM(183, 100, 50, 512) #context
model = NARM(551, 100, 50, 512)
# model = NARM(3, 100, 50, 512)

encode_dict = joblib.load('/home/ubuntu/michelle_ML/s3_file_processing/' + 'encoder_dicts/all-label_encoder_dict.joblib')

model.load_state_dict(checkpoint['state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer'])
model.cuda()
model.eval()
from torch.utils.data import DataLoader
# input_dat = [[['0cb64af1-3442-35e3-9b2a-9baff30f8598', '7be9f4ee-2475-3a1c-b5bc-909e473a5d81', 'c85852ca-2d67-3d62-ae2c-b36f8f6b4625', '014be8c4-c354-339b-8f30-c4b49fe9a795', 'cbaee633-cb19-3b94-b33f-c03ed4cf786a', 'abcd584f-46ea-3e3f-8073-d36efd0436d9']]]
# input_seq = input_dat[0][0]
# lens = len(input_dat[0][0])
# print(lens)
# print(encode_dict.keys())
# transformed_input_seq = encode_dict['poi_id'].transform(input_seq)
# print(transformed_input_seq)
# input_data = RecSysDataset([[transformed_input_seq]])

input_dat = [[[34, 78, 75, 12, 64, 78]]]
print(encode_dict['poi_id'].inverse_transform(input_dat[0][0]))
# input_dat = [[[0, 2, 1, 2, 2, 1]]]
# input_data = 

input_data = RecSysDataset(input_dat)
input_data_loader = DataLoader(input_data, batch_size = 512, shuffle = True, collate_fn = collate_fn)

lens = 6

with torch.no_grad():
    for i, (seq, target, lens) in tqdm(enumerate(input_data_loader), total=len(input_data_loader)):

        seq = seq.to(device='cuda:0')
        outputs = model(seq, lens)
        recommendation = torch.topk(outputs, 20)
#         print(torch.topk(outputs, 20))
        topk_values, topk_indices = torch.topk(outputs, 20)
        print(topk_indices)
        decoded_output = encode_dict['poi_id'].inverse_transform(topk_indices.cpu().detach().numpy()[0])
        print(decoded_output)

#         topk_values, topk_indices = torch.max(dim=0)[0].flatten().topk(2)
#         print(topk_indices)
#         encoded_pred = torch.nn.functional.softmax(outputs, dim = 1)
#         print(encoded_pred)
#dict_keys(['epoch', 'state_dict', 'optimizer'])
# --------------------------------------------------
# Dataset info:
# Number of sessions: 1
# --------------------------------------------------
#   0%|                                                                                                                                                          | 0/1 [00:00<?, ?it/s]torch.return_types.topk(
# values=tensor([[8.0192, 5.9741, 5.3666, 4.4292, 4.4034, 4.3018, 4.1337, 3.9993, 3.8117,
#          3.6361, 3.4624, 3.4454, 3.4317, 3.2312, 3.0832, 2.9559, 2.8750, 2.7711,
#          2.7560, 2.6624]], device='cuda:0'),
# indices=tensor([[ 78,  25,   6,  99,   4,  22, 141,  64,  43,  62, 129,  14,  34,  71,
#           32,  45, 100,  37,  83, 143]], device='cuda:0'))
# 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 84.34it/s]
# tensor([[ 0.0000e+00, -1.2210e+00, -4.7654e+00,  3.4353e-01,  4.4034e+00,
#           7.6476e-01,  5.3666e+00, -3.1059e+00, -2.0234e+00, -4.2301e-01,
#          -1.5688e+00,  1.5253e+00,  2.6206e+00,  9.7654e-02,  3.4454e+00,
#          -3.2834e-01, -3.3070e+00, -3.0645e+00, -2.3746e+00, -2.3383e+00,
#          -5.4965e+00, -3.2434e+00,  4.3018e+00,  3.6333e-01,  8.2705e-01,
#           5.9741e+00,  1.0314e+00, -3.6273e-02,  1.6163e+00, -9.0679e-01,
#           1.0745e+00,  3.2295e-03,  3.0832e+00,  8.0965e-01,  3.4317e+00,
#           2.1644e+00, -6.4601e-01,  2.7711e+00,  1.9226e-01,  2.4857e+00,
#          -2.7952e-01, -4.8118e+00,  3.9333e-01,  3.8117e+00,  2.6507e+00,
#           2.9559e+00, -1.3105e+00, -2.5777e+00,  1.2875e+00,  3.9700e-01,
#           1.9235e+00, -4.8054e+00, -2.9304e-01, -2.9126e-01,  5.5086e-01,
#          -1.0030e+00,  4.5740e-01, -5.0840e+00,  1.7514e+00,  8.7503e-01,
#           7.3451e-01, -7.0750e-01,  3.6361e+00,  1.8214e+00,  3.9993e+00,
#          -1.0608e+00,  1.6231e+00, -2.1182e+00,  1.2551e+00, -2.6357e+00,
#           2.0462e+00,  3.2312e+00, -5.5118e-01, -5.3535e+00, -4.5119e+00,
#           2.5781e+00, -1.8877e-01, -5.6302e-01,  8.0192e+00, -1.8902e+00,
#           1.0654e+00, -1.1799e+00, -7.7777e-01,  2.7560e+00, -6.4460e-01,
#           1.3529e+00, -2.5548e+00, -2.4191e-01, -7.9726e-01,  1.1603e+00,
#          -4.1061e+00, -2.0593e+00, -3.4569e+00, -1.1229e+00,  2.4675e+00,
#          -4.3466e-03, -4.3721e+00,  4.3860e-01,  2.3720e+00,  4.4292e+00,
#           2.8750e+00,  2.5978e+00,  1.3235e+00, -5.4682e+00,  1.4433e+00,
#           5.2503e-01,  8.0926e-01,  1.9298e+00,  2.3473e+00, -1.5651e+00,
#          -1.0515e+00, -1.2561e+00, -3.4626e+00, -6.9835e-01,  2.4861e+00,
#           4.1609e-01,  1.3681e-01,  4.6634e-01, -1.1632e+00,  9.4415e-01,
#          -1.5854e+00, -1.0505e+00,  5.6705e-01,  5.5218e-01, -7.7380e-01,
#          -1.1222e+00, -4.2198e+00, -1.1624e-01, -9.5314e-02,  3.4624e+00,
#          -3.6830e+00,  1.4822e+00, -2.3828e+00, -4.0699e+00, -5.5869e-01,
#          -1.9590e+00, -3.4224e+00, -3.3203e+00, -5.9722e-01,  1.5760e+00,
#           1.2984e+00,  4.1337e+00,  7.6513e-02,  2.6624e+00, -2.9699e-01,
#          -2.6274e+00, -1.7512e+00, -3.3432e+00,  8.8400e-01, -1.0646e-01,
#          -8.6946e-01,  7.7828e-01,  4.9062e-01, -2.2379e+00, -2.8933e-01,
#          -3.2034e+00, -1.1893e-01,  2.2470e+00, -5.1053e+00,  1.2764e+00,
#          -2.2163e+00, -2.7080e+00, -1.7425e+00, -6.5616e-01,  1.1786e+00,
#           1.9933e+00,  1.2546e+00, -7.6519e-01, -2.0751e+00, -1.9089e+00,
#          -2.1149e+00, -2.8616e+00, -1.7964e-01,  1.2976e+00,  2.3472e-02,
#          -1.1492e-01, -1.2391e+00,  2.1487e+00, -1.6137e+00, -2.0606e-01,
#           8.8630e-01,  9.3309e-01, -1.0652e+00]], device='cuda:0')
        
# print(outputs)
# print('rec')
# print(recommendation)
# print(outputs.cpu().data.numpy())
# decoded_output = encode_dict['poi_id'].inverse_transform(outputs.cpu().data.numpy())
# print(decoded_output)