#!/usr/bin/env python37
# -*- coding: utf-8 -*-
"""
Created on 17 Sep, 2019

Reference: https://github.com/CRIPAC-DIG/SR-GNN/blob/master/datasets/preprocess.py

预处理基本流程：
1. 创建两个字典sess_clicks和sess_date来分别保存session的相关信息。两个字典都以sessionId为键，其中session_click以一个Session中用户先后点击的物品id
构成的List为值；session_date以一个Session中最后一次点击的时间作为值，后续用于训练集和测试集的划分；
2. 过滤长度为1的Session和出现次数小于5次的物品；
3. 依据日期划分训练集和测试集。其中Yoochoose数据集以最后一天时长内的Session作为测试集，Diginetica数据集以最后一周时长内的Session作为测试集；
4. 分解每个Session生成最终的数据格式。每个Session中以不包括最后一个物品的其他物品作为特征，以最后一个物品作为标签。同时把物品的id重新编码成从1开始递增的自然数序列
"""

import argparse
import time
import csv
import pickle
import operator
import datetime
import os
from tqdm import tqdm
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='amex_explorepoi-poi_category')
args = parser.parse_args()

# add a header for yoochoose dataset
# with open('yoochoose-clicks.dat', 'r') as f, open('yoochoose-clicks-withHeader.dat', 'w') as fn:
#     fn.write('sessionId,timestamp,itemId,category'+'\n')
#     for line in f:
#         fn.write(line)


# #amex full datasets
# with open('amex_output.dat', 'r') as f, open('amex_output-withHeader.dat', 'w') as fn:
#     fn.write('sessionId,timestamp,itemId,category'+'\n')
#     for line in f:
#         fn.write(line)        
# with open('amex_category_output.dat', 'r') as f, open('amex_category_output-withHeader.dat', 'w') as fn:
#     fn.write('sessionId,timestamp,itemId,category'+'\n')
#     for line in f:
#         fn.write(line)
# with open('amex_context_output.dat', 'r') as f, open('amex_context_output-withHeader.dat', 'w') as fn:
#     fn.write('sessionId,timestamp,itemId,category'+'\n')
#     for line in f:
#         fn.write(line)

dataset_path = '/home/ec2-user/SageMaker/sequence-based-recommendation/s3_file_processing/dataset_files/amex-yoochoose-format/'
#amex log datasets
with open(os.path.join(dataset_path,'amex_logs_poi_clicks.dat'), 'r') as f, open('amex_log_poi_clicks-withHeader.dat', 'w') as fn:
    fn.write('sessionId,timestamp,itemId,category'+'\n')
    for line in f:
        fn.write(line)        
with open(os.path.join(dataset_path,'amex_logs_category_clicks.dat'), 'r') as f, open('amex_log_category_clicks-withHeader.dat', 'w') as fn:
    fn.write('sessionId,timestamp,itemId,category'+'\n')
    for line in f:
        fn.write(line)
with open(os.path.join(dataset_path,'amex_logs_context_clicks.dat'), 'r') as f, open('amex_log_context_clicks-withHeader.dat', 'w') as fn:
    fn.write('sessionId,timestamp,itemId,category'+'\n')
    for line in f:
        fn.write(line)
        
#amex explore_POI datasets       
with open(os.path.join(dataset_path,'AMEX_explorepoi_poi_category_clicks.dat'), 'r') as f, open('AMEX_explorepoi_poi_category_clicks-withHeader.dat', 'w') as fn:
    fn.write('sessionId,timestamp,itemId,category'+'\n')
    for line in f:
        fn.write(line)
with open(os.path.join(dataset_path,'AMEX_explorepoi_poi_context_clicks.dat'), 'r') as f, open('AMEX_explorepoi_poi_context_clicks-withHeader.dat', 'w') as fn:
    fn.write('sessionId,timestamp,itemId,category'+'\n')
    for line in f:
        fn.write(line)
with open(os.path.join(dataset_path,'AMEX_explorepoi_category_clicks.dat'), 'r') as f, open('AMEX_explorepoi_category_clicks-withHeader.dat', 'w') as fn:
    fn.write('sessionId,timestamp,itemId,category'+'\n')
    for line in f:
        fn.write(line)
with open(os.path.join(dataset_path,'AMEX_explorepoi_context_clicks.dat'), 'r') as f, open('AMEX_explorepoi_context_clicks-withHeader.dat', 'w') as fn:
    fn.write('sessionId,timestamp,itemId,category'+'\n')
    for line in f:
        fn.write(line)

if args.dataset == 'diginetica':
    dataset = 'train-item-views.csv'
elif args.dataset =='yoochoose':
    dataset = 'yoochoose-clicks-withHeader.dat'
# elif args.dataset =='amex_explorepoi':
#     dataset = 'AMEX_yoochoose_clicks-withHeader.dat'
# elif args.dataset =='amex_explorepoi_category':
#     dataset = 'AMEX_explorepoi_category_clicks-withHeader.dat'
# elif args.dataset =='amex_explorepoi_context':
#     dataset = 'AMEX_explorepoi_context_clicks-withHeader.dat'
# elif args.dataset =='amex_poi':
#     print('in amex poi')
#     dataset = 'amex_output-withHeader.dat'
# elif args.dataset =='amex_category':
#     dataset = 'amex_category_output-withHeader.dat'
# elif args.dataset =='amex_context':
#     dataset = 'amex_context_output-withHeader.dat'
# else:
#     print('invalid dataset name')
elif args.dataset =='amex_explorepoi-poi_category':
    dataset = 'AMEX_explorepoi_poi_category_clicks-withHeader.dat'
elif args.dataset =='amex_explorepoi-poi_context':
    dataset = 'AMEX_explorepoi_poi_context_clicks-withHeader.dat'
elif args.dataset =='amex_explorepoi-category':
    dataset = 'AMEX_explorepoi_category_clicks-withHeader.dat'
elif args.dataset =='amex_explorepoi-context':
    dataset = 'AMEX_explorepoi_context_clicks-withHeader.dat'
# elif args.dataset =='amex_poi':
#     dataset = 'amex_output-withHeader.dat'
# elif args.dataset =='amex_category':
#     dataset = 'amex_category_output-withHeader.dat'
# elif args.dataset =='amex_context':
#     dataset = 'amex_context_output-withHeader.dat'
    
elif args.dataset =='amex_log-poi':
    dataset = 'amex_log_poi_clicks-withHeader.dat'
elif args.dataset =='amex_log-category':
    dataset = 'amex_log_category_clicks-withHeader.dat'
elif args.dataset =='amex_log-context':
    dataset = 'amex_log_context_clicks-withHeader.dat'
else:
    print('invalid dataset name')

preprocess_output_log = dict({'dataset name':  args.dataset})

amex_sets = set(['amex_explorepoi-poi_category', 'amex_explorepoi-poi_context', 'amex_explorepoi-category', 'amex_explorepoi-context', 'amex_log-poi', 'amex_log-category', 'amex_log-context'])

print("-- Starting @ %ss" % datetime.datetime.now())
with open(dataset, "r") as f:
    if args.dataset == 'yoochoose' or args.dataset in amex_sets:
        reader = csv.DictReader(f, delimiter=',')
    else:
        reader = csv.DictReader(f, delimiter=';')
    sess_clicks = {}
    sess_date = {}
    ctr = 0
    curid = -1
    curdate = None
    for data in tqdm(reader):
        
        sessid = data['sessionId']
        if curdate and not curid == sessid:
            date = ''
            if args.dataset == 'yoochoose' or args.dataset in amex_sets:
                date = time.mktime(time.strptime(curdate[:19], '%Y-%m-%dT%H:%M:%S'))
#             elif args.dataset in amex_sets: 
#                 date = curdate
            else:
                date = time.mktime(time.strptime(curdate, '%Y-%m-%d'))
            sess_date[curid] = date
        curid = sessid
        if args.dataset == 'yoochoose' or args.dataset in amex_sets:
            item = data['itemId']
        else:
            item = data['itemId'], int(data['timeframe'])
        curdate = ''
        if args.dataset == 'yoochoose' or args.dataset in amex_sets:
            curdate = data['timestamp']
        else:
            curdate = data['eventdate']

        if sessid in sess_clicks:
            sess_clicks[sessid] += [item]
        else:
            sess_clicks[sessid] = [item]
        ctr += 1
    date = ''
    if args.dataset == 'yoochoose' or args.dataset in amex_sets:
        date = time.mktime(time.strptime(curdate[:19], '%Y-%m-%dT%H:%M:%S'))
    else:
        date = time.mktime(time.strptime(curdate, '%Y-%m-%d'))
        for i in list(sess_clicks):
            sorted_clicks = sorted(sess_clicks[i], key=operator.itemgetter(1))
            sess_clicks[i] = [c[0] for c in sorted_clicks]
    sess_date[curid] = date
print("-- Reading data @ %ss" % datetime.datetime.now())

preprocess_output_log['start time'] = datetime.datetime.now()

len1_sessions_filtered_count = 0 
# Filter out length 1 sessions
for s in list(sess_clicks):
    if len(sess_clicks[s]) == 1:
        len1_sessions_filtered_count += 1
        del sess_clicks[s]
        del sess_date[s]
        
preprocess_output_log['length 1 sessions filtered'] = len1_sessions_filtered_count

# Count number of times each item appears
iid_counts = {}
for s in sess_clicks:
    seq = sess_clicks[s]
    for iid in seq:
        if iid in iid_counts:
            iid_counts[iid] += 1
        else:
            iid_counts[iid] = 1

sorted_counts = sorted(iid_counts.items(), key=operator.itemgetter(1))

# preprocess_output_log['sorted item counts'] = sorted_counts

length = len(sess_clicks)
for s in list(sess_clicks):
    curseq = sess_clicks[s]
    filseq = list(filter(lambda i: iid_counts[i] >= 5, curseq))
    if len(filseq) < 2:
        del sess_clicks[s]
        del sess_date[s]
    else:
        sess_clicks[s] = filseq

# Split out test set based on dates
dates = list(sess_date.items())
maxdate = dates[0][1]

for _, date in dates:
    if maxdate < date:
        maxdate = date
preprocess_output_log['max date'] = maxdate

amex_splitdate = 6
preprocess_output_log['amex split days before maxdate'] = amex_splitdate

# 7 days for test
splitdate = 0
if args.dataset == 'yoochoose':
    splitdate = maxdate - 86400 * 1  #the number of seconds for a day：86400
elif args.dataset in amex_sets: 
    splitdate = maxdate - 86400 * amex_splitdate #chosen to be 6 days?
else:
    splitdate = maxdate - 86400 * 7 

print('Splitting date', splitdate)      # Yoochoose: ('Split date', 1411930799.0)
tra_sess = filter(lambda x: x[1] < splitdate, dates)
tes_sess = filter(lambda x: x[1] > splitdate, dates)

# Sort sessions by date
tra_sess = sorted(tra_sess, key=operator.itemgetter(1))     # [(sessionId, timestamp), (), ]
tes_sess = sorted(tes_sess, key=operator.itemgetter(1))     # [(sessionId, timestamp), (), ]
print(len(tra_sess))    # 186670    # 7966257    explore_poi 771 explorepoi_con 1170
print(len(tes_sess))    # 15979     # 15324      explore_poi 84  explorepoi_con 148
print(tra_sess[:3])
print(tes_sess[:3])
print("-- Splitting train set and test set @ %ss" % datetime.datetime.now())

preprocess_output_log['len training sessions (tra_sess)'] = len(tra_sess)
preprocess_output_log['len test sessions (tes_sess)'] = len(tes_sess)

# Choosing item count >=5 gives approximately the same number of items as reported in paper
item_dict = {}
# Convert training sessions to sequences and renumber items to start from 1
def obtian_tra():
    train_ids = []
    train_seqs = []
    train_dates = []
    item_ctr = 1
    for s, date in tra_sess:
        seq = sess_clicks[s]
        outseq = []
        for i in seq:
            if i in item_dict:
                outseq += [item_dict[i]]
            else:
                outseq += [item_ctr]
                item_dict[i] = item_ctr
                item_ctr += 1
        if len(outseq) < 2:  # Doesn't occur
            continue
        train_ids += [s]
        train_dates += [date]
        train_seqs += [outseq]
    print('item count: ')
    preprocess_output_log['item count:'] = item_ctr
    print(item_ctr)     # 43098, 37484, 
                        #explore_poi: 551 explorepoi_category: 4, explorepoi_context: 183
                        #amex: 1164  amexcategory: 4  amexcontext: 482
    return train_ids, train_dates, train_seqs


# Convert test sessions to sequences, ignoring items that do not appear in training set
def obtian_tes():
    test_ids = []
    test_seqs = []
    test_dates = []
    for s, date in tes_sess:
        seq = sess_clicks[s]
        outseq = []
        for i in seq:
            if i in item_dict:
                outseq += [item_dict[i]]
        if len(outseq) < 2:
            continue
        test_ids += [s]
        test_dates += [date]
        test_seqs += [outseq]
    return test_ids, test_dates, test_seqs


tra_ids, tra_dates, tra_seqs = obtian_tra()
tes_ids, tes_dates, tes_seqs = obtian_tes()


def process_seqs(iseqs, idates):
    out_seqs = []
    out_dates = []
    labs = []
    ids = []
    for id, seq, date in zip(range(len(iseqs)), iseqs, idates):
        for i in range(1, len(seq)):
            tar = seq[-i]
            labs += [tar]
            out_seqs += [seq[:-i]]
            out_dates += [date]
            ids += [id]
    return out_seqs, out_dates, labs, ids


tr_seqs, tr_dates, tr_labs, tr_ids = process_seqs(tra_seqs, tra_dates)
te_seqs, te_dates, te_labs, te_ids = process_seqs(tes_seqs, tes_dates)
tra = (tr_seqs, tr_labs)
tes = (te_seqs, te_labs)
print(len(tr_seqs))  # explore_poi: 3361 explore_cont: 9998
print(len(te_seqs))  # explore_poi: 298  explore_cont: 1432
print(tr_seqs[:3], tr_dates[:3], tr_labs[:3])
print(te_seqs[:3], te_dates[:3], te_labs[:3])
all = 0

preprocess_output_log['len training seqs (tr_seqs)'] = len(tr_seqs)
preprocess_output_log['len test seqs (te_seqs)'] = len(tr_seqs)

for seq in tra_seqs:
    all += len(seq)
for seq in tes_seqs:
    all += len(seq)
print('avg length: ', all/(len(tra_seqs) + len(tes_seqs) * 1.0))

preprocess_output_log['avg length'] = all/(len(tra_seqs) + len(tes_seqs) * 1.0)

# preprocess_output_log = dict('dataset name':  args.dataset)
# preprocess_output_log['start time'] = datetime.datetime.now()
# preprocess_output_log['length 1 sessions filtered'] = len1_sessions_filtered_count
# preprocess_output_log['sorted item counts'] = sorted_counts
# preprocess_output_log['amex split days before maxdate'] = amex_splitdate

# preprocess_output_log = {'dataset name':, 
#                          'start time':, 
#                          'length 1 sessions filtered':,
#                          'sorted item counts':, 
#                          'amex split days before maxdate':, 
#                          'len training sessions (tra_sess)':,
#                          'len test sessions (tes_sess)':,
#                          'item count':, 
#                          'len training seqs (tr_seqs)':
#                          'len test seqs (te_seqs)':, 
#                          'avg length':}
print(preprocess_output_log)
preprocess_output_log = {k:[v] for k,v in preprocess_output_log.items()}
output_df = pd.DataFrame.from_dict(preprocess_output_log) 

output_df.to_csv('preprocess_data_info.csv', mode ='a', index = 'dataset name', header=True)

if args.dataset == 'diginetica':
    if not os.path.exists('diginetica'):
        os.makedirs('diginetica')
    pickle.dump(tra, open('diginetica/train.txt', 'wb'))
    pickle.dump(tes, open('diginetica/test.txt', 'wb'))
    pickle.dump(tra_seqs, open('diginetica/all_train_seq.txt', 'wb'))
elif args.dataset == 'yoochoose':
    if not os.path.exists('yoochoose1_4'):
        os.makedirs('yoochoose1_4')
    if not os.path.exists('yoochoose1_64'):
        os.makedirs('yoochoose1_64')
    pickle.dump(tes, open('yoochoose1_4/test.txt', 'wb'))
    pickle.dump(tes, open('yoochoose1_64/test.txt', 'wb'))

    split4, split64 = int(len(tr_seqs) / 4), int(len(tr_seqs) / 64)
    print(len(tr_seqs[-split4:]))
    print(len(tr_seqs[-split64:]))

    tra4, tra64 = (tr_seqs[-split4:], tr_labs[-split4:]), (tr_seqs[-split64:], tr_labs[-split64:])
    seq4, seq64 = tra_seqs[tr_ids[-split4]:], tra_seqs[tr_ids[-split64]:]

    pickle.dump(tra4, open('yoochoose1_4/train.txt', 'wb'))
    pickle.dump(seq4, open('yoochoose1_4/all_train_seq.txt', 'wb'))

    pickle.dump(tra64, open('yoochoose1_64/train.txt', 'wb'))
    pickle.dump(seq64, open('yoochoose1_64/all_train_seq.txt', 'wb'))
    
# elif args.dataset == 'amex_explorepoi':
#     if not os.path.exists('amex_explorepoi'):
#         os.makedirs('amex_explorepoi')
#     pickle.dump(tra, open('amex_explorepoi/train.txt', 'wb'))
#     pickle.dump(tes, open('amex_explorepoi/test.txt', 'wb'))
#     pickle.dump(tra_seqs, open('amex_explorepoi/all_train_seq.txt', 'wb'))
# elif args.dataset == 'amex_explorepoi_category':
#     if not os.path.exists('amex_explorepoi_category'):
#         os.makedirs('amex_explorepoi_category')
#     pickle.dump(tra, open('amex_explorepoi_category/train.txt', 'wb'))
#     pickle.dump(tes, open('amex_explorepoi_category/test.txt', 'wb'))
#     pickle.dump(tra_seqs, open('amex_explorepoi_category/all_train_seq.txt', 'wb'))
# elif args.dataset == 'amex_explorepoi_context':
#     if not os.path.exists('amex_explorepoi_context'):
#         os.makedirs('amex_explorepoi_context')
#     pickle.dump(tra, open('amex_explorepoi_context/train.txt', 'wb'))
#     pickle.dump(tes, open('amex_explorepoi_context/test.txt', 'wb'))
#     pickle.dump(tra_seqs, open('amex_explorepoi_context/all_train_seq.txt', 'wb'))

# elif args.dataset == 'amex_poi':
#     if not os.path.exists('amex_poi'):
#         os.makedirs('amex_poi')
#     pickle.dump(tra, open('amex_poi/train.txt', 'wb'))
#     pickle.dump(tes, open('amex_poi/test.txt', 'wb'))
#     pickle.dump(tra_seqs, open('amex_poi/all_train_seq.txt', 'wb'))
# elif args.dataset == 'amex_category':
#     if not os.path.exists('amex_category'):
#         os.makedirs('amex_category')
#     pickle.dump(tra, open('amex_category/train.txt', 'wb'))
#     pickle.dump(tes, open('amex_category/test.txt', 'wb'))
#     pickle.dump(tra_seqs, open('amex_category/all_train_seq.txt', 'wb'))
# elif args.dataset == 'amex_context':
#     if not os.path.exists('amex_context'):
#         os.makedirs('amex_context')
#     pickle.dump(tra, open('amex_context/train.txt', 'wb'))
#     pickle.dump(tes, open('amex_context/test.txt', 'wb'))
#     pickle.dump(tra_seqs, open('amex_context/all_train_seq.txt', 'wb'))
    
    
    
# elif args.dataset == 'amex_explorepoi-poi':
#     if not os.path.exists('amex_explorepoi-poi'):
#         os.makedirs('amex_explorepoi-poi')
#     pickle.dump(tra, open('amex_explorepoi-poi/train.txt', 'wb'))
#     pickle.dump(tes, open('amex_explorepoi-poi/test.txt', 'wb'))
#     pickle.dump(tra_seqs, open('amex_explorepoi-poi/all_train_seq.txt', 'wb'))
elif args.dataset in amex_sets:
    if not os.path.exists(args.dataset):
        os.makedirs(args.dataset)
    pickle.dump(tra, open(args.dataset + '/train.txt', 'wb'))
    pickle.dump(tes, open(args.dataset + '/test.txt', 'wb'))
    pickle.dump(tra_seqs, open(args.dataset + '/all_train_seq.txt', 'wb'))
    pickle.dump(tes_seqs, open(args.dataset + '/all_test_seq.txt', 'wb'))
    
elif args.dataset == 'amex_explorepoi-poi_category':
    if not os.path.exists('amex_explorepoi-poi_category'):
        os.makedirs('amex_explorepoi-poi_category')
    pickle.dump(tra, open('amex_explorepoi-poi_category/train.txt', 'wb'))
    pickle.dump(tes, open('amex_explorepoi-poi_category/test.txt', 'wb'))
    pickle.dump(tra_seqs, open('amex_explorepoi-poi_category/all_train_seq.txt', 'wb'))
    pickle.dump(tes_seqs, open('amex_explorepoi-poi_category/all_test_seq.txt', 'wb'))
    
elif args.dataset == 'amex_explorepoi-poi_context':
    if not os.path.exists('amex_explorepoi-poi_context'):
        os.makedirs('amex_explorepoi-poi_context')
    pickle.dump(tra, open('amex_explorepoi-poi_context/train.txt', 'wb'))
    pickle.dump(tes, open('amex_explorepoi-poi_context/test.txt', 'wb'))
    pickle.dump(tra_seqs, open('amex_explorepoi-poi_context/all_train_seq.txt', 'wb'))
    pickle.dump(tes_seqs, open('amex_explorepoi-poi_context/all_test_seq.txt', 'wb'))
    
elif args.dataset == 'amex_explorepoi-poi':
    if not os.path.exists('amex_explorepoi-poi'):
        os.makedirs('amex_explorepoi-poi')
    pickle.dump(tra, open('amex_explorepoi-poi/train.txt', 'wb'))
    pickle.dump(tes, open('amex_explorepoi-poi/test.txt', 'wb'))
    pickle.dump(tra_seqs, open('amex_explorepoi-poi/all_train_seq.txt', 'wb'))
    pickle.dump(tes_seqs, open('amex_explorepoi-poi/all_test_seq.txt', 'wb'))
    
elif args.dataset == 'amex_explorepoi-category':
    if not os.path.exists('amex_explorepoi-category'):
        os.makedirs('amex_explorepoi-category')
    pickle.dump(tra, open('amex_explorepoi-category/train.txt', 'wb'))
    pickle.dump(tes, open('amex_explorepoi-category/test.txt', 'wb'))
    pickle.dump(tra_seqs, open('amex_explorepoi-category/all_train_seq.txt', 'wb'))
    pickle.dump(tes_seqs, open('amex_explorepoi-category/all_test_seq.txt', 'wb'))
    
elif args.dataset == 'amex_explorepoi-context':
    if not os.path.exists('amex_explorepoi-context'):
        os.makedirs('amex_explorepoi-context')
    pickle.dump(tra, open('amex_explorepoi-context/train.txt', 'wb'))
    pickle.dump(tes, open('amex_explorepoi-context/test.txt', 'wb'))
    pickle.dump(tra_seqs, open('amex_explorepoi-context/all_train_seq.txt', 'wb'))
    pickle.dump(tes_seqs, open('amex_explorepoi-context/all_test_seq.txt', 'wb'))
    
elif args.dataset == 'amex_log-poi':
    if not os.path.exists('amex_log-poi'):
        os.makedirs('amex_log-poi')
    pickle.dump(tra, open('amex_log-poi/train.txt', 'wb'))
    pickle.dump(tes, open('amex_log-poi/test.txt', 'wb'))
    pickle.dump(tra_seqs, open('amex_log-poi/all_train_seq.txt', 'wb'))
    pickle.dump(tes_seqs, open('amex_log-poi/all_test_seq.txt', 'wb'))
    
elif args.dataset == 'amex_log-category':
    if not os.path.exists('amex_log-category'):
        os.makedirs('amex_log-category')
    pickle.dump(tra, open('amex_log-category/train.txt', 'wb'))
    pickle.dump(tes, open('amex_log-category/test.txt', 'wb'))
    pickle.dump(tra_seqs, open('amex_log-category/all_train_seq.txt', 'wb'))
    pickle.dump(test_seqs, open('amex_log-category/all_test_seq.txt', 'wb'))
    
elif args.dataset == 'amex_log-context':
    if not os.path.exists('amex_log-context'):
        os.makedirs('amex_log-context')
    pickle.dump(tra, open('amex_log-context/train.txt', 'wb'))
    pickle.dump(tes, open('amex_log-context/test.txt', 'wb'))
    pickle.dump(tra_seqs, open('amex_log-context/all_train_seq.txt', 'wb'))
    pickle.dump(tes_seqs, open('amex_log-context/all_test_seq.txt', 'wb'))
    
else:
    print('dataset not created, invalid entered')
    pass

# else: 
#     if not os.path.exists(args.dataset):
#         os.makedirs(args.dataset)
#     pickle.dump(tra, open(args.dataset + '/train.txt', 'wb'))
#     pickle.dump(tes, open(args.dataset + '/test.txt', 'wb'))
#     pickle.dump(tra_seqs, open(args.dataset + '/all_train_seq.txt', 'wb'))
        
print('Done.')
