import argparse
import random
from tqdm import tqdm
from pathlib import Path
import os
import pickle
import numpy as np
import time

import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, SequentialSampler

from srgnn import SRGNN
from collate import (collate_fn_factory, seq_to_session_graph)
import metric
from dataset import load_data,RecSysDataset
import warnings 
warnings.filterwarnings('ignore')

def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def prepare_batch(batch, device):
    inputs, labels = batch
    # inputs, labels = batch
    inputs_gpu  = [x.to(device) for x in inputs]
    labels_gpu  = labels.to(device)
   
    return inputs_gpu, labels_gpu 

def validate(valid_loader, model,device):
    model.eval()
    recalls = []
    mrrs = []
    losses=[]
    with torch.no_grad():
        for step, batch in tqdm(enumerate(valid_loader), total=len(valid_loader),position=0,leave=True):
            inputs, labels = prepare_batch(batch, device)
            outputs = model(*inputs)
            # loss = criterion(outputs, labels)
            loss = nn.functional.nll_loss(outputs, labels)
            logits = F.softmax(outputs, dim = 1)
            recall, mrr = metric.evaluate(logits, labels, k = args.topk)
            recalls.append(recall)
            mrrs.append(mrr)
            losses.append(loss.item())
    
    mean_recall = np.mean(recalls)
    mean_mrr = np.mean(mrrs)
    mean_loss=np.mean(losses)
    
    return mean_recall, mean_mrr, mean_loss

def fix_weight_decay(model):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(map(lambda x: x in name, ['bias', 'batch_norm', 'activation'])):
            no_decay.append(param)
        else:
            decay.append(param)
    params = [{'params': decay}, {'params': no_decay, 'weight_decay': 0}]
    return params


def main(args,device):
    train, valid, test = load_data(args.dataset_dir, valid_portion=args.valid_split)
    train_data = RecSysDataset(train)
    valid_data = RecSysDataset(valid)
    test_data = RecSysDataset(test)

    collate_fn = collate_fn_factory(seq_to_session_graph)

    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        # shuffle=True,
        # drop_last=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        sampler=SequentialSampler(train_data)
    )

    valid_loader = DataLoader(
        valid_data,
        batch_size=args.batch_size,
        # shuffle=True,
        # drop_last=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        sampler=SequentialSampler(valid_data)
    )

    test_loader = DataLoader(
        test_data,
        batch_size=args.batch_size,
        # shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )
    print()
    print('{:<30}{:<10,} '.format("training mini-batch",len(train_loader)))
    print('{:<30}{:<10,} '.format("validation mini-batch",len(valid_loader)))
    print('{:<30}{:<10,} '.format("test mini-batch",len(test_loader)))
    
    
    model = SRGNN(args.n_items, args.embedding_dim, args.num_layers, feat_drop=args.feat_drop)
    model = model.to(device)
    print()
    print("{:<30}{:<20,}".format("Number of parameters",np.sum([p.nelement() for p in model.parameters()])))
    
    if args.weight_decay > 0:
        params = fix_weight_decay(model)
    else:
        params = model.parameters()

    optimizer = optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
    # criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    best_metric = float('inf') ## if cross-entropy loss is selected

    TRAIN_LOSS=[]
    VALID_LOSS=[]
    TEST_LOSS=[]

    TRAIN_MRR=[]
    VALID_MRR=[]
    TEST_MRR=[]

    TRAIN_RECALL=[]
    VALID_RECALL=[]
    TEST_RECALL=[]
    
    for epoch in tqdm(range(args.epochs)): #before: no leave param, now , leave=False

        scheduler.step(epoch = epoch)
        # trainForEpoch(train_loader, model, optimizer, epoch, args.epochs, criterion, device, log_aggr = 1)
        model.train()
        sum_epoch_loss = 0
        start = time.time()
        for step, batch in tqdm(enumerate(train_loader), total=len(train_loader),position=0,leave=True):
            inputs, labels = prepare_batch(batch, device)

            optimizer.zero_grad()
            logits = model(*inputs)
            # loss = criterion(logits, labels)
            loss = nn.functional.nll_loss(logits, labels)
            loss.backward()
            optimizer.step()  
            loss_val = loss.item()
            sum_epoch_loss += loss_val

#             if (step+1)%(len(train_loader)//args.log_aggr) == 0:
#                 print('Epoch {:05d} |  Loss {:.4f} | Speed (samples/sec) {:.2f}'
#                       .format(epoch, sum_epoch_loss / (step + 1), labels.shape[0] / (time.time() - start)))

#             start = time.time()

        if not os.path.exists(os.path.join(os.getcwd(),"output_metrics")):
            os.makedirs("output_metrics")
        
        train_recall, train_mrr, train_loss = validate(train_loader, model, device)
        TRAIN_LOSS.append(train_loss)
        TRAIN_MRR.append(train_mrr)
        TRAIN_RECALL.append(train_recall)
        print()
        print('Epoch {} training--loss: {:.4f}, Recall@{}: {:.4f}, MRR@{}: {:.4f} \n'\
              .format(epoch, train_loss,args.topk, train_recall, args.topk, train_mrr))

        dataset_name = args.dataset_dir.split('/')[-2]
        with open(os.path.join(os.getcwd(),"output_metrics",dataset_name+"_train_metrics.txt"),'a') as f:
            f.write(f'{epoch+1},{train_loss},{train_recall},{train_mrr}\n')        
        
        valid_recall, valid_mrr, valid_loss = validate(valid_loader, model, device)
        VALID_LOSS.append(valid_loss)
        VALID_MRR.append(valid_mrr)
        VALID_RECALL.append(valid_recall)
        print('Epoch {} validation--loss: {:.4f}, Recall@{}: {:.4f}, MRR@{}: {:.4f} \n'\
              .format(epoch, valid_loss,args.topk, valid_recall, args.topk, valid_mrr))
        
        with open(os.path.join(os.getcwd(),"output_metrics",dataset_name+"_valid_metrics.txt"),'a') as f:
            f.write(f'{epoch+1},{valid_loss},{valid_recall},{valid_mrr}\n')
        
        test_recall, test_mrr, test_loss = validate(test_loader, model, device)
        TEST_LOSS.append(test_loss)
        TEST_MRR.append(test_mrr)
        TEST_RECALL.append(test_recall)
        print('Epoch {} test--loss: {:.4f}, Recall@{}: {:.4f}, MRR@{}: {:.4f} \n'\
              .format(epoch, test_loss,args.topk, test_recall, args.topk, test_mrr))

        with open(os.path.join(os.getcwd(),"output_metrics",dataset_name+"_test_metrics.txt"),'a') as f:
            f.write(f'{epoch+1},{test_loss},{test_recall},{test_mrr}\n')

        # store best loss and save a model checkpoint
        ckpt_dict = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }

        selected_metric=valid_loss
        if selected_metric<best_metric:
            best_metric=selected_metric
            dataset_name = args.dataset_dir.split('/')[-2] 
            torch.save(ckpt_dict, dataset_name + '_' + 'latest_checkpoint.pth')
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--dataset-dir', default='../YOOCHOOSE_data/yoochoose1_64/', help='the dataset directory'
    )
    parser.add_argument("--seed",  type=int,default=101,
            help="random seed for np.random.seed, torch.manual_seed and torch.cuda.manual_seed.")
    parser.add_argument('--n_items', type=int, default=37484, help='number of unique items. 37484 for yoochoose')
    parser.add_argument('--embedding-dim', type=int, default=256, help='the embedding size')
    parser.add_argument('--num-layers', type=int, default=1, help='the number of layers')
    parser.add_argument('--feat-drop', type=float, default=0.1, help='the dropout ratio for features')
    parser.add_argument('--lr', type=float, default=1e-3, help='the learning rate')
    parser.add_argument(
        '--batch-size', type=int, default=512, help='the batch size for training'
    )
    parser.add_argument(
        '--epochs', type=int, default=30, help='the number of training epochs'
    )
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=1e-4,
        help='the parameter for L2 regularization',
    )

    parser.add_argument(
        '--valid-split',
        type=float,
        default=0.1,
        help='the fraction for the validation set',
    )

    parser.add_argument(
        '--topk', 
        type=int, 
        default=20, 
        help='number of top score items selected for calculating recall and mrr metrics',
    )

    parser.add_argument(
        '--log_aggr', 
        type=int, 
        default=1, 
        help='print the loss after this number of iterations',
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        default=0,
        help='the number of processes to load the input graphs',
    )


    args= parser.parse_args()
    print(args)
    
    seed_everything(args.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    main(args,device)
    