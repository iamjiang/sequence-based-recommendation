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

from msgifsr import MSGIFSR
from collate import (collate_fn_factory_ccs, seq_to_ccs_graph)
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
            criterion = nn.CrossEntropyLoss()
            loss = criterion(outputs, labels)
            # loss = nn.functional.nll_loss(outputs, labels)
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

def split_validation(train_data, valid_portion):
    train_x, train_y = train_data
    n_samples = len(train_x)
    sidx = np.arange(n_samples, dtype='int32')
    np.random.seed(101)
    np.random.shuffle(sidx)
    n_train = int(np.round(n_samples * (1. - valid_portion)))
    valid_x = [train_x[s] for s in sidx[n_train:]]
    valid_y = [train_y[s] for s in sidx[n_train:]]
    train_x = [train_x[s] for s in sidx[:n_train]]
    train_y = [train_y[s] for s in sidx[:n_train]]  
    
    train_set=(train_x, train_y)
    valid_set=(valid_x, valid_y)
    
    return train_set, valid_set

def main(args,device):

    train, test = load_data(args.dataset_dir)
    if args.validation:
        train,valid=split_validation(train, args.valid_split)
        test=valid
    
    train_data = RecSysDataset(train)
    test_data = RecSysDataset(test)

    collate_fn = collate_fn_factory_ccs((seq_to_ccs_graph,), order=args.order)

    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        # shuffle=True,  # Remove shuffle=True in this case as SubsetRandomSampler shuffles data already
        # drop_last=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        sampler=SequentialSampler(train_data)
    )


    test_loader = DataLoader(
        test_data,
        batch_size=args.batch_size,
        # shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    print()
    print('{:<30}{:<10,} '.format("training mini-batch",len(train_loader)))
    print('{:<30}{:<10,} '.format("test mini-batch",len(test_loader)))
    
    model = MSGIFSR(args.n_items, args.embedding_dim, args.num_layers, dropout=args.feat_drop, reducer=args.reducer, order=args.order, 
                    norm=args.norm, extra=args.extra, fusion=args.fusion, device=device)

    model = model.to(device)
    print()
    print("{:<30}{:<20,}".format("Number of parameters",np.sum([p.nelement() for p in model.parameters()])))
    
    if args.weight_decay > 0:
        params = fix_weight_decay(model)
    else:
        params = model.parameters()

    optimizer = optim.Adam(params, args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_dc_step, gamma=args.lr_dc)
    
    best_result = [0, 0]
    best_epoch = [0, 0]
    bad_counter = 0
    
    # best_metric = float('inf') ## if cross-entropy loss is selected
    best_mrr = float(0) 
    best_recall = float(0)

    TRAIN_LOSS=[]
    TEST_LOSS=[]

    TRAIN_RECALL=[]
    TEST_RECALL=[]
    
    TRAIN_MRR=[]
    TEST_MRR=[]
    
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
            loss = criterion(logits, labels)
            # loss = nn.functional.nll_loss(logits, labels)

            loss.backward()
            if args.gradient_accumulation:
                if (step+1)%args.accumulation_steps == 0 or step==len(train_loader):
                    optimizer.step()
            else:
                optimizer.step()
 
            loss_val = loss.item()
            sum_epoch_loss += loss_val

#             if (step+1)%(len(train_loader)//args.log_aggr) == 0:
#                 print('Epoch {:05d} |  Loss {:.4f} | Speed (samples/sec) {:.2f}'
#                       .format(epoch, sum_epoch_loss / (step + 1), labels.shape[0] / (time.time() - start)))

#             start = time.time()

        root_dir=os.path.join(os.getcwd(),"output_metrics")
        if not os.path.exists(root_dir):
            os.makedirs(root_dir)
        
        train_recall, train_mrr, train_loss = validate(train_loader, model, device)
        TRAIN_LOSS.append(train_loss)
        TRAIN_MRR.append(train_mrr)
        TRAIN_RECALL.append(train_recall)
        print()
        print('Epoch {} training--loss: {:.4f}, Recall@{}: {:.4f}, MRR@{}: {:.4f} \n'\
              .format(epoch, train_loss,args.topk, train_recall, args.topk, train_mrr))

        # with open(os.path.join(os.getcwd(),"output_metrics","train_"+args.output_name),'a') as f:
        #     f.write(f'{epoch+1},{train_recall},{train_mrr}\n')
                  
        test_recall, test_mrr, test_loss = validate(test_loader, model, device)
        TEST_LOSS.append(test_loss)
        TEST_MRR.append(test_mrr)
        TEST_RECALL.append(test_recall)
        print('Epoch {} test--loss: {:.4f}, Recall@{}: {:.4f}, MRR@{}: {:.4f} \n'\
              .format(epoch, test_loss,args.topk, test_recall, args.topk, test_mrr))
        
        root_dir=os.path.join(os.getcwd(),"output_metrics")
        with open(os.path.join(root_dir,args.hyper_type+args.output_name),'a') as f:
            f.write(f'{epoch+1},{test_recall},{test_mrr}\n')

#         # store best loss and save a model checkpoint
#         ckpt_dict = {
#             'epoch': epoch + 1,
#             'state_dict': model.state_dict(),
#             'optimizer': optimizer.state_dict()
#         }
        
#         if test_recall>best_recall or test_mrr>best_mrr:
#             best_recall=test_recall
#             best_mrr=test_mrr
#             torch.save(ckpt_dict, args.model_checkpoint) 
            
        flag = 0
        if round(test_recall,2) > round(best_result[0],2):
            best_result[0] = test_recall
            best_epoch[0] = epoch
            flag = 1
        if round(test_mrr,2) > round(best_result[1],2):
            best_result[1] = test_mrr
            best_epoch[1] = epoch
            flag = 1
        print('Best Result:')
        print('\tRecall@20:\t%.4f\tMMR@20:\t%.4f\tEpoch:\t%d,\t%d'% (best_result[0], best_result[1], best_epoch[0], best_epoch[1]))
        bad_counter += 1 - flag
        if bad_counter >= args.patience:
            break
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--dataset-dir', default='../YOOCHOOSE_data/yoochoose1_64/', help='the dataset directory'
    )
    parser.add_argument("--seed",  type=int,default=101,
            help="random seed for np.random.seed, torch.manual_seed and torch.cuda.manual_seed.")
    parser.add_argument('--n_items', type=int, default=37484, help='number of unique items. 37484 for yoochoose')
    parser.add_argument('--embedding-dim', type=int, default=128, help='the embedding size')
    parser.add_argument('--num-layers', type=int, default=1, help='the number of layers')
    parser.add_argument('--feat-drop', type=float, default=0.1, help='the dropout ratio for features')
    parser.add_argument('--lr', type=float, default=1e-3, help='the learning rate')
    parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay rate')
    parser.add_argument('--lr_dc_step', type=int, default=3, help='the number of steps after which the learning rate decay')
    parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')  # [0.001, 0.0005, 0.0001, 0.00005, 0.00001]
    parser.add_argument("--gradient_accumulation",action='store_true', help='gradient accumulation or not')
    parser.add_argument("--accumulation_steps",type=int,default=2,
                               help="Number of updates steps to accumulate before performing a backward/update pass.")
    
    parser.add_argument('--step', type=int, default=1, help='gnn propogation steps')
    parser.add_argument('--patience', type=int, default=5, help='the number of epoch to wait before early stop ')
    
    parser.add_argument('--batch-size', type=int, default=100, help='the batch size for training')
    parser.add_argument('--epochs', type=int, default=30, help='the number of training epochs')
    parser.add_argument("--output_name", type=str, default="amex_metrics.txt")
    parser.add_argument("--model_checkpoint", type=str, default="amex_checkpoint.pth")    
    parser.add_argument('--weight-decay',type=float,default=1e-5,help='the parameter for L2 regularization')
    parser.add_argument('--validation', action='store_true', help='validation')
    parser.add_argument('--valid-split',type=float,default=0.1,help='the fraction for the validation set')
    parser.add_argument('--topk', type=int, default=20, help='number of top score items selected for calculating recall and mrr metrics')
    parser.add_argument('--log_aggr', type=int, default=1, help='print the loss after this number of iterations')
    parser.add_argument('--num-workers',type=int,default=0,help='the number of processes to load the input graphs')
    parser.add_argument('--order',type=int,default=3,help='order of msg',)
    parser.add_argument('--reducer',type=str,default='mean',help='method for reducer')
    parser.add_argument('--norm', type=bool,default=True,help='whether use l2 norm')
    parser.add_argument('--extra',action='store_true',help='whether use REnorm.')
    parser.add_argument('--fusion',action='store_true',help='whether use IFR.')
    parser.add_argument('--hyper_type',type=str,default="num_layer_1_")
    
    args= parser.parse_args()
    print(args)
    
    seed_everything(args.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    main(args,device)
    