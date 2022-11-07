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
    
def trainForEpoch(train_loader, model, optimizer, epoch, num_epochs, criterion, log_aggr=10):
    model.train()

    sum_epoch_loss = 0

    start = time.time()
    for i, (seq, target, lens) in tqdm(enumerate(train_loader), total=len(train_loader)):
        seq = seq.to(device)
        target = target.to(device)
        
        optimizer.zero_grad()
        outputs = model(seq, lens)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step() 
        
        loss_val = loss.item()
        sum_epoch_loss += loss_val

        iter_num = epoch * len(train_loader) + i + 1
        
        # writer.add_scalar("Loss/train", loss, iter_num)

        if i%(len(train_loader)//log_aggr) == 0 and not i==0:
            print('[TRAIN] epoch %d/%d batch loss: %.4f (avg %.4f) (%.2f im/s)'
                % (epoch + 1, num_epochs, loss_val, sum_epoch_loss / (i + 1),
                  len(seq) / (time.time() - start)))

        start = time.time()


def validate(valid_loader, model,device):
    model.eval()
    recalls = []
    mrrs = []
    losses=[]
    with torch.no_grad():
        for seq, target, lens in valid_loader:
            seq = seq.to(device)
            target = target.to(device)
            outputs = model(seq, lens)
            criterion = nn.CrossEntropyLoss()
            loss = criterion(outputs, target)
            logits = F.softmax(outputs, dim = 1)
            recall, mrr = metric.evaluate(logits, target, k = args.topk)
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
    
    print('Loading data...')
    train, test = load_data(args.dataset_path)
    if args.validation:
        train,valid=split_validation(train, args.valid_split)
        test=valid
        
    train_data = RecSysDataset(train)
    test_data = RecSysDataset(test)
    
    train_loader = DataLoader(train_data, batch_size = args.batch_size, shuffle = True, collate_fn = collate_fn)
    test_loader = DataLoader(test_data, batch_size = args.batch_size, shuffle = False, collate_fn = collate_fn)
    
    print('{:<30}{:<10,} '.format("training batch",len(train_loader)))
    print('{:<30}{:<10,} '.format("test batch",len(test_loader)))
    
    model = NARM(args.n_items, args.hidden_size, args.embed_dim, args.batch_size).to(device)
    print()
    print("{:<30}{:<20,}".format("Number of parameters",np.sum([p.nelement() for p in model.parameters()])))

    if args.l2 > 0:
        params = fix_weight_decay(model)
    else:
        params = model.parameters()
        
    optimizer = optim.Adam(params, args.lr, weight_decay=args.l2)
    
    scheduler = StepLR(optimizer, step_size = args.lr_dc_step, gamma = args.lr_dc)

    best_result = [0, 0]
    best_epoch = [0, 0]
    bad_counter = 0
    
    # best_metric = float('inf') ## if cross-entropy loss is selected
    best_mrr = float(0) 
    best_recall = float(0)
    
    TRAIN_LOSS=[]
    TEST_LOSS=[]

    TRAIN_MRR=[]
    TEST_MRR=[]

    TRAIN_RECALL=[]
    TEST_RECALL=[]

    for epoch in tqdm(range(args.epoch)): #before: no leave param, now , leave=False
    #         time.sleep(1)
        # train for one epoch
        scheduler.step(epoch = epoch)
        # trainForEpoch(train_loader, model, optimizer, epoch, args.epoch, criterion, log_aggr = 5)
        model.train()
        sum_epoch_loss = 0
        start = time.time()
        for i, (seq, target, lens) in tqdm(enumerate(train_loader), total=len(train_loader)):
            seq = seq.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            outputs = model(seq, lens)
            criterion = nn.CrossEntropyLoss()
            loss = criterion(outputs, target)
            loss.backward()
            if args.gradient_accumulation:
                if (step+1)%args.accumulation_steps == 0 or step==len(train_loader):
                    optimizer.step()
            else:
                optimizer.step() 

            loss_val = loss.item()
            sum_epoch_loss += loss_val

            iter_num = epoch * len(train_loader) + i + 1

            # writer.add_scalar("Loss/train", loss, iter_num)

#             if i%(len(train_loader)//5) == 0 and not i==0:
#                 print('[TRAIN] epoch %d/%d batch loss: %.4f (avg %.4f) (%.2f im/s)'
#                     % (epoch + 1, args.epoch, loss_val, sum_epoch_loss / (i + 1),
#                       len(seq) / (time.time() - start)))

#             start = time.time()
        
        root_dir=os.path.join(os.getcwd(),"output_metrics")
        if not os.path.exists(root_dir):
            os.makedirs(root_dir)
            
        train_recall, train_mrr, train_loss = validate(train_loader, model, device)
        TRAIN_LOSS.append(train_loss)
        TRAIN_MRR.append(train_mrr)
        TRAIN_RECALL.append(train_recall)
        print()
        print('Epoch {} training-- loss: {:.4f}, Recall@{}: {:.4f}, MRR@{}: {:.4f} \n'\
              .format(epoch, train_loss,args.topk, train_recall, args.topk, train_mrr))

        test_recall, test_mrr, test_loss = validate(test_loader, model, device)
        TEST_LOSS.append(test_loss)
        TEST_MRR.append(test_mrr)
        TEST_RECALL.append(test_recall)
        print('Epoch {} test-- loss: {:.4f}, Recall@{}: {:.4f}, MRR@{}: {:.4f} \n'\
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
        print('\tRecall@20:\t%.4f\tMMR@20:\t%.4f\tEpoch:\t%d,\t%d'% (best_result[0], best_result[1], best_epoch[0], 
                                                                     best_epoch[1]))
        bad_counter += 1 - flag
        if bad_counter >= args.patience:
            break

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', default='../YOOCHOOSE_data/yoochoose1_64/', 
                        help='dataset directory path: datasets/amex/yoochoose1_4/yoochoose1_64')
    
    parser.add_argument("--seed",  type=int,default=101,
            help="random seed for np.random.seed, torch.manual_seed and torch.cuda.manual_seed.")
    
    parser.add_argument('--n_items', type=int, default=37484, help='number of unique items. 37484 for yoochoose')
    parser.add_argument('--batch_size', type=int, default=100, help='input batch size')
    parser.add_argument('--hidden_size', type=int, default=128, help='hidden state size of gru module')
    parser.add_argument('--embed_dim', type=int, default=128, help='the dimension of item embedding')
    parser.add_argument('--epoch', type=int, default=30, help='the number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')  
    parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay rate')
    parser.add_argument('--lr_dc_step', type=int, default=80, help='the number of steps after which the learning rate decay') 
    parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')  # [0.001, 0.0005, 0.0001, 0.00005, 0.00001]
    parser.add_argument("--gradient_accumulation",action='store_true', help='gradient accumulation or not')
    parser.add_argument("--accumulation_steps",type=int,default=2,
                               help="Number of updates steps to accumulate before performing a backward/update pass.")
    
    parser.add_argument('--test', action='store_true', help='test')
    parser.add_argument('--patience', type=int, default=5, help='the number of epoch to wait before early stop ')
    parser.add_argument('--topk', type=int, default=20, help='number of top score items selected for calculating recall and mrr metrics')
    parser.add_argument('--validation',action='store_true', help='validation')
    parser.add_argument('--valid_portion', type=float, default=0.1, help='split the portion of training set as validation set')
    
    parser.add_argument("--output_name", type=str, default="amex_metrics.txt")
    parser.add_argument("--model_checkpoint", type=str, default="amex_checkpoint.pth")    
    parser.add_argument('--hyper_type',type=str,default="num_layer_1_")
    
    args = parser.parse_args()
    print(args)
    
    seed_everything(args.seed)
    
    # here = os.path.dirname(os.path.abspath(__file__))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    main(args,device)
    
    