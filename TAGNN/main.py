import argparse
import os
from tqdm import tqdm
import pickle
import time
from utils import build_graph, Data, split_validation
from model import *

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
def main(opt):
    
    if not os.path.exists(os.path.join(os.getcwd(),"long_seq")):
        os.makedirs("long_seq")
    if not os.path.exists(os.path.join(os.getcwd(),"short_seq")):
        os.makedirs("short_seq")
    
    if opt.sequence_type=="all":
        train_path_data=os.path.join(opt.dataset,"train.txt")
        test_path_data=os.path.join(opt.dataset,"test.txt")
    elif opt.sequence_type=="long":
        data_dir=os.path.join(opt.dataset,"long_short_seq")
        train_path_data=os.path.join(data_dir,"long_train.txt")
        test_path_data=os.path.join(data_dir,"long_test.txt")        
    elif opt.sequence_type=="short":
        data_dir=os.path.join(opt.dataset,"long_short_seq")
        train_path_data=os.path.join(data_dir,"short_train.txt")
        test_path_data=os.path.join(data_dir,"short_test.txt") 
    else:
        raise ValueError("unknown sequence type")        
        
    with open(train_path_data, 'rb') as f1:
        train_data = pickle.load(f1)
    
    if opt.validation:
        train_data, valid_data = split_validation(train_data, opt.valid_portion)
        test_data = valid_data
    else:
        with open(test_path_data, 'rb') as f2:
            test_data = pickle.load(f2)
            
    # all_train_seq = pickle.load(open('../datasets/' + opt.dataset + '/all_train_seq.txt', 'rb'))
    # g = build_graph(all_train_seq)
    train_data = Data(train_data, shuffle=True)
    test_data = Data(test_data, shuffle=False)
    # del all_train_seq, g
    if opt.dataset.split("/")[-2] == 'diginetica_data':
        n_node = 43098
    elif opt.dataset.split("/")[-2] == 'yoochoose1_64' or opt.dataset.split("/")[-2] == 'yoochoose1_4':
        n_node = 37484
    else:
        n_node = 556

    model = trans_to_cuda(SessionGraph(opt, n_node))

    start = time.time()
    best_result = [0, 0]
    best_epoch = [0, 0]
    bad_counter = 0
    
    # best_metric = float('inf') ## if cross-entropy loss is selected
    best_mrr = float(0) 
    best_recall = float(0)
    
    for epoch in tqdm(range(opt.epoch)):
        print('-------------------------------------------------------')
        print('epoch: ', epoch)
        hit, mrr = train_test(model, train_data, test_data)
        
        if opt.sequence_type=="all":
            root_dir=os.path.join(os.getcwd(),"output_metrics")
            if not os.path.exists(root_dir):
                os.makedirs(root_dir)
        elif opt.sequence_type=="long":
            root_dir=os.path.join(os.getcwd(),"long_seq","output_metrics")
            if not os.path.exists(root_dir):
                os.makedirs(root_dir)
        elif opt.sequence_type=="short":
            root_dir=os.path.join(os.getcwd(),"short_seq","output_metrics")
            if not os.path.exists(root_dir):
                os.makedirs(root_dir)
        else:
            raise ValueError("unknown sequence type")

        if opt.sequence_type=="all":
            root_dir=os.path.join(os.getcwd(),"output_metrics")
            with open(os.path.join(root_dir,"test_"+opt.output_name),'a') as f:
                f.write(f'{epoch+1},{hit},{mrr}\n')
            
        elif opt.sequence_type=="long":
            root_dir=os.path.join(os.getcwd(),"long_seq","output_metrics")
            with open(os.path.join(root_dir,"test_"+opt.output_name),'a') as f:
                f.write(f'{epoch+1},{hit},{mrr}\n')
                
        elif opt.sequence_type=="short":
            root_dir=os.path.join(os.getcwd(),"short_seq","output_metrics")
            with open(os.path.join(root_dir,"test_"+opt.output_name),'a') as f:
                f.write(f'{epoch+1},{hit},{mrr}\n')
        else:
            raise ValueError("unknown sequence type")
        
        # store best loss and save a model checkpoint
        ckpt_dict = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': model.optimizer.state_dict()
        }

        if hit>best_recall or mrr>best_mrr:
            best_recall=hit
            best_mrr=mrr
            if opt.sequence_type=="all":
                torch.save(ckpt_dict, opt.model_checkpoint)
            elif opt.sequence_type=="long":
                save_dir=os.path.join(os.getcwd(),"long_seq")
                torch.save(ckpt_dict, os.path.join(save_dir,opt.model_checkpoint))
            elif opt.sequence_type=="short":
                save_dir=os.path.join(os.getcwd(),"short_seq")
                torch.save(ckpt_dict, os.path.join(save_dir,opt.model_checkpoint))                
            else:
                raise ValueError("unknown sequence type") 
        
        flag = 0
        if round(hit,2) > round(best_result[0],2):
            best_result[0] = hit
            best_epoch[0] = epoch
            flag = 1
        if round(mrr,2) > round(best_result[1],2):
            best_result[1] = mrr
            best_epoch[1] = epoch
            flag = 1
        print('Best Result:')
        print('\tRecall@20:\t%.4f\tMMR@20:\t%.4f\tEpoch:\t%d,\t%d'% (best_result[0], best_result[1], best_epoch[0], best_epoch[1]))
        bad_counter += 1 - flag
        if bad_counter >= opt.patience:
            break
    print('-------------------------------------------------------')
    end = time.time()
    print("Run time: %f s" % (end - start))


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='../YOOCHOOSE_data/yoochoose1_64/', help='the dataset directory')
    parser.add_argument('--batchSize', type=int, default=100, help='input batch size')
    parser.add_argument('--hiddenSize', type=int, default=100, help='hidden state size')
    parser.add_argument('--epoch', type=int, default=30, help='the number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')  # [0.001, 0.0005, 0.0001]
    parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay rate')
    parser.add_argument('--lr_dc_step', type=int, default=3, help='the number of steps after which the learning rate decay')
    parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')  # [0.001, 0.0005, 0.0001, 0.00005, 0.00001]
    parser.add_argument("--gradient_accumulation",action='store_true', help='gradient accumulation or not')
    parser.add_argument("--accumulation_steps",type=int,default=2,
                               help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--step', type=int, default=1, help='gnn propogation steps')
    parser.add_argument('--patience', type=int, default=10, help='the number of epoch to wait before early stop ')
    parser.add_argument('--nonhybrid', action='store_true', help='only use the global preference to predict')
    parser.add_argument('--validation', action='store_true', help='validation')
    parser.add_argument('--valid_portion', type=float, default=0.1, help='split the portion of training set as validation set')
    parser.add_argument("--output_name", type=str, default="amex_metrics.txt")
    parser.add_argument("--model_checkpoint", type=str, default="amex_checkpoint.pth")
    parser.add_argument('--sequence_type',type=str,default="all",help='all sequence or longer only sequence(>5) or short only sequence(<=5)')
    opt = parser.parse_args()
    print(opt)

    seed_everything(101)
    
    main(opt)
