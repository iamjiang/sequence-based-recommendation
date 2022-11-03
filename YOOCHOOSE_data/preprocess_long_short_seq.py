import argparse
import pickle
import os

def process_seqs(iseqs):
    out_seqs = []
    labs = []

    for id, seq in zip(range(len(iseqs)), iseqs):
        for i in range(1, len(seq)):
            tar = seq[-i]
            labs += [tar]
            out_seqs += [seq[:-i]]
    return out_seqs, labs


def main(args):
    train_path_data=os.path.join(args.dataset_dir,"all_train_seq.txt")
    test_path_data=os.path.join(args.dataset_dir,"all_test_seq.txt")

    with open(train_path_data, 'rb') as f1:
        train_seq = pickle.load(f1)

    with open(test_path_data, 'rb') as f2:
        test_seq = pickle.load(f2)
        
    if not os.path.exists(os.path.join(args.dataset_dir,"long_short_seq")):
        os.makedirs(os.path.join(args.dataset_dir,"long_short_seq"))
        
    long_train_seq=list(filter(lambda k: len(k)>5,train_seq))
    long_test_seq=list(filter(lambda k: len(k)>5,test_seq))
    
    short_train_seq=list(filter(lambda k: len(k)<=5,train_seq))
    short_test_seq=list(filter(lambda k: len(k)<=5,test_seq))
    
    long_tr_seqs, long_tr_labs = process_seqs(long_train_seq)
    long_te_seqs, long_te_labs = process_seqs(long_test_seq)
    long_tra = (long_tr_seqs, long_tr_labs)
    long_tes = (long_te_seqs, long_te_labs)
    
    short_tr_seqs, short_tr_labs = process_seqs(short_train_seq)
    short_te_seqs, short_te_labs = process_seqs(short_test_seq)
    short_tra = (short_tr_seqs, short_tr_labs)
    short_tes = (short_te_seqs, short_te_labs)
    
    output_dir=os.path.join(args.dataset_dir,"long_short_seq")
    
    pickle.dump(long_tra, open(os.path.join(output_dir,'long_train.txt'), 'wb'))
    pickle.dump(long_tes, open(os.path.join(output_dir,'long_test.txt'), 'wb'))
    pickle.dump(long_tr_seqs, open(os.path.join(output_dir,'long_train_seq.txt'), 'wb'))
    pickle.dump(long_te_seqs, open(os.path.join(output_dir,'long_test_seq.txt'), 'wb'))
    
    pickle.dump(short_tra, open(os.path.join(output_dir,'short_train.txt'), 'wb'))
    pickle.dump(short_tes, open(os.path.join(output_dir,'short_test.txt'), 'wb'))
    pickle.dump(short_tr_seqs, open(os.path.join(output_dir,'short_train_seq.txt'), 'wb'))
    pickle.dump(short_te_seqs, open(os.path.join(output_dir,'short_test_seq.txt'), 'wb'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--dataset-dir', default='./yoochoose1_64/', help='the dataset directory'
    )
 
    args=parser.parse_args()
    print(args)
    
    main(args)
    