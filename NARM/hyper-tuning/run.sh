CUDA_VISIBLE_DEVICES=2 python main.py \
--dataset_path  ../../dataset/amex_explorepoi-poi_category/ \
--n_items 556 \
--batch_size 32 \
--epoch 30 \
--embed_dim 64 \
--hidden_size 64 \
--output_name amex.txt \
--hyper_type emb_64_ \
--patience 5 

CUDA_VISIBLE_DEVICES=2 python main.py \
--dataset_path  ../../YOOCHOOSE_data/yoochoose1_64/ \
--n_items 37484 \
--batch_size 100 \
--epoch 30 \
--embed_dim 64 \
--hidden_size 64 \
--output_name yh.txt \
--hyper_type emb_64_ \
--patience 5 


CUDA_VISIBLE_DEVICES=2 python main.py \
--dataset_path  ../../diginetica_data/ \
--n_items 43098 \
--batch_size 100 \
--epoch 30 \
--embed_dim 64 \
--hidden_size 64 \
--output_name dg.txt \
--hyper_type emb_64_ \
--patience 5 

########################################################################

CUDA_VISIBLE_DEVICES=2 python main.py \
--dataset_path  ../../dataset/amex_explorepoi-poi_category/ \
--n_items 556 \
--batch_size 32 \
--epoch 30 \
--embed_dim 128 \
--hidden_size 128 \
--output_name amex.txt \
--hyper_type emb_128_ \
--patience 5 

CUDA_VISIBLE_DEVICES=2 python main.py \
--dataset_path  ../../YOOCHOOSE_data/yoochoose1_64/ \
--n_items 37484 \
--batch_size 100 \
--epoch 30 \
--embed_dim 128 \
--hidden_size 128 \
--output_name yh.txt \
--hyper_type emb_128_ \
--patience 5 


CUDA_VISIBLE_DEVICES=2 python main.py \
--dataset_path  ../../diginetica_data/ \
--n_items 43098 \
--batch_size 100 \
--epoch 30 \
--embed_dim 128 \
--hidden_size 128 \
--output_name dg.txt \
--hyper_type emb_128_ \
--patience 5 

########################################################################

CUDA_VISIBLE_DEVICES=2 python main.py \
--dataset_path  ../../dataset/amex_explorepoi-poi_category/ \
--n_items 556 \
--batch_size 32 \
--epoch 30 \
--embed_dim 256 \
--hidden_size 256 \
--output_name amex.txt \
--hyper_type emb_256_ \
--patience 5 

CUDA_VISIBLE_DEVICES=2 python main.py \
--dataset_path  ../../YOOCHOOSE_data/yoochoose1_64/ \
--n_items 37484 \
--batch_size 100 \
--epoch 30 \
--embed_dim 256 \
--hidden_size 256 \
--output_name yh.txt \
--hyper_type emb_256_ \
--patience 5 


CUDA_VISIBLE_DEVICES=2 python main.py \
--dataset_path  ../../diginetica_data/ \
--n_items 43098 \
--batch_size 100 \
--epoch 30 \
--embed_dim 256 \
--hidden_size 256 \
--output_name dg.txt \
--hyper_type emb_256_ \
--patience 5 

########################################################################

CUDA_VISIBLE_DEVICES=2 python main.py \
--dataset_path  ../../dataset/amex_explorepoi-poi_category/ \
--n_items 556 \
--batch_size 32 \
--epoch 30 \
--embed_dim 512 \
--hidden_size 512 \
--output_name amex.txt \
--hyper_type emb_512_ \
--patience 5 

CUDA_VISIBLE_DEVICES=2 python main.py \
--dataset_path  ../../YOOCHOOSE_data/yoochoose1_64/ \
--n_items 37484 \
--batch_size 100 \
--epoch 30 \
--embed_dim 512 \
--hidden_size 512 \
--output_name yh.txt \
--hyper_type emb_512_ \
--patience 5 


CUDA_VISIBLE_DEVICES=2 python main.py \
--dataset_path  ../../diginetica_data/ \
--n_items 43098 \
--batch_size 100 \
--epoch 30 \
--embed_dim 512 \
--hidden_size 512 \
--output_name dg.txt \
--hyper_type emb_512_ \
--patience 5 

########################################################################

CUDA_VISIBLE_DEVICES=2 python main.py \
--dataset_path  ../../dataset/amex_explorepoi-poi_category/ \
--n_items 43098 \
--batch_size 100 \
--epoch 30 \
--embed_dim 768 \
--hidden_size 768 \
--output_name amex.txt \
--hyper_type emb_768_ \
--patience 5 

CUDA_VISIBLE_DEVICES=2 python main.py \
--dataset_path  ../../dataset/amex_explorepoi-poi_category/ \
--n_items 43098 \
--batch_size 100 \
--epoch 30 \
--embed_dim 1024 \
--hidden_size 1024 \
--output_name amex.txt \
--hyper_type emb_1024_ \
--patience 5 


