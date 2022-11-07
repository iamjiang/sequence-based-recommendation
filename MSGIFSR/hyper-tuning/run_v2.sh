CUDA_VISIBLE_DEVICES=1 python main_msgifsr.py \
--dataset-dir  ../../dataset/amex_explorepoi-poi_category/ \
--n_items 556 \
--batch-size 32 \
--epochs 30 \
--embedding-dim 128 \
--order 1 \
--output_name amex.txt \
--hyper_type order_1_emb_128_ \
--patience 5 

CUDA_VISIBLE_DEVICES=1 python main_msgifsr.py \
--dataset-dir  ../../dataset/amex_explorepoi-poi_category/ \
--n_items 556 \
--batch-size 32 \
--epochs 30 \
--embedding-dim 128 \
--order 2 \
--output_name amex.txt \
--hyper_type order_2_emb_128_ \
--patience 5 

########################################################################
CUDA_VISIBLE_DEVICES=1 python main_msgifsr.py \
--dataset-dir  ../../dataset/amex_explorepoi-poi_category/ \
--n_items 556 \
--batch-size 32 \
--epochs 30 \
--embedding-dim 512 \
--order 1 \
--output_name amex.txt \
--hyper_type order_1_emb_512_ \
--patience 5 

CUDA_VISIBLE_DEVICES=1 python main_msgifsr.py \
--dataset-dir  ../../dataset/amex_explorepoi-poi_category/ \
--n_items 556 \
--batch-size 32 \
--epochs 30 \
--embedding-dim 512 \
--order 2 \
--output_name amex.txt \
--hyper_type order_2_emb_512_ \
--patience 5 

CUDA_VISIBLE_DEVICES=1 python main_msgifsr.py \
--dataset-dir  ../../dataset/amex_explorepoi-poi_category/ \
--n_items 556 \
--batch-size 32 \
--epochs 30 \
--embedding-dim 512 \
--order 3 \
--output_name amex.txt \
--hyper_type order_3_emb_512_ \
--patience 5 

########################################################################
CUDA_VISIBLE_DEVICES=1 python main_msgifsr.py \
--dataset-dir  ../../YOOCHOOSE_data/yoochoose1_64/ \
--n_items 37484 \
--batch-size 100 \
--epochs 30 \
--embedding-dim 256 \
--order 1 \
--output_name yh.txt \
--hyper_type order_1_emb_256_ \
--patience 5 


CUDA_VISIBLE_DEVICES=1 python main_msgifsr.py \
--dataset-dir  ../../diginetica_data/ \
--n_items 43098 \
--batch-size 100 \
--epochs 30 \
--embedding-dim 256 \
--order 1 \
--output_name dg.txt \
--hyper_type order_1_emb_256_ \
--patience 5 


CUDA_VISIBLE_DEVICES=1 python main_msgifsr.py \
--dataset-dir  ../../YOOCHOOSE_data/yoochoose1_64/ \
--n_items 37484 \
--batch-size 100 \
--epochs 30 \
--embedding-dim 512 \
--order 1 \
--output_name yh.txt \
--hyper_type order_1_emb_512_ \
--patience 5 


CUDA_VISIBLE_DEVICES=1 python main_msgifsr.py \
--dataset-dir  ../../diginetica_data/ \
--n_items 43098 \
--batch-size 100 \
--epochs 30 \
--embedding-dim 512 \
--order 1 \
--output_name dg.txt \
--hyper_type order_1_emb_512_ \
--patience 5 
########################################################################
CUDA_VISIBLE_DEVICES=1 python main_msgifsr.py \
--dataset-dir  ../../YOOCHOOSE_data/yoochoose1_64/ \
--n_items 37484 \
--batch-size 100 \
--epochs 30 \
--embedding-dim 256 \
--order 2 \
--output_name yh.txt \
--hyper_type order_2_emb_256_ \
--patience 5 


CUDA_VISIBLE_DEVICES=1 python main_msgifsr.py \
--dataset-dir  ../../diginetica_data/ \
--n_items 43098 \
--batch-size 100 \
--epochs 30 \
--embedding-dim 256 \
--order 2 \
--output_name dg.txt \
--hyper_type order_2_emb_256_ \
--patience 5 


CUDA_VISIBLE_DEVICES=1 python main_msgifsr.py \
--dataset-dir  ../../YOOCHOOSE_data/yoochoose1_64/ \
--n_items 37484 \
--batch-size 100 \
--epochs 30 \
--embedding-dim 512 \
--order 2 \
--output_name yh.txt \
--hyper_type order_2_emb_512_ \
--patience 5 


CUDA_VISIBLE_DEVICES=1 python main_msgifsr.py \
--dataset-dir  ../../diginetica_data/ \
--n_items 43098 \
--batch-size 100 \
--epochs 30 \
--embedding-dim 512 \
--order 2 \
--output_name dg.txt \
--hyper_type order_2_emb_512_ \
--patience 5 