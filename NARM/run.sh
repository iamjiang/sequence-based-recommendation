CUDA_VISIBLE_DEVICES=1 python main.py \
--dataset_path  ../dataset/amex_explorepoi-poi_category/ \
--n_items 556 \
--batch_size 32 \
--epoch 30 \
--embed_dim 128 \
--hidden_size 128 \
--output_name amex_metrics.txt \
--model_checkpoint amex_checkpoint.pth \
--patience 5 \
--sequence_type long \
--l2 1e-4

CUDA_VISIBLE_DEVICES=1 python main.py \
--dataset_path  ../YOOCHOOSE_data/yoochoose1_64/ \
--n_items 37484 \
--batch_size 100 \
--epoch 30 \
--embed_dim 128 \
--hidden_size 128 \
--output_name yoochoose1_64_metrics.txt \
--model_checkpoint yoochoose1_64_checkpoint.pth \
--patience 5 \
--sequence_type long \
--l2 1e-4 

CUDA_VISIBLE_DEVICES=1 python main.py \
--dataset_path  ../diginetica_data/ \
--n_items 43098 \
--batch_size 100 \
--epoch 30 \
--embed_dim 128 \
--hidden_size 128 \
--output_name diginetica_metrics.txt \
--model_checkpoint diginetica_checkpoint.pth \
--patience 5 \
--sequence_type long \
--l2 1e-4


CUDA_VISIBLE_DEVICES=1 python main.py \
--dataset_path  ../dataset/amex_explorepoi-poi_category/ \
--n_items 556 \
--batch_size 32 \
--epoch 30 \
--embed_dim 128 \
--hidden_size 128 \
--output_name amex_metrics.txt \
--model_checkpoint amex_checkpoint.pth \
--patience 5 \
--sequence_type short \
--l2 1e-4

CUDA_VISIBLE_DEVICES=1 python main.py \
--dataset_path  ../YOOCHOOSE_data/yoochoose1_64/ \
--n_items 37484 \
--batch_size 100 \
--epoch 30 \
--embed_dim 128 \
--hidden_size 128 \
--output_name yoochoose1_64_metrics.txt \
--model_checkpoint yoochoose1_64_checkpoint.pth \
--patience 5 \
--sequence_type short \
--l2 1e-4 

CUDA_VISIBLE_DEVICES=1 python main.py \
--dataset_path  ../diginetica_data/ \
--n_items 43098 \
--batch_size 100 \
--epoch 30 \
--embed_dim 128 \
--hidden_size 128 \
--output_name diginetica_metrics.txt \
--model_checkpoint diginetica_checkpoint.pth \
--patience 5 \
--sequence_type short \
--l2 1e-4


# CUDA_VISIBLE_DEVICES=1 python main.py \
# --dataset_path  ../dataset/amex_explorepoi-poi_category/ \
# --n_items 556 \
# --batch_size 32 \
# --epoch 30 \
# --embed_dim 256 \
# --hidden_size 256 \
# --output_name amex_metrics.txt \
# --model_checkpoint amex_checkpoint.pth \
# --patience 5 \
# --sequence_type all 

# CUDA_VISIBLE_DEVICES=1 python main.py \
# --dataset_path  ../YOOCHOOSE_data/yoochoose1_64/ \
# --n_items 37484 \
# --batch_size 100 \
# --epoch 30 \
# --embed_dim 128 \
# --hidden_size 128 \
# --output_name yoochoose1_64_metrics.txt \
# --model_checkpoint yoochoose1_64_checkpoint.pth \
# --patience 5 \
# --sequence_type all 

# CUDA_VISIBLE_DEVICES=1 python main.py \
# --dataset_path  ../diginetica_data/ \
# --n_items 43098 \
# --batch_size 100 \
# --epoch 30 \
# --embed_dim 128 \
# --hidden_size 128 \
# --output_name diginetica_metrics.txt \
# --model_checkpoint diginetica_checkpoint.pth \
# --patience 5 \
# --sequence_type all 




