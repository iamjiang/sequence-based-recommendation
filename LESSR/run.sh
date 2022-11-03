CUDA_VISIBLE_DEVICES=1 python main_lessr.py \
--dataset-dir  ../dataset/amex_explorepoi-poi_category/ \
--n_items 556 \
--batch-size 32 \
--epochs 30 \
--embedding-dim 128 \
--num-layers 3 \
--output_name amex_metrics.txt \
--model_checkpoint amex_checkpoint.pth \
--patience 5 \
--sequence_type long \
--weight-decay 1e-4

CUDA_VISIBLE_DEVICES=1 python main_lessr.py \
--dataset-dir  ../YOOCHOOSE_data/yoochoose1_64/ \
--n_items 37484 \
--batch-size 100 \
--epochs 30 \
--embedding-dim 128 \
--num-layers 3 \
--output_name yoochoose1_64_metrics.txt \
--model_checkpoint yoochoose1_64_checkpoint.pth \
--patience 5 \
--sequence_type long


CUDA_VISIBLE_DEVICES=1 python main_lessr.py \
--dataset-dir  ../diginetica_data/ \
--n_items 43098 \
--batch-size 100 \
--epochs 30 \
--embedding-dim 128 \
--num-layers 3 \
--output_name diginetica_metrics.txt \
--model_checkpoint diginetica_checkpoint.pth \
--patience 5 \
--sequence_type long

CUDA_VISIBLE_DEVICES=1 python main_lessr.py \
--dataset-dir  ../dataset/amex_explorepoi-poi_category/ \
--n_items 556 \
--batch-size 32 \
--epochs 30 \
--embedding-dim 128 \
--num-layers 3 \
--output_name amex_metrics.txt \
--model_checkpoint amex_checkpoint.pth \
--patience 5 \
--sequence_type short \
--weight-decay 1e-4

CUDA_VISIBLE_DEVICES=1 python main_lessr.py \
--dataset-dir  ../YOOCHOOSE_data/yoochoose1_64/ \
--n_items 37484 \
--batch-size 100 \
--epochs 30 \
--embedding-dim 128 \
--num-layers 3 \
--output_name yoochoose1_64_metrics.txt \
--model_checkpoint yoochoose1_64_checkpoint.pth \
--patience 5 \
--sequence_type short


CUDA_VISIBLE_DEVICES=1 python main_lessr.py \
--dataset-dir  ../diginetica_data/ \
--n_items 43098 \
--batch-size 100 \
--epochs 30 \
--embedding-dim 128 \
--num-layers 3 \
--output_name diginetica_metrics.txt \
--model_checkpoint diginetica_checkpoint.pth \
--patience 5 \
--sequence_type short

# CUDA_VISIBLE_DEVICES=0 python main_lessr.py \
# --dataset-dir  ../dataset/amex_explorepoi-poi_category/ \
# --n_items 556 \
# --batch-size 32 \
# --epochs 30 \
# --embedding-dim 256 \
# --num-layers 3 \
# --output_name amex_metrics.txt \
# --model_checkpoint amex_checkpoint.pth \
# --patience 5

# CUDA_VISIBLE_DEVICES=0 python main_lessr.py \
# --dataset-dir  ../YOOCHOOSE_data/yoochoose1_64/ \
# --n_items 37484 \
# --batch-size 100 \
# --epochs 30 \
# --embedding-dim 256 \
# --num-layers 3 \
# --output_name yoochoose1_64_metrics.txt \
# --model_checkpoint yoochoose1_64_checkpoint.pth \
# --patience 5


# CUDA_VISIBLE_DEVICES=0 python main_lessr.py \
# --dataset-dir  ../diginetica_data/ \
# --n_items 43098 \
# --batch-size 100 \
# --epochs 30 \
# --embedding-dim 256 \
# --num-layers 3 \
# --output_name diginetica_metrics.txt \
# --model_checkpoint diginetica_checkpoint.pth \
# --patience 5
