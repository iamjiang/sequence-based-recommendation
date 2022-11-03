CUDA_VISIBLE_DEVICES=2 python main.py \
--dataset  ../dataset/amex_explorepoi-poi_category/  \
--batchSize 32 \
--epoch 30 \
--hiddenSize 128 \
--step 1 \
--output_name amex_metrics.txt \
--model_checkpoint amex_checkpoint.pth \
--patience 5 \
--sequence_type long \
--l2 1e-4

CUDA_VISIBLE_DEVICES=2 python main.py \
--dataset  ../YOOCHOOSE_data/yoochoose1_64/  \
--batchSize 100 \
--epoch 30 \
--hiddenSize 128 \
--step 1 \
--output_name yoochoose1_64_metrics.txt \
--model_checkpoint yoochoose1_64_checkpoint.pth \
--patience 5 \
--sequence_type long

CUDA_VISIBLE_DEVICES=2 python main.py \
--dataset  ../diginetica_data/  \
--batchSize 100 \
--epoch 30 \
--hiddenSize 128 \
--step 1 \
--output_name diginetica_metrics.txt \
--model_checkpoint diginetica_checkpoint.pth \
--patience 5 \
--sequence_type long


CUDA_VISIBLE_DEVICES=2 python main.py \
--dataset  ../dataset/amex_explorepoi-poi_category/  \
--batchSize 32 \
--epoch 30 \
--hiddenSize 128 \
--step 1 \
--output_name amex_metrics.txt \
--model_checkpoint amex_checkpoint.pth \
--patience 5 \
--sequence_type short \
--l2 1e-4

CUDA_VISIBLE_DEVICES=2 python main.py \
--dataset  ../YOOCHOOSE_data/yoochoose1_64/  \
--batchSize 100 \
--epoch 30 \
--hiddenSize 128 \
--step 1 \
--output_name yoochoose1_64_metrics.txt \
--model_checkpoint yoochoose1_64_checkpoint.pth \
--patience 5 \
--sequence_type short

CUDA_VISIBLE_DEVICES=2 python main.py \
--dataset  ../diginetica_data/  \
--batchSize 100 \
--epoch 30 \
--hiddenSize 128 \
--step 1 \
--output_name diginetica_metrics.txt \
--model_checkpoint diginetica_checkpoint.pth \
--patience 5 \
--sequence_type short

# CUDA_VISIBLE_DEVICES=2 python main.py \
# --dataset  ../dataset/amex_explorepoi-poi_category/  \
# --batchSize 32 \
# --epoch 30 \
# --hiddenSize 256 \
# --step 1 \
# --output_name amex_metrics.txt \
# --model_checkpoint amex_checkpoint.pth \
# --patience 5

# CUDA_VISIBLE_DEVICES=2 python main.py \
# --dataset  ../YOOCHOOSE_data/yoochoose1_64/  \
# --batchSize 100 \
# --epoch 30 \
# --hiddenSize 128 \
# --step 1 \
# --output_name yoochoose1_64_metrics.txt \
# --model_checkpoint yoochoose1_64_checkpoint.pth \
# --patience 5

# CUDA_VISIBLE_DEVICES=2 python main.py \
# --dataset  ../diginetica_data/  \
# --batchSize 100 \
# --epoch 30 \
# --hiddenSize 128 \
# --step 1 \
# --output_name diginetica_metrics.txt \
# --model_checkpoint diginetica_checkpoint.pth \
# --patience 5