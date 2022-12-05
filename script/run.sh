TASK="mnist_classification_cnum10_dist2_skew0.7_seed0"
GPU_IDS=( 0 )
NUM_THREADS=1
BATCH_SIZE=20

python main.py \
    --task $TASK \
    --model mlp \
    --algorithm sv_fedavg \
    --num_rounds 50 \
    --num_epochs 2 \
    --learning_rate 0.01 \
    --lr_scheduler 0 \
    --learning_rate_decay 0.998 \
    --proportion 1 \
    --batch_size $BATCH_SIZE \
    --eval_interval 1 \
    --gpu $GPU_IDS \
    --num_threads $NUM_THREADS \
    --aggregate weighted_scale \
    --sample full \
    --num_partitions 2 \
    --const_lambda \
    --optimal_lambda \