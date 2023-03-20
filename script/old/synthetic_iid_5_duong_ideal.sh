TASK="synthetic_classification_cnum5_duong_skew0_seed7"
GPU_IDS=( 1 )
NUM_THREADS=1
BATCH_SIZE=10
NUM_ROUNDS=50
PROPORTION=1.0
    
python main_ideal.py \
    --task $TASK \
    --model lr \
    --algorithm fedavg \
    --num_rounds $NUM_ROUNDS \
    --num_epochs 2 \
    --learning_rate 0.1 \
    --lr_scheduler 0 \
    --learning_rate_decay 1.0 \
    --proportion 1 \
    --batch_size $BATCH_SIZE \
    --eval_interval 1 \
    --gpu $GPU_IDS \
    --num_threads $NUM_THREADS \
    --aggregate weighted_scale \
    --sample full