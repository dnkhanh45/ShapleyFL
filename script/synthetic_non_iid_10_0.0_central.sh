TASK="synthetic_classification"
DIST=10
SKEW=0.0
NUM_CLIENTS=10
SEED=0
python generate_fedtask.py --benchmark $TASK --dist $DIST --skew $SKEW --num_clients $NUM_CLIENTS --seed $SEED

TASK="${TASK}_cnum${NUM_CLIENTS}_dist${DIST}_skew${SKEW}_seed${SEED}"
GPU_IDS=( 1 )
NUM_THREADS=1
BATCH_SIZE=10
NUM_ROUNDS=100
PROPORTION=1.0

python main_central.py \
    --task $TASK \
    --model lr \
    --algorithm sv_central \
    --num_rounds $NUM_ROUNDS \
    --num_epochs 2 \
    --learning_rate 0.5 \
    --lr_scheduler 0 \
    --learning_rate_decay 1.0 \
    --proportion 1 \
    --batch_size $BATCH_SIZE \
    --eval_interval 1 \
    --gpu $GPU_IDS \
    --num_threads $NUM_THREADS \
    --aggregate weighted_scale \
    --sample full