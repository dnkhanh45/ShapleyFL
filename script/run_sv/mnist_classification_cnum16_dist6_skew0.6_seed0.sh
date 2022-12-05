# GPU variables
GPU_IDS=( 1 )
NUM_THREADS=1
BATCH_SIZE=20

# Generate fedtask
python generate_fedtask.py \
    --benchmark mnist_classification \
    --dist 6 \
    --skew 0.6 \
    --num_clients 16 \
    --seed 0

# Run
TASK="mnist_classification_cnum16_dist2_skew0.6_seed0"

# ROOT_PATH="$SGE_LOCALDIR/$JOB_ID/"
# mkdir $ROOT_PATH

# DATA_PATH="$ROOT_PATH/data"
# mkdir $DATA_PATH
# cp -r ./benchmark/RAW_DATA/MNIST ${DATA_PATH}
# DATA_PATH="$DATA_PATH/MNIST"

# FEDTASK_PATH="$ROOT_PATH/fedtask"
# mkdir $FEDTASK_PATH
# cp -r ./fedtask/$TASK ${FEDTASK_PATH}

python main.py \
    --task $TASK \
    --model mlp \
    --algorithm sv_fedavg \
    --num_rounds 200 \
    --num_epochs 5 \
    --learning_rate 0.01 \
    --lr_scheduler 0 \
    --learning_rate_decay 0.998 \
    --proportion 1.0 \
    --batch_size $BATCH_SIZE \
    --eval_interval 1 \
    --gpu $GPU_IDS \
    --num_threads $NUM_THREADS \
    --aggregate weighted_scale \
    --sample uniform \
    --num_partitions 2 \
    --const_lambda \
    --optimal_lambda \
    --optimal_lambda_samples 300 \
    --exact \
    # --fedtask_path $FEDTASK_PATH \
    # --data_path $DATA_PATH \