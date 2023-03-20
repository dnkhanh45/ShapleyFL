GPU_IDS=( 0 )
NUM_THREADS=2
BATCH_SIZE=20
# Generate fedtask
python generate_fedtask.py \
    --benchmark mnist_classification \
    --dist 2 \
    --skew 0.6 \
    --num_clients 6 \
    --seed 0

# Run
TASK="mnist_classification_cnum6_dist2_skew0.6_seed0"

ROOT_PATH="./tmp"
mkdir $ROOT_PATH

DATA_PATH="$ROOT_PATH/data"
mkdir $DATA_PATH
cp -r ./benchmark/RAW_DATA/MNIST ${DATA_PATH}
DATA_PATH="$DATA_PATH/MNIST"

FEDTASK_PATH="$ROOT_PATH/fedtask"
mkdir $FEDTASK_PATH
cp -r ./fedtask/$TASK ${FEDTASK_PATH}

python main.py \
    --task $TASK \
    --model mlp \
    --algorithm sv_fedavg \
    --num_rounds 1 \
    --num_epochs 1 \
    --learning_rate 0.01 \
    --lr_scheduler 0 \
    --learning_rate_decay 0.998 \
    --proportion 0.7 \
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
    --fedtask_path $FEDTASK_PATH \
    --data_path $DATA_PATH \