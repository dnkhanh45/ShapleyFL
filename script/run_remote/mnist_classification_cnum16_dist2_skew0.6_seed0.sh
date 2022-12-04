#!/bin/bash
#$ -cwd
#$ -l rt_G.small=1
#$ -l h_rt=36:00:00
#$ -o /home/aaa10078nj/Federated_Learning/Shapley_value/logs/$JOB_NAME_$JOB_ID.log
#$ -j y

source /etc/profile.d/modules.sh
module load gcc/11.2.0
module load openmpi/4.1.3
module load cuda/11.5/11.5.2
module load cudnn/8.3/8.3.3
module load nccl/2.11/2.11.4-1
module load python/3.10/3.10.4
source ~/venv/pytorch1.11+horovod/bin/activate

LOG_DIR="/home/aaa10078nj/Federated_Learning/Shapley_value/logs/cifar10/$JOB_NAME_$JOB_ID"
rm -r ${LOG_DIR}
mkdir ${LOG_DIR}

# GPU variables
GPU_IDS=( 0 1 )
NUM_THREADS=1
BATCH_SIZE=20

# Generate fedtask
python generate_fedtask.py \
    --benchmark mnist_classification \
    --dist 2 \
    --skew 0.6 \
    --num_clients 16 \
    --seed 0

# Run
TASK="mnist_classification_cnum16_dist2_skew0.6_seed0"

ROOT_PATH="$SGE_LOCALDIR/$JOB_ID/"
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
    --fedtask_path $FEDTASK_PATH \
    --data_path $DATA_PATH \