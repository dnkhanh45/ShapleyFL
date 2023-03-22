#!/bin/bash
#$ -cwd
#$ -l rt_G.small=1
#$ -l h_rt=36:00:00
#$ -o /home/aaa10078nj/Federated_Learning/Khanh_SV_FL/logs/cifar100/$JOB_NAME_$JOB_ID.log
#$ -j y

source /etc/profile.d/modules.sh
module load gcc/11.2.0
module load openmpi/4.1.3
module load cuda/11.5/11.5.2
module load cudnn/8.3/8.3.3
module load nccl/2.11/2.11.4-1
module load python/3.10/3.10.4
source ~/venv/pytorch1.11+horovod/bin/activate
​
LOG_DIR="/home/aaa10078nj/Federated_Learning/Khanh_SV_FL/logs/cifar100/$JOB_NAME_$JOB_ID"
rm -r ${LOG_DIR}
mkdir ${LOG_DIR}

# #Dataset
DATA_DIR="$SGE_LOCALDIR/$JOB_ID/"
cp -r ./ShapleyFL/benchmark/RAW_DATA/CIFAR10 ${DATA_DIR}
cd ./ShapleyFL
# DATA_DIR=benchmark/RAW_DATA/CIFAR100

mkdir ./fedtask  #### Should add the save directory to be the option of generate_fedtask. And using this directory to be the input of main.py

TASK="cifar100_classification"
DIST=1
SKEW=0.5
NUM_CLIENTS=10
SEED=0
python generate_fedtask.py --benchmark $TASK --dist $DIST --skew $SKEW --num_clients $NUM_CLIENTS --seed $SEED

TASK="${TASK}_cnum${NUM_CLIENTS}_dist${DIST}_skew${SKEW}_seed${SEED}"
GPU_IDS=( 0 )
NUM_THREADS=1
BATCH_SIZE=64
NUM_ROUNDS=50
PROPORTION=1.0
    
python main_ideal.py \
    --task $TASK \
    --model resnet18 \
    --algorithm fedavg \
    --num_rounds $NUM_ROUNDS \
    --num_epochs 5 \
    --learning_rate 0.01 \
    --lr_scheduler 0 \
    --learning_rate_decay 1.0 \
    --proportion 1 \
    --batch_size $BATCH_SIZE \
    --eval_interval 1 \
    --gpu $GPU_IDS \
    --num_threads $NUM_THREADS \
    --aggregate weighted_scale \
    --sample full \
    --start 0 \
    --end 200 \
    --data_path $DATA_DIR \
    --fedtask_path fedtask \
    --log_folder $LOG_DIR