#!/bin/bash
source /etc/profile.d/modules.sh
module load gcc/11.2.0
module load openmpi/4.1.3
module load cuda/11.5/11.5.2
module load cudnn/8.3/8.3.3
module load nccl/2.11/2.11.4-1
module load python/3.10/3.10.4
source ~/venv/pytorch1.11+horovod/bin/activate

TASK="cifar10_cnum50_dist0_skew0_seed0"
#Dataset
ROOT_DIR="$SGE_LOCALDIR/$JOB_ID/"
FEDTASK_DIR="$ROOT_DIR/fedtask"
mkdir $FEDTASK_DIR
# cp -r ./benchmark/cifar10/data ${DATA_DIR}
cp -r ./fedtask/$TASK ${FEDTASK_DIR}

python main.py --task $TASK --fedtask_folder $FEDTASK_DIR --model resnet18 --algorithm mp_fedavg --num_rounds 400 --num_epochs 5 --learning_rate 0.01 --lr_scheduler 0 --learning_rate_decay 0.998 --proportion 0.8 --batch_size 20 --eval_interval 1 --gpu 0 --num_threads_per_gpu 1 --log_folder "./chkpts"