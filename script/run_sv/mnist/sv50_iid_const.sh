#!/bin/bash
source /etc/profile.d/modules.sh
module load gcc/11.2.0
module load openmpi/4.1.3
module load cuda/11.5/11.5.2
module load cudnn/8.3/8.3.3
module load nccl/2.11/2.11.4-1
module load python/3.10/3.10.4
source ~/venv/pytorch1.11+horovod/bin/activate

TASK="mnist_cnum50_dist0_skew0_seed0"
#Dataset
ROOT_DIR="$SGE_LOCALDIR/$JOB_ID/"
FEDTASK_DIR="$ROOT_DIR/fedtask"
mkdir $FEDTASK_DIR
# cp -r ./benchmark/cifar10/data ${DATA_DIR}
cp -r ./fedtask/$TASK ${FEDTASK_DIR}

python sv_main.py --task $TASK --fedtask_folder $FEDTASK_DIR --model mlp --algorithm mp_fedavg --num_rounds 400 --proportion 0.8 --gpu 0 --start 1 --end 401 --method const_lambda --log_folder "sv_result"
