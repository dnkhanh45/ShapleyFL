#!/bin/bash
source /etc/profile.d/modules.sh
module load gcc/11.2.0
module load openmpi/4.1.3
module load cuda/11.5/11.5.2
module load cudnn/8.3/8.3.3
module load nccl/2.11/2.11.4-1
module load python/3.10/3.10.4
source ~/venv/pytorch1.11+horovod/bin/activate

python generate_fedtask.py --benchmark cifar10_classification --dist 2 --skew 0.6 --num_clients 16 --seed 0
python generate_fedtask.py --benchmark cifar10_classification --dist 6 --skew 0.6 --num_clients 16 --seed 0
python generate_fedtask.py --benchmark cifar10_classification --dist 0 --skew 0 --num_clients 50 --seed 0
python generate_fedtask.py --benchmark cifar10_classification --dist 2 --skew 0.6 --num_clients 50 --seed 0
python generate_fedtask.py --benchmark mnist_classification --dist 2 --skew 0.6 --num_clients 16 --seed 0
python generate_fedtask.py --benchmark mnist_classification --dist 6 --skew 0.6 --num_clients 16 --seed 0
python generate_fedtask.py --benchmark mnist_classification --dist 0 --skew 0 --num_clients 50 --seed 0
python generate_fedtask.py --benchmark mnist_classification --dist 2 --skew 0.6 --num_clients 50 --seed 0