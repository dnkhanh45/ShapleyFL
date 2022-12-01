#!/bin/bash

# step 1: generate fedtask
python generate_fedtask.py --dataset mnist --dist 2 --skew 0.6 --num_clients 16 --zipf_skew 1.6 --seed 0
# step 2: run experiment to get client's weight per round
python main.py --task mnist_cnum16_dist2_skew0.6_seed0 --model mlp --algorithm mp_fedavg --num_rounds 200 --num_epochs 5 --learning_rate 0.01 --lr_scheduler 0 --learning_rate_decay 0.998 --proportion 1 --batch_size 10 --eval_interval 1 --gpu 1 --num_threads_per_gpu 1 
# step 3: calculate shapley value in 3 cases: const lambda, optimal lambda and exact
python sv_main.py --task mnist_cnum16_dist2_skew0.6_seed0 --model mlp --algorithm mp_fedavg --num_rounds 200 --proportion 1 --gpu 1 --start 1 --end 201 --method const_lambda --log_folder "sv_result"
python sv_main.py --task mnist_cnum16_dist2_skew0.6_seed0 --model mlp --algorithm mp_fedavg --num_rounds 200 --proportion 1 --gpu 1 --start 1 --end 201 --method optimal_lambda --log_folder "sv_result"
python sv_main.py --task mnist_cnum16_dist2_skew0.6_seed0 --model mlp --algorithm mp_fedavg --num_rounds 200 --proportion 1 --gpu 1 --start 1 --end 201 --method exact --log_folder "sv_result"

# TASK_NAME="mnist_cnum16_dist2_skew0.6_seed0"
# rm -r chkpts/$TASK_NAME