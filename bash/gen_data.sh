# python generate_fedtask.py --dataset mnist --dist 0 --skew 0 --num_clients 100
# python generate_fedtask.py --dataset mnist --dist 1 --skew 0.8 --num_clients 10
# python generate_fedtask.py --dataset mnist --dist 2 --skew 0.8 --num_clients 10
python generate_fedtask.py --dataset synthetic --dist 11 --skew 1 --num_clients 16 --num_samples 2400 --zipf_skew 1.6 --seed 1