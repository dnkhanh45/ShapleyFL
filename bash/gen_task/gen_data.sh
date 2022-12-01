# python generate_fedtask.py --dataset mnist --dist 0 --skew 0 --num_clients 100
# python generate_fedtask.py --dataset mnist --dist 1 --skew 0.8 --num_clients 10
# python generate_fedtask.py --dataset mnist --dist 2 --skew 0.8 --num_clients 10
# python generate_fedtask.py --dataset synthetic --dist 12 --skew 1 --num_clients 16 --num_samples 2400 --zipf_skew 1.6 --seed 1
python generate_fedtask.py --dataset mnist --dist 6 --skew 0.6 --num_clients 16 --zipf_skew 1.6 --seed 0
python generate_fedtask.py --dataset cifar10 --dist 6 --skew 0.6 --num_clients 16 --zipf_skew 1.6 --seed 0
python generate_fedtask.py --dataset mnist --dist 2 --skew 0.6 --num_clients 16 --zipf_skew 1.6 --seed 0
python generate_fedtask.py --dataset cifar10 --dist 2 --skew 0.6 --num_clients 16 --zipf_skew 1.6 --seed 0
# python generate_fedtask.py --dataset fashion_mnist --dist 6 --skew 0.5 --num_clients 16 --zipf_skew 1.6 --seed 0