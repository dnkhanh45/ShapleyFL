TASK="synthetic_classification"
DIST=12
SKEW=1.0
NUM_CLIENTS=300
SEED=0
python generate_fedtask.py --benchmark $TASK --dist $DIST --skew $SKEW --num_clients $NUM_CLIENTS --seed $SEED