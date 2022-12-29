FEDTASK='synthetic_classification_cnum10_dist11_skew1.0_seed0'
for N in 6 7 9
do
    echo $N
    python fill.py \
        --fedtask $FEDTASK \
        --exact_dir "exact-${N}"
done