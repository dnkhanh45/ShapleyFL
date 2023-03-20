python central_sv.py \
    --task mnist_classification_cnum4_dist2_skew0.7_seed0 \
    --model mlp \
    --algorithm sv_fedavg \
    --num_epochs 10 \
    --batch_size 10 \
    --gpu 1 \
    --start 0 \
    --end -1