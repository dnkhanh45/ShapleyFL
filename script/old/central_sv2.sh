python central_sv.py \
    --task mnist_cnum6_dist2_skew0.8_seed0 \
    --model mlp \
    --algorithm mp_fedavg \
    --num_epochs 250\
    --batch_size 10 \
    --gpu 1 \
    --start 50 \
    --end 51