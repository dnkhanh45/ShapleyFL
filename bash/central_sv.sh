cd ..
python central_sv.py \
    --task synthetic_cnum16_dist12_skew1.0_seed1 \
    --model lr \
    --algorithm mp_fedavg \
    --num_epochs 4 \
    --batch_size 10 \
    --gpu 0 \
    --start 0 \
    --end 1