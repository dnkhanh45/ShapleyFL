algorithm = ['mp_fedkdr', 'mp_fedkdrv2']
dataset = ['mnist']
model = ['cnn']
cnum = [10]
dist = [1,2,3]
skew = [0.2, 0.4, 0.6, 0.8]
round = [50]
epoch = [4]
batch = [4]
proportion = [0.1]

with open("ex.sh", "w") as file:
    for data in dataset:
        file.write(f"#{data}\n")
        for m in model:
            for c in cnum:
                for d in dist:
                    file.write(f"#dist {d}\n")
                    for s in skew:
                        for r in round:
                            for e in epoch:
                                for b in batch:
                                    for p in proportion:
                                        for a in algorithm:
                                            task = f"{data}_cnum{c}_dist{d}_skew{s}_seed0"
                                            file.write(f"CUDA_VISIBLE_DEVICES=0,1 python main.py --task {task} --model {m} --algorithm {a} --num_rounds {r} --num_epochs {e} --learning_rate 0.001 --proportion {p} --batch_size {b} --eval_interval 1 --num_threads_per_gpu 1  --num_gpus 2 --server_gpu_id 0\n")
