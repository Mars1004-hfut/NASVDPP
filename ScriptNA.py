import subprocess as sp

x = [1e-1]
y = [1e-1]
z = [1,0]
for xIdx in x:
    for yIdx in y:
        for zIdx in z:
            sp.call("python NASVDPP.py --dataset yelp-91 --pretrain " + str(zIdx) + " --verbose 1 --batch_" "choice user --epochs 500 \
            --weight_size 10 --embed_size 10 --data_alpha 0 --regs [1e-1,1e-0,1e-1,1e-7,1e-2,1e-3] --alpha 0.5 --train_loss 0 \
            --beta 0 --num_neg 4 --lr 0.1 --activation 0 --algorithm 0", shell=True)

            # + " + str(zIdx) + "," + str(yIdx) + "," + str(xIdx)" +
            # 1r 0.03