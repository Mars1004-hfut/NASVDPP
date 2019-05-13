import subprocess as sp

x = [1e-5]
y = [1e-0]
z = [1e-0] # 0:0.9300  0.03/0.08:0.9260 0.1:0.9241 0.2:9201 0.3:0.9199 0.6:0.9193
for xIdx in x:
    for yIdx in y:
        for zIdx in z:
            sp.call("python SVDPP.py --dataset yelp-91 --pretrain 0 --verbose 1 --batch_choice user --epochs 500 --weight_size 80 --embed_size 80 --data_alpha 0 --regs [" + str(zIdx) + "," + str(yIdx) + "," + str(xIdx) + ",1e-5,1e-5,1e-5] --alpha 0.5 --train_loss 0 --beta 0.5 --num_neg 4 --lr 0.01 --activation 0 --algorithm 0", shell=True)

# " + str(xIdx) + " 1e-1 1e-1 1e-4