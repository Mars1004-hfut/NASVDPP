import subprocess as sp

x = [1e-7]
y = [1e-5]
z = [1,0]
# w = 1
# for i in range(8):
#     z.append(w)
#     w = w/10
for xIdx in x:
    for yIdx in y:
        for zIdx in z:
            sp.call("python DeepNASVDPP.py --dataset yelp-91 --pretrain " +str(zIdx) + " --verbose 1 --batch_choice user --epochs 500 --weight_size 10 \
             --embed_size 10 --data_alpha 0 --regs [1e-7,1e-5,1e-0,1e-5,1e-4,1e-6] --alpha 0.5 --train_loss 0 --beta 0 --num_neg 4 \
             --lr 0.0001 --activation 0 --algorithm 0 --layers [40,20,10] --batch_norm 0 --reg_W [1e-4,1e-5,1e-6]",
                    shell=True)
# 1r 0.0001 0.01
# " + str(yIdx) + "," + str(yIdx/10) + "," + str(yIdx/100) + "
# " + str(xIdx) + "," + str(yIdx) + "," + str(zIdx) + ", # 1e-7,1e-5,1e-3