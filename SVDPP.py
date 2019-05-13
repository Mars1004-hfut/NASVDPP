'''
Created on Oct 30, 2017

@author: Jiandong-XU
'''
from __future__ import absolute_import
from __future__ import division

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
import math as mt
import logging
from time import time
from time import strftime
from time import localtime
from Dataset import Dataset
from DatasetAE import DatasetAE
from utils import early_stop
import Batch_gen as data
import Evaluate as evaluate

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Run NAIS.")
    parser.add_argument('--path', nargs='?', default='Data/',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='pinterest-20',
                        help='Choose a dataset.')
    parser.add_argument('--pretrain', type=int, default=1,
                        help='0: No pretrain, 1: Pretrain with updating FISM variables, 2:Pretrain with fixed FISM variables.')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Interval of evaluation.')
    parser.add_argument('--batch_choice', nargs='?', default='user',
                        help='user: generate batches by user, fixed:batch_size: generate batches by batch size')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs.')
    parser.add_argument('--weight_size', type=int, default=16,
                        help='weight size.')
    parser.add_argument('--embed_size', type=int, default=16,
                        help='Embedding size.')
    parser.add_argument('--data_alpha', type=float, default=0,
                        help='Index of coefficient of embedding vector')
    parser.add_argument('--regs', nargs='?', default='[1e-7,1e-7,0]',
                        help='Regularization for user and item embeddings.')
    parser.add_argument('--alpha', type=float, default=0,
                        help='Index of coefficient of embedding vector')
    parser.add_argument('--train_loss', type=float, default=1,
                        help='Caculate training loss or nor')
    parser.add_argument('--beta', type=float, default=0.5,
                        help='Index of coefficient of sum of exp(A)')
    parser.add_argument('--num_neg', type=int, default=4,
                        help='Number of negative instances to pair with a positive instance.')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate.')
    parser.add_argument('--activation', type=int, default=0,
                        help='Activation for ReLU, sigmoid, tanh.')
    parser.add_argument('--algorithm', type=int, default=0,
                        help='0 for NAIS_prod, 1 for NAIS_concat')
    return parser.parse_args()

class SVDPP:

    def __init__(self, num_users, num_items, args):
        self.pretrain = args.pretrain
        self.num_users = num_users
        self.num_items = num_items
        self.dataset_name = args.dataset
        self.learning_rate = args.lr
        self.embedding_size = args.embed_size
        self.weight_size = args.weight_size
        self.alpha = args.alpha
        self.beta = args.beta
        self.data_alpha = args.data_alpha
        self.verbose = args.verbose
        self.activation = args.activation
        self.algorithm = args.algorithm
        self.batch_choice = args.batch_choice
        regs = eval(args.regs)
        self.alpha_bilinear = regs[0]
        self.beta_bilinear = regs[1]
        self.gamma_bilinear = regs[2]
        self.eta_bilinear = regs[3]
        self.eta_bilinear1 = regs[4]
        self.t = regs[5]
        self.train_loss = args.train_loss

    def _create_placeholders(self):
        with tf.name_scope("input_data"):
            self.user_input = tf.placeholder(tf.int32, shape=[None, 1])     #the index of users
            self.user_hist = tf.placeholder(tf.int32, shape=[None, None])   #the index of users' historical items
            self.num_idx = tf.placeholder(tf.float32, shape=[None, 1])      #the number of items rated by users
            self.item_input = tf.placeholder(tf.int32, shape=[None, 1])     #the index of items
            self.labels = tf.placeholder(tf.float32, shape=[None, 1])        #the ground truth

    def _create_variables(self):
        with tf.name_scope("embedding"):  # The embedding initialization is unknown now
            trainable_flag = (self.pretrain != 2)
            self.c1 = tf.Variable(tf.truncated_normal(shape=[self.num_items, self.embedding_size], mean=0.0, stddev=0.01), name='c1', dtype=tf.float32, trainable=trainable_flag)
            self.c2 = tf.constant(0.0, tf.float32, [1, self.embedding_size], name='c2')
            self.embedding_Y_ = tf.concat([self.c1, self.c2], 0, name='embedding_Y_')
            self.embedding_P  = tf.Variable(tf.truncated_normal(shape=[self.num_users, self.embedding_size], mean=0.0, stddev=0.02), name='embedding_P', dtype=tf.float32, trainable=trainable_flag)
            self.embedding_Q  = tf.Variable(tf.truncated_normal(shape=[self.num_items, self.embedding_size], mean=0.0, stddev=0.02), name='embedding_Q', dtype=tf.float32, trainable=trainable_flag)
            self.embedding_Q1 = tf.Variable(dataset.features, name='embedding_Q1', dtype=tf.float32, trainable=False)
            # bias
            self.biases_u = tf.Variable(tf.random_uniform([self.num_users, 1], 0.0, 0.0), name='biases_u', trainable=trainable_flag)
            self.biases_i = tf.Variable(tf.random_normal([self.num_items, 1], 0.0, 0.0),  name='biases_i', trainable=trainable_flag)
            self.biases_global = tf.Variable(tf.constant(dataset.globalMean), name='biases_global', trainable=trainable_flag)   # fixme

            # Variables for attention
            if self.algorithm == 0:
                self.W = tf.Variable(tf.truncated_normal(shape=[self.embedding_size, self.weight_size], mean=0.0, stddev=tf.sqrt(tf.div(2.0, self.weight_size + self.embedding_size))),name='Weights_for_MLP', dtype=tf.float32, trainable=True)
            else:
                self.W = tf.Variable(tf.truncated_normal(shape=[2*self.embedding_size, self.weight_size], mean=0.0, stddev=tf.sqrt(tf.div(2.0, self.weight_size + (2*self.embedding_size)))),name='Weights_for_MLP', dtype=tf.float32, trainable=True)
            self.b = tf.Variable(tf.truncated_normal(shape=[1, self.weight_size], mean=0.0, stddev=tf.sqrt(tf.div(2.0, self.weight_size + self.embedding_size))),name='Bias_for_MLP', dtype=tf.float32, trainable=True)
            self.h = tf.Variable(tf.ones([self.weight_size, 1]), name='H_for_MLP', dtype=tf.float32)
            self.W_AE_1 = tf.Variable(tf.random_normal([943, 10], mean=0, stddev=np.sqrt(2.0 / (943+10))),
                                  name='Weights_for_AE1', dtype=tf.float32, trainable=True)
            self.b_AE_1 = tf.Variable(tf.truncated_normal(shape=[1, 10], mean=0.0, stddev=np.sqrt(2.0 / (1+10))),
                                  name='Bias_for_AE1', dtype=tf.float32, trainable=True)

    def _create_inference(self):
        with tf.name_scope("inference"):
            self.embedding_y_ = tf.nn.embedding_lookup(self.embedding_Y_, self.user_hist)  # (b, n, e)
            self.embedding_p  = tf.nn.embedding_lookup(self.embedding_P,  self.user_input) # (b, 1, e)  select user
            self.embedding_q  = tf.nn.embedding_lookup(self.embedding_Q,  self.item_input) # (b, 1, e)  select item
            self.embedding_q1 = tf.nn.embedding_lookup(self.embedding_Q1, self.item_input)


            self.embedding_y = tf.reduce_sum(self.embedding_y_, 1)
            self.embedding_p = tf.reduce_sum(self.embedding_p, 1) # pu
            self.embedding_q = tf.reduce_sum(self.embedding_q, 1) # qi
            self.embedding_q1 = tf.reduce_sum(self.embedding_q1, 1)  # qi


            self.bias_u = tf.reduce_sum(tf.nn.embedding_lookup(self.biases_u, self.user_input), 1) #bu
            self.bias_i = tf.reduce_sum(tf.nn.embedding_lookup(self.biases_i, self.item_input), 1) #bi
            # self.bias_global = self.biases_global * tf.ones_like(self.labels)    #bg
            self.bias_global = tf.cast(self.biases_global, tf.float32) * tf.ones_like(self.labels)  # bg  todo

            self.coeff = tf.pow(self.num_idx+1, -tf.constant(self.alpha, tf.float32, [1]))  # + 0.1*self.embedding_q1
            inner_product = tf.reduce_sum(tf.multiply((self.embedding_q), (self.embedding_p + self.coeff*self.embedding_y)), 1, keep_dims=True)  # fixme
            # inner_product = tf.reduce_sum(tf.multiply(self.embedding_q,self.embedding_p),1, keep_dims=True)
            self.output = tf.add_n([inner_product, self.bias_u, self.bias_i, self.bias_global])   #fixme 0.06

    def _create_loss(self):
        with tf.name_scope("loss"):

            self.loss = tf.nn.l2_loss(tf.subtract(self.output, self.labels)) + \
                        self.alpha_bilinear * tf.nn.l2_loss(self.embedding_p) + \
                        self.beta_bilinear * tf.nn.l2_loss(self.embedding_q) + \
                        self.eta_bilinear * tf.nn.l2_loss(self.bias_u) + \
                        self.eta_bilinear1 * tf.nn.l2_loss(self.bias_i) + \
                        self.gamma_bilinear * tf.nn.l2_loss(self.embedding_y)


    def _create_optimizer(self):
        with tf.name_scope("optimizer"):
            self.optimizer = tf.train.AdagradOptimizer(learning_rate=self.learning_rate, initial_accumulator_value=1e-8).minimize(self.loss)

    def build_graph(self):
        self._create_placeholders()
        self._create_variables()
        self._create_inference()
        self._create_loss()
        self._create_optimizer()
        logging.info("already build the computing graph...")

def training(flag, model, dataset, epochs):
    
    saver = tf.train.Saver()

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())  # initialized NN
        logging.info("initialized")
        print("initialized")

        #initialize for training batches
        batch_begin = time()
        batches = data.shuffle(dataset, model.batch_choice)
        batch_time = time() - batch_begin
        num_batch = len(batches[1])  # 6040
        batch_index = list(range(num_batch))  #todo

        #initialize the evaluation feed_dicts
        validDict, testDict = evaluate.init_evaluate_model(model, sess, dataset.validRatings, dataset.testRatings, dataset.trainList)

        best_valid = 100000
        last_valid = 10000
        validCount = 0
        #train by epoch
        for epoch_count in range(epochs):

            train_begin = time()
            training_batch(batch_index, model, sess, batches)  #todo
            train_time = time() - train_begin

            if epoch_count % model.verbose == 0:
                
                if model.train_loss:
                    loss_begin = time()
                    train_loss = training_loss(model, sess, batches)
                    loss_time = time() - loss_begin
                else:
                    loss_time, train_loss = 0, 0  
              
                eval_begin = time()
                rmse_valid = evaluate.eval(model, sess, dataset.validRatings, validDict)
                rmse_test = 0
                # rmse_test  = evaluate.eval(model, sess, dataset.testRatings,  testDict)

                if(rmse_valid > last_valid): #if rmse have incresed then stop
                    validCount = validCount + 1
                else:
                    validCount = 0
                if(validCount > 2):
                    print(str(model.embedding_size) + "," + str(mt.log10(model.alpha_bilinear)) + "," + str(mt.log10(model.beta_bilinear)) + ","
                          + str(mt.log10(model.gamma_bilinear)) + "," + str(mt.log10(model.eta_bilinear)) + ","
                          + str(mt.log10(model.eta_bilinear1)) + "," + str(model.t))
                    exit()

                last_valid = rmse_valid

                if rmse_valid < best_valid:
                    best_valid = rmse_valid
                    saver.save(sess, 'PretrainSVDPP/%s/valid_rmse_%.4f.ckpt' % (model.dataset_name, rmse_valid))

                eval_time = time() - eval_begin
                valids = []
                valids.append(rmse_valid)
                logging.info(
                    "Epoch %d [%.1fs + %.1fs]: valid_rmse = %.4f, test_rmse = %.4f, [%.1fs] train_loss = %.4f [%.1fs]" % (
                    epoch_count, batch_time, train_time, rmse_valid, rmse_test, eval_time, train_loss, loss_time))
                print("Epoch %d [%.1fs + %.1fs]: valid_rmse = %.4f, test_rmse = %.4f, [%.1fs] train_loss = %.4f [%.1fs]" % (
                    epoch_count, batch_time, train_time, rmse_valid, rmse_test, eval_time, train_loss, loss_time))

                with open('log/%s/%d,,,,%d,,,,%d,,,,%d,,,,%d-%d.txt' % (model.dataset_name, mt.log10(model.alpha_bilinear), mt.log10(model.beta_bilinear), mt.log10(model.gamma_bilinear),
                                                         mt.log10(model.eta_bilinear), mt.log10(model.eta_bilinear1), model.embedding_size
                                                         ), 'a') as flog:
                    flog.write("Epoch %d [%.1fs + %.1fs]: valid_rmse = %.4f, test_rmse = %.4f, [%.1fs] train_loss = %.4f [%.1fs]\n" % (
                    epoch_count, batch_time, train_time, rmse_valid, rmse_test, eval_time, train_loss, loss_time))
                flog.close()

                if early_stop(valids) == True:
                    break;

            batch_begin = time()
            batches = data.shuffle(dataset, model.batch_choice)
            np.random.shuffle(batch_index)
            batch_time = time() - batch_begin


def  training_batch(batch_index, model, sess, batches):
    for index in batch_index:
        user_input, user_hist, num_idx, item_input, labels = data.batch_gen(batches, index)  # index(0-6039)
        feed_dict = {model.user_input: user_input[:, None], model.user_hist: user_hist, model.num_idx: num_idx[:, None],
                     model.item_input: item_input[:, None], model.labels: labels[:, None]}
        sess.run([model.loss, model.optimizer], feed_dict)


def training_loss(model, sess, batches):
    train_loss = 0.0
    num_batch = len(batches[1])
    for index in range(num_batch):
        user_input, user_hist, num_idx, item_input, labels = data.batch_gen(batches, index)
        feed_dict = {model.user_input: user_input[:, None], model.user_hist: user_hist, model.num_idx: num_idx[:, None], 
                     model.item_input: item_input[:, None], model.labels: labels[:, None]}

        train_loss += sess.run(model.loss, feed_dict)
    return train_loss / num_batch

if __name__=='__main__':

    args = parse_args()
    regs = eval(args.regs)
    
    #logging.basicConfig(filename="Log/%s/log_beta%.2f_pre%dweight_size%d_reg%.7f_%s" %(args.dataset, args.beta, args.pretrain, args.weight_size, regs[2], strftime('%Y-%m-%d%H:%M:%S', localtime())), level = logging.INFO)
    
    if args.algorithm == 0:
        logging.info("begin training NAIS_prod model ......")
    else:
        logging.info("begin training NAIS_concat model ......")

    logging.info("dataset:%s  pretrain:%d   weight_size:%d  embedding_size:%d" 
                % (args.dataset, args.pretrain,  args.weight_size, args.embed_size))
    logging.info("regs:%.8f, %.8f, %.8f, %.8f  beta:%.1f  learning_rate:%.4f  train_loss:%d  activation:%d" 
                % (regs[0], regs[1], regs[2], regs[3], args.beta, args.lr, args.train_loss, args.activation)) 

    dataset = Dataset(args.path + args.dataset)
    model = SVDPP(dataset.num_users, dataset.num_items, args)
    model.build_graph()
    training(args.pretrain, model, dataset, args.epochs)
