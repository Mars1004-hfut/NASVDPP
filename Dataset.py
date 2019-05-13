'''
Created on Aug 8, 2016
Processing datasets. 

@author: Xiangnan He (xiangnanhe@gmail.com)
'''
import scipy.sparse as sp
import numpy as np
import pandas as pd

class Dataset(object):
    '''
    Loading the data file
        trainMatrix: load rating records as sparse matrix for class Data
        trianList: load rating records as list to speed up user's feature retrieval
        testRatings: load leave-one-out rating test for class Evaluate
        testNegatives: sample the items not rated by user
    '''

    def __init__(self, path):
        '''
        Constructor
        '''
        self.trainMatrix = self.load_training_file_as_matrix(path + ".train.dat")
        self.trainList = self.load_training_file_as_list(path + ".train.dat")
        self.validRatings = self.load_rating_file_as_list(path + ".valid.dat")
        self.testRatings = self.load_rating_file_as_list(path + ".test.dat")
        self.num_users, self.num_items = self.trainMatrix.shape
        self.globalMean = self.mean(path + ".train.dat")
        self.features = self.load_features('features80.csv')
        # self.testNegatives = self.load_negative_file(path + ".test.negative")


    def load_rating_file_as_list(self, filename):
        ratingList = []
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                user, item, rating = int(arr[0]), int(arr[1]), float(arr[2])
                ratingList.append([user, item, rating])
                line = f.readline()
        return ratingList

    def load_training_file_as_matrix(self, filename):
        '''
        Read .rating file and Return dok matrix.
        The first line of .rating file is: num_users\t num_items
        '''
        # Get number of users and items
        num_users, num_items = 0, 0
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                u, i = int(arr[0]), int(arr[1])
                num_users = max(num_users, u)
                num_items = max(num_items, i)
                line = f.readline()
        # Construct matrix
        mat = sp.dok_matrix((num_users + 1, num_items + 1), dtype=np.float32)
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                user, item, rating = int(arr[0]), int(arr[1]), float(arr[2])
                if (rating > 0):
                    mat[user, item] = float(rating)
                    # mat[user, item] = 1.0
                    # print('[' + str(user) + ',' + str(item)+']')
                line = f.readline()
        print ("already load the trainMatrix...")
        return mat

    def load_training_file_as_list(self, filename):
        # Get all items rated by the corresponding user
        u_ = 0
        lists, items = [], []
        with open(filename, "r") as f:
            line = f.readline()
            index = 0
            while line != None and line != "":
                arr = line.split("\t")
                u, i = int(arr[0]), int(arr[1])
                if u_ < u:
                    index = 0
                    if u_ != 0:
                        lists.append(items)
                    items = []
                    u_ += 1
                index += 1
                # if index < 300:
                items.append(i)
                line = f.readline()
        lists.append(items)
        print ("already load the trainList...")
        return lists

    def mean(self,filename):
        with open(filename, "r") as f:
            line = f.readline()
            sum = 0
            count = 0
            while line != None and line != "":
                arr = line.split("\t")
                sum += float(arr[2])
                count += 1
                line = f.readline()
        return sum/count

    def load_features(self,filename):
        raw_feature = pd.read_csv(filename,header=None)
        zero_featrue = [list(np.zeros(shape=[80]))]
        zero_featrue.extend(raw_feature.values.tolist())
        return zero_featrue

    def load_negative_file(self, filename):
        negativeList = []
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                negatives = []
                for x in arr[1:]:
                    negatives.append(int(x))
                negativeList.append(negatives)
                line = f.readline()
        return negativeList

# if __name__ == '__main__':
#     dataset = Dataset('Data/ml-100k-91')
    # print(dataset.features)
    # print(dataset.trainMatrix.keys())
    # print(dataset.trainMatrix[2])
    # print('helloWorld!')