# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 21:39:42 2018

@author: wangpeng884112
"""

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Flatten
from keras import optimizers
from keras.callbacks import EarlyStopping, ModelCheckpoint
from scipy.stats import pearsonr
import numpy as np
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='The parameters for select highly-expressed promoters',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input_file', type=str, default='./seq/predicted_promoters.fa', help='Promoters will be predicted by predictors')
 
    # params
    args = parser.parse_args()
    return args

class PREDICT():   # Predict the sequence expression
    
    def __init__(self, file_input):  
        self.file = file_input
        self.model_weight = 'weight_CNN.h5'
        self.CNN_train_num = 10000
        self.shuffle_flag = 2
        
    def CNN_model(self, promoter_length):
        model = Sequential()
        model.add(
                Conv2D(100, (6, 1),
                padding='same',
                input_shape=(promoter_length, 1, 4))
                )
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 1)))
        model.add(Conv2D(200, (5, 1), padding='same'))
        model.add(Activation('relu'))
        model.add(Conv2D(200, (5, 1), padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 1)))
        model.add(Flatten())
        model.add(Dense(1024))
        model.add(Activation('relu'))
        model.add(Dense(1))
        return model

    def seq2onehot(self, seq):     # 将序列转换为one-hot编码
        module = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
        promoter_onehot = []
        for s in seq:
            tmp = []
            for item in s:
                if item == 't' or item == 'T':
                    tmp.append(module[0])
                elif item == 'c' or item == 'C':
                    tmp.append(module[1])
                elif item == 'g' or item == 'G':
                    tmp.append(module[2])
                elif item == 'a' or item == 'A':
                    tmp.append(module[3])
                else:
                    tmp.append([0,0,0,0])
            promoter_onehot.append(tmp)
        data = np.zeros((len(seq), 50, 1, 4), dtype=np.float32)
        for i in range(len(seq)):
            for j in range(len(seq[0])):
                data[i, j, 0, :] = promoter_onehot[i][j]
        return data
    
    def CNN_predict(self, seq_onehot):  # Using trained CNN model to predict the expression of promoters.
        model = self.CNN_model(len(seq_onehot[0]))
        model.load_weights(self.model_weight)
        batch_exp = model.predict(seq_onehot, verbose=0)
        return batch_exp   # return the expression of promoters in this batch
        
    def open_exp(self):    # open the first biological experimental result
        # predeal part, load the file
        with open('./seq/seq_exp_94.txt', 'r') as f:
            seq = []
            exp = []
            for item in f:
                item = item.split()
                seq.append(item[0][5:-1])
                exp.append(item[1])
        seq = seq[6:]
        exp = exp[6:]
        
        # transform the exp into array format
        expression = np.zeros((len(exp), 1))
        for i in range(len(exp)):
            expression[i] = float(exp[i])
        expression = np.log2(expression)
        return seq, expression
    
    def open_fa(self, file):
        record = []
        with open(file, 'r') as f:
            for item in f:
                if '>' not in item:
                    record.append(item.strip())
        return record
        
    def random_perm(self, seq, exp, shuffle_flag):  # random perm the sequence and expression data
        indices = np.arange(seq.shape[0])
        np.random.seed(shuffle_flag)
        np.random.shuffle(indices)
        seq = seq[indices]
        exp = exp[indices]
        return seq, exp
        
    def SVR_train(self):
        seq, expression = self.open_exp()
        data = self.seq2onehot(seq)
        data, expression = self.random_perm(data, expression, self.shuffle_flag)
        data = data.reshape(len(data), data.shape[1] * 4)
        from sklearn.svm import SVR  
        from sklearn.model_selection import GridSearchCV 
        svr = GridSearchCV(SVR(kernel='rbf', gamma=0.1), cv=5,  
                           param_grid={"C": [1e0, 1e1, 1e2, 1e3],  
                                       "gamma": np.logspace(-2, 2, 5)})  
        svr.fit(data, expression[:, 0])  
        return svr
    
    def CNN_train(self):
        promoter = np.load('./seq/promoter.npy')
        expression = np.load('./seq/gene_expression.npy')
        data = self.seq2onehot(promoter)
        
        expression_new = np.zeros((len(expression),))
        for i in range(len(expression)):
            expression_new[i] = float(expression[i])
        
        expression = np.log2(expression_new)
        
        data, expression = self.random_perm(data, expression, self.shuffle_flag + 1) # Used different shuffle flag from SVR_train
        # Split training/validation and testing set
        r = self.CNN_train_num
        train_feature = data[:r]
        test_feature = data[r:]
        train_label = expression[:r]
        test_label = expression[r:]
        # construct CNN model and training 
        model = self.CNN_model(50)
        sgd = optimizers.SGD(learning_rate=0.0005, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss='mean_squared_error', optimizer=sgd)
        model.fit(train_feature, train_label, epochs=1000, batch_size=128, validation_split=0.1, callbacks=[EarlyStopping(patience=10), ModelCheckpoint(filepath=self.model_weight, save_best_only=True)], shuffle=True)
        model.load_weights(self.model_weight)
        result = model.predict(test_feature, verbose=0)
        result = result[:, 0]
        cor_pearsonr = pearsonr(test_label, result)
        print(cor_pearsonr)
    
    def predict(self):
        seq = self.open_fa(self.file)
        seq_onehot = self.seq2onehot(seq)
        
        # Predict by SVR
        svr = self.SVR_train()
        seq_svr = seq_onehot.reshape(len(seq_onehot), seq_onehot.shape[1] * 4)
        exp_SVR = svr.predict(seq_svr)
        exp_SVR = exp_SVR / np.max(exp_SVR)
        
        # Predict by CNN
        self.CNN_train()
        exp_CNN = self.CNN_predict(seq_onehot)
        exp_CNN = exp_CNN / np.max(exp_CNN)
        
        return seq, exp_SVR, exp_CNN

if __name__ == '__main__':
    args = parse_args()
    input_file = args.input_file
    
    predictor = PREDICT(input_file)
    
    seq, exp_SVR, exp_CNN = predictor.predict()
    
    with open('seq_exp_CNN.txt', 'w') as f:
        for i in range(len(seq)):
            f.write(seq[i] + '   ' + str(round(exp_CNN[i], 5)) + '\n')
    
    with open('seq_exp_SVR.txt', 'w') as f:
        for i in range(len(seq)):
            f.write(seq[i] + '   ' + str(round(exp_SVR[i], 5)).strip('[').strip(']') + '\n')
























