import random
import os
import numpy as np
good_curve_list = os.listdir('choosed_data/classification/good')
bad_curve_list = os.listdir('choosed_data/classification/bad')

total_list = np.array(good_curve_list+bad_curve_list)
total_number = len(total_list)
label = np.concatenate((np.zeros(len(good_curve_list),dtype=int),np.ones(len(bad_curve_list),dtype=int)))
rd = np.random.permutation(total_number)

sample_rate=0.9

test_list = total_list[rd[:int(sample_rate*total_number)]]
test_labels = label[rd[:int(sample_rate*total_number)]]

test_file = open('./choosed_data/classification/test.txt','w')
for test_file_name,test_label in zip(test_list,test_labels):
    test_file.write(test_file_name.split('.')[0]+' '+str(test_label)+'\n')
test_file.close()


train_list = total_list[rd[int(sample_rate*total_number):]]
train_labels = label[rd[int(sample_rate*total_number):]]
train_file = open('./choosed_data/classification/train.txt','w')
for train_file_name,train_label in zip(train_list,train_labels):
    train_file.write(train_file_name.split('.')[0]+' '+str(train_label)+'\n')
train_file.close()
