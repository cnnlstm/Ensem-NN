import numpy as np
import lmdb
import threading
import h5py
np.random.seed(seed=1234)
import os











data_root = "/data2/nturgbd/relative/relative_17_1/"
train_list = "/home/siat/xyy/3d/cross_view/view_train_video_list.txt"

scale = 1


f = open(train_list,"r")  
lines_ = f.readlines()
train_lines = []
for i in range(len(lines_)//scale):
  train_lines.append(lines_[i*scale])
train_lines = sorted(train_lines)
np.random.shuffle(train_lines)
len_train = len(train_lines)
print "successfully,len(train_list)",len_train



# h = open(test_list,"r")  
# test_lines_ = h.readlines()
# test_lines = [] 
# for i in range(len(test_lines_)//scale):
#   test_lines.append(test_lines_[i*scale])
# test_lines = sorted(test_lines)
# len_test = len(test_lines)
# print "successfully,len(test_list)",len_test

def read_h5(path):
    #print path
    g = h5py.File(path)
    data = g['data']
    data = data[:]
    label = g['label']
    label = label[:]
    return data,label

def compute_dataset_mean():
    x_yed = []
    total_sum = 0
    total_count = 0
    sum_file = []
    #while 1:
    file_num = 0
    for line in train_lines:
        data_1,label = read_h5(data_root+line[:-1]+".h5")
        nonzeros = np.where(np.array([np.sum(data_1[i])>0 for i in range(0,data_1.shape[0])])==False)[0]

        if len(nonzeros) == 0:
             continue

        last_time = nonzeros[0]
        total_count += last_time
        total_sum += np.sum(data_1[:last_time])
        file_num = file_num+1
        print(file_num)
    final_mean = total_sum / total_count
    final_mean = final_mean / 144
    print(total_count)
    print(final_mean)
compute_dataset_mean()
