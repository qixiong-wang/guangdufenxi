import numpy as np
import torch
import torch.utils.data as data
import torch.nn.init
from model import *
import pandas as pd
import os
import matplotlib.pyplot as plt


import random

data_root='choosed_data/alldata'
test_txt_path =  'choosed_data/classification/test.txt'
visualization_dir = 'visualization/'
window_size = 1
fs_ = 20


def biliner_interpolation(curve):
    nonzero_idx = np.array(np.where(curve != 0))[0]
    max_idx = np.max(nonzero_idx)
    min_idx = np.min(nonzero_idx)
    for i in range(min_idx, max_idx):
        if curve[i] == 0:
            nearest_idx1 = nonzero_idx[np.argsort(np.abs(nonzero_idx - i))[0]]
            if nearest_idx1 < i:
                nearest_idx2 = nonzero_idx[np.argsort(np.abs(nonzero_idx - i))[0] + 1]
            else:
                nearest_idx2 = nonzero_idx[np.argsort(np.abs(nonzero_idx - i))[0] - 1]
            distance = np.abs(nearest_idx2 - nearest_idx1)
            curve[i] = np.abs(i - nearest_idx1) / distance * curve[nearest_idx1] + np.abs(nearest_idx2 - i) / distance * \
                       curve[nearest_idx2]
            pass
    return curve

def random_drop(curve):
    nonzero_idx = np.array(np.where(curve[0] != 0))[0]
    max_idx = np.max(nonzero_idx)
    min_idx = np.min(nonzero_idx)
    drop_length = int((max_idx-min_idx)*0.2)
    start_point = random.randint(min_idx,max_idx)
    curve[:,start_point:start_point+drop_length]=0
    return curve



def preprocess(phase, mag, window_size = 0.1, fs=20):
    new_data = np.zeros((2, 180 * fs))
    mean_mag = np.zeros((180 * fs))
    std_mag = np.zeros((180 * fs))

    ## analysis feature in a phase window
    for i in range(180 * fs):
        phase_i = i / (fs)
        mag_in_window = mag[np.multiply(phase < phase_i + window_size, phase > phase_i - window_size)]
        num_points = len(mag_in_window)
        if num_points == 0:
            mean_mag[i] = 0
            std_mag[i] = 0
        else:
            mean_mag[i] = np.mean(mag_in_window)
            std_mag[i] = np.std(mag_in_window)

    ## culculate gradiant for curve
    epsilon = 0.0001
    mag_gradient_abs = np.abs((mag[1:] - mag[:-1]) / (phase[1:] - phase[:-1] + epsilon))
    mean_mag_gradient_abs = np.zeros((180 * fs))

    for i in range(180 * fs):
        phase_i = i / (fs)
        mag_gradients_in_window = mag_gradient_abs[
            np.multiply(phase[:-1] < phase_i + window_size, phase[:-1] > phase_i - window_size)]
        num_points = len(mag_gradients_in_window)

        if num_points == 0:
            mean_mag_gradient_abs[i] = 0
        else:
            mean_mag_gradient_abs[i] = np.mean(mag_gradients_in_window)


    plt.scatter(phase,mag,c='red',marker='d')
    plt.xlim(0,180)
    plt.show()

    x = np.linspace(0, 180, 3600)
    plt.plot(x,mean_mag)
    plt.xlim(0, 180)
    plt.show()
    mean_mag = biliner_interpolation(mean_mag)
    std_mag = biliner_interpolation(std_mag)

    plt.plot(x,mean_mag)
    plt.xlim(0, 180)
    plt.show()
    plt.plot(x,std_mag)
    plt.xlim(0, 180)
    plt.show()

    new_data[0] = mean_mag
    new_data[1] = std_mag
    new_data = random_drop(new_data)
    plt.plot(x,new_data[0])
    plt.xlim(0, 180)
    plt.show()
    plt.plot(x,new_data[1])
    plt.xlim(0, 180)
    plt.show()
    # new_data[2] = mean_mag_gradient_abs
    new_data = np.reshape(new_data, (2, 60, 60))
    return new_data

with open(test_txt_path) as f:
    for line in f.readlines():
        filename = line.split(' ')[0] + '.xlsx'
        png_save_path = os.path.join(visualization_dir,line.split(' ')[0])
        filename = os.path.join(data_root, filename)
        df = pd.read_excel(filename)
        phase = df.iloc[:, 13].values
        mag = df.iloc[:, 9].values
        ### normalization
        mag = (mag - np.min(mag)) / (np.max(mag) - np.min(mag))
        ### remove redundence
        descend = phase[1] - phase[0] < 0
        if descend:  # 递减
            cut_index = np.argmin(phase)
        else:  # 递增
            cut_index = np.argmax(phase)
        if cut_index > phase.__len__() / 2:
            phase = phase[0:cut_index]
            mag = mag[0:cut_index]
        else:
            phase = phase[cut_index:]
            mag = mag[cut_index:]

        processed_data = preprocess(phase, mag, window_size, fs_)  # fs为每个角度切成多少份
        processed_data = processed_data.astype(np.float32)

