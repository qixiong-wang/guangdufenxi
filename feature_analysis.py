
# imports and stuff
import numpy as np
from skimage import io
from glob import glob
from tqdm import tqdm_notebook as tqdm
from sklearn.metrics import confusion_matrix
import random
import itertools
# Matplotlib
import matplotlib.pyplot as plt
# Torch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
import torch.optim.lr_scheduler
import torch.nn.init
import pandas as pd
from torch.autograd import Variable
from torch.backends import cudnn
import cv2
import os

fpath = os.path.join(os.curdir, 'data', 'train')

for f in os.listdir(fpath):

    labelList_0 = ['36287', '39157', '38352']  # 工作
    labelList_1 = ['34779']  # 异常
    lableString = f[2:7]

    if lableString in labelList_0:
        label = 0
        df = pd.read_excel(os.path.join(fpath, f))
        phase = df.iloc[:, 13].values
        mag = df.iloc[:, 9].values
        mag = (mag - np.min(mag)) / (np.max(mag) - np.min(mag))
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
        plt.plot(mag)
        plt.show()
        plt.plot(phase,mag)
        plt.show()
        print(1)
    if lableString in labelList_1:
        label = 1
        df = pd.read_excel(os.path.join(fpath, f))
        phase = df.iloc[:, 13].values
        mag = df.iloc[:, 9].values
        mag = (mag - np.min(mag)) / (np.max(mag) - np.min(mag))
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
        plt.plot(mag)
        plt.show()
        plt.plot(phase,mag)
        plt.show()
        print(1)
    # mag = np.array(self.norm_data(phase, mag, self.fs))
    #
    # mag_per_phase = mag.reshape(18, 10)
    # x.append(mag_per_phase)
    #
    # labelList_0 = ['36287', '39157', '38352']  # 工作
    # labelList_1 = ['34779']  # 异常
    # lableString = f[2:7]
    #
    # if lableString in labelList_0:
    #     label = 0
    # if lableString in labelList_1:
    #     label = 1
    #
    # plt.plot(mag)
    # plt.show()
    # plt.title(label)
    # y.append(np.array(label))

index = np.arange(y.__len__())
np.random.shuffle(index)
x = np.array(x)[index]
y = np.array(y)[index]
x = x.astype(np.float32)