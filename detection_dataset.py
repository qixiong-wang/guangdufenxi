
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

class Light_Curve_dataset(torch.utils.data.Dataset):
    def __init__(self, txt_path,window_size,fs):
        super(Light_Curve_dataset, self).__init__()

        self.window_size=window_size
        self.fs = fs
        self.curve_list=[]
        self.label_list=[]
        self.data_root='choosed_data/alldata'
        with open(txt_path) as f:
            for line in f.readlines():
                filename = line.split(' ')[0]+'.xlsx'
                filename = os.path.join(self.data_root, filename)
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

                processed_data = self.preprocess(phase, mag, self.window_size, self.fs)  # fs为每个角度切成多少份
                processed_data = processed_data.astype(np.float32)
                label = int(line.split(' ')[1])
                self.curve_list.append(processed_data)
                self.label_list.append(label)
        pass


    def __len__(self):
        # Default epoch size is 10 000 samples
        return len(self.label_list)


    def preprocess(self,phase, mag, window_size=0.1, fs=20):

        new_data = np.zeros((2,180 * fs))
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
        mag_gradient_abs = np.abs((mag[1:]-mag[:-1])/(phase[1:]-phase[:-1]+epsilon))
        mean_mag_gradient_abs = np.zeros((180 * fs))

        for i in range(180 * fs):
            phase_i = i / (fs)
            mag_gradients_in_window = mag_gradient_abs[np.multiply(phase[:-1] < phase_i + window_size, phase[:-1] > phase_i - window_size)]
            num_points = len(mag_gradients_in_window)

            if num_points == 0:
                mean_mag_gradient_abs[i] = 0
            else:
                mean_mag_gradient_abs[i] = np.mean(mag_gradients_in_window)

        new_data[0]=mean_mag
        new_data[1] = std_mag
        # new_data[2] = mean_mag_gradient_abs
        new_data = np.reshape(new_data,(2,60,60))
        return new_data


    def __getitem__(self, i):

        processed_data= self.curve_list[i]
        label = self.label_list[i]


        # plt.imshow(processed_data[1])
        # plt.show()
        # plt.title(label)
        processed_data = torch.from_numpy(processed_data)
        # data.unsqueeze(dim=0)
        # data = data.expand((3,60,60))
        label = torch.tensor(label)

        return processed_data,label