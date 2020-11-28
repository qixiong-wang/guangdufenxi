import numpy as np
import torch
import torch.utils.data as data
import torch.nn.init
# from cam.scorecam import *
from model import *
import pandas as pd
import os
import matplotlib.pyplot as plt

mynet =  Baseline(input_channels=3, n_classes=2,dropout=True)
model_dir = './best_model.pth'
mynet.load_state_dict(torch.load(model_dir,map_location= 'cpu'))
device = torch.device('cuda:{}'.format(0))
mynet.to(device)
mynet.eval()


torch.manual_seed(7)
torch.cuda.manual_seed(7)
np.random.seed(7)

for name, module in mynet._modules.items():
    # x = module(x)
    print("名称:{}".format(name))


# mynet_model_dict = dict(type='mynet', arch=mynet, layer_name='bn3',input_size=(60, 60))
# mynet_scorecam = my_CAM(mynet_model_dict)

data_root='choosed_data/alldata'
test_txt_path =  'choosed_data/detection/test.txt'
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

def preprocess(phase, mag, window_size = 0.1, fs=20):
    new_data = np.zeros((3, 180 * fs))
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
    mean_mag = biliner_interpolation(mean_mag)
    std_mag = biliner_interpolation(std_mag)

    new_data[0] = mean_mag
    new_data[1] = std_mag
    new_data[2] = std_mag
    return new_data

with open(test_txt_path) as f:
    for line in f.readlines():
        filename = line.split('\n')[0] + '.xlsx'
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
        x = np.linspace(0, 180, 3600)
        vline = np.zeros((3600))
        with torch.no_grad():
            for i in range(1,35):
                input_data = np.zeros((3, 3600),dtype=np.float32)
                input_data[:,(i-1)*100:(i+1)*100] = processed_data[:,(i-1)*100:(i+1)*100]
                input_data = np.reshape(input_data, (3, 60, 60))
                input_data = torch.from_numpy(input_data).unsqueeze(0)
                # label = int(line.split(' ')[1])
                input_data = input_data.cuda()
                output = mynet(input_data).cpu().numpy()
                output = np.argmax(output)
                if output==1:
                    print(i)
                    vline[(i-1)*100]=1
                    vline[(i + 1) * 100] = 1
                    plt.plot(x,vline)

        # scorecam_map = mynet_scorecam(input_data,label)
        # weight_phase = scorecam_map.reshape(3600).cpu().detach().numpy()

        # plt.figure()
        # plt.plot(x, weight_phase.squeeze(), 'g')
        plt.plot(phase, mag, 'b-*')
        plt.show()