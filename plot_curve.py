import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

dir_path = os.path.join(os.curdir, 'data', 'alldata')
save_path = os.path.join(os.curdir, 'data', 'curve_img')
filename_list = [os.path.join(dir_path, f) for f in os.listdir(dir_path)]
i = 0
for filename in filename_list:
    i = i + 1
    img_name = filename.split('\\')[3].split('.')[0]
    img_dir = os.path.join(save_path,img_name)
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


    plt.figure()
    plt.plot(phase,mag,'-p',color='k',markerfacecolor='r')
    plt.savefig(img_dir)

