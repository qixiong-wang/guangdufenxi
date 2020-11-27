import random
import os
import numpy as np
decection_curve_list = os.listdir('choosed_data/detection')

total_list = np.array(decection_curve_list)
total_number = len(total_list)

test_file = open('./choosed_data/detection/test.txt','w')
for test_file_name in total_list:
    test_file.write(test_file_name.split('.')[0]+'\n')
test_file.close()
