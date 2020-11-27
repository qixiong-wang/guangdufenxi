
import numpy as np
from skimage import io
from glob import glob
from tqdm import tqdm_notebook as tqdm
from sklearn.metrics import confusion_matrix
import random
import itertools

def metrics(predictions, gts, number_class):
    cm = confusion_matrix(
        gts,
        predictions,
        range(number_class))
    print("Confusion matrix :")
    print(cm)

    print("---")

    # Compute global accuracy
    total = sum(sum(cm))
    accuracy = sum([cm[x][x] for x in range(len(cm))])
    accuracy *= 100 / float(total)
    print("{} curve processed".format(total))
    print("Total accuracy : {}%".format(accuracy))

    #
    # # Compute F1 score
    # F1Score = np.zeros(len(label_values))
    # #     F1Score = np.zeros(len(label_values)-1)
    # for i in range(len(label_values)):
    #     #     for i in range(len(label_values)-1):
    #     try:
    #         F1Score[i] = 2. * cm[i, i] / (np.sum(cm[i, :]) + np.sum(cm[:, i]))
    #     except:
    #         # Ignore exception if there is no element in class i for test set
    #         pass
    # print("F1Score :")
    # for l_id, score in enumerate(F1Score):
    #     print("{}: {}".format(label_values[l_id], score))
    #
    # print("---")
    #
    # # Compute kappa coefficient
    # total = np.sum(cm)
    # pa = np.trace(cm) / float(total)
    # pe = np.sum(np.sum(cm, axis=0) * np.sum(cm, axis=1)) / float(total * total)
    # kappa = (pa - pe) / (1 - pe);
    # print("Kappa: " + str(kappa))
    return accuracy
