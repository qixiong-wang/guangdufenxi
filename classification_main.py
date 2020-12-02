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
from torch.autograd import Variable
from torch.backends import cudnn
import os
from dataset import Light_Curve_dataset
from resnet import resnet18
from utilis import metrics
import model
from sklearn.metrics import confusion_matrix
import sklearn

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
random.seed(4)
torch.manual_seed(4)
torch.cuda.manual_seed(4)
np.random.seed(4)
cudnn.deterministic = True

train_path = 'choosed_data/classification/train.txt'

test_path = 'choosed_data/classification/test.txt'

fs=20
window_size=0.5
BATCH_SIZE=10
epochs =500

train_set = Light_Curve_dataset(train_path,window_size,fs)
train_loader = torch.utils.data.DataLoader(train_set,batch_size=BATCH_SIZE,num_workers=4,shuffle=True)

test_set = Light_Curve_dataset(test_path,window_size,fs)
test_loader = torch.utils.data.DataLoader(test_set,batch_size=BATCH_SIZE,num_workers=4,shuffle=True)


criterion=nn.CrossEntropyLoss()
criterion.cuda()

model = model.Baseline(input_channels=3, n_classes=2,dropout=True)
model.cuda()
base_lr = 0.01
optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0005)



def adjust_learning_rate(optimizer,base_lr,max_iter,cur_ites,power=0.9):
    lr =base_lr*((1-float(cur_ites)/max_iter)**(power))
    for param_group in optimizer.param_groups:
        param_group['lr']=lr



def test(net, dataloader):
    net.eval()
    outputs=[]
    targets=[]
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(dataloader):
            data = Variable(data.cuda())
            output = net(data)
            output = output.cpu().numpy()
            output = np.argmax(output,axis=1)
            outputs.append(output)
            target = target.numpy()
            targets.append(target)

        outputs = np.concatenate([p.ravel() for p in outputs])
        targets = np.concatenate([p.ravel() for p in targets])
        accuracy = metrics(outputs,targets,2)
    return accuracy


def train(net, optimizer, epochs,best_acc=0):
    losses = np.zeros(100000)
    mean_losses = np.zeros(100000)

    iter_ = 0
    for e in tqdm(range(1, epochs + 1)):

        adjust_learning_rate(optimizer,base_lr,epochs,e,power=0.9)
        # adjust_learning_rate_milestone(optimizer, base_lr, e, changeepochs=[20.30,40], power=0.1)
        net.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = Variable(data.cuda()), Variable(target.cuda())
            optimizer.zero_grad()
            output = net(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            losses[iter_] = loss.item()
            mean_losses[iter_] = np.mean(losses[max(0, iter_ - 100):iter_])

            if iter_ % 100 == 0:
                plt.plot(mean_losses[:iter_])
                plt.show()
            iter_ += 1

        if e%20==0:
            # test(model, train_loader)
            acc = test(model, test_loader)
            if acc > best_acc:
                best_acc = acc
                torch.save(model.state_dict(),'./best_model.pth')
                print('best_acc:',best_acc)

train(model, optimizer, epochs)