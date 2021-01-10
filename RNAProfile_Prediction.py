import torch
from torch.autograd import Variable
from torch import nn, optim
import h5py as h5
import numpy as np
from torch.utils.data import DataLoader
import Bidirectional_Lstm_Moudel
import DataSet_Demo
PATH = 'Model_Save/Model'
hf = h5.File("Merge_Data/Merge_train.h5", "r")
hf1 = h5.File("Merge_Data/Merge_validate.h5", "r")
train_x = hf.get('Sequence')
train_y = hf.get('Label')
test_x = hf1.get('Sequence')
test_y = hf1.get('Label')
train_y = torch.from_numpy(np.array(train_y)).long()
test_y = torch.from_numpy(np.array(test_y)).long()
train_x = torch.from_numpy(np.array(train_x)).float()
test_x = torch.from_numpy(np.array(test_x)).float()
train_data = DataSet_Demo.subDataset(train_x, train_y)
train_Loader = DataLoader(train_data, 128, shuffle=True, num_workers=4)
test_data = DataSet_Demo.subDataset(test_x, test_y)
test_Loader = DataLoader(test_data, 128, shuffle=False, num_workers=4)
model = Bidirectional_Lstm_Moudel.RNAProfileModel(Bidirectional_Lstm_Moudel.Residual_Block, [2, 2, 2, 2])
#model = torch.load(PATH)
Acc = 0
if torch.cuda.is_available():
    model = model.cuda()
criterion = nn.CrossEntropyLoss()
optimer = optim.Adam(model.parameters(), lr=0.001)
total_step = len(train_Loader)
def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
curr_lr = 0.001
prediction_Rate = 0
for j in range(50):
    model.train()
    for i, item in enumerate(train_Loader):
        data, label = item
        num = label.shape[0]
        data = Variable(data)
        label = Variable(label)
        if torch.cuda.is_available():
            data = data.cuda()
            label = label.cuda()
        out = model(data)
        _, pred_acc = torch.max(out.data, 1)
        #print(pred_acc)
        correct = (pred_acc == label).sum()
        loss = criterion(out, label)
        optimer.zero_grad()
        loss.backward()
        optimer.step()
        if(i %1000 == 0 or i == total_step):
            print(('Epoch:[{}/{}], Step[{}/{}], loss:{:.4f}, Accuracy:{:.4f}'.format(j+1,50,i, total_step, loss.data.item(), 100 * correct / num)))
    curr_lr =  curr_lr *0.95
    update_lr(optimer, curr_lr)
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        loss_total = 0
        ii = 0
        for i_test, item_test in enumerate(test_Loader):
            data_test, label_test = item_test
            data_test = Variable(data_test)
            label_test = Variable(label_test)
            if torch.cuda.is_available():
                data_test = data_test.cuda()
                label_test = label_test.cuda()
            outputs = model(data_test)
            loss = criterion(outputs, label_test)
            _, predicted = torch.max(outputs.data, 1)
            # print(predicted)
            loss_total += loss.data.item()
            ii += 1
            total += label_test.size(0)
            correct += (predicted == label_test).sum().item()
        print('Accuracy of the test Data:{}%'.format(100 * correct / total))
        print('Loss of the test Data:{}%'.format(loss_total / ii))
torch.save(model, PATH)