import torch
from torch.autograd import Variable
from torch import nn, optim
import h5py as h5
import numpy as np
from torch.utils.data import DataLoader
import Bidirectional_Lstm_Moudel
import DataSet_Demo
#PATH = 'Model_Save/High_Low_Model_30'
hf = h5.File("Human_Data/Human_train.h5", "r")
hf1 = h5.File("Mouse_Data/Mouse_validate.h5", "r")
hf2 = h5.File("Yeast_Data/Yeast_Data.h5", "r")
hf3 = h5.File("Fish_Data/Fish_validate_150.h5", "r")
hf4 = h5.File("Human_Data/Human_validate_150.h5", "r")
hf5 = h5.File("PDB_Data/PDB_validate.h5", "r")
train_x = hf.get('Sequence')
train_y = hf.get('Label')
test_x = hf1.get('Sequence')
test_y = hf1.get('Label')
test_x_1 = hf2.get('Sequence')
test_y_1 = hf2.get('Label')
test_x_2 = hf3.get('Sequence')
test_y_2 = hf3.get('Label')
test_x_3 = hf4.get('Sequence')
test_y_3 = hf4.get('Label')
test_x_4 = hf5.get('Sequence')
test_y_4 = hf5.get('Label')

train_y = torch.from_numpy(np.array(train_y)).long()
test_y = torch.from_numpy(np.array(test_y)).long()
test_y_1 = torch.from_numpy(np.array(test_y_1)).long()

test_y_2 = torch.from_numpy(np.array(test_y_2)).long()
test_y_3 = torch.from_numpy(np.array(test_y_3)).long()
test_y_4 = torch.from_numpy(np.array(test_y_4)).long()

train_x = torch.from_numpy(np.array(train_x)).float()
test_x = torch.from_numpy(np.array(test_x)).float()
test_x_1 = torch.from_numpy(np.array(test_x_1)).float()

test_x_2 = torch.from_numpy(np.array(test_x_2)).float()
test_x_3 = torch.from_numpy(np.array(test_x_3)).float()
test_x_4 = torch.from_numpy(np.array(test_x_4)).float()

train_data = DataSet_Demo.subDataset(train_x, train_y)
train_Loader = DataLoader(train_data, 128, shuffle=True, num_workers=4)
test_data = DataSet_Demo.subDataset(test_x, test_y)
test_Loader = DataLoader(test_data, 128, shuffle=False, num_workers=4)

test_data_1 = DataSet_Demo.subDataset(test_x_1, test_y_1)
test_Loader_1 = DataLoader(test_data_1, 128, shuffle=False, num_workers=4)

test_data_2 = DataSet_Demo.subDataset(test_x_2, test_y_2)
test_Loader_2 = DataLoader(test_data_2, 128, shuffle=False, num_workers=4)

test_data_3 = DataSet_Demo.subDataset(test_x_3, test_y_3)
test_Loader_3 = DataLoader(test_data_3, 128, shuffle=False, num_workers=4)

test_data_4 = DataSet_Demo.subDataset(test_x_4, test_y_4)
test_Loader_4 = DataLoader(test_data_4, 128, shuffle=False, num_workers=4)
model = Bidirectional_Lstm_Moudel.RNAProfileModel(Bidirectional_Lstm_Moudel.Residual_Block, [2, 2, 2, 2])
#model = torch.load(PATH)
Acc = 0
if torch.cuda.is_available():
    model = model.cuda()
criterion = nn.CrossEntropyLoss()
optimer = optim.Adam(model.parameters(), lr=0.001)
total_step = len(train_Loader)
# for updating learning rate
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
        if(i %500 == 0 or i == total_step):
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

            loss_total += loss.data.item()
            ii += 1
            total += label_test.size(0)
            correct += (predicted == label_test).sum().item()

        print('Accuracy of the Mouse Data:{}%'.format(100 * correct / total))
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for i_test, item_test in enumerate(test_Loader_1):
            data_test, label_test = item_test
            data_test = Variable(data_test)
            label_test = Variable(label_test)
            if torch.cuda.is_available():
                data_test = data_test.cuda()
                label_test = label_test.cuda()
            outputs = model(data_test)
            _, predicted = torch.max(outputs.data, 1)
            # print(predicted)
            total += label_test.size(0)
            correct += (predicted == label_test).sum().item()
        print('Accuracy of the model on the Yeast Data:{}%'.format(100 * correct / total))
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for i_test, item_test in enumerate(test_Loader_2):
            data_test, label_test = item_test
            data_test = Variable(data_test)
            label_test = Variable(label_test)
            if torch.cuda.is_available():
                data_test = data_test.cuda()
                label_test = label_test.cuda()
            outputs = model(data_test)
            _, predicted = torch.max(outputs.data, 1)
            # print(predicted)
            total += label_test.size(0)
            correct += (predicted == label_test).sum().item()
        print('Accuracy of the model on the Fish Data:{}%'.format(100 * correct / total))
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for i_test, item_test in enumerate(test_Loader_3):
            data_test, label_test = item_test
            data_test = Variable(data_test)
            label_test = Variable(label_test)
            if torch.cuda.is_available():
                data_test = data_test.cuda()
                label_test = label_test.cuda()
            outputs = model(data_test)
            _, predicted = torch.max(outputs.data, 1)
            # print(predicted)
            total += label_test.size(0)
            correct += (predicted == label_test).sum().item()
        print('Accuracy of the model on the Human Data:{}%'.format(100 * correct / total))
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for i_test, item_test in enumerate(test_Loader_4):
            data_test, label_test = item_test
            data_test = Variable(data_test)
            label_test = Variable(label_test)
            if torch.cuda.is_available():
                data_test = data_test.cuda()
                label_test = label_test.cuda()
            outputs = model(data_test)
            _, predicted = torch.max(outputs.data, 1)
            # print(predicted)
            total += label_test.size(0)
            correct += (predicted == label_test).sum().item()
        print('Accuracy of the model on the PDB  Data:{}%'.format(100 * correct / total))
