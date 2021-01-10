import torch
import torch.nn.functional as F
from torch import nn, optim
import h5py as h5
import numpy as np
import DataSet_Demo
from torch.utils.data import DataLoader
from torch.autograd import Variable
class Residual_Block(nn.Module):
    def __init__(self, i_channel, o_channel, stride = 1, downsample = None):
        super(Residual_Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=i_channel, out_channels = o_channel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(o_channel)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=o_channel, out_channels=o_channel, kernel_size=3, stride=1,padding=1, bias = False)
        self.bn2 = nn.BatchNorm2d(o_channel)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out
class RNAProfileModel(nn.Module):
    def __init__(self, block, layers, num_classes = 2):
        super(RNAProfileModel, self).__init__()
        self.lstm_profile = nn.LSTM(input_size=4, hidden_size=512, bidirectional=True, num_layers=1, batch_first=True)
        self.in_channels = 16
        self.conv = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(block, 16, layers[0])
        self.layer2 = self.make_layer(block, 32, layers[0], 2)
        self.layer3 = self.make_layer(block, 64, layers[1], 2)
        self.avg_pool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64, num_classes)
    def make_layer(self, block, out_channels, blocks, stride =1):
        downsample = None
        if(stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(nn.Conv2d(self.in_channels, out_channels, kernel_size=3, stride = stride, padding=1, bias=False),
                                       nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)
    def attention_net(self, lstm_output, final_state):
        batch_size = len(lstm_output)
        hidden = final_state.view(batch_size, -1,
                                  1)  # hidden : [batch_size, n_hidden * num_directions(=2), n_layer(=1)]
        attn_weights = torch.bmm(lstm_output, hidden).squeeze(2)  # attn_weights : [batch_size, n_step]
        soft_attn_weights = F.softmax(attn_weights, 1)
        # context : [batch_size, n_hidden * num_directions(=2)]
        context = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)
        return context, soft_attn_weights
    def forward(self, x):
        outputs, h_2 = self.lstm_profile(x)
        out = outputs[:, 149, :]
        out = out.reshape(-1, 1, 32, 32)
        out = self.conv(out)
        out = self.bn(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        out = F.softmax(out, dim=1)
        return out

