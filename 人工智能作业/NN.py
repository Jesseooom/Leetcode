# -*- coding:utf8 -*-

from sklearn.datasets import load_iris
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F


hl = 6
lr = 0.05
num_epoch = 10000


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(4, hl)
        self.fc2 = nn.Linear(hl, 3)
        self.softmax = torch.nn.Softmax(dim=1)


    def forward(self, x):
        out = self.fc1(x)
        out = F.sigmoid(out)
        out = self.fc2(out)
        out = self.softmax(out)
        return out


if __name__ == '__main__':
    iris = load_iris()
    x, y = shuffle(iris.data,iris.target)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=33)

    net = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=lr)

    # train
    for epoch in range(num_epoch):
        x = torch.Tensor(x_train).float()
        y = torch.Tensor(y_train).long()

        optimizer.zero_grad()
        y_pred = net(x)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()

        if epoch % 50 is 0:
            print(loss) # cross entropy


    # test
    x = torch.Tensor(x_test).float()
    y = torch.Tensor(y_test).long()
    y_pred = net(x)
    _, predicted = torch.max(y_pred, 1)

    acc = torch.sum(y == predicted).numpy() / len(x_test)
    print(acc)