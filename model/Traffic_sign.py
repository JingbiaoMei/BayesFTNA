import torch
import torch.nn as nn
import torch.nn.functional as F
from model.randomnoise import GaussLayer, LaplaceLayer
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 43)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class BayesFT(nn.Module):
    def __init__(self, p1, p2, p3, p4):
        super(BayesFT, self).__init__()
    
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.do1 = nn.Dropout2d(p1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.do2 = nn.Dropout2d(p2)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.do3 = nn.Dropout(p3)
        self.fc2 = nn.Linear(120, 84)
        self.do4 = nn.Dropout(p4)
        self.fc3 = nn.Linear(84, 43)

    def forward(self, x):
        x = self.pool(self.do1(F.relu(self.conv1(x))))
        x = self.pool(self.do2(F.relu(self.conv2(x))))
        x = x.view(-1, 16 * 5 * 5)
        x = self.do3(F.relu(self.fc1(x)))
        x = self.do4(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x


class GaussFT(nn.Module):
    def __init__(self, p1, p2, p3, p4):
        super(GaussFT, self).__init__()

        self.conv1 = nn.Conv2d(3, 6, 5)
        self.do1 = GaussLayer(p1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.do2 = GaussLayer(p2)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.do3 = GaussLayer(p3)
        self.fc2 = nn.Linear(120, 84)
        self.do4 = GaussLayer(p4)
        self.fc3 = nn.Linear(84, 43)

    def forward(self, x):
        x = self.pool(self.do1(F.relu(self.conv1(x))))
        x = self.pool(self.do2(F.relu(self.conv2(x))))
        x = x.view(-1, 16 * 5 * 5)
        x = self.do3(F.relu(self.fc1(x)))
        x = self.do4(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x


class LaplaceFT(nn.Module):
    def __init__(self, p1, p2, p3, p4):
        super(LaplaceFT, self).__init__()

        self.conv1 = nn.Conv2d(3, 6, 5)
        self.do1 = LaplaceLayer(p1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.do2 = LaplaceLayer(p2)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.do3 = LaplaceLayer(p3)
        self.fc2 = nn.Linear(120, 84)
        self.do4 = LaplaceLayer(p4)
        self.fc3 = nn.Linear(84, 43)

    def forward(self, x):
        x = self.pool(self.do1(F.relu(self.conv1(x))))
        x = self.pool(self.do2(F.relu(self.conv2(x))))
        x = x.view(-1, 16 * 5 * 5)
        x = self.do3(F.relu(self.fc1(x)))
        x = self.do4(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x



nclasses = 43  # GTSRB as 43 classes


class TransformerNet(nn.Module):
    def __init__(self):
        super(TransformerNet, self).__init__()
        self.stn = Stn()
        self.conv1 = nn.Conv2d(3, 100, 5)
        self.conv1_bn = nn.BatchNorm2d(100)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(100, 150, 3)
        self.conv2_bn = nn.BatchNorm2d(150)
        self.conv3 = nn.Conv2d(150, 250, 1)
        self.conv3_bn = nn.BatchNorm2d(250)
        self.fc1 = nn.Linear(250 * 3 * 3, 350)
        self.fc1_bn = nn.BatchNorm1d(350)
        self.fc2 = nn.Linear(350, 43)
        

    def forward(self, x):
        x = self.stn(x)
        x = self.pool(F.elu(self.conv1(x)))
        x = self.conv1_bn(x)
        x = self.pool(F.elu(self.conv2(x)))
        x = self.conv2_bn(x)
        x = self.pool(F.elu(self.conv3(x)))
        x = self.conv3_bn(x)
        x = x.view(-1, 250 * 3 * 3)
        x = F.elu(self.fc1(x))
        x = self.fc1_bn(x)
        x = self.fc2(x)
        return x


class Stn(nn.Module):
    def __init__(self):
        super(Stn, self).__init__()
        # Spatial transformer localization-network
        self.loc_net = nn.Sequential(
            nn.Conv2d(3, 50, 7),
            nn.MaxPool2d(2, 2),
            nn.ELU(),
            nn.Conv2d(50, 100, 5),
            nn.MaxPool2d(2, 2),
            nn.ELU()
        )
        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(100 * 4 * 4, 100),
            nn.ELU(),
            nn.Linear(100, 3 * 2)
        )
        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor(
            [1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, x):
        xs = self.loc_net(x)
        xs = xs.view(-1, 100 * 4 * 4)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x






class TransformerNet_B(nn.Module):
    def __init__(self):
        super(TransformerNet_B, self).__init__()
        self.stn = Stn_B()
        self.conv1 = nn.Conv2d(3, 100, 5)
        self.conv1_bn = nn.BatchNorm2d(100)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(100, 150, 3)
        self.conv2_bn = nn.BatchNorm2d(150)
        self.conv3 = nn.Conv2d(150, 250, 1)
        self.conv3_bn = nn.BatchNorm2d(250)
        self.fc1 = nn.Linear(250 * 3 * 3, 350)
        self.fc1_bn = nn.BatchNorm1d(350)
        self.fc2 = nn.Linear(350, 43)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.stn(x)
        x = self.pool(F.elu(self.conv1(x)))
        x = self.dropout(self.conv1_bn(x))
        x = self.pool(F.elu(self.conv2(x)))
        x = self.dropout(self.conv2_bn(x))
        x = self.pool(F.elu(self.conv3(x)))
        x = self.dropout(self.conv3_bn(x))
        x = x.view(-1, 250 * 3 * 3)
        x = F.elu(self.fc1(x))
        x = self.dropout(self.fc1_bn(x))
        x = self.fc2(x)
        return x


class Stn_B(nn.Module):
    def __init__(self):
        super(Stn_B, self).__init__()
        # Spatial transformer localization-network
        self.loc_net = nn.Sequential(
            nn.Conv2d(3, 50, 7),
            nn.MaxPool2d(2, 2),
            nn.ELU(),
            nn.Dropout2d(0.5),
            nn.Conv2d(50, 100, 5),
            nn.MaxPool2d(2, 2),
            nn.ELU()
        )
        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(100 * 4 * 4, 100),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(100, 3 * 2)
        )
        # Initialize the weights/bias with identity transformation
        self.fc_loc[3].weight.data.zero_()
        self.fc_loc[3].bias.data.copy_(torch.tensor(
            [1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, x):
        xs = self.loc_net(x)
        xs = xs.view(-1, 100 * 4 * 4)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x