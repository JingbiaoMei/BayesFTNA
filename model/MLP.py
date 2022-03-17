import torch.nn as nn
import torch.nn.functional as F


class MLP3L(nn.Module):
    """Feedfoward neural network with 1 hidden layer"""

    def __init__(self, p1=0.5, p2=0.5, p3=0.5):
        super().__init__()
        # hidden layer1
        self.linear1 = nn.Linear(784, 192)
        # hidden layer2
        self.linear2 = nn.Linear(192, 32)
        # output layer
        self.linear3 = nn.Linear(32, 10)

    def forward(self, xb):
        # Flatten the image tensors
        xb = xb.view(xb.size(0), -1)
        # Get intermediate outputs using hidden layer1
        out = self.linear1(xb)
        # Apply activation function
        out = F.relu(out)
        # Get intermediate outputs using hidden layer2
        out = self.linear2(out)
        # Apply activation function
        out = F.relu(out)
        # Get predictions using output layer
        out = self.linear3(out)
        return out


class MLP_Dropout(nn.Module):
    """Feedfoward neural network with 1 hidden layer"""

    def __init__(self, p1=0.2, p2=0.2, p3=0):
        super().__init__()
        # hidden layer1
        self.linear1 = nn.Linear(784, 192)
        self.do1 = nn.Dropout(p=p1)
        # hidden layer2
        self.linear2 = nn.Linear(192, 32)
        self.do2 = nn.Dropout(p=p2)
        # output layer
        self.linear3 = nn.Linear(32, 10)
        self.do3 = nn.Dropout(p=p3)

    def forward(self, xb):
        # Flatten the image tensors
        xb = xb.view(xb.size(0), -1)
        # Get intermediate outputs using hidden layer1
        out = self.do1(self.linear1(xb))
        # Apply activation function
        out = F.relu(out)
        # Get intermediate outputs using hidden layer2
        out = self.do2(self.linear2(out))
        # Apply activation function
        out = F.relu(out)
        # Get predictions using output layer
        out = self.linear3(out)
        return out


class MLP_Alpha(nn.Module):
    """Feedfoward neural network with 1 hidden layer"""

    def __init__(self):
        super().__init__()
        # hidden layer1
        self.linear1 = nn.Linear(784, 192)
        self.do1 = nn.AlphaDropout(p=0.2)
        # hidden layer2
        self.linear2 = nn.Linear(192, 32)
        self.do2 = nn.AlphaDropout(p=0.2)
        # output layer
        self.linear3 = nn.Linear(32, 10)

    def forward(self, xb):
        # Flatten the image tensors
        xb = xb.view(xb.size(0), -1)
        # Get intermediate outputs using hidden layer1
        out = self.do1(self.linear1(xb))
        # Apply activation function
        out = F.relu(out)
        # Get intermediate outputs using hidden layer2
        out = self.do2(self.linear2(out))
        # Apply activation function
        out = F.relu(out)
        # Get predictions using output layer
        out = self.linear3(out)
        return out


class MnistModel_GroupNorm(nn.Module):
    """Feedfoward neural network with 1 hidden layer"""

    def __init__(self, in_size=784, hidden_size_1=196, hidden_size_2=32, out_size=10):
        super().__init__()
        # hidden layer1
        self.linear1 = nn.Linear(in_size, hidden_size_1)
        self.bn1 = nn.GroupNorm(hidden_size_1 / 2, hidden_size_1)
        # hidden layer2
        self.linear2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.bn2 = nn.GroupNorm(hidden_size_2 / 2, hidden_size_2)
        # output layer
        self.linear3 = nn.Linear(hidden_size_2, out_size)
        self.bn3 = nn.GroupNorm(out_size / 2, out_size)

    def forward(self, xb):
        # Flatten the image tensors
        xb = xb.view(xb.size(0), -1)
        # Get intermediate outputs using hidden layer1
        out = self.bn1(self.linear1(xb))
        # Apply activation function
        out = F.relu(out)
        # Get intermediate outputs using hidden layer2
        out = self.bn2(self.linear2(out))
        # Apply activation function
        out = F.relu(out)
        # Get predictions using output layer
        out = self.bn3(self.linear3(out))
        return out


class MnistModel_InstanceNorm(nn.Module):
    """Feedfoward neural network with 1 hidden layer"""

    def __init__(self, in_size=784, hidden_size_1=196, hidden_size_2=32, out_size=10):
        super().__init__()
        # hidden layer1
        self.linear1 = nn.Linear(in_size, hidden_size_1)
        self.bn1 = nn.InstanceNorm1d(hidden_size_1)
        # hidden layer2
        self.linear2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.bn2 = nn.InstanceNorm1d(hidden_size_2)
        # output layer
        self.linear3 = nn.Linear(hidden_size_2, out_size)
        self.bn3 = nn.InstanceNorm1d(out_size)

    def forward(self, xb):
        # Flatten the image tensors
        xb = xb.view(xb.size(0), -1)
        # Get intermediate outputs using hidden layer1
        out = self.bn1(self.linear1(xb))
        # Apply activation function
        out = F.relu(out)
        # Get intermediate outputs using hidden layer2
        out = self.bn2(self.linear2(out))
        # Apply activation function
        out = F.relu(out)
        # Get predictions using output layer
        out = self.bn3(self.linear3(out))
        return out


class MnistModel_LayerNorm(nn.Module):
    """Feedfoward neural network with 1 hidden layer"""

    def __init__(self, in_size=784, hidden_size_1=196, hidden_size_2=32, out_size=10):
        super().__init__()
        # hidden layer1
        self.linear1 = nn.Linear(in_size, hidden_size_1)
        self.bn1 = nn.GroupNorm(hidden_size_1, 1)
        # hidden layer2
        self.linear2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.bn2 = nn.GroupNorm(hidden_size_2, 1)
        # output layer
        self.linear3 = nn.Linear(hidden_size_2, out_size)
        self.bn3 = nn.GroupNorm(out_size, 1)

    def forward(self, xb):
        # Flatten the image tensors
        xb = xb.view(xb.size(0), -1)
        # Get intermediate outputs using hidden layer1
        out = self.bn1(self.linear1(xb))
        # Apply activation function
        out = F.relu(out)
        # Get intermediate outputs using hidden layer2
        out = self.bn2(self.linear2(out))
        # Apply activation function
        out = F.relu(out)
        # Get predictions using output layer
        out = self.bn3(self.linear3(out))
        return out


class MLP6L(nn.Module):
    """Feedfoward neural network with 1 hidden layer"""

    def __init__(self, p1=0.5, p2=0.5, p3=0.5):
        super().__init__()

        self.linear1 = nn.Linear(784, 392)

        self.linear2 = nn.Linear(392, 256)

        self.linear3 = nn.Linear(256, 128)

        self.linear4 = nn.Linear(128, 64)

        self.linear5 = nn.Linear(64, 32)

        self.linear6 = nn.Linear(32, 10)

    def forward(self, xb):
        # Flatten the image tensors
        xb = xb.view(xb.size(0), -1)
        # Get intermediate outputs using hidden layer1
        out = self.linear1(xb)
        # Apply activation function
        out = F.relu(out)
        # Get intermediate outputs using hidden layer2
        out = self.linear2(out)
        # Apply activation function
        out = F.relu(out)
        # Get predictions using output layer
        out = F.relu(self.linear3(out))
        out = F.relu(self.linear4(out))
        out = F.relu(self.linear5(out))
        out = self.linear6(out)
        return out


class MLP9L(nn.Module):
    """Feedfoward neural network with 1 hidden layer"""

    def __init__(self, p1=0.5, p2=0.5, p3=0.5):
        super().__init__()
        # hidden layer1
        self.linear1 = nn.Linear(784, 512)

        self.linear2 = nn.Linear(512, 256)

        self.linear3 = nn.Linear(256, 128)

        self.linear4 = nn.Linear(128, 64)

        self.linear5 = nn.Linear(64, 64)

        self.linear6 = nn.Linear(64, 32)

        self.linear7 = nn.Linear(32, 16)

        self.linear8 = nn.Linear(16, 16)

        self.linear9 = nn.Linear(16, 10)

    def forward(self, xb):
        # Flatten the image tensors
        xb = xb.view(xb.size(0), -1)
        # Get intermediate outputs using hidden layer1
        out = self.linear1(xb)
        # Apply activation function
        out = F.relu(out)
        # Get intermediate outputs using hidden layer2
        out = self.linear2(out)
        # Apply activation function
        out = F.relu(out)
        # Get predictions using output layer

        out = F.relu(self.linear3(out))
        out = F.relu(self.linear4(out))
        out = F.relu(self.linear5(out))
        out = F.relu(self.linear6(out))
        out = F.relu(self.linear7(out))
        out = F.relu(self.linear8(out))
        out = F.relu(self.linear9(out))
        return out
