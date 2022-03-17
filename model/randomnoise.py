import torch
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as td
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR


class MyDropout(nn.Module):
    def __init__(self, p: float = 0.5):
        super(MyDropout, self).__init__()
        if p < 0 or p > 1:
            raise ValueError("dropout probability should be between 0 and 1")
        self.p = p

    def forward(self, X):
        if self.training:
            binomial = torch.distributions.binomial.Binomial(probs=1 - self.p)
            return X * binomial.sample(X.size()) * (1.0 / (1 - self.p))


class pureSigmaLayer(nn.Module):
    def __init__(self, sigma=0.01):
        super(pureSigmaLayer, self).__init__()
        self.sigma = sigma

    def forward(self, X):
        return X + self.sigma
    

class BetaLayer(nn.Module):
    def __init__(self, sigma=0.01, alpha=0.5, beta=0.5):
        super(BetaLayer, self).__init__()
        self.sigma = sigma
        self.alpha = alpha
        self.beta = beta

    def forward(self, X):
        beta = td.Beta(concentration1=self.alpha, concentration0=self.beta).sample()
        return X + self.sigma * beta


class PoissonLayer(nn.Module):
    def __init__(self, sigma=0.01, rate=3):
        super(PoissonLayer, self).__init__()
        self.sigma = sigma
        self.rate = rate

    def forward(self, X):
        poisson = td.Poisson(rate=self.rate).sample()
        return X + self.sigma * poisson


class ParetoLayer(nn.Module):
    def __init__(self, sigma=0.01, scale=1, alpha=1):
        super(ParetoLayer, self).__init__()
        self.sigma = sigma
        self.scale = scale
        self.alpha = alpha

    def forward(self, X):
        pareto = td.Pareto(scale=self.scale, alpha=self.alpha).sample()
        return X + self.sigma * pareto


class GaussLayer(nn.Module):
    def __init__(self, sigma: float = 0.01):
        super(GaussLayer, self).__init__()
        if sigma < 0:
            raise ValueError("sigma value should be above 0")
        self.sigma = sigma

    def forward(self, X):
        # if self.training:
        gauss = torch.distributions.Normal(0, self.sigma).sample()
        return X + self.sigma * gauss


class LaplaceLayer(nn.Module):
    def __init__(self, sigma: float = 0.01):
        super(LaplaceLayer, self).__init__()
        if sigma < 0:
            raise ValueError("sigma value should be above 0")
        self.sigma = sigma

    def forward(self, X):
        laplace = torch.distributions.Laplace(0, self.sigma).sample()
        return X + self.sigma * laplace


class GammaLayer(nn.Module):
    def __init__(self, sigma=0.01, alpha=1.0, beta=1.0):
        super(GammaLayer, self).__init__()
        self.sigma = sigma
        self.alpha = alpha
        self.beta = beta

    def forward(self, X):
        gamma = td.Gamma(concentration=self.alpha, rate=self.beta).sample()
        return X + self.sigma * gamma


class BernoulliLayer(nn.Module):
    def __init__(self, sigma: float = 0.01, prob: float = 0.3):
        super(BernoulliLayer, self).__init__()
        if sigma < 0:
            raise ValueError("sigma value should be above 0")
        self.sigma = sigma
        self.prob = prob

    def forward(self, X):
        bernoulli = torch.distributions.Bernoulli(self.prob).sample()
        return X + self.sigma * bernoulli


class MultinomialLayer(nn.Module):
    def __init__(self, sigma=0.01):
        super(MultinomialLayer, self).__init__()
        self.sigma = sigma

    def forward(self, X):
        multi = td.Multinomial(total_count=10, probs=torch.tensor([1., 1., 1., 1.])).sample()
        return X + self.sigma * multi[0]


if __name__ == '__main__':
    print("Test for noise implementation functions")


    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(1, 32, 3, 1)
            self.dropout1 = nn.Dropout(0.25)
            self.fc = nn.Linear(9216, 128)

        def forward(self, x):
            x = self.conv1(x)
            x = F.relu(x)
            x = self.fc(x)
            output = F.log_softmax(x, dim=1)
            return output


    import torch
    import torch.nn as nn
    import torch.utils.data as data
    import torchvision
    import torchvision.transforms as transforms
    from tqdm import tqdm
    import time

    BATCH_SIZE = 128
    NUM_EPOCHS = 10
    # preprocessing
    normalize = transforms.Normalize(mean=[.5], std=[.5])
    transform = transforms.Compose([transforms.ToTensor(), normalize])

    # download and load the data
    train_dataset = torchvision.datasets.MNIST(root='./mnist/', train=True, transform=transform, download=True)
    test_dataset = torchvision.datasets.MNIST(root='./mnist/', train=False, transform=transform, download=False)

    # encapsulate them into dataloader form
    train_loader = data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    test_loader = data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)

    model = Net()
    for epoch in range(10):
        for images, labels in tqdm(train_loader):
            pred = Net(images)
            print(pred)
            exit()
