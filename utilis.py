import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import ToTensor
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
import torchvision.transforms as tt
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
from torch.utils.data.dataset import Dataset
import pickle
from utilis_gpu import get_default_device, to_device
# Adv lib
from pgd import attack_pgd
from awp import AdvWeightPerturb
from utils_awp import AdvWeightPerturb
from copy import deepcopy
from torchvision.datasets import MNIST
from torchvision.datasets import CIFAR10

import models_nas
from assets import architecture_code

# Adding Noise to the Model
device = get_default_device()


class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""

    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)


# Traffic Sign Data Wrapper
class PickledDataset(Dataset):
    def __init__(self, file_path, transform=None):
        with open(file_path, mode='rb') as f:
            data = pickle.load(f)
            self.features = data['features']
            self.labels = data['labels']
            self.count = len(self.labels)
            self.transform = transform

    def __getitem__(self, index):
        feature = self.features[index]
        if self.transform is not None:
            feature = self.transform(feature)
        return (feature, self.labels[index])

    def __len__(self):
        return self.count


class WrappedDataLoader:
    def __init__(self, dl, func):
        self.dl = dl
        self.func = func

    def __len__(self):
        return len(self.dl)

    def __iter__(self):
        batches = iter(self.dl)
        for b in batches:
            yield (self.func(*b))


def select_Data(data_Set):
    if data_Set == 'MNIST' or data_Set == 'Mnist':
        dataset = MNIST(root='data/', download=True, transform=ToTensor())
        val_size = 10000
        train_size = len(dataset) - val_size
        train_ds, valid_ds = random_split(dataset, [train_size, val_size])
        batch_size = 2048
        print("Dataset: Mnist")
        # PyTorch data loaders
        train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=8, pin_memory=True)
        valid_dl = DataLoader(valid_ds, batch_size * 4, num_workers=8, pin_memory=True)
        # Move to GPU
        train_dl = DeviceDataLoader(train_dl, device)
        valid_dl = DeviceDataLoader(valid_dl, device)
    elif data_Set == "CIFAR10" or data_Set == "CIFAR-10":
        transform_train = tt.Compose([
            # tt.RandomCrop(32, padding=4),
            # tt.RandomHorizontalFlip(),
            tt.ToTensor(),
            tt.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

        # Normalize the test set same as training set without augmentation
        transform_test = tt.Compose([
            tt.ToTensor(),
            tt.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

        train_ds = CIFAR10(root='data/', train=True, download=True, transform=transform_train)
        valid_ds = CIFAR10(root='data/', train=False, download=True, transform=transform_test)
        batch_size = 128
        print("Dataset: Cifar10")
        # PyTorch data loaders

        train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=8, pin_memory=True)
        valid_dl = DataLoader(valid_ds, batch_size * 4, num_workers=8, pin_memory=True)
        # print(train_dl)
        # Move to GPU
        train_dl = DeviceDataLoader(train_dl, device)
        valid_dl = DeviceDataLoader(valid_dl, device)
    elif data_Set == "CIFAR10-AWP":
        transform_train = tt.Compose([
            # tt.RandomCrop(32, padding=4),
            # tt.RandomHorizontalFlip(),
            tt.ToTensor()
        ])

        # Normalize the test set same as training set without augmentation
        transform_test = tt.Compose([
            tt.ToTensor()
        ])

        train_ds = CIFAR10(root='data/', train=True, download=True, transform=transform_train)
        valid_ds = CIFAR10(root='data/', train=False, download=True, transform=transform_test)
        batch_size = 128
        print("Dataset: Cifar10")
        # PyTorch data loaders
        train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=8, pin_memory=True)
        valid_dl = DataLoader(valid_ds, batch_size * 4, num_workers=8, pin_memory=True)
        # Move to GPU
        train_dl = DeviceDataLoader(train_dl, device)
        valid_dl = DeviceDataLoader(valid_dl, device)
    elif data_Set == "CIFAR-10-256":
        transform_train = tt.Compose([
            # tt.RandomCrop(32, padding=4),
            # tt.RandomHorizontalFlip(),
            tt.ToTensor(),
            tt.Resize(256),
            tt.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

        # Normalize the test set same as training set without augmentation
        transform_test = tt.Compose([
            tt.ToTensor(),
            tt.Resize(256),
            tt.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

        train_ds = CIFAR10(root='data/', train=True, download=True, transform=transform_train)
        valid_ds = CIFAR10(root='data/', train=False, download=True, transform=transform_test)
        batch_size = 1000
        print("Dataset: Cifar10")
        # PyTorch data loaders
        train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=8, pin_memory=True)
        valid_dl = DataLoader(valid_ds, batch_size * 4, num_workers=8, pin_memory=True)
        # Move to GPU
        train_dl = DeviceDataLoader(train_dl, device)
        valid_dl = DeviceDataLoader(valid_dl, device)
    elif data_Set == "CIFAR10-256-AWP":
        transform_train = tt.Compose([
            # tt.RandomCrop(32, padding=4),
            # tt.RandomHorizontalFlip(),
            tt.ToTensor(),
            tt.Resize(256)
            # tt.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

        # Normalize the test set same as training set without augmentation
        transform_test = tt.Compose([
            tt.ToTensor(),
            tt.Resize(256)
            # tt.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

        train_ds = CIFAR10(root='data/', train=True, download=True, transform=transform_train)
        valid_ds = CIFAR10(root='data/', train=False, download=True, transform=transform_test)
        batch_size = 1000
        print("Dataset: Cifar10")
        # PyTorch data loaders
        train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=8, pin_memory=True)
        valid_dl = DataLoader(valid_ds, batch_size * 4, num_workers=8, pin_memory=True)
        # Move to GPU
        train_dl = DeviceDataLoader(train_dl, device)
        valid_dl = DeviceDataLoader(valid_dl, device)
    elif data_Set == "Traffic-Sign":
        training_file = "data/Traffic-Sign/train.p"
        validation_file = "data/Traffic-Sign/valid.p"
        # testing_file = "data/Traffic_Sign/test.p"
        train_dataset = PickledDataset(training_file, transform=tt.ToTensor())
        valid_dataset = PickledDataset(validation_file, transform=tt.ToTensor())
        # test_dataset = PickledDataset(testing_file, transform=tt.ToTensor())

        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=False)

        # test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
        def to_device_func(x, y):
            return x.to(device), y.to(device, dtype=torch.int64)

        train_dl = WrappedDataLoader(train_loader, to_device_func)
        valid_dl = WrappedDataLoader(valid_loader, to_device_func)
        # test_dl = WrappedDataLoader(test_loader, to_device_func)

    else:
        print("No such dataset:", data_Set)

    return train_dl, valid_dl


def select_Model(name, num_classes):
    if name == 'MLP-3L':
        from model import MLP
        model = MLP.MLP3L()
    elif name == 'MLP-6L':
        from model import MLP
        model = MLP.MLP6L()
    elif name == 'MLP-9L':
        from model import MLP
        model = MLP.MLP9L()
    elif name == 'MLP-DO':
        from model import MLP
        model = MLP.MLP_Dropout()
    elif name == 'MLP-Alpha':
        from model import MLP
        model = MLP.MLP_Alpha()
    elif name == 'ResNet-18':
        model = models.resnet18()
        model.fc = nn.Linear(512, num_classes)
    elif name == 'ResNet-18-Bayes':
        from model import Res18Bayes
        model = Res18Bayes.res18bayes()
        model.fc = nn.Linear(512, num_classes)
    elif name == 'VGG':
        model = models.vgg11_bn()
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, num_classes)
    elif name == "VGG-Best":
        from model import VGGBayes
        model = VGGBayes.vggBayes()
    elif name == "VGG-Naive":
        from model import VGGBayes
        model = VGGBayes.vgg_naive()
    elif name == 'LeNet':
        from model import LeNet
        model = LeNet.LeNet5()
    elif name == 'LeNet-Best':
        from model import LeNet
        model = LeNet.LeNet_Best()
    elif name == 'LeNet-12':
        from model import LeNet
        model = LeNet.LeNet_12()
    elif name == 'AlexNet':
        from model import AlexNet
        model = AlexNet.alex()
    elif name == 'AlexNet-Best':
        from model import AlexNet
        model = AlexNet.alexBayes()
    # Naive version of Bayes
    elif name == 'AlexNet-1':
        from model import AlexNet
        model = AlexNet.alex1()
    elif name == 'SqueezeNet':
        from model.SqueezeNet import squeezenet
        model = squeezenet()
    elif name == 'SqueezeNet-Naive':
        from model.SqueezeNet import squeezenet_naive
        model = squeezenet_naive()
    elif name == 'MobileNet':
        model = models.mobilenet_v2(pretrained=True)
        model.classifier[1] = nn.Linear(1280, 10)
    # Naive version of Bayes
    elif name == 'MobileNet1':
        from model import MobileNet
        model = MobileNet.mobilenet()
    elif name == 'Traffic-Sign':
        from model import Traffic_sign
        model = Traffic_sign.Model()
    elif name == 'Traffic-Sign-Transform':
        from model import Traffic_sign
        model = Traffic_sign.TransformerNet()
    elif name == 'Traffic-Sign-Transform-Bayes':
        from model import Traffic_sign
        model = Traffic_sign.TransformerNet_B()
    elif name == 'PreAct-Res18-Bayes':
        from model import PreAct_Bayes
        model = PreAct_Bayes.PreActResNet18()
    elif name == 'PreAct-Res18':
        from model import PreAct
        model = PreAct.PreActResNet18()
    elif name == 'PreAct-Res34-Bayes':
        from model import PreAct_Bayes
        model = PreAct_Bayes.PreActResNet34()
    elif name == 'PreAct-Res34':
        from model import PreAct
        model = PreAct.PreActResNet34()
    elif name == 'PreAct-Res50-Bayes':
        from model import PreAct_Bayes
        model = PreAct_Bayes.PreActResNet50()
    elif name == 'PreAct-Res50':
        from model import PreAct
        model = PreAct.PreActResNet50()
    elif name == 'PreAct-Res152-Bayes':
        from model import PreAct_Bayes
        model = PreAct_Bayes.PreActResNet152()
    elif name == 'PreAct-Res152':
        from model import PreAct
        model = PreAct.PreActResNet152()
    elif name == 'RobNet_free':
        model = models_nas.robnet(architecture_code.robnet_free)
    elif name == 'robnet_large_v1':
        model = models_nas.robnet(architecture_code.robnet_large_v1, share=True)
    else:
        print("model not found")

    to_device(model, device)
    return model


# Helper Function For training


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


def training_step(model, batch):
    images, labels = batch
    out = model(images)  # Generate predictions
    loss = F.cross_entropy(out, labels)  # Calculate loss
    return loss


def validation_step(model, batch):
    images, labels = batch
    out = model(images)  # Generate predictions
    loss = F.cross_entropy(out, labels)  # Calculate loss
    acc = accuracy(out, labels)  # Calculate accuracy
    return {'val_loss': loss.detach(), 'val_acc': acc}


def validation_epoch_end(outputs):
    batch_losses = [x['val_loss'] for x in outputs]
    epoch_loss = torch.stack(batch_losses).mean()  # Combine losses
    batch_accs = [x['val_acc'] for x in outputs]
    epoch_acc = torch.stack(batch_accs).mean()  # Combine accuracies
    return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}


def epoch_end(epoch, result):
    print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
        epoch, result['train_loss'], result['val_loss'], result['val_acc']))


# Traning
@torch.no_grad()
def evaluate(model, val_loader):
    # Tell PyTorch validation start, disable all regularization
    model.eval()
    # Take a Batch loss and Accuracy and Average through all the batches  
    outputs = [validation_step(model, batch) for batch in val_loader]
    return validation_epoch_end(outputs)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def fit_one_cycle(epochs, max_lr, model, train_loader, val_loader,
                  weight_decay=0, grad_clip=None, opt_func=torch.optim.SGD):
    torch.cuda.empty_cache()
    history = []

    # Set up cutom optimizer with weight decay
    optimizer = opt_func(model.parameters(), max_lr, weight_decay=weight_decay)
    # Set up one-cycle learning rate scheduler
    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epochs,
                                                steps_per_epoch=len(train_loader))

    for epoch in range(epochs):
        # Training Phase 
        model.train()
        train_losses = []
        lrs = []
        for batch in train_loader:
            loss = training_step(model, batch)
            train_losses.append(loss)
            loss.backward()

            # Gradient clipping
            if grad_clip:
                nn.utils.clip_grad_value_(model.parameters(), grad_clip)

            optimizer.step()
            optimizer.zero_grad()

            # Record & update learning rate
            lrs.append(get_lr(optimizer))
            sched.step()

        # Validation phase
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        result['lrs'] = lrs
        epoch_end(epoch, result)
        history.append(result)
    return history


def fit_one_cycle_adv(epochs, max_lr, model, train_loader, val_loader,
                      weight_decay=0, grad_clip=None, opt_func=torch.optim.SGD, epsilon=0.02):
    torch.cuda.empty_cache()
    history = []

    # awp = AdvWeightPerturb(model, eta=0.01, nb_iter=5)

    # Set up cutom optimizer with weight decay
    optimizer = opt_func(model.parameters(), max_lr, weight_decay=weight_decay)
    # Set up one-cycle learning rate scheduler
    # sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epochs,
    # steps_per_epoch=len(train_loader))

    proxy = deepcopy(model)
    proxy_opt = opt_func(proxy.parameters(), max_lr, weight_decay=weight_decay)
    awp_adversary = AdvWeightPerturb(model=model, proxy=proxy, proxy_optim=proxy_opt, gamma=0.01)
    print("epislon used:", epsilon)
    for epoch in range(epochs):
        # Training Phase 
        model.train()
        train_losses = []
        # lrs = []
        for batch in train_loader:
            # loss = training_step(model, batch)
            images, labels = batch

            delta = attack_pgd(model, images, labels, epsilon, 0.001, 3, 1, "l_inf")

            X_adv = torch.clamp(images + delta, 0, 1)
            awp = awp_adversary.calc_awp(inputs_adv=X_adv, targets=labels)
            awp_adversary.perturb(awp)

            out = model(X_adv)  # Generate predictions
            loss = F.cross_entropy(out, labels)  # Calculate loss

            train_losses.append(loss)
            loss.backward()

            # Gradient clipping
            if grad_clip:
                nn.utils.clip_grad_value_(model.parameters(), grad_clip)

            optimizer.step()
            optimizer.zero_grad()

            # Record & update learning rate
            # lrs.append(get_lr(optimizer))
            # sched.step()

        # Validation phase
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        # result['lrs'] = lrs

        epoch_end(epoch, result)
        history.append(result)
    return history


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def add_noise_to_weights(mean, std, model):
    """
    with torch.no_grad():
        if hasattr(m, 'weight'):
            m.weight.add_(torch.randn(m.weight.size()) * 0.1)
    """
    gassian_kernel = torch.distributions.Normal(mean, std)
    with torch.no_grad():
        for param in model.parameters():
            param.mul_(to_device(torch.exp(gassian_kernel.sample(param.size())), device))
    # print("Noise added, the standard deviation is: ", std)


def evaluate_robustness(model, model_path, valid_dl):
    # Pick different value for sigma
    sigma = np.linspace(0., 1.5, 31)

    # Initialize Empty list for accuracy under different std
    accu = []

    # Run several time for a smoother curve
    num = 20
    evaluated = np.zeros(num)

    for std in sigma:
        for i in range(num):
            model.load_state_dict(torch.load(model_path))
            add_noise_to_weights(0, std, model)
            evaluated[i] = evaluate(model, valid_dl)['val_acc']
        print("Finshed sigma=", std, np.sum(evaluated) / num)
        accu.append(np.sum(evaluated) / num)
    return sigma, accu


def fit_one_cycle_Bayes(epochs, max_lr, model, train_loader, val_loader,
                        weight_decay=0, grad_clip=None, opt_func=torch.optim.SGD):
    torch.cuda.empty_cache()
    history = []

    # Set up cutom optimizer with weight decay
    optimizer = opt_func(model.parameters(), max_lr, weight_decay=weight_decay)
    # Set up one-cycle learning rate scheduler
    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epochs,
                                                steps_per_epoch=len(train_loader))

    for epoch in range(epochs):
        # Training Phase 
        model.train()
        train_losses = []
        lrs = []
        for batch in train_loader:
            loss = training_step(model, batch)
            train_losses.append(loss)
            loss.backward()

            # Gradient clipping
            if grad_clip:
                nn.utils.clip_grad_value_(model.parameters(), grad_clip)

            optimizer.step()
            optimizer.zero_grad()

            # Record & update learning rate
            lrs.append(get_lr(optimizer))
            sched.step()

        # Validation phase
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        result['lrs'] = lrs
        # if (epoch % 9) == 0:
        epoch_end(epoch, result)
        history.append(result)
    return history


def add_noise_to_weights_out(mean, std, model):
    """
    with torch.no_grad():
        if hasattr(m, 'weight'):
            m.weight.add_(torch.randn(m.weight.size()) * 0.1)
    """
    gassian_kernel = torch.distributions.Normal(mean, std)
    with torch.no_grad():
        for param in model.parameters():
            param.mul_(to_device(torch.exp(gassian_kernel.sample(param.size())), device))
    # print("Noise added, the standard deviation is: ", std)
    return model


def enumerate_robnet_large(num):
    from itertools import product
    arch_list = list(product(['01', '10', '11'], repeat=14))
    arch_list = [list(ele) for ele in arch_list]
    import random
    random.shuffle(arch_list)
    return arch_list[:num]
