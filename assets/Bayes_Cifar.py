from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
import torchvision.models as models
import torch.nn as nn
import numpy as np
import torch
from utilis import add_noise_to_weights, fit_one_cycle_Bayes, evaluate, select_Data
from utilis_gpu import to_device, get_default_device
from model.randomnoise import GaussLayer

device = get_default_device()

train_dl, valid_dl = select_Data("CIFAR-10")


def run(model_name, n_iter):
    if model_name == 'VGG':
        step_VGG(n_iter)
    elif model_name == 'ResNet-18':
        step_Res18(n_iter)
    elif model_name == 'SqueezeNet':
        step_Squeeze(n_iter)
    else:
        print("Not Found Bayes FT model")


def bbf_VGG(p1, p2, p3, p4, p5, p6):
    p1 = round(p1, 2)
    p2 = round(p2, 2)
    p3 = round(p3, 2)
    p4 = round(p4, 2)
    p5 = round(p5, 2)
    p6 = round(p6, 2)
    model = models.vgg11_bn(pretrained=True)
    model.features[1] = nn.Identity()
    model.features[2] = nn.Sequential(model.features[2], nn.Dropout2d(p1))

    model.features[5] = nn.Identity()
    model.features[6] = nn.Sequential(model.features[6], nn.Dropout2d(p1))

    model.features[9] = nn.Identity()
    model.features[10] = nn.Sequential(model.features[10], nn.Dropout2d(p1))

    model.features[12] = nn.Identity()
    model.features[13] = nn.Sequential(model.features[13], nn.Dropout2d(p2))

    model.features[16] = nn.Identity()
    model.features[17] = nn.Sequential(model.features[17], nn.Dropout2d(p2))

    model.features[19] = nn.Identity()
    model.features[20] = nn.Sequential(model.features[20], nn.Dropout2d(p2))

    model.features[23] = nn.Identity()
    model.features[24] = nn.Sequential(model.features[24], nn.Dropout2d(p3))

    model.features[26] = nn.Identity()
    model.features[27] = nn.Sequential(model.features[27], nn.Dropout2d(p3))

    model.classifier[2] = nn.Dropout(p4)
    model.classifier[5] = nn.Dropout(p5)
    model.classifier[6] = nn.Linear(4096, 10)
    model.classifier[6] = nn.Sequential(model.classifier[6], nn.Dropout(p6))

    to_device(model, device)
    history = fit_one_cycle_Bayes(20, 0.001, model, train_dl, valid_dl, opt_func=torch.optim.Adam)
    torch.save(model.state_dict(), './results/Bayes/VGG/VGG-{}-{}-{}-{}-{}.pth'.format(p1, p2, p3, p4, p5))
    # AVERAGE RUN 30 TIMES
    num = 20
    std = 0.8
    evaluated = np.zeros(num)
    for i in range(num):
        model.load_state_dict(torch.load('./results/Bayes/VGG/VGG-{}-{}-{}-{}-{}.pth'.format(p1, p2, p3, p4, p5)))
        add_noise_to_weights(0, std, model)
        evaluated[i] = evaluate(model, valid_dl)['val_acc']

    accu = np.sum(evaluated) / num
    print(accu)
    if accu > 0.2:
        torch.save(model.state_dict(), './results/Bayes/VGG/VGG-{}@0.8.pth'.format(accu))
        with open('./results/Bayes/VGG/VGG-{}@0.8.txt'.format(accu), "w") as f:
            print(model, file=f)

    return accu


def step_VGG(n_iter):
    # Bounded region of parameter space
    pbounds = {'p1': (0.2, 0.8), 'p2': (0.2, 0.8), 'p3': (0.2, 0.8), 'p4': (0.2, 0.8), 'p5': (0.2, 0.8),
               'p6': (0.1, 0.5)}

    optimizer = BayesianOptimization(
        f=bbf_VGG,
        pbounds=pbounds,
        verbose=2,  # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
        random_state=1,
    )
    optimizer.probe(
        params={'p1': 0.2, 'p2': 0.2, 'p3': 0.2, 'p4': 0.5, 'p5': 0.5, 'p6': 0.2},
        lazy=True,
    )

    logger = JSONLogger(path="./results/Bayes/log/VGG/logs1.json")
    optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

    optimizer.maximize(
        init_points=3,
        n_iter=n_iter,
    )


def bbf_Res18(p1, p2, p3, p4, p5, p6):
    p1 = round(p1, 2)
    p2 = round(p2, 2)
    p3 = round(p3, 2)
    p4 = round(p4, 2)
    p5 = round(p5, 2)
    p6 = round(p6, 2)

    model = models.resnet18(pretrained=True)
    # Initial Layer
    model.bn1 = nn.Identity()

    # Layer1 
    model.layer1[0].bn1 = nn.Identity()
    model.layer1[0].relu = nn.Sequential(model.layer1[0].relu, nn.Dropout2d(p1))
    model.layer1[0].bn2 = nn.Identity()
    model.layer1[1].bn1 = nn.Identity()
    model.layer1[1].relu = nn.Sequential(model.layer1[0].relu, nn.Dropout2d(p1))
    model.layer1[1].bn2 = nn.Identity()

    model.layer2[0].bn1 = nn.Identity()
    model.layer2[0].relu = nn.Sequential(model.layer1[0].relu, nn.Dropout2d(p2))
    model.layer2[0].bn2 = nn.Identity()
    model.layer2[0].downsample[1] = nn.Identity()
    model.layer2[1].bn1 = nn.Identity()
    model.layer2[1].relu = nn.Sequential(model.layer1[0].relu, nn.Dropout2d(p2))
    model.layer2[1].bn2 = nn.Identity()

    model.layer3[0].bn1 = nn.Identity()
    model.layer3[0].relu = nn.Sequential(model.layer1[0].relu, nn.Dropout2d(p3))
    model.layer3[0].bn2 = nn.Identity()
    model.layer3[0].downsample[1] = nn.Identity()
    model.layer3[1].bn1 = nn.Identity()
    model.layer3[1].relu = nn.Sequential(model.layer1[0].relu, nn.Dropout2d(p3))
    model.layer3[1].bn2 = nn.Identity()

    model.layer4[0].bn1 = nn.Identity()
    model.layer4[0].relu = nn.Sequential(model.layer1[0].relu, nn.Dropout2d(p4))
    model.layer4[0].bn2 = nn.Identity()
    model.layer4[0].downsample[1] = nn.Identity()
    model.layer4[1].bn1 = nn.Identity()
    model.layer4[1].relu = nn.Sequential(model.layer1[0].relu, nn.Dropout2d(p4))
    model.layer4[1].bn2 = nn.Identity()

    model.fc = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Dropout(p5), nn.Linear(256, 10), nn.Dropout(p6))

    to_device(model, device)
    history = fit_one_cycle_Bayes(20, 0.0003, model, train_dl, valid_dl, opt_func=torch.optim.Adam)
    torch.save(model.state_dict(), './results/Bayes/Res18/Res-18-{}-{}-{}-{}-{}.pth'.format(p1, p2, p3, p4, p5))
    # AVERAGE RUN 30 TIMES
    num = 20
    std = 0.8
    evaluated = np.zeros(num)
    for i in range(num):
        model.load_state_dict(torch.load('./results/Bayes/Res18/Res-18-{}-{}-{}-{}-{}.pth'.format(p1, p2, p3, p4, p5)))
        add_noise_to_weights(0, std, model)
        evaluated[i] = evaluate(model, valid_dl)['val_acc']

    accu = np.sum(evaluated) / num
    print(accu)
    if accu > 0.2:
        torch.save(model.state_dict(), './results/Bayes/Res18/Res-18-{}@0.8.pth'.format(accu))
        with open('./results/Bayes/Res18/Res-18-{}@0.8.txt'.format(accu), "w") as f:
            print(model, file=f)

    return accu


def step_Res18(n_iter):
    # Bounded region of parameter space
    pbounds = {'p1': (0.2, 0.8), 'p2': (0.2, 0.8), 'p3': (0.2, 0.8), 'p4': (0.2, 0.8), 'p5': (0.2, 0.8),
               'p6': (0.1, 0.5)}

    optimizer = BayesianOptimization(
        f=bbf_Res18,
        pbounds=pbounds,
        verbose=2,  # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
        random_state=1,
    )
    optimizer.probe(
        params={'p1': 0.2, 'p2': 0.2, 'p3': 0.2, 'p4': 0.5, 'p5': 0.5, 'p6': 0.2},
        lazy=True,
    )

    logger = JSONLogger(path="./results/Bayes/log/Res18/logs1.json")
    optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

    optimizer.maximize(
        init_points=3,
        n_iter=n_iter,
    )


def bbf_SqueezeNet(p1, p2, p3, p4, p5, p6):
    p1 = round(p1, 2)
    p2 = round(p2, 2)
    p3 = round(p3, 2)
    p4 = round(p4, 2)
    p5 = round(p5, 2)
    p6 = round(p6, 2)
    from model.SqueezeNet import SqueezeNet_Bayes
    model = SqueezeNet_Bayes(p1, p2, p3, p4, p5, p6)

    model.bn1 = nn.Identity()
    to_device(model, device)
    history = fit_one_cycle_Bayes(20, 0.0003, model, train_dl, valid_dl, opt_func=torch.optim.Adam)
    torch.save(model.state_dict(), './results/Bayes/SqueezeNet/Squeeze-{}-{}-{}-{}-{}.pth'.format(p1, p2, p3, p4, p5))
    # AVERAGE RUN 30 TIMES
    num = 15
    std = 0.8
    evaluated = np.zeros(num)
    for i in range(num):
        model.load_state_dict(
            torch.load('./results/Bayes/SqueezeNet/Squeeze-{}-{}-{}-{}-{}.pth'.format(p1, p2, p3, p4, p5)))
        add_noise_to_weights(0, std, model)
        evaluated[i] = evaluate(model, valid_dl)['val_acc']

    accu = np.sum(evaluated) / num
    print(accu)
    if accu > 0.2:
        torch.save(model.state_dict(), './results/Bayes/SqueezeNet/Squeeze-{}@0.8.pth'.format(accu))
        with open('./results/Bayes/Res18/Res-18-{}@0.8.txt'.format(accu), "w") as f:
            print(model, file=f)

    return accu


def step_Squeeze(n_iter):
    # Bounded region of parameter space
    pbounds = {'p1': (0.2, 0.8), 'p2': (0.2, 0.8), 'p3': (0.2, 0.8), 'p4': (0.2, 0.8), 'p5': (0.2, 0.8),
               'p6': (0.1, 0.5)}

    optimizer = BayesianOptimization(
        f=bbf_SqueezeNet,
        pbounds=pbounds,
        verbose=2,  # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
        random_state=1,
    )
    optimizer.probe(
        params={'p1': 0.2, 'p2': 0.2, 'p3': 0.2, 'p4': 0.5, 'p5': 0.5, 'p6': 0.2},
        lazy=True,
    )

    logger = JSONLogger(path="./results/Bayes/log/SqueezeNet/logs1.json")
    optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

    optimizer.maximize(
        init_points=3,
        n_iter=n_iter,
    )
