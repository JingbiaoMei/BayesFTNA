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
import matplotlib.pyplot as plt

device = get_default_device()

train_dl, valid_dl = select_Data("CIFAR-10")
best_accu = 0


def run(model_name, n_iter):
    if model_name == 'VGG':
        step_VGG(n_iter)
    elif model_name == 'ResNet-18':
        step_Res18(n_iter)
    elif model_name == 'AlexNet':
        step_AlexNet(n_iter)
    else:
        print("Not supported model!")


def bbf_VGG(p1, p2, p3, p4, p5, p6):
    p1 = round(p1, 10)
    p2 = round(p2, 10)
    p3 = round(p3, 10)
    p4 = round(p4, 10)
    p5 = round(p5, 10)
    p6 = round(p6, 10)
    print("Sigma update: ", p1, p2, p3, p4, p5, p6)
    model = models.vgg11_bn(pretrained=True)
    model.features[1] = nn.Identity()
    model.features[2] = nn.Sequential(model.features[2], GaussLayer(p1))

    model.features[5] = nn.Identity()
    model.features[6] = nn.Sequential(model.features[6], GaussLayer(p1))

    model.features[9] = nn.Identity()
    model.features[10] = nn.Sequential(model.features[10], GaussLayer(p1))

    model.features[12] = nn.Identity()
    model.features[13] = nn.Sequential(model.features[13], GaussLayer(p2))

    model.features[16] = nn.Identity()
    model.features[17] = nn.Sequential(model.features[17], GaussLayer(p2))

    model.features[19] = nn.Identity()
    model.features[20] = nn.Sequential(model.features[20], GaussLayer(p2))

    model.features[23] = nn.Identity()
    model.features[24] = nn.Sequential(model.features[24], GaussLayer(p3))

    model.features[26] = nn.Identity()
    model.features[27] = nn.Sequential(model.features[27], GaussLayer(p3))

    model.classifier[2] = GaussLayer(p4)
    model.classifier[5] = GaussLayer(p5)
    model.classifier[6] = nn.Linear(4096, 10)
    model.classifier[6] = nn.Sequential(model.classifier[6], GaussLayer(p6))

    to_device(model, device)
    history = fit_one_cycle_Bayes(20, 0.001, model, train_dl, valid_dl, opt_func=torch.optim.Adam)
    torch.save(model.state_dict(), './results/Gauss/VGG/VGG-{}-{}-{}-{}-{}.pth'.format(p1, p2, p3, p4, p5))
    # AVERAGE RUN 30 TIMES
    num = 10
    std = 0.8
    evaluated = np.zeros(num)
    for i in range(num):
        model.load_state_dict(torch.load('./results/Gauss/VGG/VGG-{}-{}-{}-{}-{}.pth'.format(p1, p2, p3, p4, p5)))
        add_noise_to_weights(0, std, model)
        evaluated[i] = evaluate(model, valid_dl)['val_acc']

    accu = np.sum(evaluated) / num
    print(accu)
    if accu > 0.19:
        torch.save(model.state_dict(), './results/Gauss/VGG/VGG-{}@0.8.pth'.format(accu))
        with open('./results/Gauss/VGG/VGG-{}@0.8.txt'.format(accu), "w") as f:
            print(model, file=f)

    global best_accu
    if accu > best_accu and accu > 0.19:
        N = 10
        S = np.linspace(0., 1.5, 31)
        A = []
        E = np.zeros(N)
        for s in S:
            for n in range(N):
                # print(s)
                model.load_state_dict(torch.load('./results/Gauss/VGG/VGG-{}-{}-{}-{}-{}.pth'.format(p1, p2, p3, p4, p5)))
                add_noise_to_weights(0, s, model)
                E[n] = evaluate(model, valid_dl)['val_acc']
            A.append(np.sum(E) / N)

        fig, ax = plt.subplots(1)
        ax.set_xlabel('$sigma$')
        ax.set_ylabel('Accuracy')
        ax.set_xticks([0, 0.3, 0.6, 0.9, 1.2, 1.5])
        ax.grid(True)
        ax.plot(S, A)
        ax.set_title("Gauss Evaluation with sigma {}".format(std))
        fig.savefig("./results/Gauss/VGG/best_accu{}_constraint_e-6_e-3.png".format(accu), dpi=320, bbox_inches='tight')
        results = np.vstack((S, np.array(A)))
        np.save("./results/Gauss/VGG/best_accu{}_constraint_e-6_e-3.npy".format(accu), results)
        best_accu = accu
        print("best accu: {}".format(best_accu))

    return accu


def step_VGG(n_iter):
    # Bounded region of parameter space
    pbounds = {'p1': (0, 1e-3), 'p2': (0, 1e-3), 'p3': (0, 1e-3), 'p4': (0, 1e-3), 'p5': (0, 1e-3),
               'p6': (0, 1e-3)}

    optimizer = BayesianOptimization(
        f=bbf_VGG,
        pbounds=pbounds,
        verbose=2,  # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
        random_state=1,
    )
    optimizer.probe(
        params={'p1': 1e-6, 'p2': 1e-6, 'p3': 1e-6, 'p4': 1e-6, 'p5': 1e-6, 'p6': 1e-6},
        lazy=True,
    )

    logger = JSONLogger(path="./results/Gauss/log/VGG/logs1.json")
    optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

    optimizer.maximize(
        init_points=3,
        n_iter=n_iter,
    )


def bbf_Res18(p1, p2, p3, p4, p5, p6):
    p1 = round(p1, 10)
    p2 = round(p2, 10)
    p3 = round(p3, 10)
    p4 = round(p4, 10)
    p5 = round(p5, 10)
    p6 = round(p6, 10)

    print("Sigma update: ", p1, p2, p3, p4, p5, p6)
    model = models.resnet18(pretrained=True)
    # Initial Layer
    model.bn1 = nn.Identity()

    # Layer1
    model.layer1[0].bn1 = nn.Identity()
    model.layer1[0].relu = nn.Sequential(model.layer1[0].relu, GaussLayer(p1))
    model.layer1[0].bn2 = nn.Identity()
    model.layer1[1].bn1 = nn.Identity()
    model.layer1[1].relu = nn.Sequential(model.layer1[0].relu, GaussLayer(p1))
    model.layer1[1].bn2 = nn.Identity()

    model.layer2[0].bn1 = nn.Identity()
    model.layer2[0].relu = nn.Sequential(model.layer1[0].relu, GaussLayer(p2))
    model.layer2[0].bn2 = nn.Identity()
    model.layer2[0].downsample[1] = nn.Identity()
    model.layer2[1].bn1 = nn.Identity()
    model.layer2[1].relu = nn.Sequential(model.layer1[0].relu, GaussLayer(p2))
    model.layer2[1].bn2 = nn.Identity()

    model.layer3[0].bn1 = nn.Identity()
    model.layer3[0].relu = nn.Sequential(model.layer1[0].relu, GaussLayer(p3))
    model.layer3[0].bn2 = nn.Identity()
    model.layer3[0].downsample[1] = nn.Identity()
    model.layer3[1].bn1 = nn.Identity()
    model.layer3[1].relu = nn.Sequential(model.layer1[0].relu, GaussLayer(p3))
    model.layer3[1].bn2 = nn.Identity()

    model.layer4[0].bn1 = nn.Identity()
    model.layer4[0].relu = nn.Sequential(model.layer1[0].relu, GaussLayer(p4))
    model.layer4[0].bn2 = nn.Identity()
    model.layer4[0].downsample[1] = nn.Identity()
    model.layer4[1].bn1 = nn.Identity()
    model.layer4[1].relu = nn.Sequential(model.layer1[0].relu, GaussLayer(p4))
    model.layer4[1].bn2 = nn.Identity()

    model.fc = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), GaussLayer(p5), nn.Linear(256, 10), GaussLayer(p6))

    to_device(model, device)
    history = fit_one_cycle_Bayes(20, 0.0003, model, train_dl, valid_dl, opt_func=torch.optim.Adam)
    torch.save(model.state_dict(), './results/Gauss/Res18/Res-18-{}-{}-{}-{}-{}.pth'.format(p1, p2, p3, p4, p5))
    # AVERAGE RUN 30 TIMES
    num = 20
    std = 0.8
    evaluated = np.zeros(num)
    for i in range(num):
        model.load_state_dict(torch.load('./results/Gauss/Res18/Res-18-{}-{}-{}-{}-{}.pth'.format(p1, p2, p3, p4, p5)))
        add_noise_to_weights(0, std, model)
        evaluated[i] = evaluate(model, valid_dl)['val_acc']

    accu = np.sum(evaluated) / num
    print(accu)

    global best_accu
    if accu > best_accu:
        best_accu = accu

        model.load_state_dict(torch.load('./results/Gauss/Res18/Res-18-{}-{}-{}-{}-{}.pth'.format(p1, p2, p3, p4, p5)))
        torch.save(model.state_dict(), './results/Gauss/Res18/Res18-state-BEST.pth'.format(accu))

        N = 10
        S = np.linspace(0., 1.5, 31)
        A = []
        E = np.zeros(N)
        for s in S:
            for n in range(N):
                # print(s)
                model.load_state_dict(torch.load('./results/Gauss/Res18/Res18-state-BEST.pth'.format(accu)))
                add_noise_to_weights(0, s, model)
                E[n] = evaluate(model, valid_dl)['val_acc']
            A.append(np.sum(E) / N)
        
        fig, ax = plt.subplots(1)
        ax.set_xlabel('$sigma$')
        ax.set_ylabel('Accuracy')
        ax.set_xticks([0, 0.3, 0.6, 0.9, 1.2, 1.5])
        ax.grid(True)
        ax.plot(S, A)
        ax.set_title("Gauss Evaluation with sigma {}".format(std))
        fig.savefig("./results/Gauss/Res18/best_accu_{}.png".format(accu), dpi=320, bbox_inches='tight')
        results = np.vstack((S, np.array(A)))
        np.save("./results/Gauss/Res18/best_accu_{}.npy".format(accu), results)
        best_accu = accu
        print("best accu: {}".format(best_accu))

    return accu


def step_Res18(n_iter):
    # Bounded region of parameter space
    pbounds = {'p1': (0, 1e-3), 'p2': (0, 1e-3), 'p3': (0, 1e-3), 'p4': (0, 1e-3), 'p5': (0, 1e-3),
               'p6': (0, 1e-3)}

    optimizer = BayesianOptimization(
        f=bbf_Res18,
        pbounds=pbounds,
        verbose=2,  # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
        random_state=1,
    )
    optimizer.probe(
        params={'p1': 1e-6, 'p2': 1e-6, 'p3': 1e-6, 'p4': 1e-6, 'p5': 1e-6, 'p6': 1e-6},
        lazy=True,
    )

    logger = JSONLogger(path="./results/Gauss/log/Res18/logs1.json")
    optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

    optimizer.maximize(
        init_points=3,
        n_iter=n_iter,
    )


def bbf_AlexNet(p1, p2, p3, p4, p5, p6, p7, p8):
    p1 = round(p1, 10)
    p2 = round(p2, 10)
    p3 = round(p3, 10)
    p4 = round(p4, 10)
    p5 = round(p5, 10)
    p6 = round(p6, 10)
    p7 = round(p7, 10)
    p8 = round(p8, 10)

    print("Sigma update: ", p1, p2, p3, p4, p5, p6, p7, p8)
    model = models.alexnet(pretrained=True)

    model.features[1] = nn.Sequential(model.features[1], GaussLayer(p1))
    model.features[4] = nn.Sequential(model.features[4], GaussLayer(p2))
    model.features[7] = nn.Sequential(model.features[7], GaussLayer(p3))
    model.features[9] = nn.Sequential(model.features[9], GaussLayer(p4))
    model.features[11] = nn.Sequential(model.features[11], GaussLayer(p5))

    model.classifier[0] = GaussLayer(p6)
    model.classifier[3] = GaussLayer(p7)
    model.classifier[6] = nn.Sequential(model.classifier[6], nn.ReLU(inplace=True), GaussLayer(p8),
                                        nn.Linear(1000, 10))
    to_device(model, device)
    history = fit_one_cycle_Bayes(20, 0.0003, model, train_dl, valid_dl, opt_func=torch.optim.Adam)
    torch.save(model.state_dict(), './results/Gauss/AlexNet/AlexNet-{}-{}-{}-{}-{}.pth'.format(p1, p2, p3, p4, p5))
    # AVERAGE RUN 30 TIMES
    num = 20
    std = 0.8
    evaluated = np.zeros(num)
    for i in range(num):
        model.load_state_dict(torch.load('./results/Gauss/AlexNet/AlexNet-{}-{}-{}-{}-{}.pth'.format(p1, p2, p3, p4, p5)))
        add_noise_to_weights(0, std, model)
        evaluated[i] = evaluate(model, valid_dl)['val_acc']

    accu = np.sum(evaluated) / num
    print(accu)

    global best_accu
    if accu > best_accu:
        model.load_state_dict(
            torch.load('./results/Gauss/AlexNet/AlexNet-{}-{}-{}-{}-{}.pth'.format(p1, p2, p3, p4, p5)))
        torch.save(model.state_dict(), './results/Gauss/AlexNet/AlexNet-BEST.pth'.format(accu))

        N = 10
        S = np.linspace(0., 1.5, 31)
        A = []
        E = np.zeros(N)
        for s in S:
            for n in range(N):
                # print(s)
                model.load_state_dict(torch.load('./results/Gauss/AlexNet/AlexNet-BEST.pth'.format(accu)))
                add_noise_to_weights(0, s, model)
                E[n] = evaluate(model, valid_dl)['val_acc']
            A.append(np.sum(E) / N)

        fig, ax = plt.subplots(1)
        ax.set_xlabel('$sigma$')
        ax.set_ylabel('Accuracy')
        ax.set_xticks([0, 0.3, 0.6, 0.9, 1.2, 1.5])
        ax.grid(True)
        ax.plot(S, A)
        ax.set_title("Gauss Evaluation with sigma {}".format(std))
        fig.savefig("./results/Gauss/AlexNet/best_accu_{}.png".format(accu), dpi=320, bbox_inches='tight')
        results = np.vstack((S, np.array(A)))
        np.save("./results/Gauss/AlexNet/best_accu_{}.npy".format(accu), results)

        best_accu = accu
        print("best accu: {}".format(best_accu))

    return accu


def step_AlexNet(n_iter):
    # Bounded region of parameter space
    pbounds = {'p1': (0, 1e-3), 'p2': (0, 1e-3), 'p3': (0, 1e-3), 'p4': (0, 1e-3), 'p5': (0, 1e-3),
               'p6': (0, 1e-3), 'p7': (0, 1e-3), 'p8': (0, 1e-3)}

    optimizer = BayesianOptimization(
        f=bbf_AlexNet,
        pbounds=pbounds,
        verbose=2,  # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
        random_state=1,
    )
    optimizer.probe(
        params={'p1': 1e-6, 'p2': 1e-6, 'p3': 1e-6, 'p4': 1e-6, 'p5': 1e-6, 'p6': 1e-6, 'p7': 1e-6, 'p8': 1e-6},
        lazy=True,
    )

    logger = JSONLogger(path="./results/Gauss/log/AlexNet/logs1.json")
    optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

    optimizer.maximize(
        init_points=3,
        n_iter=n_iter,
    )
