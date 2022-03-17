from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
import numpy as np
import torch
from utilis import add_noise_to_weights, fit_one_cycle_Bayes, evaluate, select_Data
from utilis_gpu import to_device, get_default_device

device = get_default_device()

train_dl, valid_dl = select_Data("Mnist")


def run(model_name, n_iter):
    if model_name == 'LeNet':
        step_LeNet(n_iter)


def bbf_LeNet(p1, p2, p3, p4, p5):
    from model import LeNet
    model = LeNet.LeNet_Bayes(p1, p2, p3, p4, p5)
    to_device(model, device)
    history = fit_one_cycle_Bayes(20, 0.01, model, train_dl, valid_dl,
                                  grad_clip=1e-1, weight_decay=1e-4, opt_func=torch.optim.Adam)
    torch.save(model.state_dict(), './results/Bayes/LeNet/LeNet-{}-{}-{}-{}-{}.pth'.format(p1, p2, p3, p4, p5))
    # AVERAGE RUN 30 TIMES
    num = 30
    std = 1.5
    evaluated = np.zeros(num)
    for i in range(num):
        model.load_state_dict(torch.load('./results/Bayes/LeNet/LeNet-{}-{}-{}-{}-{}.pth'.format(p1, p2, p3, p4, p5)))
        add_noise_to_weights(0, std, model)
        evaluated[i] = evaluate(model, valid_dl)['val_acc']

    accu = np.sum(evaluated) / num
    print(accu)
    if accu > 0.4:
        torch.save(model.state_dict(), './results/Bayes/LeNet/LeNet-{}@1.5.pth'.format(accu))
        with open('./results/Bayes/LeNet/LeNet-{}@1.5.txt'.format(accu), "w") as f:
            print(model, file=f)

    return accu


def step_LeNet(n_iter):
    # Bounded region of parameter space
    pbounds = {'p1': (0.2, 0.8), 'p2': (0.2, 0.8), 'p3': (0.2, 0.8), 'p4': (0.2, 0.8), 'p5': (0.2, 0.8)}

    optimizer = BayesianOptimization(
        f=bbf_LeNet,
        pbounds=pbounds,
        verbose=2,  # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
        random_state=1,
    )
    optimizer.probe(
        params={'p1': 0.2, 'p2': 0.2, 'p3': 0.2, 'p4': 0.5, 'p5': 0.5},
        lazy=True,
    )

    logger = JSONLogger(path="./results/Bayes/log/LeNet/logs1.json")
    optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

    optimizer.maximize(
        init_points=3,
        n_iter=n_iter,
    )
