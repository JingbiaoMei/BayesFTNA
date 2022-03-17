from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
import numpy as np
import torch
from utilis import add_noise_to_weights, fit_one_cycle_Bayes, evaluate, select_Data
from utilis_gpu import to_device, get_default_device
import matplotlib.pyplot as plt

device = get_default_device()
best_accu = 0
train_dl, valid_dl = select_Data("Mnist")


def run(model_name, n_iter):
    if model_name == 'LeNet':
        step_LeNet(n_iter)


def bbf_LeNet(p1, p2, p3, p4, p5):
    from model import LeNet
    p1 = round(p1, 10)
    p2 = round(p2, 10)
    p3 = round(p3, 10)
    p4 = round(p4, 10)
    p5 = round(p5, 10)
    print("Sigma update: ", p1, p2, p3, p4, p5)

    model = LeNet.LeNet_Laplace(p1, p2, p3, p4, p5)
    to_device(model, device)
    history = fit_one_cycle_Bayes(20, 0.01, model, train_dl, valid_dl,
                                  grad_clip=1e-1, weight_decay=1e-4, opt_func=torch.optim.Adam)
    torch.save(model.state_dict(), './results/Laplace/LeNet/LeNet-{}-{}-{}-{}-{}.pth'.format(p1, p2, p3, p4, p5))
    # AVERAGE RUN 30 TIMES
    num = 30
    std = 1.5
    evaluated = np.zeros(num)
    for i in range(num):
        model.load_state_dict(torch.load('./results/Laplace/LeNet/LeNet-{}-{}-{}-{}-{}.pth'.format(p1, p2, p3, p4, p5)))
        add_noise_to_weights(0, std, model)
        evaluated[i] = evaluate(model, valid_dl)['val_acc']

    accu = np.sum(evaluated) / num
    print(accu)

    global best_accu
    if accu > best_accu:
        model.load_state_dict(torch.load('./results/Laplace/LeNet/LeNet-{}-{}-{}-{}-{}.pth'.format(p1, p2, p3, p4, p5)))
        torch.save(model.state_dict(), './results/Laplace/LeNet/LeNet-state-BEST.pth'.format(accu))

        N = 10
        S = np.linspace(0., 1.5, 31)
        A = []
        E = np.zeros(N)
        for s in S:
            for n in range(N):
                # print(s)
                model.load_state_dict(torch.load('./results/Laplace/LeNet/LeNet-state-BEST.pth'.format(accu)))
                add_noise_to_weights(0, s, model)
                E[n] = evaluate(model, valid_dl)['val_acc']
            A.append(np.sum(E) / N)

        fig, ax = plt.subplots(1)
        ax.set_xlabel('$sigma$')
        ax.set_ylabel('Accuracy')
        ax.set_xticks([0, 0.3, 0.6, 0.9, 1.2, 1.5])
        ax.grid(True)
        ax.plot(S, A)
        ax.set_title("Laplace Evaluation with sigma {}".format(std))
        fig.savefig("./results/Laplace/LeNet/best_accu_{}.png".format(accu), dpi=320, bbox_inches='tight')
        results = np.vstack((S, np.array(A)))
        np.save("./results/Laplace/LeNet/best_accu_{}.npy".format(accu), results)
        best_accu = accu
        print("best accu: {}".format(best_accu))

    return accu


def step_LeNet(n_iter):
    # Bounded region of parameter space
    pbounds = {'p1': (0, 1e-3), 'p2': (0, 1e-3), 'p3': (0, 1e-3), 'p4': (0, 1e-3), 'p5': (0, 1e-3)}

    optimizer = BayesianOptimization(
        f=bbf_LeNet,
        pbounds=pbounds,
        verbose=2,  # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
        random_state=1,
    )
    optimizer.probe(
        params={'p1': 1e-6, 'p2': 1e-6, 'p3': 1e-6, 'p4': 1e-6, 'p5': 1e-6},
        lazy=True,
    )

    logger = JSONLogger(path="./results/Laplace/log/LeNet/logs1.json")
    optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

    optimizer.maximize(
        init_points=3,
        n_iter=n_iter,
    )
