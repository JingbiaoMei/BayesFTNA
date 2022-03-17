import os
import numpy as np

# Automatic select the gpu with lowest memory load
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
memory_gpu = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
print('Using GPU:' + str(np.argmax(memory_gpu)))
os.environ["CUDA_VISIBLE_DEVICES"] = str(np.argmax(memory_gpu))
os.system('rm tmp')

# imports
import argparse
import torch
from mmcv import Config
from utilis import select_Data, select_Model, fit_one_cycle, fit_one_cycle_adv, evaluate_robustness
from assets import FTNA, confusion_matrix, ReRam
import os

# Initialise argparser
parser = argparse.ArgumentParser()
parser.add_argument('--config', '-c', type=str, default='./experiments/config.py',
                    help='location of the config file')
# Get the argument
args = parser.parse_args()

# main
def main():
    # define global variable for the confid file and data loader
    global cfg, train_dl, valid_dl

    """# Cuda/CPU setup
    device = get_default_device()"""

    # Get config file
    cfg = Config.fromfile(args.config)

    # Set seed
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)

    # Select DataSet 
    # MNIST CIFAR10 ..
    # print(cfg.dataSet)
    train_dl, valid_dl = select_Data(cfg.dataSet)

    # Select Model 
    model = select_Model(cfg.model, cfg.model_param['num_classes'])
    print(model)
    model_ERM_path = os.path.join(cfg.save_path, cfg.model_ERM_saved_name)
    # Method
    # ERM / FTNA / RERAM (ACC-Friendly Paper) / ADVTRAIN / Bayesian / NAS
    for method in cfg.methods:
        if method == "ERM":
            print("<------ERM training started------>")
            # Continue training from the last saved model path
            if cfg.contin == True:
                model.load_state_dict(torch.load(model_ERM_path))
            # Train the normal way  
            model, model_ERM_path = train_normal(model)

            # Evaluation the trained model
            if cfg.eval == True:
                # Go straight to Evaluation
                sigma, accu = evaluate_robustness(model, model_ERM_path, valid_dl)
                
                results = np.vstack((sigma, np.array(accu)))

                np.save(os.path.join(cfg.save_path, cfg.Robust_ERM_saved_name), results)

            print("<------ERM Evaluation Completed------>")

        elif method == "FTNA":
            # Train FTNA
            print("<------FTNA training started------>")
            # Train the normal way first to get confusion matrix
            try:
                model.load_state_dict(torch.load(model_ERM_path))
            except:
                model, _ = train_normal(model)
            cm = confusion_matrix.Confusion_Matrix(model, valid_dl, cfg.model_param['num_classes'])
            codebook = FTNA.CodeToClass(cm, cfg.model_param['num_classes'],
                                        FTNA.Searching_code(cfg.model_param['num_classes'], 7, 3))

            # Change FC layer for FTNA model
            model_FTNA = FTNA.model_alter(model, cfg.model)

            # Train FTNA
            FTNA.fit_one_cycle_FTNA(cfg.epoch, cfg.max_lr_FTNA, model_FTNA,
                                    train_dl, valid_dl, codebook, cfg.weight_decay,
                                    cfg.grad_clip, cfg.opt_func)

            # Save and evaluate 
            model_save_path = os.path.join(cfg.save_path, cfg.model_FTNA_saved_name)
            torch.save(model_FTNA.state_dict(), model_save_path)

            sigma, accu = FTNA.evaluate_FTNA_robustness(model, model_save_path, valid_dl, codebook)

            results = np.vstack((sigma, np.array(accu)))

            np.save(os.path.join(cfg.save_path, cfg.Robust_FTNA_saved_name), results)

            print("<------FTNA Evaluation Completed------>")

        elif method == "ReRam":
            print("<------ReRam training started------>")

            try:
                model = select_Model(cfg.model, cfg.model_param['num_classes'])
                model.load_state_dict(torch.load(model_ERM_path))
            except:
                model = select_Model(cfg.model, cfg.model_param['num_classes'])
                model, _ = train_normal(model)
            # Train ReRam
            # Check if already trained if interrupted during evaluation 
            try:
                model_ReRam = select_Model(cfg.model, cfg.model_param['num_classes'])
                model_ReRam.load_state_dict(torch.load(os.path.join(cfg.save_path, cfg.model_ReRam_saved_name)))
            except:
                model_ReRam = ReRam.step(model, train_dl, valid_dl, cfg.ReRam_std)

            # Save and Evaluate  
            model_save_path = os.path.join(cfg.save_path, cfg.model_ReRam_saved_name)

            torch.save(model_ReRam.state_dict(), model_save_path)

            sigma, accu = evaluate_robustness(model_ReRam, model_save_path, valid_dl)

            results = np.vstack((sigma, np.array(accu)))

            np.save(os.path.join(cfg.save_path, cfg.Robust_ReRam_saved_name), results)

            print("<------ReRam Evaluation Completed------>")

        elif method == "Adv":
            print("<------Adv training started------>")
            model = select_Model(cfg.model, cfg.model_param['num_classes'])
            model_Adv_path = os.path.join(cfg.save_path, cfg.model_Adv_saved_name)

            if cfg.contin_adv == True:
                model.load_state_dict(torch.load(model_Adv_path))

            model, model_Adv_path = train_adv(model, cfg.epsilon)
            # model, model_ERM_path = train_normal(model)

            if cfg.eval_adv == True:
                # Go straight to Evaluation
                sigma, accu = evaluate_robustness(model, model_Adv_path, valid_dl)

                results = np.vstack((sigma, np.array(accu)))

                np.save(os.path.join(cfg.save_path, cfg.Robust_Adv_saved_name), results)

            print("<------Adv Evaluation Completed------>")

        elif method == "Bayes":

            print("<------Bayes optimization started------>")
            if cfg.dataSet == 'Mnist':
                from assets import Bayes_Mnist
                Bayes_Mnist.run(cfg.model, cfg.Bayes_iter)
            elif cfg.dataSet == 'CIFAR10':
                from assets import Bayes_Cifar
                Bayes_Cifar.run(cfg.model, cfg.Bayes_iter)
            elif cfg.dataSet == 'Traffic-Sign':
                from assets import Bayes_Traffic
                Bayes_Traffic.run(cfg.model, cfg.Bayes_iter)
            print("<------Bayes optimization Completed------>")

        elif method == "Gauss":
            print("<------Gauss optimization started------>")
            if cfg.dataSet == 'CIFAR10':
                from assets import Gauss_Cifar
                Gauss_Cifar.run(cfg.model, cfg.Bayes_iter)
            elif cfg.dataSet == 'Traffic-Sign':
                from assets import Gauss_Traffic
                Gauss_Traffic.run(cfg.model, cfg.Bayes_iter)
            elif cfg.dataSet == 'Mnist':
                from assets import Gauss_Mnist
                Gauss_Mnist.run(cfg.model, cfg.Bayes_iter)
            else:
                print("Not supported data!")
            print("<------Gauss optimization Completed------>")
        elif method == "Laplace":
            print("<------Laplace optimization started------>")
            if cfg.dataSet == 'CIFAR10':
                from assets import Laplace_Cifar
                Laplace_Cifar.run(cfg.model, cfg.Bayes_iter)
            elif cfg.dataSet == 'Traffic-Sign':
                from assets import Laplace_Traffic
                Laplace_Traffic.run(cfg.model, cfg.Bayes_iter)
            elif cfg.dataSet == 'Mnist':
                from assets import Laplace_Mnist
                Laplace_Mnist.run(cfg.model, cfg.Bayes_iter)
            else:
                print("Not supported data!")
            print("<------Laplace optimization Completed------>")

        elif method == "Bernoulli":
            print("<------Bernoulli optimization started------>")
            if cfg.dataSet == 'CIFAR10':
                from assets import Bernoulli_Cifar
                Bernoulli_Cifar.run(cfg.model, cfg.Bayes_iter)
            else:
                print("Not supported data!")
            print("<------Bernoulli optimization Completed------>")
        elif method == "Gamma":
            print("<------Gamma optimization started------>")
            if cfg.dataSet == 'CIFAR10':
                from assets import Gamma_Cifar
                Gamma_Cifar.run(cfg.model, cfg.Bayes_iter)
            else:
                print("Not supported data!")
            print("<------Gamma optimization Completed------>")
        elif method == "pureSigma":
            print("<------pureSigma optimization started------>")
            if cfg.dataSet == 'CIFAR10':
                from assets import Sigma_Cifar
                Sigma_Cifar.run(cfg.model, cfg.Bayes_iter)
            else:
                print("Not supported data!")
            print("<------pureSigma optimization Completed------>")
        elif method == "Beta":
            print("<------Beta optimization started------>")
            if cfg.dataSet == 'CIFAR10':
                from assets import Beta_Cifar
                Beta_Cifar.run(cfg.model, cfg.Bayes_iter)
            else:
                print("Not supported data!")
            print("<------Beta optimization Completed------>")
        elif method == "Poisson":
            print("<------Poisson optimization started------>")
            if cfg.dataSet == 'CIFAR10':
                from assets import Poisson_Cifar
                Poisson_Cifar.run(cfg.model, cfg.Bayes_iter, cfg.sParamNum)
            else:
                print("Not supported data!")
            print("<------Poisson optimization Completed------>")
        elif method == "Pareto":
            print("<------Pareto optimization started------>")
            if cfg.dataSet == 'CIFAR10':
                from assets import Pareto_Cifar
                Pareto_Cifar.run(cfg.model, cfg.Bayes_iter)
            else:
                print("Not supported data!")
            print("<------Pareto optimization Completed------>")
        elif method == "Multinomial":
            print("<------Multinomial optimization started------>")
            if cfg.dataSet == 'CIFAR10':
                from assets import Multinomial_Cifar
                Multinomial_Cifar.run(cfg.model, cfg.Bayes_iter)
            else:
                print("Not supported data!")
            print("<------Multinomial optimization Completed------>")
        else:
            raise NotImplementedError("Method not implemented")


def train_normal(model):
    # Train 
    model_save_path = os.path.join(cfg.save_path, cfg.model_ERM_saved_name)
    if cfg.train == True:
        History = fit_one_cycle(cfg.epoch, cfg.max_lr, model, train_dl,
                                valid_dl, cfg.weight_decay, cfg.grad_clip, cfg.opt_func)
        torch.save(model.state_dict(), model_save_path)
    return model, model_save_path


def train_adv(model, epsilon):
    # Train 
    model_save_path = os.path.join(cfg.save_path, cfg.model_Adv_saved_name)
    if cfg.train == True:
        History = fit_one_cycle_adv(cfg.epoch_adv, cfg.lr_adv, model, train_dl,
                                    valid_dl, cfg.weight_decay_adv, cfg.grad_clip, epsilon=epsilon)
        torch.save(model.state_dict(), model_save_path)
    return model, model_save_path


if __name__ == '__main__':
    main()
