import os
import numpy as np
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
memory_gpu=[int(x.split()[2]) for x in open('tmp','r').readlines()]
print('Using GPU:'+str(np.argmax(memory_gpu)))
os.environ["CUDA_VISIBLE_DEVICES"]=str(np.argmax(memory_gpu))
os.system('rm tmp')

import torch
import torch.backends.cudnn as cudnn
import argparse
from mmcv import Config
import assets.architecture_code as archiecture_code
import models_nas_fast
import models_nas
from utilis import select_Data, get_default_device, to_device, add_noise_to_weights_out, enumerate_robnet_large
import json


parser = argparse.ArgumentParser()
parser.add_argument('--config', '-c', type=str, default='./experiments/config.py',
                    help='location of the config file')
args = parser.parse_args()

device = get_default_device()


# combine the code from
#neural architecture search without training
#ref:https://github.com/BayesWatch/nas-without-training/blob/master/search.py
def get_batch_jacobian(net, x, target):
    net.zero_grad()

    x.requires_grad_(True)

    y = net(x)

    y.backward(torch.ones_like(y))
    jacob = x.grad.detach()

    return jacob, target.detach()

def eval_score(jacob, labels):
    corrs = np.corrcoef(jacob)
    v, _  = np.linalg.eig(corrs)
    k = 1e-5
    return -np.sum(np.log(v + k) + 1./(v + k))    

def main():
    global cfg, rank, world_size

    cfg = Config.fromfile(args.config)
    train_dl, valid_dl = select_Data(cfg.dataSet)
   
    # Set seed
    np.random.seed(cfg.seed)
    cudnn.benchmark = True
    torch.manual_seed(cfg.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(cfg.seed)

    # Model
    #print('==> Building model..')
    #add dropout rate/batch norm configurations set default value to be 1 for now
    #arch_code = eval('architecture_code.{}'.format(cfg.model))
    #net = models_nas.model_entry(cfg, arch_code)
    
    # create all possible arch_code_set:
    
    arch_code_set = enumerate_robnet_large(cfg.number)
    print("Loaded, Length of the arch_code:", len(arch_code_set))
    
    # 
    Good_set = [] # 10% highest scores with arch code 
    
    Var_set = [] # Variance set
    
    Good_list = [] # A list of dictionary containing all the results



    # Iterate through all possible architecture codes
    for num, arch_code in enumerate(arch_code_set) :
        #calculate the number of active parameters of the arch
        #if it is bigger than a threshold then continue, e.g. ResNet18 or Vgg or 5 million
        #Note: this step can save a lot of time by limiting the complexity of neural network
        
        # Load the model from the architecture coding
        net = models_nas.model_entry(cfg, [arch_code])
        to_device(net, device)
        
        s = eval_all(net, train_dl)
        print("Arch No. :", num , "   Evaluation Score:", s)
        # If the evaluated score is higher than top 10% of the set of scores
        if len(Good_set) < 10 or s > Good_set[len(Good_set) // 10][0]:
                
            # Append the set with the eval score and the code 
            Good_set.append([s, [arch_code]])
            
            # Sort the set with score ascending 
            Good_set.sort()
                
            # Noise set of Eval_Score    
            s_set = []
                
            for i in range(5):
                    
                # Load the model and add noise
                net = models_nas.model_entry(cfg, [arch_code])
                to_device(net, device)
                net = add_noise_to_weights_out(0, cfg.std, net)
                    
                s_noise = eval_all(net, train_dl)

                # Append the noise set    
                s_set.append(s_noise)

            #compute the variance of s_set
            Var_set.append(np.var(s_set))
            Good_list.append({'initial_score': s, 'noised_score': np.mean(s_set), 
                              'variance': np.var(s_set), 'arch_code': [arch_code] })
    
    
    # Sort the arch_code set based on Var_set, we want the network to have smaller variance with regard to the weight perturbations
    Good_list = sorted(Good_list, key= lambda i: i['variance'])
    
    # This will return the sorted good arch code with first one being 
    # the lowest variance 
    arch_code_set_good = [d['arch_code'] for d in Good_list]

    with open("./results/NAS/run.json", 'w') as f:
        json.dump(Good_list, f) 


def eval_all(net, train_dl):
    
    # List of all eval_score loop throgh train_dl
    s = []
    
    for (data, label) in train_dl:
            
        
        jacobs, labels = get_batch_jacobian(net, data, label)
        jacobs = jacobs.reshape(jacobs.size(0), -1).cpu().numpy()
        s.append(eval_score(jacobs, labels)) 
    
    return np.mean(s)




'''
    for arch_code in arch_code_set_good:
        #TODO: Bayesian optimization for dropout on the network with arch_code

    #Return the best performed neural network archiecture.
'''
if __name__ == '__main__':
    main()


