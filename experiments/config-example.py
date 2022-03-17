import torch 
model = 'ResNet-18'

# The seed used for shuffling dataset
seed = 1
# Save path 'for the trained models and results
save_path = './results/'
# List of methods to compare for the system
methods = ['ERM', 'FTNA', 'ReRam', 'Adv']
# Model parameters
model_param = dict(C=36,
                   num_classes=10,
                   layers=20,
                   steps=4,
                   multiplier=4,
                   stem_multiplier=3,
                   share=False,
                   AdPoolSize=1)

# Used dataset                   
dataSet = 'CIFAR10'


# ERM 
model_ERM_saved_name = model + '-ERM.pth'
Robust_ERM_saved_name = model + '-ERM.npy'

# Training hyperparameters
train = True
epoch = 20
max_lr = 0.01
grad_clip = 0.1
weight_decay = 1e-4
opt_func = torch.optim.Adam

# FTNA method
model_FTNA_saved_name = model + '-FTNA.pth'
Robust_FTNA_saved_name = model + '-FTNA.npy'
max_lr_FTNA = 1e-2

# ReRam method
model_ReRam_saved_name = model + '-RERAM.pth'
Robust_ReRam_saved_name = model + '-RERAM.npy'
ReRam_std = 1

# ADV training method
model_Adv_saved_name = model + '-Adv.pth'
Robust_Adv_saved_name = model + '-Adv.npy'
epoch_adv = epoch
lr_adv = max_lr


# Baye

bayes_parameter= dict(do = True)
'''
dataset_param = dict(data_root='../data/SVHN',
                     batch_size=32,
                     num_workers=2)
report_freq = 10
'''



#resume_path = dict(path='./checkpoint/RobNet_free_SVHN.pth.tar', origin_ckpt=True)




