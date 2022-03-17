import torch

from utilis import fit_one_cycle, evaluate, add_noise_to_weights
from utilis_gpu import to_device, get_default_device

device = get_default_device()

def step(model, train_dl, val_dl, std):
    #store the correct value in T
    T = [param for param in model.parameters()]
    to_device(T, device)
    FF = [torch.ones(t.size(), device=device) for t in T]
    add_noise_to_weights(0, std ,model)
    acc_f_training(model, T, FF, train_dl, val_dl, False, False)
    return model


def retrain(model, FF, train_dl, val_dl):
    i=0
    W=[]
    for param in model.parameters():
        W.append(param.mul(1-FF[i]))
        i+=1
    
    #retrain
    epochs = 1
    max_lr = 0.000
    grad_clip = 0.1
    weight_decay = 1e-4
    opt_func = torch.optim.Adam
    print('retraining...')
    history = fit_one_cycle(epochs, max_lr, model, train_dl, val_dl, grad_clip=grad_clip,   weight_decay=weight_decay, opt_func=opt_func)

    with torch.no_grad():
        i=0
        for param in model.parameters():
            param.mul_(to_device(FF[i], device))
            param.add_(to_device(W[i], device))
            #param.addcmul_(to_device((W[i], 1, param, FF[i]),device))
            i += 1
    result = evaluate(model, val_dl)
    print("retrained, val_loss: {:.4f}, val_acc: {:.4f}".format(result['val_loss'], result['val_acc']))

def batch_SWV(W,T):
    batch_size = 100
    batch_num = W.shape[0] // batch_size
    W_list = []
    T_list = []
    #split W and T into batches
    for x in range(batch_num):
        W_list.append(W[batch_size*x:batch_size*(x+1)])
        T_list.append(T[batch_size*x:batch_size*(x+1)])
    W_list.append(W[batch_size*(x+1):])
    T_list.append(T[batch_size*(x+1):])
    #calculate SWV using batches
    SWV_batch = torch.tensor([])
    SWV_line = torch.tensor([])
    for xw in range(batch_num+1):
        for xt in range(batch_num+1):
            if xt == 0:
                SWV_line = torch.cdist(W_list[xw],T_list[xt],p=1)
            else:
                SWV_line = torch.cat((SWV_line,torch.cdist(W_list[xw],T_list[xt],p=1)),1)
        if xw == 0:
            SWV_batch = SWV_line
        else:
            SWV_batch = torch.cat((SWV_batch, SWV_line),0)
    return SWV_batch



def update_SWV(W, T, SA0, SA1):
    #print(W.size())
    #print(T.size())
    print(W.shape)
    with torch.no_grad():
        if len(W.size()) == 1:
            m=W.size()[0]
            if SA0:
                SWV = torch.abs(W-torch.max(W)).repeat(m,1).transpose(0,1).to(device)
            elif SA1:
                SWV = torch.abs(W-torch.min(W)).repeat(m,1).transpose(0,1).to(device)
            else:
                SWV = torch.abs(torch.add(W.unsqueeze(0),-T.unsqueeze(0).transpose(0,1))).to(device)
        elif len(W.size()) == 2:
            m = W.size()[0]
            n = W.size()[1]
            if SA0:
                SWV = torch.sum(torch.abs(W-torch.max(W)),1).repeat(n,1).transpose(0,1).to(device)
            elif SA1:
                SWV = torch.sum(torch.abs(W-torch.min(W)),1).repeat(n,1).transpose(0,1).to(device)
            else:
                # W1 = W.unsqueeze(-1).expand(m,n,m).transpose(0,1)
                # T1 = T.unsqueeze(-1).expand(m,n,m).transpose(0,1).transpose(1,2)
                # SWV = torch.sum(torch.abs(W1 - T1),0).to(device)
                if W.shape[0] < 30000:
                    SWV = torch.cdist(W,T,p=1)
                else:
                    SWV = batch_SWV(W,T)
    return SWV

def acc_f_training(model, T, FF, train_dl, val_dl, SA0 = False, SA1 = False):
    """
    The algorithm for accelerator-friendly Training
    
    W: m*n weight matrix
    T: m*n Xbar_variation
    FF: matrix F to represent whether W_ji is fixed
    """
    #some initial values
    t=2
    Convergence = False
    test_rate = [evaluate(model, val_dl)['val_acc'],0,0,0,0]
    iter_num = 0
    test_rate_is_promoted = False
    while (not Convergence) or test_rate_is_promoted:
        iter_num += 1
        #print(test_rate)
        W = [param.to(device) for param in model.parameters()]
        matrix_mul = [torch.ones(w.size(), device=device) for w in W]
        for i in range(len(T)):
            #see if W and T has the same shape
            if W[i].size() != T[i].size():
                return "W, T doesn't have the same size"
            #update SWV
            #Flatten extra dimensions for the SWV
            flatten = False
            if W[i].ndim > 2:
                flatten = True
                shape = W[i].shape
                new_shape = torch.Size([ int(torch.prod(torch.tensor(shape))/ shape[-1]) , shape[-1] ])
                W[i] = torch.reshape(W[i], new_shape).to(device=device)
                FF[i] = torch.reshape(FF[i], new_shape).to(device=device)
                T[i] = torch.reshape(T[i], new_shape).to(device=device)
                matrix_mul[i] = torch.reshape(matrix_mul[i], new_shape).to(device=device)
            #SWV = update_SWV(W[i], T[i], SA0, SA1)
            #SWV = update_SWV(W[i], T[i], SA0, SA1)
            SWV = torch.abs(W[i]-T[i])
            #SWV = torch.abs(W[i]*T[i])
            #Find max variation and its position



            #Find max variation and its position
            num = torch.argmax(SWV)
            if W[i].ndim == 1:
                pos = num // W[i].size()[0]
                matrix_mul[i][pos] = 1/t
                #Revise the fix matrix
                FF[i][pos] = 0
            elif W[i].ndim == 2:
                #pos = [int(num // SWV.size()[1]), int(num % SWV.size()[1])]
                #print('original:',pos)
                row = num // SWV.size()[1]
                col = torch.argmax(W[i][row]-T[i][row])
                pos=[row,col]
                #print('final:',pos)
                matrix_mul[i][pos[0]][pos[1]] = 1/t
                #Revise the fix matrix
                FF[i][pos[0]][pos[1]] = 0
            #Reshape the flattened parameters to the original shape
            if flatten:
                W[i] = torch.reshape(W[i], shape).to(device=device)
                matrix_mul[i] = torch.reshape(matrix_mul[i], shape).to(device=device)
                FF[i] = torch.reshape(FF[i], shape).to(device=device)
                T[i] = torch.reshape(T[i], shape).to(device=device)
        #Reduce the weight W_ij
        with torch.no_grad():
            j=0
            for param in model.parameters():                 
                param.mul_(to_device(matrix_mul[j],device))
                j+=1
        print('finished rescaling')
        #Retrain and test NN
        retrain(model, FF, train_dl, val_dl)
        #Derive the Test_rate (accuracy)
        new_test_rate = evaluate(model, val_dl)['val_acc']
        print('Accuracy: ', new_test_rate)
        test_rate_is_promoted = False
        if new_test_rate > test_rate[0]:
            test_rate_is_promoted = True
        #Checking Convergence
        flag = False
        #if iter_num % 5 ==0:
        if True:
            print('iter:',iter_num)
            test_rate[4] = test_rate[3]
            test_rate[3] = test_rate[2]
            test_rate[2] = test_rate[1]
            test_rate[1] = test_rate[0]
            test_rate[0] = new_test_rate
            aver = sum(test_rate)/5
            flag = True
            for r in range(5):
                if abs(test_rate[r] - aver) > 0.005:
                    flag = False
                    #print(abs(test_rate[r] - aver))
                    break
        if flag and test_rate[4] != 0:
            print('end at iter',iter_num,', test rate:',test_rate[0])
            Convergence = True
