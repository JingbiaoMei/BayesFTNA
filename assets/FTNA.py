import itertools
import numpy as np 
import torch
# Some helper functions 
from utilis import add_noise_to_weights
from utilis_gpu import to_device, get_default_device
device = get_default_device()
import torch.nn as nn
# 1. Generate all the conbination of bits of certain length
def all_Combination_Bits(bit_len):
    result = []
    # Iterate for probable 0-0s 1-0s 2-0s till 4-0s for bit_length of 4
    for i in range(bit_len + 1):   
        # Iterate for combinations 
        for j in itertools.combinations(range(bit_len), i):
            s = ['0'] * bit_len
            # j would be a tuple 
            # Iterate through the combination (j) to set the position to be 1 
            for k in j:
                s[k] = '1'
            result.append(''.join(s))
    return result

# 2. Find the Hamming distance between two coding in string 
def Hamming(str1, str2):
    ham = 0
    if len(str1) != len(str2):
        return "Length error"
    for i in range(len(str1)):
        if str1[i] != str2[i]:
            ham += 1
    return ham


# Code length number of classifiers 
#l = 7    
# Initial Hamming distance 
#h = 7

# Testing for above helper fucntion
"""seven_bit_combination = all_Combination_Bits(l)
print(type(seven_bit_combination))        
print(Hamming(seven_bit_combination[0], seven_bit_combination[1]))
print(Hamming('0000','1111'))
print(Hamming('0000','11110'))"""


# Create searching table for codings
def Generate_Search_Table(l, h):
    '''
    Param l: collaborative classifier number
    Param h: Minimum required hamming distance
    return: List of coding table
    '''
    # Generate all coding for a bit length
    coding_list = all_Combination_Bits(l) 
    
    # Initialize searching list as the last one (all ones)
    T = []
    T.append(coding_list[-1])
    
    # Find all the codings list until h = 3
    while h >= 3:
        #print(h)
        # Iterate through all the codings in the coding list 
        for coding in coding_list:
            # Initialize the flag for satisfying hamming distance to be true
            # flag here because need to be true for all current flag before append
            flag_Hamm = True
            
            # Iterate through current searching table 
            for t in T:
                if Hamming(coding, t) < h:
                    # If the Hamming distance is lower than the setted value then set flag to be false
                    flag_Hamm = False
                    
        
            # if this coding satisfies the Hamming distance w.r.t all the previous codings t
            # then append this current codings to the list 
            if flag_Hamm == True:
                T.append(coding)
                #print("Appended T is:", T)
        
        # If the Hamming distance set to be to large resulted in no newly added coding
        # then reduce the Hamming distance
        h = h - 1
    return T    


def Searching_code(m, l, h):
    '''
    param m: Number of classes
    param l: Number of classifiers
    param h: Number of Minimum Hamming distance
    return: List of coding length of m
    '''
    # Generate search table
    table = Generate_Search_Table(l, h)

    # Randomly chosen m codings for the problem
    np.random.shuffle(table)

    o = table[:m]
    o.sort()
    """
    # Iterate through m-1 classes as first one is chosen randomly 
    for i in range(m-1):
        # set a lower hamming distance than chosen the greatest one 
        hamm = 2
            
        for j in range(len(table)):
    """
    # Sort from smallest to largest
    return o
# test
#print(Searching_code(3, 7, 3))

# Assign searching code to class 
def CodeToClass(confusion_mat, num_classes, code_book):
    '''
    confusion_mat: The input confusion matrix, numpy array m * m
    num_classes: number of classes m
    code_book: generated codebook 
    return: dictionary of coding list

    '''
    
    # Initialize j as number of classes, as j reaches 0, then stop
    j = num_classes
    # Initialize the returned dictionary
    S = {} 

    # Loop Until j=0
    while j > 0:
        # Pop Maximum value in the confusion matrix
        # No such operation in numpy matrix library
        # Instead, get the max value and
        # y-Vertical, True, Row No. / x-Horizontal, Predicted, Column No.
        # After that, set that value to be zero
        
        # Find current max 
        max = np.amax(confusion_mat)
        # Find index or indices for max
        indices = np.where(confusion_mat == max)
        
        # Check if same max value occured more than once
        # if max is only onece
        if indices[0].size == 1:
            yindex = int(indices[0]) # Assign row number
            xindex = int(indices[1]) # Assign column number
        elif indices[0].size != 1:
            yindex = int(indices[0][0])
            xindex = int(indices[1][0])
        
        # Update the confusion matrix 
        confusion_mat[yindex][xindex] = 0

        if (xindex != yindex) and (not(xindex in S.keys())):
            # The code book is sorted from smallest to greatest 
            # Pop the last one in table
            S[xindex] = code_book.pop(-1)
            j = j - 1
            #print(code_book)
    return S
    
#Helper Functions For training and Fine tuning 


# Predict the Label
def predict(outputs, codebook):
    '''
    outputs: Tensor of (batch_size * 7) output from the final layer length of 7 
    codebook: Python Dictionary mapping 0-9 to a string of coding 
    return: Tensor of (batch_size,) label from 0-9 

    This code could be complicated but for now just do the basics 
    '''
    # Initialize output 
    pred = to_device(torch.zeros(outputs.shape[0], ), device)

    # Iterate through batches 
    for i in range(outputs.shape[0]):
        # Single is Tensor of length seven 
        # The output is right after Linear Layer + Sigmoid(Loistic Function)
        # Therefore if value larger or equal to 0.5, treated as 1 
        # If smaller than 0.5 treated as 0, 
        # This could be modified if in range 0.4 to 0.6 output both and compare      

        single = outputs[i]
        # Large than 0.5 output 1 smaller than 0.5 output 0 
        pred_coding = (single>=0.5).int() 
        # Turn the coding tensor into string and compare with coding book
        pred_coding_list = pred_coding.tolist()
        pred_coding_string = ''.join(str(s) for s in pred_coding_list)
        

        '''
        # If find the exact then return the coding book's key
        #if pred_coding_string in codebook.values():
        for key, value in codebook.items():
            if pred_coding_string == value:
                pred[i] = value
                found = True
                break
        
        # If not found then search for the least distanced
        if found == False:
        '''    
        
        # Initialize empty list of 10 by 2 each row is key(label) and hamming distance
        l = np.zeros([len(codebook),2])
        j = 0 # index for 
        found_exact = False
        # Loop through the codebook Find the label with the least Hamming distance
        for key, value in codebook.items():
            ham = Hamming(pred_coding_string, value)
            # If exact then assign and break out of the loop to save conputation power
            if ham == 0:
                pred[i] = key
                found_exact = True
                break
            l[j] = [key, ham]
            j += 1
        
        if found_exact == False:
            #  Column 0 of l is 0-9, Column 1 is hamming distance we need to find the index
            # for the lowest hamming distance
            Hamm_Vec = l[:,1]
            # Find the index of the maximum occured
            index = np.where(Hamm_Vec.min() == Hamm_Vec)[0][0]
            # The index is the row number for the label, the column should be zero
            pred[i] = l[index,0]
    
    return pred 


# Helper Function for turn labels into codings 
def label2coding(labels, codebook):
    '''
    labels: Tensor of (batch_size,) 0-9
    codebook: Python Dictionary mapping 0-9 to a string of coding 
    return: Tensor of (batch_size, len(dict)) 
    '''
    
    # Get the number of CLC by finding the length of one codings in the coding book
    length = len(codebook[0])
    # Initialize the true coding with shape (Batch_size, No.CLC) 
    true_coding = to_device(torch.zeros(labels.shape[0], length), device) 
    

    # Iterate through batches of the label
    for i in range(labels.shape[0]):
        # Get the codings in string
        string_coding = codebook[int(labels[i])]
        
        # Iterate though the String 
        for j in range(len(string_coding)):
            # Assign values to the Tensor
            true_coding[i][j] = int(string_coding[j])
 
    return true_coding

# Modified for Fine Tuning using coded label
def accuracy_FTNA(outputs, labels, codebook):
    #_, preds = torch.max(outputs, dim=1)
    '''
    outputs: Tensor of batch_size * 7 output from the final layer length of 7 
    labels: Tensor of (batch_size,) 0-9
    codebook: Python Dictionary mapping 0-9 to a string of coding  
    return: Tensor accuracy a percentage

    '''

    # Similarly retrieve the outputs and labels first and then get the prediction 
    # However the prediction is now length of 7 and act collaboratively
    # Use the function 'predict' to find the predict labels and compare with true labels  

    preds = predict(outputs, codebook)

    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


def training_step_FTNA(model, batch, codebook):
    images, labels = batch 
    out = model(images)                  # Generate predictions
    loss = nn.MSELoss()(label2coding(labels, codebook), out)

    return loss
    
def validation_step_FTNA(model, batch, codebook):
    images, labels = batch 
    out = model(images)                    # Generate predictions
    loss = nn.MSELoss()(label2coding(labels, codebook), out)   # Calculate loss
    acc = accuracy_FTNA(out, labels, codebook)           # Calculate accuracy
    return {'val_loss': loss.detach(), 'val_acc': acc}
        
def validation_epoch_end_FTNA(outputs):
    '''
    outputs: Dictionary from top one function
    '''

    batch_losses = [x['val_loss'] for x in outputs]
    epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
    batch_accs = [x['val_acc'] for x in outputs]
    epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
    return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
def epoch_end_FTNA(epoch, result):
    print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
        epoch, result['train_loss'], result['val_loss'], result['val_acc']))


#Traning 
@torch.no_grad()
def evaluate_FTNA(model, val_loader, codebook):
    model.eval()
    outputs = [validation_step_FTNA(model, batch, codebook) for batch in val_loader]
    return validation_epoch_end_FTNA(outputs)

def get_lr_FTNA(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def fit_one_cycle_FTNA(epochs, max_lr, model, train_loader, val_loader, codebook,
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
            loss = training_step_FTNA(model, batch, codebook)
            train_losses.append(loss)
            loss.backward()
            
            # Gradient clipping
            if grad_clip: 
                nn.utils.clip_grad_value_(model.parameters(), grad_clip)
            
            optimizer.step()
            optimizer.zero_grad()
            
            # Record & update learning rate
            lrs.append(get_lr_FTNA(optimizer))
            sched.step()
        
        # Validation phase
        result = evaluate_FTNA(model, val_loader, codebook)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        result['lrs'] = lrs
        epoch_end_FTNA(epoch, result)
        history.append(result)
    return history        

def model_alter(model, model_name):
    if model_name == "MLP-3L":
        model.linear3 = nn.Sequential(nn.Linear(32, 7), nn.Sigmoid())
    if model_name == "ResNet-18":
        model.fc = nn.Linear(512, 7)
    if model_name == "VGG":
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, 7)
    if model_name == "LeNet":
        model.fc2 = nn.Linear(84, 7)

    to_device(model, device)
    return model


def evaluate_FTNA_robustness(model, model_path, valid_dl, codebook):
    
    # Pick different value for sigma
    sigma = np.linspace(0., 1.5, 31)

    #Initialize Empty list for accuracy under different std
    accu = []

    # Run several time for a smoother curve
    num = 20
    evaluated = np.zeros(num)

    for std in sigma:
        for i in range(num):
            
            model.load_state_dict(torch.load(model_path))  
            add_noise_to_weights(0, std, model)
            evaluated[i] = evaluate_FTNA(model, valid_dl, codebook)['val_acc']
        print("Finshed sigma=", std)      
        accu.append(np.sum(evaluated)/num)
    return sigma, accu