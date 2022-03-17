import torch 
from utilis_gpu import to_device, get_default_device
# Confusion Matrix
device = get_default_device()
# Get  all the predictions in one tensor
@torch.no_grad()
def get_all_predAndLabel(model, val_loader):
    model.eval()
    all_preds = to_device(torch.tensor([]), device)   
    all_labels = to_device(torch.tensor([]), device)
    for batch in val_loader:
        image, label = batch
        out = model(image)
        all_preds = torch.cat((all_preds, out), dim = 0)
        all_labels = torch.cat((all_labels, label), dim = 0)
    return all_preds, torch.tensor(all_labels,dtype=int,device=device) 

def Confusion_Matrix(model, val_loader, m):
    '''
    model: Model
    val_loader: validation set
    m: number of classes
    '''
    pred, valid_targets = get_all_predAndLabel(model, val_loader)
    valid_pred = pred.argmax(dim=1)
    # Stack the true Label and Prediced Label
    stacked = torch.stack((valid_targets,valid_pred), dim=1)
    # Initialize all zeros for the Confusion Matrix
    conf_mat = torch.zeros(m,m)
    # Iterate through all the pairs (here 10000) in the stacked lists
    for pair in stacked:
        # Get hte True Label and Predicted Label
        tl, pl = pair.tolist()
        # Add 1 for each row and column
        conf_mat[tl, pl] = conf_mat[tl, pl] + 1
        # Vertically is the True Label
        # Horizontally is the Prediced Label  
    return conf_mat.numpy()
