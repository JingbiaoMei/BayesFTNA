import torchvision.models as models
import torch.nn as nn
def vggBayes():
    p1, p2, p3, p4, p5, p6 = 0.35, 0.36, 0.2, 0.2, 0.2, 0.14
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
    print(model)
    return model

def vgg_naive():
    p1, p2, p3, p4, p5, p6 = 0.5, 0.5, 0.5, 0.5, 0.5, 0.14
    p6 = round(p6, 2)
    model = models.vgg11_bn()
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
    print(model)
    return model

