import torchvision.models as models
import torch.nn as nn


def alex():
    model = models.alexnet(pretrained=True)
    model.classifier[6] = nn.Linear(4096, 10)
    return model


def alexBayes():
    model = models.alexnet(pretrained=True)
    # print(model)
    para0, para1, para2, para3, para4 = 0, 0, 0, 0, 0

    do1, do2, do3, do4, do5, do6, do7, do8 = 0.5, 0.2, 0.2, 0.2, 0.5, 0.5, 0.5, 0.5
    if para0 == 0:
        model.features[1] = nn.Sequential(model.features[1], nn.Dropout2d(do1))
    if para1 == 0:
        model.features[4] = nn.Sequential(model.features[4], nn.Dropout2d(do2))
    if para2 == 0:
        model.features[7] = nn.Sequential(model.features[7], nn.Dropout2d(do3))
    if para3 == 0:
        model.features[9] = nn.Sequential(model.features[9], nn.Dropout2d(do4))
    if para4 == 0:
        model.features[11] = nn.Sequential(model.features[11], nn.Dropout2d(do5))
    model.classifier[0] = nn.Dropout(do6)
    model.classifier[3] = nn.Dropout(do7)
    model.classifier[6] = nn.Sequential(model.classifier[6], nn.ReLU(inplace=True), nn.Dropout(do8),
                                        nn.Linear(1000, 10))
    # print(model)
    return model


def alex1():
    model = models.alexnet()
    print(model)
    para0, para1, para2, para3, para4 = 0, 0, 0, 0, 0

    do1, do2, do3, do4, do5, do6, do7 = 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5
    if para0 == 0:
        model.features[1] = nn.Sequential(model.features[1], nn.Dropout2d(do1))
    if para1 == 0:
        model.features[4] = nn.Sequential(model.features[4], nn.Dropout2d(do2))
    if para2 == 0:
        model.features[7] = nn.Sequential(model.features[7], nn.Dropout2d(do3))
    if para3 == 0:
        model.features[9] = nn.Sequential(model.features[9], nn.Dropout2d(do4))
    if para4 == 0:
        model.features[11] = nn.Sequential(model.features[11], nn.Dropout2d(do5))
    model.classifier[0] = nn.Dropout(do6)
    model.classifier[3] = nn.Dropout(do7)
    model.classifier[6] = nn.Linear(4096, 10)
    # print(model)
    return model
