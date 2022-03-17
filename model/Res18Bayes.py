import torchvision.models as models
import torch.nn as nn
def res18bayes():
    model = models.resnet18(pretrained=True)
    model.bn1 = nn.Dropout(p=0.344531768634169)
    model.layer1[0].bn1 = nn.Dropout2d(p=0.6537668353119641)
    model.layer1[0].bn2 = nn.Identity()
    model.layer1[1].bn1 = nn.Sequential(model.layer1[1].bn1, nn.Dropout2d(0.6537668353119641)) 
    model.layer1[1].bn2 = nn.Dropout2d(p=0.6537668353119641)

    model.layer2[0].bn1 = nn.Dropout2d(p=0.4117262265132121)
    model.layer2[0].bn2 = nn.Dropout2d(p=0.4117262265132121)
    model.layer2[0].downsample[1] = nn.Dropout2d(p=0.5104401480078931)
    model.layer2[1].bn1 = nn.Sequential(model.layer2[1].bn1, nn.Dropout2d(p=0.4117262265132121)) 
    model.layer2[1].bn2 = nn.Dropout2d(p=0.4117262265132121)
    

    model.layer3[0].bn1 = nn.Dropout2d(p=0.491663584789946)
    #model.layer3[0].bn2 = nn.Identity() comment is use original layer
    model.layer3[0].downsample[1] = nn.Dropout2d(0.3)
    #model.layer3[1].bn1 = nn.Sequential(model.layer1[1].bn1, nn.Dropout2d(0.6537668353119641)) 
    #model.layer3[1].bn2 = nn.Dropout2d(p=0.6537668353119641)

    #model.layer4[0].bn1 = nn.Dropout2d(p=0.491663584789946)
    model.layer4[0].bn2 = nn.Sequential(model.layer4[0].bn2, nn.Dropout2d(p=0.3)) 
    model.layer4[0].downsample[1] = nn.Dropout2d(p=0.5568058171779489)
    #model.layer4[1].bn1 = nn.Sequential(model.layer1[1].bn1, nn.Dropout2d(0.6537668353119641)) 
    model.layer4[1].bn2 = nn.Dropout2d(p=0.3)

    return model

#Model 
def conv_block(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),  
              nn.ReLU(inplace=True), nn.Dropout2d(p=0.5)]
    if pool: layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)

class ResNet18_Best(nn.modules):
    def __init__(self, in_channels=3, num_classes=10):
        super().__init__()
        
        """orders:
        -> CONV/FC -> BatchNorm -> ReLu(or other activation) -> Dropout -> CONV/FC ->
        """

        self.conv1 = conv_block(in_channels, 64)
        self.conv2 = conv_block(64, 64)
        self.res1 = nn.Sequential(conv_block(64, 64), conv_block(64, 64))
        self.conv3 = conv_block(64, 128, pool=True)
        self.conv4 = conv_block(128, 128)
        self.res2 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))        
        self.conv5 = conv_block(128, 256, pool=True)
        self.conv6 = conv_block(256, 256)
        self.res3 = nn.Sequential(conv_block(256, 256), conv_block(256, 256))
        self.conv7 = conv_block(256, 512, pool=True)
        self.conv8 = conv_block(512, 512)
        self.res4 = nn.Sequential(conv_block(512, 512), conv_block(512, 512))
        
        self.classifier = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                        nn.Flatten(), 
                                        nn.Linear(512, num_classes), nn.Dropout(p=0.5))
        
    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.conv5(out)
        out = self.conv6(out)
        out = self.res3(out) + out
        out = self.conv7(out)
        out = self.conv8(out)
        out = self.res4(out) + out
        out = self.classifier(out)
        return out
model = to_device(ResNet18(3,10), device)
print(model)
