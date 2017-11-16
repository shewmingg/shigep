# coding:utf8
import torchvision
from torch import nn
from .basic_module import BasicModule
import torch


class Inception(BasicModule):
    def __init__(self, model, opt=None, feature_dim=, name='inception'):
        super(ResNet, self).__init__(opt)
        self.model_name = name

        model.avgpool = nn.AdaptiveAvgPool2d(1)
        del model.fc
        model.fc = lambda x: x
        self.features = model
        self.classifier = nn.Linear(feature_dim, 80)
    def forward(self, x):
        features = self.features(x)
        return self.classifier(features)

def inceptionV3(opt):
    model = torchvision.models.inception_v3(pretrained=not opt.load_path)
    return Inception(model, opt, feature_dim=512, name='inceptionV3')


