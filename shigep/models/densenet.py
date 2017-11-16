# coding:utf8
import torchvision
from torch import nn
from .basic_module import BasicModule
import torch


class DenseNet(BasicModule):
    def __init__(self, model,opt, name='densenet'):
        super(DenseNet, self).__init__(opt)
        self.model_name = name
        feature_dim = model.classifier.in_features
        del model.classifier
        model.classifier = lambda x : x
        self.features = model
        self.classifier = nn.Linear(feature_dim, 80)
    def forward(self, x):
        features = self.features(x)
        return self.classifier(features)



def densenet365(opt):
    model = torch.load('checkpoints/whole_densenet161_places365.pth.tar')
    # model = tv.models.resnet50()
    return DenseNet(model, opt, name='dense365')
