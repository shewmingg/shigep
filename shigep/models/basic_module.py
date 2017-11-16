#coding:utf8
import torch
import time
import os
class BasicModule(torch.nn.Module):
    def __init__(self,opt=None):
        super(BasicModule,self).__init__()
        self.model_name=str(type(self).__name__)# 默认名字
        self.opt = opt
        
    def load(self, path,map_location=lambda storage, loc: storage):
        checkpoint = torch.load(path,map_location=map_location)
        if 'opt' in checkpoint:
            self.load_state_dict(checkpoint['d'])
            #print('old config：')
            #print(checkpoint['opt'])
        else:
            self.load_state_dict(checkpoint)


    def save(self, name=''):
        if not os.path.exists('checkpoints/'+self.model_name):
            os.makedirs('checkpoints/'+self.model_name)
        format = 'checkpoints/'+self.model_name+'/'+self.model_name+'_%m%d_%H%M_'
        file_name = time.strftime(format) + str(name)
        state_dict = self.state_dict()
        opt_state_dict = dict(self.opt.state_dict())
        optimizer_state_dict = self.optimizer.state_dict()

        torch.save({'d':state_dict,'opt':opt_state_dict,'optimizer':optimizer_state_dict,'accuracy':name}, file_name)
        return file_name

    def get_optimizer(self,model_name,lr1,lr2,lr3):

        """self.optimizer =  torch.optim.Adam(
            [
            {'params': self.features.parameters(), 'lr': lr1},
             #{'params': self.features.layer4.parameters(), 'lr': lr2*0.1},
             #{'params': self.features.avgpool.parameters(), 'lr': lr2*0.1},
             {'params': self.classifier.parameters(), 'lr':lr2}
            ] )"""
        if model_name =='densenet365':
            ignored_params = list(map(id, self.classifier.parameters())) + \
                             list(map(id, self.features.features.norm5.parameters())) + \
                             list(map(id, self.features.features.denseblock4.parameters())) + \
                             list(map(id, self.features.features.transition3.parameters()))
            base_params = filter(lambda p: id(p) not in ignored_params,
                                 self.parameters())
            self.optimizer = torch.optim.Adam([
                {'params': base_params, 'lr': lr1},
                {'params': self.classifier.parameters(), 'lr': lr2}
            ], lr=lr3)
        else:
            self.optimizer = torch.optim.Adam(
                [
                    {'params': self.features.parameters(), 'lr': lr1},
                    {'params': self.classifier.parameters(), 'lr': lr2}
                ])
        return self.optimizer

    def update_optimizer(self,lr1,lr2):
        param_groups = self.optimizer.param_groups
        param_groups[0]['lr']=lr1
        param_groups[1]['lr']=lr2
        return self.optimizer
