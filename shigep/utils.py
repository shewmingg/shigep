#coding:utf8
import visdom
import numpy as np
class Config:
    #load_path='checkpoints/res_365_1114_1836_0.927830717489'
    load_path = None
    model = 'densenet365'  # 具体名称查看 models/__init__.py
    result_path = 'result365.json'  # 提交保存路径
    lr1 = 0
    lr2 = 0.0005 # lr2 = 5e-10
    batch_size = 16
    env = 'res34_adappool'  # visdom env
    lr3 = 0

    train_dir = '../data/train/scene_train_images_20170904'
    test_dir = '../data/test/scene_test_a_images_20170922'
    val_dir = '../data/val/scene_validation_images_20170908'
    meta_path = 'scene1.pth'
    img_size=256
    shuffle = True
    lr_decay = 0.5
    max_epoch = 100
    debug_file = '/tmp/debugc'
    plot_every = 10 # 每10步可视化一次
    workers = 8 # CPU多线程加载数据
    loss='celoss'


class Visualizer():
    '''
    对可视化工具visdom的封装
    '''
    def __init__(self, env, **kwargs):
        import visdom
        self.vis = visdom.Visdom(env=env,port=8887, **kwargs)
        self.index = {}

    def plot_many(self, d):
        for k, v in d.items():
            self.plot(k, v)

    def plot(self, name, y):
        x = self.index.get(name, 0)
        self.vis.line(Y=np.array([y]), X=np.array([x]),
                      win=name,
                      opts=dict(title=name),
                      update=None if x == 0 else 'append'
                      )
        self.index[name] = x + 1
def topk_acc(score,label,k=3):
    '''
    topk accuracy,默认是top3准确率
    '''
    topk = score.topk(k)[1]
    label = label.view(-1,1).expand_as(topk)
    acc = (label == topk).float().sum()/(0.0+label.size(0))
    return acc

def parse(self,kwargs,print_=True):
        '''
        根据字典kwargs 更新 config参数
        '''
        for k,v in kwargs.items():
            if not hasattr(self,k):
                raise Exception("opt has not attribute <%s>" %k)
            setattr(self,k,v) 
        if print_:
            print('user config:')
            print('#################################')
            for k in dir(self):
                if not k.startswith('_') and k!='parse' and k!='state_dict':
                    print (k,getattr(self,k))
            print('#################################')
        return self

def state_dict(self):
    return  {k:getattr(self,k) for k in dir(self) if not k.startswith('_') and k!='parse' and k!='state_dict' }

Config.parse = parse
Config.state_dict = state_dict
opt = Config()
