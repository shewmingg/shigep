# coding:utf8
from __future__ import division
import torch
from torchvision import transforms
from torch.utils import data
from torch.utils.data import sampler

import os
import PIL
import random
import numpy as np
from PIL import Image, ImageOps, ImageEnhance
import types
from .functional import adjust_brightness, adjust_contrast, adjust_saturation, ten_crop
import math
from PIL import Image, ImageOps, ImageEnhance
try:
    import accimage
except ImportError:
    accimage = None
import numpy as np
import numbers
import types
import collections
import warnings

train_dir = '../data/train/scene_train_images_20170904'
test_dir = '../data/test/scene_test_a_images_20170922'
val_dir = '../data/val/scene_validation_images_20170908'
meta_path = 'scene1.pth'
img_size = 256

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
imagenet_pca = {
    'eigval': torch.Tensor([0.2175, 0.0188, 0.0045]),
    'eigvec': torch.Tensor([
        [-0.5675, 0.7192, 0.4009],
        [-0.5808, -0.0045, -0.8140],
        [-0.5836, -0.6948, 0.4203],
    ])
}

def load(path):
    return PIL.Image.open(path).convert('RGB')
class Lambda(object):
    """Apply a user-defined lambda as a transform.
    Args:
        lambd (function): Lambda/function to be used for transform.
    """

    def __init__(self, lambd):
        assert isinstance(lambd, types.LambdaType)
        self.lambd = lambd

    def __call__(self, img):
        return self.lambd(img)
class ColorJitter(object):
    """Randomly change the brightness, contrast and saturation of an image.
    Args:
        brightness (float): How much to jitter brightness. brightness_factor
            is chosen uniformly from [max(0, 1 - brightness), 1 + brightness].
        contrast (float): How much to jitter contrast. contrast_factor
            is chosen uniformly from [max(0, 1 - contrast), 1 + contrast].
        saturation (float): How much to jitter saturation. saturation_factor
            is chosen uniformly from [max(0, 1 - saturation), 1 + saturation].
        hue(float): How much to jitter hue. hue_factor is chosen uniformly from
            [-hue, hue]. Should be >=0 and <= 0.5.
    """

    def __init__(self, brightness=0, contrast=0, saturation=0):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation

    def get_params(self, brightness, contrast, saturation):
        """Get a randomized transform to be applied on image.
        Arguments are same as that of __init__.
        Returns:
            Transform which randomly adjusts brightness, contrast and
            saturation in a random order.
        """
        Transforms = []
        if brightness > 0:
            brightness_factor = np.random.uniform(max(0, 1 - brightness), 1 + brightness)
            Transforms.append(Lambda(lambda img: adjust_brightness(img, brightness_factor)))
        if contrast > 0:
            contrast_factor = np.random.uniform(max(0, 1 - contrast), 1 + contrast)
            Transforms.append(Lambda(lambda img: adjust_contrast(img, contrast_factor)))
        if saturation > 0:
            saturation_factor = np.random.uniform(max(0, 1 - saturation), 1 + saturation)
            Transforms.append(Lambda(lambda img: adjust_saturation(img, saturation_factor)))
        np.random.shuffle(Transforms)
        Transform = transforms.Compose(Transforms)
        return Transform

    """def Compose(self, transforms):
        for t in transforms:
            img = t(img)
        return img"""

    def __call__(self, img):
        transform = self.get_params(self.brightness, self.contrast, self.saturation)
        return transform(img)
class Lighting(object):
    """Lighting noise(AlexNet - style PCA - based noise)"""

    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = eigval
        self.eigvec = eigvec
    def __call__(self, img):
        if self.alphastd == 0:
            return img
        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(img).clone() \
            .mul(alpha.view(1, 3).expand(3, 3)) \
            .mul(self.eigval.view(1, 3).expand(3, 3)) \
            .sum(1).squeeze()
        return img.add(rgb.view(3, 1, 1).expand_as(img))
class TenCrop(object):
    def __init__(self, size, vertical_flip=False):
        self.size = size
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            assert len(size) == 2, "Please provide only two dimensions (h, w) for size."
            self.size = size
        self.vertical_flip = vertical_flip

    def __call__(self, img):
        return ten_crop(img, self.size, self.vertical_flip)


class LabelShufflingSampler(Sampler):
    """
    label shuffling technique aimed to deal with imbalanced class problem

    argument:
    labels: array of labels of the dataset e.g. [3,2,0,1,2,4]
    """

    def __init__(self, labels):

        # mapping between label index and sorted label index
        sorted_labels = sorted(enumerate(labels), key=lambda x: x[1])

        count = 1
        count_of_each_label = []
        tmp = -1
        # get count of each label
        for (x, y) in sorted_labels:
            if y == tmp:
                count += 1
            else:
                if tmp != -1:
                    count_of_each_label.append(count)
                    count = 1
            tmp = y
        count_of_each_label.append(count)
        # picture number that the label classw with the highest quantity of picture has
        largest = int(np.amax(count_of_each_label))
        self.count_of_each_label = count_of_each_label
        self.enlarged_index = []

        # preidx used for find the mapping beginning of arg "sorted_labels"
        preidx = 0
        for x in range(len(self.count_of_each_label)):
            idxes = np.remainder(t.randperm(largest).numpy(), self.count_of_each_label[x]) + preidx
            for y in idxes:
                self.enlarged_index.append(sorted_labels[y][0])

            preidx += int(self.count_of_each_label[x])


    def __iter__(self):
        shuffle(self.enlarged_index)
        return iter(self.enlarged_index)

    def __len__(self):
        return np.amax(self.count_of_each_label)*len(self.count_of_each_label)

class ClsDataset(data.Dataset):
    def __init__(self):
        self.datas = torch.load(meta_path)

        self.val_transforms = transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            normalize,
            #TenCrop(224),  # this is a list of PIL Images
            #Lambda(lambda crops: [transforms.ToTensor()(crop) for crop in crops]),
            #Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops])),  # 4dtensor
        ])
        self.train_transforms = transforms.Compose([
            #transforms.Scale(256),
            transforms.RandomSizedCrop(256),
            transforms.RandomHorizontalFlip(),
            ColorJitter(0.4, 0.4, 0.4),
            transforms.ToTensor(),
            Lighting(0.1, imagenet_pca['eigval'], imagenet_pca['eigvec']),
            normalize,
        ])
        self.test_transforms = transforms.Compose([
            transforms.Scale(256),
            TenCrop(224),  # this is a list of PIL Images
            Lambda(lambda crops: [transforms.ToTensor()(crop) for crop in crops]),
            Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops])),  # 4dtensor
        ])
        self.train()

    def __getitem__(self, index):
        img_path = os.path.join(self.path, self.imgs[index])
        img = load(img_path)
        img = self.transforms(img)
        return img, self.labels[index], self.imgs[index]

    def train(self):
        data = self.datas['train']
        self.imgs, self.labels = data['ids'], data['labels']
        self.path = train_dir
        self.transforms = self.train_transforms
        return self

    def test(self):
        data = self.datas['test1']
        self.imgs, self.labels = data['ids'], data['labels']
        self.path = test_dir
        self.transforms = self.test_transforms
        return self

    def val(self):
        data = self.datas['val']
        self.imgs, self.labels = data['ids'], data['labels']
        self.path = val_dir
        self.transforms = self.val_transforms
        return self

    def __len__(self):
        return len(self.imgs)


if __name__ == '__main__':
    test
