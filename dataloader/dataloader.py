from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from PIL import Image

import os
import torch
import numpy as np

import torch.utils.data as data
from misc.utils import to_var



class Cifar10Dataloader(data.Dataset):

    def __init__(self, configs):
        split = configs['split']
        root = configs['root']
        self.transform = configs.get('transform', None)
        if self.transform is None:
            print ('Warning, transform is None!')
        self.split = split
        data_dict = torch.load(os.path.join(root, 'data', 'cifar10', split + '.pth'))
        labels = []
        data = []
        for label, data_list in data_dict.items():
            n_samples = len(data_list)
            labels.extend([label] * n_samples)
            data.extend(data_list)
        print ('Loaded %d data, %d labels'%(len(labels), len(data)))
        self.data = np.concatenate([x.reshape(1, -1) for x in data])
        print ('Concatenated shape:', self.data.shape)
        self.data = self.data.reshape((-1, 3, 32, 32))
        self.data = self.data.transpose((0, 2, 3, 1)) # convert to HWC
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        img, target = self.data[index], self.labels[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def collate_fn(self, data):
        inputs, labels = zip(*data)
        # print (len(inputs), inputs[0].shape)
        labels = torch.LongTensor(labels)
        inputs = torch.cat([x.view(1, 3, 32, 32) for x in inputs], 0)
        return inputs, labels


def get_dataloader(configs):
    if configs['dataset'] == 'cifar10':
        batch_size = configs['batch_size']
        shuffle = configs['shuffle']
        dataset = Cifar10Dataloader(configs)
        data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                  batch_size=batch_size,
                                                  shuffle=shuffle,
                                                  num_workers=2,
                                                  collate_fn=dataset.collate_fn)
    else:
        raise NotImplementedError

    return data_loader