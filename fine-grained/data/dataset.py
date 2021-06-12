import os
import numpy as np
import torch
from torch.utils.data import Dataset
import xml.etree.ElementTree as ET
from PIL import Image
import torchvision

class DataSet(Dataset):
    def __init__(self, img_path, img_txt, transform=None):
        super(Dataset, self).__init__()
        self.img_path = img_path
        self.transform = transform

        self.img_list = []
        self.label_list = []

        with open(img_txt) as f:
            for lines in f:
                _name = (lines.split('\n')[0]).split(';')[0]
                _label = (lines.split('\n')[0]).split(';')[1]
                self.img_list.append(img_path + _name)
                self.label_list.append(_label)

        self.num_classes = 200
        # self.one_hot_list = []
        # for anno in self.label_list:
        #     one_hot = np.zeros(self.num_classes)
        #     one_hot[int(anno)] = 1
        #     self.one_hot_list += [one_hot]

    def __getitem__(self, item):
        img = Image.open(self.img_list[item]).convert('RGB')
        # label = np.array(self.one_hot_list[item])
        label = int(self.label_list[item])
        if self.transform is not None:
            img = self.transform(img)
        return img, torch.LongTensor([label])

    def __len__(self):
        return len(self.img_list)

# img_path = './CUB_200_2011/images/'
# img_txt = './CUB_200_2011/train_list.txt'
# transforms = torchvision.transforms.Compose(
#     [
#         torchvision.transforms.Resize((224, 224)),
#         torchvision.transforms.RandomHorizontalFlip(p=0.5),
#         torchvision.transforms.RandomHorizontalFlip(p=0.5),
#         torchvision.transforms.ToTensor()
#     ]
# )
# dataset = DataSet(img_path, img_txt, transforms)

# dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

# for i, (img, label) in enumerate(dataloader):
#     print(img.shape, label.shape)
#     print(label)
