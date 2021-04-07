import torch
import os
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image


class CelebDataset(Dataset):
    def __init__(self, image_path, metadata_path, transform, mode, crop_size):
        self.image_path = image_path
        self.transform = transform
        self.mode = mode
        self.lines = open(metadata_path, 'r').readlines()
        self.num_data = int(self.lines[0])
        self.crop_size = crop_size

        print ('Start preprocessing dataset..!')
        random.seed(1234)
        self.preprocess()
        print ('Finished preprocessing dataset..!')

        if self.mode == 'train':
            self.num_data = len(self.train_filenames)
        elif self.mode == 'test':
            self.num_data = len(self.test_filenames)

    def preprocess(self):
        self.train_filenames = []
        self.test_filenames = []

        lines = self.lines[2:]
        random.shuffle(lines)   # random shuffling
        for i, line in enumerate(lines):

            splits = line.split()
            filename = splits[0]

            if (i+1) < 20000:
                self.test_filenames.append(filename)
            else:
                self.train_filenames.append(filename)

    def __getitem__(self, index):
        if self.mode == 'train':
            image = Image.open(os.path.join(self.image_path, self.train_filenames[index]))
        elif self.mode in ['test']:
            image = Image.open(os.path.join(self.image_path, self.test_filenames[index]))
        # self.check_size(image, index)
        return self.transform(image)

    def __len__(self):
        return self.num_data

# IMAGE_PATH: ./data/CelebA/images, METADATA_PATH: ./data/list_attr_celeba.txt, CROP_SIZE: 256, IMG_SIZE: 128, BATCH_SIZE: 32,
def get_loader(image_path, metadata_path, crop_size, image_size, batch_size, dataset='CelebA', mode='train'):
    """Build and return data loader."""

    if mode == 'train':
        # 이미지 변형
        transform = transforms.Compose([
            transforms.CenterCrop(crop_size),                               # 가운데 부분을 crop_size 크기로 자름
            transforms.Resize(image_size, interpolation=Image.ANTIALIAS),   # 이미지 크기 변경
            transforms.RandomHorizontalFlip(),                              # 랜덤으로 수평으로 뒤집기
            transforms.ToTensor(),                                          # 이미지 데이터를 텐서로 변경
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
    else:   # text 모드
        # 이미지 변형
        transform = transforms.Compose([
            transforms.CenterCrop(crop_size),                               # 가운데 부분을 crop_size 크기로 자름
            transforms.Scale(image_size, interpolation=Image.ANTIALIAS),    #
            transforms.ToTensor(),                                          # 이미지 데이터를 텐서로 변경
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

    if dataset == 'CelebA':
        dataset = CelebDataset(image_path, metadata_path, transform, mode, crop_size)

    shuffle = False
    if mode == 'train':
        shuffle = True

    data_loader = DataLoader(dataset=dataset,batch_size=batch_size, shuffle=shuffle)
    return data_loader