from paddle.io import Dataset
from paddle.io import DataLoader
from paddle.vision import datasets
from paddle.vision import transforms

def get_transforms(mode='train'):
    if mode == "train":
        deta_transforms = transforms.Compose([
            transforms.RandomCrop(32,padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914,0.4822,0.4465],std=[])
        ])
    else:
        