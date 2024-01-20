from typing import Self
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch


class Data:
    def __init__(self, data: str = "cifar10", batch_size: int = 100):
        self.data_name_ = (data,)
        self.batch_size = batch_size

    def transform_train(self):
        return transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )

    def transform_val(self):
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )

    def make_data(self):
        self.train = datasets.CIFAR10(
            root="./data", train=True, download=True, transform=self.transform_train()
        )

        print(len(self.train))

        self.len_train = len(self.train)

        self.val = datasets.CIFAR10(
            root="./data", train=False, download=True, transform=self.transform_val()
        )

        self.len_val = len(self.val)

        trainloader = torch.utils.data.DataLoader(
            self.train, batch_size=self.batch_size, shuffle=True, num_workers=4
        )

        valloader = torch.utils.data.DataLoader(
            self.val, batch_size=self.batch_size, shuffle=True, num_workers=4
        )

        return trainloader, valloader

    def get_length(self):
        return self.len_train, self.len_val
