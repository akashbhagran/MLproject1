import torch
from torchvision.models.detection import retinanet_resnet50_fpn
from torchvision.ops import batched_nms
from torchvision.transforms import functional as F
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.models import resnet50, ResNet50_Weights
import torch.optim as optim
from torch import nn
from tqdm import tqdm
import numpy as np


import torchvision.datasets as datasets

import torchvision.transforms as transforms

import data


if __name__ == "__main__":
    # get data
    batch_size = 100
    d = data.Data(batch_size=batch_size)
    trainloader, valloader = d.make_data()
    len_train, len_val = d.get_length()

    # define model
    resnet50(weights=ResNet50_Weights.DEFAULT)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = resnet50()







    
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    epochs = 2

    acc_train = []
    loss_train = []
    acc_val = []
    loss_val = []

    for epoch in range(epochs):  # loop over the dataset multiple times
        with tqdm(total=len(trainloader)) as pbar_train:
            model.train()
            train_acc_collect = []
            train_loss_collect = []

            for i, data in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]

                inputs, labels = data
                inputs = inputs.cuda()
                labels = labels.cuda()

                # zero the parameter gradients

                optimizer.zero_grad()

                # forward + backward + optimize

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics

                _, predicted = torch.max(outputs.data, 1)  # get argmax of predictions
                accuracy = np.mean(
                    list(predicted.eq(labels.data).cpu())
                )  # compute accuracy
                train_acc_collect.append(accuracy)
                train_loss_collect.append(loss.cpu().data.numpy())

                # print('LOSS_TRAIN: ',loss.cpu().data.numpy(),'  ','ACC_TRAIN: ',accuracy, flush = True)

                pbar_train.update(1)
                pbar_train.set_description("Iterations")

            avg_acc_train = np.mean(train_acc_collect)
            avg_loss_train = np.mean(train_loss_collect)
            print(
                "LOSS_TRAIN_epoch {}: ".format(epoch),
                avg_acc_train,
                "  ",
                "ACC_TRAIN_epoch {}: ".format(epoch),
                avg_loss_train,
            )
            acc_train.append(avg_acc_train)
            loss_train.append(avg_loss_train)

        with tqdm(total=len(valloader)) as pbar_val:
            model.eval()
            vall_acc_collect = []
            vall_loss_collect = []

            for i, data in enumerate(valloader, 0):
                # get the inputs; data is a list of [inputs, labels]

                inputs, labels = data
                inputs = inputs.cuda()
                labels = labels.cuda()

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                _, predicted = torch.max(outputs.data, 1)  # get argmax of predictions
                accuracy = np.mean(
                    list(predicted.eq(labels.data).cpu())
                )  # compute accuracy
                vall_acc_collect.append(accuracy)
                vall_loss_collect.append(loss.cpu().data.numpy())

                pbar_val.update(1)
                pbar_val.set_description("Iterations")

            avg_acc_val = np.mean(vall_acc_collect)
            avg_loss_val = np.mean(vall_loss_collect)
            print(
                "LOSS_VAL_epoch {}: ".format(epoch),
                avg_loss_val,
                "  ",
                "ACC_VAL_epoch {}: ".format(epoch),
                avg_acc_val,
            )
            acc_val.append(avg_acc_val)
            loss_val.append(np.mean(avg_loss_val))
