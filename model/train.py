"""
train.py

Trains the classifier and logs results.

"""

# Imports

import torch
from torchvision.ops import batched_nms
from torchvision.transforms import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import data
from torcheval.metrics import MulticlassPrecision, MulticlassAccuracy, MulticlassRecall
import mlflow
import json
from classifiers import get_cco

# Load config

with open('config.json') as f:

    config = json.load(f)

# Load model and optimizers etc
 
objects = get_cco()
model = objects.make_model()
optimizer = objects.select_optimizer()
criterion = objects.define_criterion()

# epochs

epochs = config['epochs']
batch_size = config['batch_size']

# MLFLOW settings

mlflow.set_tracking_uri(uri = 'http://127.0.0.2:8090')
mlflow.set_experiment('First Run')

# main

if __name__ == "__main__":
    # get data
    
    d = data.Data(batch_size=batch_size)
    trainloader, valloader = d.make_data()
    len_train, len_val = d.get_length()

    # collect metrics through arrays.

    acc_train = []
    loss_train = []
    acc_val = []
    loss_val = []
    prec_train = []
    rec_train = []

    #define metrics precision and recall

    metricP = MulticlassPrecision(average = 'macro',num_classes=10)
    metricR = MulticlassRecall(average = 'macro',num_classes=10)

    with mlflow.start_run():

        mlflow.log_params(config)

        for epoch in range(epochs):  # loop over the dataset multiple times
            with tqdm(total=len(trainloader)) as pbar_train:
                model.train()
                train_acc_collect = []
                train_loss_collect = []
                precision_collect = []
                recall_collect = []

                for i, data in enumerate(trainloader, 0):
                    # get the inputs; data is a list of [inputs, labels]

                    inputs, labels = data
                    inputs = inputs.cuda()
                    labels = labels.cuda()

                    print(labels.size())

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

                    metricP.update(predicted,labels)
                    metricR.update(predicted,labels)

                    precision_collect.append(float(metricP.compute().detach()))
                    recall_collect.append(float(metricR.compute().detach()))

                    pbar_train.update(1)
                    pbar_train.set_description("Iterations")

                avg_acc_train = np.mean(train_acc_collect)
                avg_loss_train = np.mean(train_loss_collect)
                avg_prec_train = np.mean(precision_collect)
                avg_rec_train =  np.mean(recall_collect)

                print(
                    "LOSS_TRAIN_epoch {}: ".format(epoch),
                    avg_acc_train,
                    "  ",
                    "ACC_TRAIN_epoch {}: ".format(epoch),
                    avg_loss_train,
                )

                acc_train.append(avg_acc_train)
                loss_train.append(avg_loss_train)
                prec_train.append(avg_prec_train)
                rec_train.append(avg_rec_train)

                if epoch == epochs-1:
                    mlflow.log_metric("accuracyT", acc_train[-1])
                    mlflow.log_metric("precisionT",prec_train[-1])
                    mlflow.log_metric("recallT",rec_train[-1])

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

    mlflow.end_run()
