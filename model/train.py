import torch
from torchvision.models.detection import retinanet_resnet50_fpn
from torchvision.transforms import functional as F
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.models import resnet50, ResNet50_Weights
import torch.optim as optim
from torch import nn
import tqdm


import torchvision.datasets as datasets
import torchvision.transforms as transforms



if __name__ == '__main__':

 
    resnet50(weights=ResNet50_Weights.DEFAULT)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(device)

    model = resnet50()
    
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    
    for epoch in range(2):  # loop over the dataset multiple times

        running_loss = 0.0
    
        with tqdm.tqdm(total=len(cifar_train)/batch_size) as pbar_train:
            
            for i, data in enumerate(trainloader, 0):
        
                # get the inputs; data is a list of [inputs, labels]
        
                inputs, labels = data
                
                inputs = inputs.cuda()
                
                labels = labels.cuda()
                
                #print(len(data))
        
                # zero the parameter gradients
        
                optimizer.zero_grad()

                # forward + backward + optimize
        
                outputs = model(inputs)
                
                loss = criterion(outputs, labels)
                
                loss.backward()
                
                optimizer.step()
        
                # print statistics
                
                running_loss += loss.item()
                    
                pbar_train.update(1)
                
                pbar_train.set_description("loss: {:.4f}")




