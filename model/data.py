import torchvision.datasets as datasets
import torchvision.transforms as transforms

class ExperimentBuilder(nn.Module):
    def __init__(self, network_model, experiment_name, num_epochs, train_data, val_data,
                 test_data, weight_decay_coefficient, use_gpu, continue_from_epoch=-1):

class Data():
    
    def __init__(self, 
                data : str = 'cifar10'
                batch_size: int = 100):
    
    self.data = data,
    
    self.batch_size = batch_size
        
    def make_data():
        
        transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        
        transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        
        cifar_train= datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        cifar_val = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_val)    
        trainloader = torch.utils.data.DataLoader(cifar_train, batch_size=self.batch_size,
                                          shuffle=True, num_workers=4)
        valloader = torch.utils.data.DataLoader(cifar_val, batch_size=self.batch_size,
                                         shuffle=True, num_workers=4)
        
        return trainloader, valloader