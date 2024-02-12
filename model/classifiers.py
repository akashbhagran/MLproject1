from torch import device, optim, cuda, nn
from torchvision.models import resnet50
import json

with open('config.json') as f:
    c = json.load(f)

class get_cco():


    def make_model(self):

        d = device("cuda" if cuda.is_available() else "cpu")

        if c['model'] == 'resnet50':

            self.model = resnet50(num_classes = c['num_classes'])

        self.model.to(d)

        return self.model
    
    
    def define_criterion(self):

        if c['criterion'] == 'crossentropy':
            
            criterion = nn.CrossEntropyLoss()

        return criterion
    
    
    def select_optimizer(self):

        if c['optimizer'] == "sgd":

            optimizer = optim.SGD(self.model.parameters(), lr=c['learning_rate'], momentum=c['momentum'], weight_decay=c['weight_decay'])

        return optimizer
    