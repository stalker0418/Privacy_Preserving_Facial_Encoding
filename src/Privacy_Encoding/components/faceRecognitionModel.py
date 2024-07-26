import torchvision.models as models
import torch
from src.Privacy_Encoding.config.configuration import ModelInitializerConfig
from src.Privacy_Encoding.logging import logger

class FRModel:
    def __init__(self, config: ModelInitializerConfig):
        self.resnet18 = models.resnet18(pretrained=True)
        self.config = config
        
    def set_first_layer(self):
        if self.config.image_channels == 1:
            #It is intially taking 3 channels resnet, so we convert the first layer to take only one channel here
            self.resnet18.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            logger.info(f"Initialized the first layer of the Facial Recognition Model to take {self.config.image_channels} image channels")
    
    def unfreeze_layers(self):
        for name, param in self.resnet18.named_parameters():
            if any(layer_name in name for layer_name in self.config.unfreeze_layers.layers):
                param.requires_grad = True
                logger.info(f"Unfreezed layer: {name}")
            else:
                param.requires_grad = False

    def create_fc_layer(self, no_of_labels):
        self.resnet18.fc = torch.nn.Linear(self.config.fc_layer_size,no_of_labels)
        logger.info(f"The last layer is {no_of_labels}")
    
    def get_model_loss_optimizer(self):
        lossfun = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.resnet18.parameters(), lr = self.config.learning_rate, momentum= self.config.momentum)
        return self.resnet18, lossfun, optimizer
    
    


    
