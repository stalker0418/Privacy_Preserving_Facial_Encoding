from src.Privacy_Encoding.components.model_trainer import modelTrainer
from src.Privacy_Encoding.logging import logger

class modelTrainerPipeline:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, num_epochs=25):
        self.model =model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.num_epochs = num_epochs

    def train_model(self):
        model_training = modelTrainer()
        train_loss_hist, train_acc_hist, val_loss_hist, val_acc_hist = model_training.train_model(self.model, self.train_loader, self.val_loader, self.criterion, self.optimizer)
        return train_loss_hist, train_acc_hist, val_loss_hist, val_acc_hist


