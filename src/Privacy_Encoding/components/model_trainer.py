

import time

import torch


class ModelTrainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer,  num_epochs=25):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = num_epochs
        # self.batch_size = batch_size
    
    def train_model(self, is_train=True):
        since = time.time()
        
        acc_history = []
        loss_history = []

        train_loss_history = []
        val_loss_history = []
        train_accuracy_history = []
        val_accuracy_history = []

        best_acc = 0.0
        
        for epoch in range(self.num_epochs):
            self.model.train()  # Set the model to training mode
            running_loss = 0.0
            running_corrects = 0
            correct_train = 0
            total_train = 0

            for images, labels in self.train_loader:
                self.optimizer.zero_grad()  # Zero the gradients

                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                # Backward pass and optimization
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

                # Calculate training accuracy
                _, predicted = torch.max(outputs.data, 1)
                total_train += labels.size(0)
                correct_train += (predicted == labels).sum().item()

            train_loss = running_loss / len(self.train_loader)
            train_accuracy = correct_train / total_train
            train_loss_history.append(train_loss)
            train_accuracy_history.append(train_accuracy)

            print('Loss: {:.4f} Acc: {:.4f}'.format(train_loss, train_accuracy))

            # Validation
            self.model.eval()  # Set the model to evaluation mode
            correct_val = 0
            total_val = 0
            val_running_loss = 0.0
            

            with torch.no_grad():
                for images, labels in self.val_loader:
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)

                    val_running_loss += loss.item()

                    # Calculate validation accuracy
                    _, predicted = torch.max(outputs.data, 1)
                    total_val += labels.size(0)
                    correct_val += (predicted == labels).sum().item()

                val_loss = val_running_loss / len(self.val_loader)
                val_accuracy = correct_val / total_val
                val_loss_history.append(val_loss)
                val_accuracy_history.append(val_accuracy)

            # scheduler.step(val_loss)

            print(f"Epoch [{epoch + 1}/{self.num_epochs}] Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best Acc: {:4f}'.format(best_acc))
        
        return train_loss_history, train_accuracy_history, val_loss_history, val_accuracy_history   