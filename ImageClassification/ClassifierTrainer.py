
from __future__ import annotations
import os, torch, torchvision
from PIL import Image
import pandas as pd
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


__author_info__: dict[str, Union[str, list[str]]] = {
    'Name': 'Rahul Sawhney',
    'Mail': [
        'sawhney.rahulofficial@outlook.com', 
        'rahulsawhney321@gmail.com'
    ]
}

__license__: str = r'''
    MIT License
    Copyright (c) 2023 Rahul Sawhney
    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:
    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.
    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
'''


__doc__: str = r'''
    The Image Classifier Trainer Class in PyTorch is a custom class that encapsulates the training and evaluation processes for an image classification 
    model. It simplifies the model training, validation, and performance evaluation processes by organizing the code in a structured way. Here's an 
    overview of the key components of the Image Classifier Trainer Class:

    1) Initialization: In the constructor (__init__) method, you'll define the model, loss function (criterion), optimizer, learning rate scheduler (if needed),
       and any other necessary components. You'll also set up the device (CPU or GPU) for running the computations.

    2) Training loop: The training loop is a method that iterates through the training dataset, feeding input images to the model and updating the model's weights 
       based on the calculated loss. The loop generally consists of the following steps:

        *    Set the model to training mode
        *    Iterate through batches of input images and labels
        *    Zero the gradients of the optimizer
        *    Forward pass through the model to obtain predictions
        *    Calculate the loss based on the predictions and ground truth labels
        *    Perform a backward pass to compute gradients
        *    Update the model's weights using the optimizer
        *    Track the training loss and accuracy


    3) Validation loop: The validation loop is a method that iterates through the validation dataset, feeding input images to the model and calculating the 
       loss and accuracy without updating the model's weights. This helps to evaluate the model's performance on unseen data and monitor overfitting. The loop 
       generally consists of the following steps:

        *    Set the model to evaluation mode
        *    Iterate through batches of input images and labels
        *    Forward pass through the model to obtain predictions
        *    Calculate the loss and accuracy based on the predictions and ground truth labels
        *    Track the validation loss and accuracy


    4) Training and validation process: This method combines the training and validation loops to train the model for a specified number of epochs. It 
       records the training and validation loss and accuracy at each epoch, and optionally saves the best model based on validation performance.


    5) Model saving and performance plotting: Include methods for saving the trained model to disk and plotting the training and validation loss and 
       accuracy over time. This allows you to visualize the model's learning progress and diagnose potential issues like overfitting or underfitting.

    
    Using the Image Classifier Trainer Class in PyTorch, you can streamline the training and evaluation of image classification models, making it easier 
    to experiment with different architectures, hyperparameters, and data augmentations.

'''


#@: Image Classifier Trainer Class 
class ImageClassifierTrainer:
    def __init__(self, model: torch.nn.Module, dataset: ImageClassificationDataset, batch_size: Optional[int] = 64, 
                                                                                    learning_rate: Optional[float] = 0.001, 
                                                                                    num_epochs: Optional[int] = 10, 
                                                                                    device: Optional[str] = 'cuda' ) -> None:
        
        self.model = model.to(device)
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.num_epochs = num_epochs
        
        train_index, val_index = train_test_split(range(len(dataset)), test_size= 0.2, random_state= 42)
        train_sampler = torch.utils.data.SubsetRandomSampler(train_index)
        val_sampler = torch.utils.data.SubsetRandomSampler(val_index)
        
        self.train_loader = DataLoader(dataset, batch_size= batch_size, sampler= train_sampler)
        self.val_loader = DataLoader(dataset, batch_size= batch_size, sampler= val_sampler)
        
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr= learning_rate)
        
        self.history: dict[str, list[float]] = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': []
        }
    
    
    
    
    def __repr__(self) -> str(dict[str, Any]):
        return str({
            x: y for x, y in zip(['Module', 'Name', 'ObjectID'], [self.__module__, type(self).__name__, hex(id(self))])
        })
        
    
    
    
    def train_loop(self) -> dict[str, list[float]]:
        for epoch in range(self.num_epochs):
            self.model.train()
            running_loss: float = 0.0
            running_corrects: int = 0
            
            for batch in self.train_loader:
                inputs: torch.tensor = batch['independent_variable'].to(self.device)
                labels: torch.tensor = batch['dependent_variable'].to(self.device)
                
                self.optimizer.zero_grad()
                
                outputs: torch.tensor = self.model(inputs)
                loss: Any = self.criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                loss.backward()
                self.optimizer.step()
                
                running_loss += loss.item()
                running_corrects += torch.sum(preds == labels.data)
            
            epoch_loss: float = running_loss / len(self.train_loader)
            epoch_acc: float = running_corrects.double() / len(self.train_loader.dataset)
            
            self.history['train_loss'].append(epoch_loss)
            self.history['train_acc'].append(epoch_acc.item())
            
            print(f"Epoch {epoch + 1}/{self.num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")
            
            self.validate_loop()
        
        return self.history
    
    
    
    
    def validate_loop(self) -> None:
        self.model.eval()
        running_loss: float = 0.0
        running_corrects: int = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                inputs: torch.tensor = batch['independent_variable'].to(self.device)
                labels: torch.tensor = batch['dependent_variable'].to(self.device)
                
                outputs: torch.tensor = self.model(inputs)
                loss: Any = self.criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                
                running_loss += loss.item()
                running_corrects += torch.sum(preds == labels.data)
                
            epoch_loss: float = running_loss / len(self.val_loader)
            epoch_acc: float = running_corrects / len(self.val_loader.dataset)
            
            self.history['val_loss'].append(epoch_loss)
            self.history['val_acc'].append(epoch_acc.item())
            
            print(f"Validation Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")
        
            
    
    
    
    
    def plot_metrics(self, save_path: Optional[str] = None) -> None:
        epochs: int = range(1, self.num_epochs + 1)

        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(epochs, self.history['train_loss'], label='Training Loss')
        plt.plot(epochs, self.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(epochs, self.history['train_acc'], label='Training Accuracy')
        plt.plot(epochs, self.history['val_acc'], label='Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
        
            
    
    
    def save_history(self, save_path: str) -> None:
        with open(save_path, 'w') as f:
            f.write(str(self.history))
            
    
    
    
    def save_model(self, save_path: str) -> None:
        torch.save(self.model.state_dict(), save_path)
    
    
    
    
        

#@: Driver Code
if __name__.__contains__('__main__'):
    
    #@: Example use case 
    dataset = ...
    model: torch.nn.Module = ...
    trainer = Trainer(model= model, dataset= dataset, batch_size= 64, learning_rate= 0.001, num_epochs= 10)
    history = trainer.train_loop()

    # Save the model and plot the metrics
    trainer.save_model('model_weights.pth')
    trainer.plot_metrics('training_metrics.png')
    
    
    
    
    
    