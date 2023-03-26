from __future__ import annotations
import os, torch, torchvision
from PIL import Image
import numpy as np
import pandas as pd


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
>>> What is Object Recognition Problem ?

        * Object recognition is a fundamental problem in computer vision. It involves the process of identifying and classifying objects within 
          images or videos. Object recognition has numerous practical applications, such as autonomous vehicles, robotics, surveillance, image search, 
          and virtual or augmented reality systems.

        * There are several approaches to tackle object recognition problems, including traditional computer vision techniques and modern deep learning methods. 
          Among the deep learning techniques, Convolutional Neural Networks (CNNs) have become the state-of-the-art method for object recognition tasks, thanks to 
          their ability to learn hierarchical features from raw image data.

        * Popular object recognition datasets, such as ImageNet and CIFAR, have been widely used to benchmark and compare different models' performance. 
          Additionally, pre-trained models like VGG, ResNet, and MobileNet can be fine-tuned or used as feature extractors for specific object recognition tasks.
          
          
>>> what are the various steps in pytorch for dealing with Object Recognition problem ?
    In PyTorch, handling an image classification problem involves several key steps:

        1) Data preparation:

            * Collect and preprocess a labeled dataset, including data augmentation and normalization.
            * Create a custom Dataset class or use existing ones like ImageFolder or CIFAR-10.
            * Use DataLoader to create iterable batches for training and validation.
        
        2) Model architecture:

            * Define the neural network architecture (e.g., CNN, ResNet, or VGG) by subclassing nn.Module.
            * Initialize the model with the desired input and output dimensions, and layers.
            * Loss function and optimizer:
            * Choose a loss function (e.g., CrossEntropyLoss) to measure classification errors.
            * Select an optimizer (e.g., SGD or Adam) to update model weights based on the gradients.
        
        3) Training loop:

            * Iterate through the epochs, processing training batches.
            * Perform forward propagation, calculate the loss, and backpropagate gradients.
            * Update the model weights using the optimizer, and reset gradients.
            * Optionally, monitor training and validation loss, accuracy, and other metrics.
        
        
        4) Model evaluation and testing:

            * Evaluate the model on a separate test dataset to assess its performance.
            * Calculate metrics (e.g., accuracy, F1 score, precision, and recall) to quantify the model's performance.
        
        
        5) Model tuning and optimization:

            * Perform hyperparameter tuning, weight regularization, or learning rate scheduling to improve the model.
            * Use techniques like early stopping or model checkpointing to prevent overfitting and save the best-performing model.
        
        
        6) Deployment and inference:

            * Save the trained model using PyTorch's torch.save or torch.jit.
            * Load the model for inference and perform predictions on new, unseen images.

'''

__code_info__: str = r'''
    This ObjectRecognitionDataset class is a custom PyTorch dataset for handling object recognition tasks. It inherits from torch.utils.data.Dataset 
    and implements the required methods, including __init__, __len__, and __getitem__.

            * __init__: The constructor takes a data_directory argument, which is the path to the dataset, and an optional transform_dict argument, 
              which contains a dictionary of data transformations to apply to the images.

            * __len__: This method returns the number of samples in the dataset.

            * __repr__: This method returns a string representation of the dataset object, including its module, name, and object ID.

            * prepare_dataset: This is a static method that prepares the dataset by reading the data directory, collecting the image paths, and creating a DataFrame
              containing the image paths and their corresponding labels.

            * __getitem__: This method retrieves a single sample from the dataset, given its index. It reads the image from the corresponding image path, applies 
              the provided transformations (if any), and returns a dictionary containing the processed image (as 'independent_variable') and its label (as 'dependent_variable').


    This class is useful for creating an object recognition dataset in PyTorch, which can be used in combination with DataLoader for efficient data loading and processing during 
    training and evaluation of deep learning models.

'''
#@: Object Recognition data Class
class ObjectRecognitionDataset(torch.utils.data.Dataset):
    def __init__(self, data_directory: str, transform_dict: Optional[dict[str, Callable[Any]]] = None) -> None:
        self.data_dict: pd.DataFrame = ObjectRecognitionDataset.prepare_dataset(
            data_directory= data_directory
        )
        
        self.transform_dict = transform_dict
        
        
    
    def __len__(self) -> int:
        return len(self.data_dict)
    
    
    
    def __repr__(self) -> str(dict[str, Any]):
        return str({
            x: y for x, y in zip(['Module', 'Name', 'ObjectID'], [self.__module__, type(self).__name__, hex(id(self))])
        })



    @staticmethod
    def prepare_dataset(data_directory: str) -> pd.DataFrame:
        class_names: list[str] = os.listdir(data_directory)
        
        image_paths: list[str] = [
            os.path.join(data_directory, class_name, image_name)
            for class_name in class_names
            for image_name in os.listdir(os.path.join(data_directory, class_name))
        ]
        
        labels: list[str] = [
            index
            for index, class_name in enumerate(class_names)
            for _ in os.listdir(os.path.join(data_directory, class_name))
        ]
        
        data_dict: dict[str, list[str]] = {
            'image_path': image_paths, 'label': labels
        }
        
        data_dict: pd.DataFrame = pd.DataFrame(data_dict)
        return data_dict
    
    
    
    
    def __getitem__(self, index: int) -> dict[str, torch.tensor]:
        image_path: str = self.data_dict.iloc[index]['image_path']
        label: str = self.data_dict.iloc[index]['label']
        
        image: PIL.Image = Image.open(image_path).convert('RGB')
        
        if self.transform_dict is not None:
            for transform_func in self.transform_dict.values():
                image: Any = transform_func(image)
                
        return {
            'independent_variable': image, 'dependent_variable': label
        }
        
        
        

#@: Driver Code
if __name__.__contains__('__main__'):
    ...





