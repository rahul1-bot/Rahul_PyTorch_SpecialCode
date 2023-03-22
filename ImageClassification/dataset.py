
from __future__ import annotations
import os, torch, torchvision
from PIL import Image
import pandas as pd
import numpy as np

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

>>> What is Image Classification Problem ? 

    Image classification is a fundamental problem in computer vision that involves assigning a pre-defined label 
    to an input image based on its visual content. The objective is to train a model that can accurately identify 
    the main subject or category of an image, such as distinguishing between cats and dogs, or recognizing handwritten 
    digits. This task is typically achieved through supervised learning, where the model is trained on a dataset of 
    labeled images and learns to recognize patterns and features associated with each class.


>>> what are the various steps in pytorch for dealing with image classification problem ?
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



#@: Image Classification Dataset
class ImageClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, data_directory: str, transform_dict: Optional[dict[str, Callable[Any]]] = None) -> None:
        
        self.data_dict: pd.DataFrame = ImageClassificationDataset.prepare_dataset(
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
        image_paths: list[str] = [
            os.path.join(data_directory, filename) 
            for filename in os.listdir(data_directory)
        ]    

        labels: list[Union[str, int]] = [
            filename.split('_')[0] for filename in os.listdir(data_directory)
        ]
        
        data_dict: dict[str, list[Union[str, int]]] = {
            'image_path': image_paths,
            'label': labels
        } 
        
        data_dict: pd.DataFrame = pd.DataFrame(data_dict)
        return data_dict
    
    
    
    
    def __getitem__(self, index: int) -> dict[str, torch.tensor]:
        image_path: str = self.data_dict.loc[index, 'image_path']
        label: Union[int, str] = self.data_dict.loc[index, 'label']
        
        image: PIL.Image = Image.open(image_path).convert('RGB')
        
        if self.transform_dict is not None:
            for transform_func in self.transform_dict.values():
                image: Any = transform_func(image)
        
        return {
            'independent_variable': image, 'dependent_variable': label
        }



__code_info__: str = r'''
    The ImageClassificationDataset class is a subclass of torch.utils.data.Dataset, designed for handling image classification data. 
    It accepts a data_directory containing images and an optional transform_dict for preprocessing. The constructor prepares a DataFrame 
    with image paths and labels. The len method returns the dataset length, while repr provides a string representation. The static method 
    prepare_dataset processes the data_directory to create the DataFrame. The getitem method loads images, applies transformations if provided, 
    and returns a dictionary containing the image and its corresponding label.
'''
        
        
        
#@: Driver Code
if __name__.__contains__('__main__'):
    print('end')