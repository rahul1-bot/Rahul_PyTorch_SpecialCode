
from __future__ import annotations
import numpy as np
import os, torch, torchvision
from PIL import Image
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
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

__code_doc__: str = r'''
    *   The Object Recognition Predictor class is a utility class that can be used to make predictions using a trained neural network model for the task of object recognition. 
        This class takes in a PyTorch model, which is expected to be trained on a dataset of images for object recognition. It also takes in a dictionary of image transformation 
        functions that can be used to preprocess the input image before feeding it into the model.

    *   The Object Recognition Predictor class provides two main functions for making predictions. The predict() function takes an image file path as input, and returns a dict 
        containing the predicted class name and the corresponding probability. The predict_top_k() function takes an image file path and an integer k as input, and returns a dictionary 
        containing the top k predicted class names and corresponding probabilities.

    *   Additionally, the Object Recognition Predictor class also provides a visualize_predictions() function that can be used to visualize the input image along with the predicted 
        class name and probability. The class also has a repr() method that returns a string representation of the class object.
'''

#@: Object Recognition Predictor Class 

class ObjectRecognitionPredictor:
    def __init__(self, model: nn.Module, transform_dict: Optional[dict[str, Callable]] = None, 
                                         class_names: Optional[list[str]] = None) -> None:
        
        device: torch.device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
        
        self.model = model.to(device)
        self.transform_dict = transform_dict
        self.class_names = class_names or [
            'class_1', 'class_2', 'class_3' # replace with your own class names
        ] 
        
                                         
    
    def __repr__(self) -> str:
        return f'''ObjectRecognitionPredictor(
            model= {self.model}, 
            transform_dict= {self.transform_dict}, 
            class_names= {self.class_names})
        '''
    
                                         
    __method_one_doc: str = r'''
        The above code defines the predict_result method in the ObjectRecognitionPredictor class, which takes an image path as input and returns a dictionary with the predicted 
        class name and its probability.

        The method first opens the image using PIL, applies the specified transformations using the transform_dict, and converts it to a tensor. It then passes the tensor through 
        the model and calculates the softmax probabilities for each class. The class with the highest probability is selected using the torch.max function, and its index is used to retrieve the corresponding class name from the class_names list. The predicted class name and its probability are returned in the form of a dictionary.
    
    '''                                     
    def predict_result(self, image_path: str) ->  dict[str, Union[str, float]]:
        image: PIL.Image = Image.open(image_path).convert('RGB')
        
        if self.transform_dict is not None:
            for transform_func in self.transform_dict.values():
                image: Any = transform_func(image)
        
        with torch.no_grad():
            output: torch.tensor = self.model(image.unsqueeze(0))
        
        probabilities: torch.tensor = torch.softmax(output, dim= 1)
        top_probabilities, top_class = torch.max(probabilities, dim= 1)
        class_index: int = top_class.item()
        class_name: str = self.class_names[class_index]
        probability: float = top_probabilities.item()
        
        return {
            'class_name': class_name, 'probability': probability
        }
        
        
    
    
    __nethod_two_doc: str = r'''
        In the predict_top_k() method, the function takes an image path as input, preprocesses the image using the specified transformations, and feeds it to the model to get the output 
        probabilities. Then it selects the top-k probabilities and their corresponding indices and stores them in a dictionary where the class name is the key and the probability 
        is the value. The method returns this dictionary.
    
    '''
    def predict_top_k(self, image_path: str, k: Optional[int] = 5) -> dict[str, float]:
        image: PIL.Image = Image.open(image_path).convert('RGB')
        
        if self.transform_dict is not None:
            for transform_func in self.transform_dict.values():
                image: Any = transform_func(image)
        
        with torch.no_grad():
            output: torch.tensor = self.model(image.unsqueeze(0))
            
        probabilities: torch.tensor = torch.softmax(output, dim= 1)
        top_probabilities, top_indices = torch.topk(probabilities, k)
        top_probabilities: Any = top_probabilities.cpu().numpy().flatten()
        top_indices: Any = top_indices.cpu().numpy().flatten()
        
        top_classes: list[str] = [
            self.class_names[idx] for idx in top_indices
        ]
        
        result_dict: dict[str, float] = {
            top_classes[idx]: top_probabilities[idx] for idx in range(len(top_classes))
        }
        
        return result_dict
    
    
        
    
    __method_three_doc: str = r'''
        The visualize_predictions method takes an image path and an optional integer k as input, where k specifies the number of top classes to show in the visualization. 
        It uses the predict_top_k method to get the top k predicted classes and their probabilities for the input image, and then displays the input image along with a 
        horizontal bar chart showing the top classes and their probabilities.

        The bar chart is created using the Matplotlib library. The class_names and probabilities lists are extracted from the result_dict dictionary returned by the predict_top_k method.
        The y_pos array is created using numpy.arange function and ax2.barh function is used to plot horizontal bar chart. The y_pos array is passed as the first argument, 
        probabilities as the second argument, and class_names as the labels for the y-axis ticks. ax2.invert_yaxis() is called to invert the order of the classes and probabilities 
        so that the highest probability class is shown at the top. Finally, the plot is displayed using plt.show() method.
    
    '''
    def visualize_predictions(self, image_path: str, k: Optional[int] = 5) -> None:
        result_dict: dict[str, float] = self.predict_top_k(
            image_path= image_path, k= k
        )
        
        class_names: list[str] = list(result_dict.keys())
        probabilities: list[float] = list(result_dict.values())
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        image: PIL.Image = Image.open(image_path).convert('RGB')
        ax1.imshow(image)
        ax1.axis('off')
        
        y_pos: Any = np.arange(len(class_names))
        ax2.barh(y_pos, probabilities)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(class_names)
        ax2.invert_yaxis()
        ax2.set_xlabel('Probability')
        ax2.set_title('Top Classes')
        
        plt.show()
        
        
        
        

#@: Driver Code
if __name__.__contains__('__main__'):
    #@> example_use case 
    
    dataset = ...
    model: torch.nn.Module = ...
    class_names: list[str] = ...
    
    transform_dict: dict[str, Callable[Any]] = {
        'RandomCropTransform': RamdomCropTransform(),
        'RandomFlipTransform': RandomFlipTransform(),
        'RandomRotationTransform': RandomRotationTransform(),
        'ColorJitterTransform': ColorJitterTransform(),
        'GaussianBlurTransform': GaussianBlurTransform(),
        'RandomAffineTransform': RandomAffineTransform(),
        'NoiseInjectionTransform': NoiseInjectionTransform(),
        'CutoutTransform': CutoutTransform(),
        'ToTensorTransform': ToTensorTransform(),
        'NormalizeTransform': NormalizeTransform(mean= (0.485, 0.456, 0.406), std= (0.229, 0.224, 0.225))
    }
    
    #: train the model from 'Trainer' Class
    #: save the model history and weights...
    #: Predict the result from 'Predictor' class which is mentioned above...
    
    
    
    