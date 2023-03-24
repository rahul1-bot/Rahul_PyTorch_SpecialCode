from __future__ import annotations
import os, torch, torchvision
from PIL import Image
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.optim as optim



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
    A Convolutional Neural Network (CNN) is a type of deep learning model specifically designed for processing grid-like data, 
    such as images. CNNs are highly effective in image recognition and classification tasks due to their ability to learn 
    hierarchical patterns and local features in the data. The architecture of a CNN typically consists of several types of 
    layers arranged in a sequential manner, each performing a specific operation on the input data. The main components of a 
    CNN architecture are:

    * Convolutional layers: The core building block of CNNs, convolutional layers perform the convolution operation on the 
      input data using a set of learnable filters (also known as kernels). Each filter is responsible for capturing a specific 
      feature or pattern in the input image, such as edges, textures, or shapes. As the network goes deeper, the filters learn 
      to recognize more complex and abstract features. The output of a convolutional layer is called a feature map.

    * Activation functions: Non-linear activation functions are applied to the output of convolutional layers, introducing 
      non-linearity into the model. The most commonly used activation function in CNNs is the Rectified Linear Unit (ReLU),
      which sets all negative values to zero and maintains positive values unchanged. This helps to mitigate the vanishing 
      gradient problem and speeds up the training process.

    * Pooling layers: Pooling layers are used to reduce the spatial dimensions of the feature maps, thereby decreasing the 
      number of parameters and computational cost in the network. This also helps to achieve translation invariance and control 
      overfitting. The most common pooling operation is max pooling, which takes the maximum value in each non-overlapping local 
      region of the input feature map.

    * Fully connected layers: After the convolutional and pooling layers have extracted hierarchical features from the input data, 
      the output feature maps are flattened and fed into one or more fully connected layers (also known as dense layers). These layers 
      perform the final classification by mapping the learned features to the output classes. The last fully connected layer typically 
      has as many neurons as there are classes in the classification problem and uses a softmax activation function to produce class 
      probabilities.

    * Dropout layers (optional): Dropout is a regularization technique that helps prevent overfitting by randomly dropping out neurons 
      during training. This forces the network to learn more robust and generalized features. Dropout layers can be added between fully 
      connected layers or, less commonly, between convolutional layers.

    * Batch normalization layers (optional): Batch normalization is a technique used to improve the training process by normalizing the 
      input to each layer. This can help reduce internal covariate shift and allows for using higher learning rates, leading to faster
      convergence and better generalization. Batch normalization layers can be added after convolutional or fully connected layers, 
      usually before the activation function.

    The specific architecture of a CNN depends on the problem at hand and may involve varying the number of layers, filter sizes, 
    and layer configurations. Some well-known CNN architectures include LeNet-5, AlexNet, VGG, ResNet, and Inception, which have 
    demonstrated state-of-the-art performance on various image classification benchmarks.

'''


__note__: str = r'''
    * the models I provided earlier are beginner-friendly and designed to introduce you to the basics of creating custom CNN 
      architectures in PyTorch. They start from a simple model with a single convolutional layer and gradually increase in complexity, 
      allowing you to understand the role of each layer in the network and how they can be combined to create more advanced architectures.
'''


#@: Six Custom Deep learning models for Image Classification, starting from a simple one and gradually increasing in complexity 

#@: Model with a Single Linear Layer
class SimplestModel(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super(SimplestModel, self).__init__()
        self.fc = nn.Linear(224 * 224 * 3, num_classes)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
    
    
    
#@: Model with a single Convolutional layer followed by a linear layer:
class SimpleConvModel(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super(SimpleConvModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.fc = nn.Linear(16 * 224 * 224, num_classes)



    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    



#@: Model with two convolutional layers and a linear layer:
class TwoConvModel(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super(TwoConvModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc = nn.Linear(32 * 224 * 224, num_classes)



    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x




#@: Model with two convolutional layers, max pooling, and a linear layer:
class ConvPoolModel(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super(ConvPoolModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(32 * 112 * 112, num_classes)



    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x




#@: Model with three convolutional layers, max pooling, and a linear layer:
class ThreeConvModel(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super(ThreeConvModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(64 * 56 * 56, num_classes)



    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x




#@: Driver Code
if __name__.__contains__('__main__'):
    ...
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    