from __future__ import annotations
import os, torch, torchvision
from PIL import Image
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
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
    * the models, I provided are beginner-friendly and designed to introduce you to the basics of creating custom CNN 
      architectures in PyTorch. They start from a simple model with a single convolutional layer and gradually increase in complexity, 
      allowing you to understand the role of each layer in the network and how they can be combined to create more advanced architectures.
'''


#@: Complexity: 1> Simple CNNs
__model_one__: str = r'''
        This code defines a simple convolutional neural network (CNN) class in PyTorch for object recognition tasks. The SimpleCNN class inherits 
        from PyTorch's nn.Module class, which is the base class for all neural network modules in PyTorch. This network has one convolutional layer, 
        one pooling layer, and one fully connected layer.

'''
class SimpleCNN(nn.Module):
    def __init__(self, num_classes: Optional[int] = 10) -> None:
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding= 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(16 * 16 * 16, num_classes)



    def forward(self, x: torch.tensor) -> None:
        x: torch.tensor = self.pool(F.relu(self.conv1(x)))
        x: torch.tensor = x.view(-1, 16 * 16 * 16)
        x: torch.tensor = self.fc(x)
        return x



#@: Complexity: 2> Two-Layers CNN
__model_two__: str = r'''
    This code defines a simple two-layer Convolutional Neural Network (CNN) for an object recognition problem in PyTorch. Let me break down the code for you:

        *   class TwoLayerCNN(nn.Module): - This line defines a new class called TwoLayerCNN that inherits from PyTorch's nn.Module class. nn.Module is the 
            base class for all neural network modules in PyTorch.

        *   def __init__(self, num_classes: Optional[int] = 10) -> None: - This is the constructor for the TwoLayerCNN class, which takes an optional argument 
            num_classes that defaults to 10. The num_classes variable defines the number of output classes for the object recognition problem.

        *   self.conv1 = nn.Conv2d(3, 16, 3, padding= 1) - This line defines the first convolutional layer, which takes an input with 3 channels (e.g., RGB images), 
            applies 16 filters with a kernel size of 3x3, and uses padding of 1 to maintain the input size.

        *   self.conv2 = nn.Conv2d(16, 32, 3, padding= 1) - This line defines the second convolutional layer, which takes an input with 16 channels (output from 
            the first convolutional layer), applies 32 filters with a kernel size of 3x3, and uses padding of 1 to maintain the input size.

        *   self.pool = nn.MaxPool2d(2, 2) - This line defines a max-pooling layer with a kernel size of 2x2 and a stride of 2. This layer is used to downsample 
            the feature maps after each convolutional layer.

        *   self.fc = nn.Linear(32 * 8 * 8, num_classes) - This line defines a fully connected (linear) layer that takes a flattened input of size 32 * 8 * 8 
            (output from the second pooling layer) and produces an output with the size of num_classes.


        *   def forward(self, x: torch.tensor) -> torch.tensor: - This method defines the forward pass of the neural network.

            
        The following lines in the forward method apply the defined layers in sequence:
        *   Apply the first convolutional layer followed by ReLU activation and max-pooling.
        *   Apply the second convolutional layer followed by ReLU activation and max-pooling.
        *   Flatten the output feature map.
        *   Apply the fully connected layer.


    This simple two-layer CNN can be used for object recognition problems, and its architecture can be easily adjusted for more complex problems by adding more layers or modifying the existing layers' parameters.

'''
class TwoLayerCNN(nn.Module):
    def __init__(self, num_classes: Optional[int] = 10) -> None:
        super(TwoLayerCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding= 1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding= 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(32 * 8 * 8, num_classes)



    def forward(self, x: torch.tensor) -> torch.tensor:
        x: torch.tensor = self.pool(F.relu(self.conv1(x)))
        x: torch.tensor = self.pool(F.relu(self.conv2(x)))
        x: torch.tensor = x.view(-1, 32 * 8 * 8)
        x: torch.tensor = self.fc(x)
        return x




#@: Complexity: 3> Three-layer CNN with dropout:
__model_three__: str = r'''
    This code defines a three-layer CNN for object recognition in PyTorch. It has three convolutional layers with ReLU activations and max-pooling for feature extraction. 
    A dropout layer prevents overfitting, and a fully connected layer classifies the extracted features into object categories.

'''
class ThreeLayerCNN(nn.Module):
    def __init__(self, num_classes: Optional[int] = 10) -> None:
        super(ThreeLayerCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding= 1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding= 1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding= 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        self.fc = nn.Linear(64 * 4 * 4, num_classes)
        


    def forward(self, x: torch.tensor) -> torch.tensor:
        x: torch.tensor = self.pool(F.relu(self.conv1(x)))
        x: torch.tensor = self.pool(F.relu(self.conv2(x)))
        x: torch.tensor = self.pool(F.relu(self.conv3(x)))
        x: torch.tensor = x.view(-1, 64 * 4 * 4)
        x: torch.tensor = self.dropout(x)
        x: torch.tensor = self.fc(x)
        return x
    
    


#@: Complexity: 4> Four-layer CNN with batch normalization:
__model_four__: str = r'''
    This code defines a four-layer CNN for object recognition in PyTorch. It has four convolutional layers with ReLU activations and max-pooling for feature extraction. 
    Batch normalization is applied after each convolutional layer to improve training speed and stability. A fully connected layer classifies the extracted features 
    into object categories.

'''
class FourLayerCNN(nn.Module):
    def __init__(self, num_classes: Optional[int] = 10) -> None:
        super(FourLayerCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding= 1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding= 1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding= 1)
        self.conv4 = nn.Conv2d(64, 128, 3, padding= 1)
        
        self.pool = nn.MaxPool2d(2, 2)
        
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(128)
        
        self.fc = nn.Linear(128 * 2 * 2, num_classes)



    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x: torch.Tensor = self.pool(F.relu(self.bn1(self.conv1(x))))
        x: torch.Tensor = self.pool(F.relu(self.bn2(self.conv2(x))))
        x: torch.Tensor = self.pool(F.relu(self.bn3(self.conv3(x))))
        x: torch.Tensor = self.pool(F.relu(self.bn4(self.conv4(x))))
        x: torch.Tensor = x.view(-1, 128 * 2 * 2)
        x: torch.Tensor = self.fc(x)
        return x

    
    
#@: Complexity 5: Eight-Layer CNN with Batch Normalization and Dropout layers
__model_five__: str = r'''
        * This code defines an eight-layer Convolutional Neural Network (CNN) for object recognition in PyTorch. The model architecture consists of two primary 
          components: the features extractor and the classifier. The features component is a sequence of four pairs of convolutional layers, each followed by batch 
          normalization, ReLU activation, and max-pooling. This sequence of layers extracts complex features from the input image by repeatedly applying convolutions 
          and pooling operations.


        * The classifier component is responsible for converting the extracted features into class probabilities. It consists of two fully connected layers with 
          dropout and ReLU activation applied between them. Dropout helps prevent overfitting by randomly dropping connections between neurons during training. 
          The ReLU activation function introduces non-linearity into the model and helps it learn more complex relationships in the data.


        * The forward pass of the model starts by passing the input image through the features component, which processes the image and extracts useful features. 
          The output of the features component is then flattened and passed through the classifier component to obtain the class probabilities.


    In summary, the EightLayerCNN model is a deep learning architecture specifically designed for object recognition tasks. It uses a combination of convolutional layers, 
    batch normalization, ReLU activations, max-pooling, dropout, and fully connected layers to learn and classify input images effectively.
'''

class EightLayerCNN(nn.Module):
    def __init__(self, num_classes: Optional[int] = 10) -> None:
        super(EightLayerCNN, self).__init__()
        
        self.features: torch.nn.Sequential = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size= 3, padding= 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace= True),
            nn.Conv2d(64, 64, kernel_size= 3, padding= 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace= True),
            nn.MaxPool2d(kernel_size= 2, stride= 2),
            
            nn.Conv2d(64, 128, kernel_size= 3, padding= 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace= True),
            nn.Conv2d(128, 128, kernel_size= 3, padding= 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace= True),
            nn.MaxPool2d(kernel_size= 2, stride= 2),

            nn.Conv2d(128, 256, kernel_size= 3, padding= 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace= True),
            nn.Conv2d(256, 256, kernel_size= 3, padding= 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace= True),
            nn.MaxPool2d(kernel_size= 2, stride= 2),
            
            nn.Conv2d(256, 512, kernel_size= 3, padding= 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace= True),
            nn.Conv2d(512, 512, kernel_size= 3, padding= 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace= True),
            nn.MaxPool2d(kernel_size= 2, stride= 2)
        )
        
        self.classifier: torch.nn.Sequential = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(inplace= True),
            nn.Dropout(),
            nn.Linear(512, num_classes)
        )



    def forward(self, x: torch.tensor) -> torch.tensor:
        x: torch.tensor = self.features(x)
        x: torch.tensor = x.view(x.size(0), -1)
        x: torch.tensor = self.classifier(x)
        return x
    
    
    

#@: Driver Code
if __name__.__contains__('__main__'):
    ...