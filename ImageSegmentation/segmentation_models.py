from __future__ import annotations
import os, torch, torchvision, random
from PIL import Image
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
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
    *   Image segmentation models are deep learning models designed to automatically segment an image into different regions or objects. These models are used extensively in computer vision 
        applications such as object detection, scene understanding, medical image analysis, and autonomous driving.

    *   There are various types of image segmentation models, but some of the most popular ones include Fully Convolutional Networks (FCN), U-Net, SegNet, RefineNet, DeepLabv3+, and Efficient 
        Neural Networks (ENet). Each of these models has its own architecture and design principles, but they all aim to achieve accurate and efficient image segmentation.

    *   Image segmentation models typically take an image as input and output a pixel-wise prediction of the image. This means that for each pixel in the input image, the model predicts a label 
        that corresponds to the class or object that the pixel belongs to. The output prediction can be either a binary mask (i.e., foreground/background segmentation) or a multi-class segmentation 
        map (i.e., each pixel is labeled with a specific object or class).

    *   These models are usually trained on large-scale datasets with pixel-level annotations to learn the mapping between the input image and the corresponding segmentation labels. Once trained, 
        they can be used to segment images in real-time and in a variety of applications.

    *   Overall, image segmentation models are an important tool for computer vision researchers and practitioners, and their continued development and improvement will enable new applications 
        and advances in the field.
'''

#@ -------------------------------------------------- UNet --------------------------------------------------------------
__UNet_doc: str = r'''
    *   U-Net is a popular architecture for image segmentation that was introduced by Olaf Ronneberger et al. in their paper "U-Net: Convolutional Networks for Biomedical Image Segmentation". This 
        architecture is widely used in various fields, including medical image segmentation, satellite imagery, and industrial inspection.

    *   The U-Net architecture consists of an encoder-decoder network that is connected by a series of skip connections. The encoder network consists of a series of convolutional layers and max-pooling 
        layers that reduce the spatial resolution of the input image, while increasing the number of feature channels. The decoder network, on the other hand, consists of a series of up-convolutional 
        layers that increase the spatial resolution of the feature maps while reducing the number of feature channels.

    *   The skip connections in U-Net are used to combine the feature maps from the encoder network with the corresponding feature maps in the decoder network. This helps to preserve spatial information 
        and prevent loss of details during the upsampling process. The skip connections allow the model to combine low-level and high-level features, which are important for accurate segmentation.

    *   The U-Net architecture has several advantages over other segmentation models. First, it has a relatively small number of parameters, which makes it computationally efficient and easier to train. 
        Second, the skip connections help to preserve spatial information, which can lead to more accurate segmentation results. Finally, the U-Net architecture is flexible and can be adapted to different
        applications by changing the number and size of the layers.

    *   Overall, U-Net is a powerful architecture for image segmentation that has been used extensively in various fields. Its encoder-decoder structure and skip connections make it effective at capturing 
        both low-level and high-level features, which is critical for accurate segmentation.

'''
class UNet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(UNet, self).__init__()

        #@: Encoder
        self.encoder1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size= 3, padding= 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size= 3, padding= 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.encoder2 = nn.Sequential(
            nn.MaxPool2d(kernel_size= 2, stride= 2),
            nn.Conv2d(64, 128, kernel_size= 3, padding= 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size= 3, padding= 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.encoder3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size= 3, padding= 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size= 3, padding= 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.encoder4 = nn.Sequential(
            nn.MaxPool2d(kernel_size= 2, stride= 2),
            nn.Conv2d(256, 512, kernel_size= 3, padding= 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size= 3, padding= 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        #@: MID Bottleneck
        self.bottleneck = nn.Sequential(
            nn.MaxPool2d(kernel_size= 2, stride= 2),
            nn.Conv2d(512, 1024, kernel_size= 3, padding= 1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size= 3, padding= 1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )

        #@: Decoder
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size= 2, stride= 2),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size= 3, padding= 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size= 3, padding= 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size= 2, stride= 2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size= 3, padding= 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size= 3, padding= 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.decoder3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size= 2, stride= 2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size= 3, padding= 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        #@: final_layer
        self.output = nn.Conv2d(64, out_channels, kernel_size= 1)




    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #@: Encoder
        x1: torch.Tensor = self.encoder1(x)
        x2: torch.Tensor = self.encoder2(x1)
        x3: torch.Tensor = self.encoder3(x2)
        x4: torch.Tensor = self.encoder4(x3)

        #@: Bottleneck
        x5: torch.Tensor = self.bottleneck(x4)

        #@: Decoder
        x6: torch.Tensor = self.decoder1(x5)
        x6: torch.Tensor = torch.cat([x6, x4], dim= 1)
        x7: torch.Tensor = self.decoder2(x6)
        x7: torch.Tensor = torch.cat([x7, x3], dim= 1)
        x8: torch.Tensor = self.decoder3(x7)
        x8: torch.Tensor = torch.cat([x8, x2], dim= 1)
        x9: torch.Tensor = self.decoder4(x8)
        x9: torch.Tensor = torch.cat([x9, x1], dim= 1)

        # Output layer
        out: torch.Tensor = self.output(x9)

        return out





#@: -------------------------------------  Feature Pyramid Network ---------------------------------------------------------
__fpn_doc: str = r'''
   *    Feature Pyramid Network (FPN) is a popular architecture for object detection and instance segmentation that was introduced by Lin et al. in their paper "Feature Pyramid Networks for Object Detection". 
        The FPN architecture is designed to address the problem of scale variation in object detection, where objects of different sizes can appear in the same image.

   *    The FPN architecture consists of a bottom-up pathway and a top-down pathway that are connected by a lateral connection. The bottom-up pathway is a standard convolutional neural network (CNN) that 
        processes the input image and generates a set of feature maps at different spatial resolutions. The top-down pathway consists of a series of upsampling layers that increase the spatial resolution of 
        the feature maps, while reducing the number of channels. The lateral connection connects the feature maps from the bottom-up pathway to the top-down pathway, allowing the model to combine low-level and 
        high-level features.

   *    The FPN architecture has several advantages over other object detection models. First, it provides a multi-scale feature representation that is effective at detecting objects of different sizes. 
        Second, it is computationally efficient because it reuses the feature maps from the bottom-up pathway, rather than generating new feature maps at every scale. Third, it is flexible and can be adapted 
        to different tasks by changing the number and size of the layers.

   *    The FPN architecture has been widely used in various applications, including object detection, instance segmentation, and semantic segmentation. It has also been extended to incorporate more advanced features, 
        such as attention mechanisms and non-local blocks, to improve performance further.

   *    Overall, Feature Pyramid Network is a powerful architecture for object detection and instance segmentation that provides a multi-scale feature representation and is computationally efficient. Its ability 
        to handle scale variation makes it well-suited for a wide range of applications, and its flexibility and adaptability make it a popular choice for researchers and practitioners in computer vision.
'''
class FCN(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(FCN, self).__init__()

        # Encoder
        self.encoder1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size= 3, padding= 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size= 3, padding= 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size= 2, stride= 2)
        )

        self.encoder2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size= 3, padding= 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size= 3, padding= 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.encoder3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size= 3, padding= 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size= 3, padding= 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size= 3, padding= 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size= 2, stride= 2)
        )

        self.encoder4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size= 3, padding= 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size= 3, padding= 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size= 3, padding= 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size= 2, stride= 2)
        )

        self.encoder5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size= 3, padding= 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size= 3, padding= 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size= 3, padding= 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Decoder
        self.decoder5 = nn.Sequential(
            nn.ConvTranspose2d(512, 512, kernel_size= 2, stride= 2),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size= 3, padding= 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size= 3, padding= 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
        self.decoder4 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size= 2, stride= 2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size= 3, padding= 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size= 3, padding= 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.decoder3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size= 2, stride= 2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size= 3, padding= 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size= 2, stride= 2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size= 3, padding= 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size= 2, stride= 2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, kernel_size= 3, padding= 1)
        )




    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        x1: torch.Tensor = self.encoder1(x)
        x2: torch.Tensor = self.encoder2(x1)
        x3: torch.Tensor = self.encoder3(x2)
        x4: torch.Tensor = self.encoder4(x3)
        x5: torch.Tensor = self.encoder5(x4)

        # Decoder
        d5: torch.Tensor = self.decoder5(x5)
        d4: torch.Tensor = self.decoder4(d5 + x4)
        d3: torch.Tensor = self.decoder3(d4 + x3)
        d2: torch.Tensor = self.decoder2(d3 + x2)
        d1: torch.Tensor = self.decoder1(d2 + x1)

        return d1





#@: ----------------------------------------------------------- SegNet ------------------------------------------------------------
__segnet_doc: str = r'''
    *   SegNet is a popular architecture for image segmentation that was introduced by Vijay Badrinarayanan et al. in their paper "SegNet: A Deep Convolutional Encoder-Decoder Architecture for 
        Image Segmentation". The SegNet architecture is designed to perform pixel-wise segmentation of images, where each pixel in the input image is assigned a label corresponding to the object or 
        region it belongs to.

    *   The SegNet architecture consists of an encoder-decoder network that is connected by a series of skip connections. The encoder network is a series of convolutional and max-pooling layers that reduce 
        the spatial resolution of the input image, while increasing the number of feature channels. The decoder network consists of a series of upsampling and convolutional layers that increase the spatial 
        resolution of the feature maps while reducing the number of feature channels.

    *   The skip connections in SegNet are used to combine the feature maps from the encoder network with the corresponding feature maps in the decoder network. This helps to preserve spatial information 
        and prevent loss of details during the upsampling process. The skip connections allow the model to combine low-level and high-level features, which are important for accurate segmentation.

    *   SegNet also includes a pooling index layer that stores the pooling indices from the encoder network. These indices are used in the decoder network to perform max-unpooling, which helps to recover 
        the spatial resolution of the feature maps. This is a unique feature of SegNet that allows it to recover fine-grained details in the segmentation.

    *   SegNet has several advantages over other segmentation models. First, it has a relatively small number of parameters, which makes it computationally efficient and easier to train. Second, the use 
        of skip connections helps to preserve spatial information, which can lead to more accurate segmentation results. Finally, the SegNet architecture is flexible and can be adapted to different applications
        by changing the number and size of the layers.

    *   Overall, SegNet is a powerful architecture for image segmentation that has been used extensively in various fields. Its encoder-decoder structure and skip connections make it effective at capturing 
        both low-level and high-level features, which is critical for accurate segmentation.

'''
class SegNet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(SegNet, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        )

        self.decoder = nn.Sequential(
            nn.MaxUnpool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.MaxUnpool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.MaxUnpool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding= 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.MaxUnpool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.MaxUnpool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, kernel_size=3, padding=1),
        )
        
    
    
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pool_indices: list[Any] = []
        pool_sizes: list[Any] = []

        for idx, layer in enumerate(self.encoder):
            if isinstance(layer, nn.MaxPool2d):
                x, indices = layer(x)
                pool_indices.append(indices)
                pool_sizes.append(x.size())
            else:
                x = layer(x)

        for idx, layer in enumerate(self.decoder):
            if isinstance(layer, nn.MaxUnpool2d):
                x = layer(x, pool_indices.pop(), output_size= pool_sizes.pop())
            else:
                x = layer(x)

        return x
    
    
    
    
    
    
#@: ------------------------------------ ENet ----------------------------------------------------------
__enet_doc: str = r'''
    *   ENet (Efficient Neural Network) is a lightweight architecture for semantic image segmentation that was introduced by Paszke et al. in their paper "ENet: A Deep Neural Network Architecture 
        for Real-Time Semantic Segmentation". ENet is designed to be computationally efficient and fast, making it well-suited for real-time applications such as autonomous driving and robotics.

    *   The ENet architecture consists of a series of encoder-decoder blocks that are connected by a bottleneck layer. The encoder network is a series of convolutional and pooling layers that reduce the 
        spatial resolution of the input image, while increasing the number of feature channels. The decoder network consists of a series of upsampling and convolutional layers that increase the spatial 
        resolution of the feature maps while reducing the number of feature channels.

    *   The bottleneck layer in ENet is a 1x1 convolutional layer that reduces the number of feature channels before passing them to the decoder network. This reduces the computational cost of the model 
        while preserving the most important features.

    *   ENet also includes several other optimizations to reduce the number of parameters and improve efficiency, such as asymmetric convolutions and factorized convolutions. Asymmetric convolutions use two 
        separate kernels for the horizontal and vertical dimensions of the feature maps, which reduces the number of parameters without sacrificing performance. Factorized convolutions split a large convolutional 
        kernel into smaller ones, which reduces the number of parameters and speeds up computation.

    *   ENet has several advantages over other segmentation models. First, it is computationally efficient and can run in real-time on low-power devices such as embedded systems and mobile devices. Second, 
        it is lightweight and has a relatively small number of parameters, which makes it easy to train and deploy. Finally, ENet is flexible and can be adapted to different applications by changing the number 
        and size of the layers.

    *   Overall, ENet is a powerful architecture for semantic image segmentation that is designed for efficiency and speed. Its use of encoder-decoder blocks and bottleneck layers make it effective at capturing 
        both low-level and high-level features, which is critical for accurate segmentation.

'''
class InitialBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(InitialBlock, self).__init__()
        self.main_branch = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.ext_branch = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
        )



    def forward(self, x: torch.Tensor) -> torch.Tensor:
        main: torch.Tensor = self.main_branch(x)
        ext: torch.Tensor = self.ext_branch(x)
        out: torch.Tensor = torch.cat((main, ext), 1)
        return out





class Bottleneck(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, internal_ratio: Optional[int] = 4, 
                                                            kernel_size: Optional[int] = 3,
                                                            padding: Optional[int] = 1, 
                                                            dilation: Optional[int] = 1, 
                                                            asymmetric: Optional[bool] = False, 
                                                            dropout_prob: Optional[float] = 0.1,
                                                            relu: Optional[bool] = True, 
                                                            downsample: Optional[bool] = False, 
                                                            upsample: Optional[bool] = False) -> None:
        super(Bottleneck, self).__init__()

        internal_channels: int = in_channels // internal_ratio

        if downsample:
            stride: int = 2
            self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)
            self.reduce_channels = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels)
            )
            
        elif upsample:
            stride = 1
            self.upsample = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
                nn.BatchNorm2d(out_channels)
            )
        
        else:
            stride = 1

        
        if asymmetric:
            conv_middle = nn.Sequential(
                nn.Conv2d(internal_channels, internal_channels, kernel_size=(kernel_size, 1), padding=(padding, 0)),
                nn.ReLU(inplace=True),
                nn.Conv2d(internal_channels, internal_channels, kernel_size=(1, kernel_size), padding=(0, padding))
            )
        else:
            conv_middle = nn.Conv2d(
                internal_channels, 
                internal_channels, 
                kernel_size= kernel_size, 
                stride= stride,
                padding= padding, 
                dilation= dilation
            )


        self.main_branch = nn.Sequential(
            nn.Conv2d(in_channels, internal_channels, kernel_size= 1),
            nn.BatchNorm2d(internal_channels),
            nn.ReLU(inplace=True) if relu else nn.PReLU(),
            conv_middle,
            nn.BatchNorm2d(internal_channels),
            nn.ReLU(inplace=True) if relu else nn.PReLU(),
            nn.Conv2d(internal_channels, out_channels, kernel_size= 1),
            nn.BatchNorm2d(out_channels),
            nn.Dropout2d(p= dropout_prob)
        )
        
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self, 'downsample'):
            identity: torch.Tensor = self.reduce_channels(self.downsample(x))
        elif hasattr(self, 'upsample'):
            identity: torch.Tensor = self.upsample(x)
        else:
            identity: torch.Tensor = x

        out: torch.Tensor = self.main_branch(x)
        out += identity
        return out







class ENet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(ENet, self).__init__()
        
        self.layers = nn.Sequential(
            InitialBlock(in_channels, 16),

            # Stage 1
            Bottleneck(16, 64, downsample=True),
            Bottleneck(64, 64),
            Bottleneck(64, 64),
            Bottleneck(64, 64),
            Bottleneck(64, 64),

            # Stage 2
            Bottleneck(64, 128, downsample=True),
            Bottleneck(128, 128),
            Bottleneck(128, 128, dilation=2, padding=2),
            Bottleneck(128, 128, asymmetric=True),
            Bottleneck(128, 128, dilation=4, padding=4),
            Bottleneck(128, 128),
            Bottleneck(128, 128, dilation=8, padding=8),
            Bottleneck(128, 128, asymmetric=True),
            Bottleneck(128, 128, dilation=16, padding=16),

            # Stage 3
            Bottleneck(128, 128),
            Bottleneck(128, 128, dilation=2, padding=2),
            Bottleneck(128, 128, asymmetric=True),
            Bottleneck(128, 128, dilation=4, padding=4),
            Bottleneck(128, 128),
            Bottleneck(128, 128, dilation=8, padding=8),
            Bottleneck(128, 128, asymmetric=True),
            Bottleneck(128, 128, dilation=16, padding=16),

            # Stage 4
            Bottleneck(128, 64, upsample=True),
            Bottleneck(64, 64),
            Bottleneck(64, 64),

            # Stage 5
            Bottleneck(64, 16, upsample=True),
            Bottleneck(16, 16),

            # Output
            nn.ConvTranspose2d(16, out_channels, kernel_size=2, stride=2)
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)





# ------------------------------------------------ DeepLabv3+ ---------------------------------------------------------
__deep_lab: str = r'''
    *   DeepLabv3+ is a state-of-the-art architecture for semantic image segmentation that was introduced by Chen et al. in their paper "Encoder-Decoder with Atrous Separable Convolution for 
        Semantic Image Segmentation". DeepLabv3+ is an extension of the DeepLabv3 architecture that is designed to improve the accuracy of semantic segmentation and reduce the computational cost.

    *   The DeepLabv3+ architecture consists of an encoder-decoder network that is connected by a series of skip connections. The encoder network is a series of convolutional and pooling layers 
        that reduce the spatial resolution of the input image, while increasing the number of feature channels. The decoder network consists of a series of upsampling and convolutional layers that 
        increase the spatial resolution of the feature maps while reducing the number of feature channels.

    *   The key innovation of DeepLabv3+ is the use of atrous (dilated) convolutions in the encoder network, which increases the receptive field of the model without increasing the number of parameters.
        Atrous convolutions allow the model to capture context and global information, which is important for accurate segmentation.

    *   DeepLabv3+ also includes several other optimizations to improve performance and reduce computational cost, such as separable convolutions and multi-scale feature fusion. Separable convolutions 
        split a convolutional layer into a depthwise convolution and a pointwise convolution, which reduces the number of parameters and speeds up computation. Multi-scale feature fusion combines 
        feature maps from different layers with different spatial resolutions, which helps to capture both fine-grained and coarse-grained information.

    *   DeepLabv3+ has several advantages over other segmentation models. First, it is highly accurate and has achieved state-of-the-art performance on several benchmark datasets. Second, the use of 
        atrous convolutions and other optimizations help to reduce the computational cost of the model, making it more efficient to train and deploy. Finally, DeepLabv3+ is flexible and can be 
        adapted to different applications by changing the number and size of the layers.

    *   Overall, DeepLabv3+ is a powerful architecture for semantic image segmentation that has achieved state-of-the-art performance on several benchmark datasets. Its use of atrous convolutions, 
        separable convolutions, and multi-scale feature fusion make it effective at capturing both low-level and high-level features, which is critical for accurate segmentation.

'''
class ASPPConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dilation: int) -> None:
        super(ASPPConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)




class ASPPPooling(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(ASPPPooling, self).__init__()
        self.pooling = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )



    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size: torch.Tensor = x.shape[-2:]
        x: torch.Tensor = self.pooling(x)
        return nn.functional.interpolate(x, size=size, mode='bilinear', align_corners=False)





class ASPP(nn.Module):
    def __init__(self, in_channels: int, out_channels: int = 256) -> None:
        super(ASPP, self).__init__()
        dilations: list[int] = [1, 6, 12, 18]
        self.aspp = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ),
            ASPPConv(in_channels, out_channels, dilations[0]),
            ASPPConv(in_channels, out_channels, dilations[1]),
            ASPPConv(in_channels, out_channels, dilations[2]),
            ASPPPooling(in_channels, out_channels)
        ])




    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([module(x) for module in self.aspp], dim=1)





class DeepLabV3Plus(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, backbone: str = 'resnet101') -> None:
        super(DeepLabV3Plus, self).__init__()

        if backbone == 'resnet101':
            self.backbone = models.resnet101(pretrained=True)
        elif backbone == 'resnet50':
            self.backbone = models.resnet50(pretrained=True)
        else:
            raise ValueError("Invalid backbone. Choose between 'resnet101' and 'resnet50'.")

        self.backbone_layers = list(self.backbone.children())
        self.feature_extractor = nn.Sequential(*self.backbone_layers[:-3])
        self.high_level_feature_extractor = nn.Sequential(*self.backbone_layers[-3])

        self.aspp = ASPP(2048, 256)
        self.decoder = nn.Sequential(
            nn.Conv2d(256 * 5, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Conv2d(256, out_channels, 1),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        )




    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        
        # Low-level features
        x_low = self.feature_extractor(x)

        # High-level features
        x_high = self.high_level_feature_extractor(x_low)

        # ASPP and Decoder
        x_aspp = self.aspp(x_high)
        x_decoded = self.decoder(x_aspp)

        # Upsample to the input size
        x_out = nn.functional.interpolate(x_decoded, size=size, mode='bilinear', align_corners=False)

        return x_out











#@: Driver Code
if __name__.__contains__('__main__'):
    ...