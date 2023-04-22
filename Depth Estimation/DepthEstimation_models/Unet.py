
from __future__ import annotations
import torch
import torch.nn as nn


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


# U-Net is a powerful deep learning architecture initially developed for biomedical image segmentation tasks. However, due to its 
# highly effective encoding and decoding structure, U-Net has been adapted for a wide range of computer vision tasks, including 
# depth estimation.

# In depth estimation tasks, the goal is to predict a depth map from a single input image. U-Net can be utilized for this purpose 
# by modifying its architecture to accommodate the depth estimation problem. The original U-Net structure consists of an encoder 
# that captures the context and a decoder that enables precise localization. For depth estimation, the architecture can be adapted 
# as follows:

#     *   Input: The input to the U-Net would be a single RGB image.
#     *   Output: The output would be a depth map corresponding to the input image.

# The U-Net architecture is particularly well-suited for depth estimation because it combines high-level contextual information 
# from the encoder with low-level details from the decoder. This combination allows U-Net to capture both global structures and 
# local details, resulting in accurate depth maps.

# In summary, U-Net is a versatile deep learning architecture that can be adapted for various computer vision tasks, including depth 
# estimation. By adjusting the input and output layers and retaining the powerful encoding and decoding structure, U-Net can 
# efficiently predict depth maps from single input images.


class UNet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, depth: int = 4, initial_filters: int = 32) -> None:
        
        # The UNet class constructor takes four parameters:

        #     *   in_channels (int): The number of input channels of the input image.
            
        #     *   out_channels (int): The number of output channels of the output image.
            
        #     *   depth (int, optional): The depth of the U-Net architecture, which determines the number of downsampling and upsampling 
        #                                layers in the encoder and decoder parts. By default, it is set to 4.

        #     *   initial_filters (int, optional): The number of filters in the first convolutional layer. It is set to 32 by default.


        # Inside the constructor, the super().__init__() method is called to initialize the parent class nn.Module.

        # Then, the following components of the UNet architecture are created:

        #     *   self.encoder: The encoder part of the U-Net architecture is created by calling the __create_encoder() method with 
        #                       in_channels, initial_filters, and depth as arguments.

        #     *   self.middle: The middle part of the U-Net architecture, which connects the encoder and decoder, is created by calling 
        #                      the __create_middle() method with initial_filters and depth as arguments.

        #     *   self.decoder: The decoder part of the U-Net architecture is created by calling the __create_decoder() method with 
        #                       out_channels, initial_filters, and depth as arguments.

        # These components, when combined, form the complete UNet architecture. The encoder progressively downsamples the input image, 
        # extracting high-level features. The middle part processes these features, and the decoder upsamples the features to generate 
        # the final depth map.
        
        super(UNet, self).__init__()
        self.encoder = self.__create_encoder(in_channels, initial_filters, depth)
        self.middle = self.__create_middle(initial_filters, depth)
        self.decoder = self.__create_decoder(out_channels, initial_filters, depth)





    @staticmethod
    def conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
        
        # The purpose of this method is to create a simple convolutional block, which consists of two convolutional layers followed by ReLU 
        # (Rectified Linear Unit) activation functions.

        # The conv_block method takes two parameters:

        #     *   in_channels (int): The number of input channels for the first convolutional layer.

        #     *   out_channels (int): The number of output channels for both the first and second convolutional layers.

        # The method returns an nn.Sequential object, which is a container in PyTorch for stacking neural network layers sequentially. 
        # The nn.Sequential object consists of the following layers:

        #     *   A 2D convolutional layer (nn.Conv2d) with in_channels input channels, out_channels output channels, a kernel size of 3, and padding of 1.

        #     *   A ReLU activation function (nn.ReLU) with the inplace=True argument, which means the input tensor is modified directly, saving memory.

        #     *   Another 2D convolutional layer (nn.Conv2d) with out_channels input channels, out_channels output channels, a kernel size of 3, and padding of 1.

        #     *   Another ReLU activation function (nn.ReLU) with the inplace=True argument.

        # This conv_block method is used in the UNet architecture to create simple convolutional blocks for the encoder and decoder parts.
        
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )





    def __create_encoder(self, in_channels: int, initial_filters: int, depth: int) -> nn.Sequential:
        
        # This code snippet defines a private method called __create_encoder within the UNet class. The purpose of this method is to create the encoder part of the UNet architecture, 
        # which consists of a series of downsampling layers, each containing a max-pooling layer followed by a convolutional block.

        # The __create_encoder() method takes three parameters:

        #     *   in_channels (int): The number of input channels for the first convolutional layer.

        #     *   initial_filters (int): The number of output channels for the first convolutional layer.

        #     *   depth (int): The number of downsampling layers in the encoder.

        # The method returns an nn.Sequential object, which is a container in PyTorch for stacking neural network layers sequentially.

        
        # A dictionary called layers is created to store the layers of the encoder. The first layer, "layer_0", is a convolutional block with in_channels input channels and 
        # initial_filters output channels, created using the UNet.conv_block static method.

        # A loop is used to create the rest of the layers (from "layer_1" to "layer_{depth-1}"). In each iteration, a downsampling layer is created as an nn.Sequential 
        # object containing:

        #     *   A 2D max-pooling layer (nn.MaxPool2d) with a kernel size of 2 and a stride of 2. This layer is used to reduce the spatial dimensions of the input.

        #     *   A convolutional block created using the UNet.conv_block static method. The number of input channels is initial_filters * (2 ** (idx - 1)), 
        #         and the number of output channels is initial_filters * (2 ** idx).

        # Finally, the layers dictionary is converted into an nn.Sequential object and returned. This object represents the encoder part of the UNet architecture.
        
        layers: dict[str, Any] = {}
        layers["layer_0"] = UNet.conv_block(in_channels, initial_filters)

        for idx in range(1, depth):
            layers[f"layer_{idx}"] = nn.Sequential(
                nn.MaxPool2d(kernel_size=2, stride=2),
                UNet.conv_block(initial_filters * (2 ** (idx - 1)), initial_filters * (2 ** idx)),
            )

        return nn.Sequential(layers)






    def __create_middle(self, initial_filters: int, depth: int) -> nn.Sequential:
        
        # This code snippet defines a private method called __create_middle within the UNet class. The purpose of this method is to create the middle part of the UNet 
        # architecture, which consists of a single convolutional block.

        # The __create_middle() method takes two parameters:

        #     *   initial_filters (int): The number of output channels for the first convolutional layer of the UNet.
        #     *   depth (int): The number of downsampling layers in the encoder.

        # The method returns an nn.Sequential object, which is a container in PyTorch for stacking neural network layers sequentially.

        # A dictionary called layers is created to store the layers of the middle part. The "middle" layer is a convolutional block with 
        # initial_filters * (2 ** (depth - 1)) input channels and initial_filters * (2 ** depth) output channels, created using the UNet.conv_block static method.

        # Finally, the layers dictionary is converted into an nn.Sequential object and returned. This object represents the middle part of the UNet architecture.
        
        layers: dict[str, Any] = {
            f"middle": UNet.conv_block(initial_filters * (2 ** (depth - 1)), initial_filters * (2 ** depth))
        }
        return nn.Sequential(layers)





    def __create_decoder(self, out_channels: int, initial_filters: int, depth: int) -> nn.Sequential:
        
        # This code snippet defines a private method called __create_decoder() within the UNet class. The purpose of this method is to create the decoder part of the 
        # UNet architecture, which is responsible for the upsampling and information fusion.

        # The __create_decoder() method takes three parameters:

        #     *   out_channels (int): The number of output channels for the final output layer of the UNet.

        #     *   initial_filters (int): The number of output channels for the first convolutional layer of the UNet.

        #     *   depth (int): The number of downsampling layers in the encoder.

        # The method returns an nn.Sequential object, which is a container in PyTorch for stacking neural network layers sequentially.

        # A dictionary called layers is created to store the layers of the decoder part. The for loop iterates in reverse order from depth - 1 to 1, 
        # creating the upsampling layers. At each iteration, a new nn.Sequential layer is added to the dictionary with a ConvTranspose2d layer for 
        # upsampling and a convolutional block created using the UNet.conv_block static method.

        # After the loop, the final "layer_0" is added to the dictionary. It includes a ConvTranspose2d layer for upsampling, a convolutional block created using 
        # the UNet.conv_block() static method, and a final Conv2d layer with the specified number of output channels and a kernel size of 1.

        # Finally, the layers dictionary is converted into an nn.Sequential object and returned. This object represents the decoder part of the UNet architecture.
        
        layers: dict[str, Any] = {}
        for idx in range(depth - 1, 0, -1):
            layers[f"layer_{idx}"] = nn.Sequential(
                nn.ConvTranspose2d(initial_filters * (2 ** idx), initial_filters * (2 ** (idx - 1)), kernel_size=2, stride=2),
                UNet.conv_block(initial_filters * (2 ** idx), initial_filters * (2 ** (idx - 1))),
            )

        layers["layer_0"] = nn.Sequential(
            nn.ConvTranspose2d(initial_filters * 2, initial_filters, kernel_size=2, stride=2),
            UNet.conv_block(initial_filters * 2, initial_filters),
            nn.Conv2d(initial_filters, out_channels, kernel_size=1),
        )

        return nn.Sequential(layers)





    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encodings: list[Any] = []

        for layer in self.encoder:
            x = layer(x)
            encodings.append(x)

        x = self.middle(encodings[-1])

        for idx, layer in enumerate(self.decoder):
            x = layer(torch.cat([x, encodings[-(idx + 2)]], dim=1))

        return x






class BasicUNet(UNet):
    def __init__(self):
        super(BasicUNet, self).__init__(in_channels=3, out_channels=1, initial_filters=32, depth=4)



class UNetWithMoreFilters(UNet):
    def __init__(self):
        super(UNetWithMoreFilters, self).__init__(in_channels=3, out_channels=1, initial_filters=64, depth=4)



class UNetWithGreaterDepth(UNet):
    def __init__(self):
        super(UNetWithGreaterDepth, self).__init__(in_channels=3, out_channels=1, initial_filters=32, depth=5)



class UNetWithMoreFiltersAndGreaterDepth(UNet):
    def __init__(self):
        super(UNetWithMoreFiltersAndGreaterDepth, self).__init__(in_channels=3, out_channels=1, initial_filters=64, depth=5)






#@: Driver Code
if __name__.__contains__('__main__'):
    ...
