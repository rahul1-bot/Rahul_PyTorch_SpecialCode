from __future__ import annotations
import os, torch, torchvision
from PIL import Image
import torchvision.transforms as transforms
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

>>> Data Transformation Pipeline
 
        *   A data transformation pipeline in PyTorch is an essential part of the preprocessing workflow 
            for many machine learning and deep learning tasks. It enables you to perform a sequence of 
            transformations on your input data before feeding it into your model. The primary goal of such 
            a pipeline is to preprocess the data, augment it, and convert it into a suitable format that 
            can be used by the model effectively.


        *   In PyTorch, the data transformation pipeline is often built using the torchvision.transforms module. 
            This module provides a collection of common image transformations, such as resizing, cropping, flipping,
            normalization, and data augmentation techniques. 

'''


__transform_one__: str = r'''
        *   The RandomResizedCropTransform class you provided is a custom transformation class that wraps the transforms.RandomResizedCrop 
            function from the torchvision.transforms module. The class has an __init__ method that takes an optional image_size argument 
            with a default value of 224. The __call__ method accepts a PIL.Image and applies the random resized crop transformation to it,
            returning a transformed PIL.Image. The __repr__ method provides a string representation of the class, which is helpful for debugging 
            purposes.
'''
class RandomResizedCropTransform:
    def __init__(self, image_size: int = 224) -> None:
        self.image_size = image_size


    def __call__(self, image: PIL.Image) -> PIL.Image:
        return transforms.RandomResizedCrop(self.image_size)(image)


    def __repr__(self) -> str:
        return f"{type(self).__name__}(image_size={self.image_size})"
        



__transform_two__: str = r'''
        *   The RandomHorizontalFlipTransform class you provided is a custom transformation class that wraps the transforms.RandomHorizontalFlip function 
            from the torchvision.transforms module. The class has a __call__ method that accepts a PIL.Image and applies the random horizontal flip transformation 
            to it, returning a transformed PIL.Image. The __repr__ method provides a string representation of the class, which is helpful for debugging purposes.
'''
class RandomHorizontalFlipTransform:
    def __call__(self, image: PIL.Image) -> PIL.Image:
        return transforms.RandomHorizontalFlip()(image)


    def __repr__(self) -> str:
        return f"{type(self).__name__}()"
        



__transform_three__: str = r'''
        *   The RandomRotationTransform class you provided is a custom transformation class that wraps the transforms.RandomRotation function from the 
            torchvision.transforms module. The class has an __init__ method that takes an optional degrees argument with a default value of 10. 
            The __call__ method accepts a PIL.Image and applies the random rotation transformation to it, returning a transformed PIL.Image. 
            The __repr__ method provides a string representation of the class, which is helpful for debugging purposes.
'''
class RandomRotationTransform:
    def __init__(self, degrees: Union[float, tuple[float, float]] = 10) -> None:
        self.degrees = degrees


    def __call__(self, image: PIL.Image) -> PIL.Image:
        return transforms.RandomRotation(self.degrees)(image)


    def __repr__(self) -> str:
        return f"{type(self).__name__}(degrees={self.degrees})"
        
        
        
        
__transform_four__: str = r'''
        *   The RandomVerticalFlipTransform class you provided is a custom transformation class that wraps the transforms.RandomVerticalFlip 
            function from the torchvision.transforms module. The class has an __init__ method that takes an optional p argument with a default
            value of 0.5, which represents the probability of the image being flipped vertically. The __call__ method accepts a PIL.Image and
            applies the random vertical flip transformation to it, returning a transformed PIL.Image. The __repr__ method provides a string 
            representation of the class, which is helpful for debugging purposes.
'''        
class RandomVerticalFlipTransform:
    def __init__(self, p: float = 0.5) -> None:
        self.p = p


    def __call__(self, image: PIL.Image) -> PIL.Image:
        return transforms.RandomVerticalFlip(self.p)(image)


    def __repr__(self) -> str:
        return f"{type(self).__name__}(p={self.p})"
    


__transform_five__: str = r'''
        *   The ColorJitterTransform class you provided is a custom transformation class that wraps the transforms.ColorJitter function 
            from the torchvision.transforms module. The class has an __init__ method that takes optional arguments for brightness, contrast, 
            saturation, and hue, all with default values of 0.1. The __call__ method accepts a PIL.Image and applies the color jitter 
            transformation to it, returning a transformed PIL.Image. The __repr__ method provides a string representation of the class,
            which is helpful for debugging purposes.
'''    
class ColorJitterTransform:
    def __init__(self, brightness: float = 0.1, contrast: float = 0.1, saturation: float = 0.1, 
                                                                       hue: float = 0.1) -> None:
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue



    def __call__(self, image: PIL.Image) -> PIL.Image:
        return transforms.ColorJitter(
            brightness= self.brightness, contrast= self.contrast,
            saturation= self.saturation, hue= self.hue
        )(image)



    def __repr__(self) -> None:
        return f"{type(self).__name__}(brightness={self.brightness}, contrast={self.contrast}, saturation={self.saturation}, hue={self.hue})"
        
        


__transform_six__: str = r'''
        *   The GrayscaleTransform class you provided is a custom transformation class that wraps the transforms.Grayscale function
            from the torchvision.transforms module. The class has a __call__ method that accepts a PIL.Image, applies the grayscale 
            transformation to it, and returns the transformed PIL.Image in grayscale. The __repr__ method provides a string representation 
            of the class, which is helpful for debugging purposes.
'''    
class GrayscaleTransform:
    def __call__(self, image: PIL.Image) -> PIL.Image:
        return transforms.Grayscale()(image)


    def __repr__(self) -> str:
        return f"{type(self).__name__}()"




__transform_seven__: str = r'''
        *   The ToTensorTransform class you provided is a custom transformation class that wraps the transforms.ToTensor function from the 
            torchvision.transforms module. The class has a __call__ method that accepts a PIL.Image, converts it into a torch.tensor, 
            and returns the resulting tensor. The __repr__ method provides a string representation of the class, which is helpful for debugging purposes.
'''
class ToTensorTransform:
    def __call__(self, image: PIL.Image) -> torch.tensor:
        return transforms.ToTensor()(image)


    def __repr__(self) -> str:
        return f"{type(self).__name__}()"
    



__transform_eight__: str = r'''
        *   The NormalizeTransform class you provided is a custom transformation class that wraps the transforms.Normalize function 
            from the torchvision.transforms module. The class has a constructor that accepts mean and standard deviation values for each 
            channel, with default values corresponding to the pre-trained models on ImageNet. The __call__ method accepts a torch.tensor (an image), 
            normalizes it using the provided mean and standard deviation values, and returns the resulting normalized tensor. The __repr__ method 
            provides a string representation of the class, which is helpful for debugging purposes.
'''        
class NormalizeTransform:
    def __init__(self, mean: tuple[float, float, float] = (0.485, 0.456, 0.406), 
                       std: tuple[float, float, float] = (0.229, 0.224, 0.225)) -> None:
        
        self.mean = mean
        self.std = std


    def __call__(self, image: torch.tensor) -> torch.tensor:
        return transforms.Normalize(mean=self.mean, std=self.std)(image)
        
        
    def __repr__(self) -> str:
        return f"{type(self).__name__}(mean={self.mean}, std={self.std})"
        
        
        
        

    
#@: Driver Code
if __name__.__contains__('__main__'):
    transform_dict: dict[str, Callable[Any]] = {
        'RandomResizedCropTransform': RandomResizedCropTransform(),
        'RandomHorizontalFlipTransform': RandomHorizontalFlipTransform(),
        'RandomRotationTransform': RandomRotationTransform(),
        'RandomVerticalFlipTransform': RandomVerticalFlipTransform(),
        'ColorJitterTransform': ColorJitterTransform(),
        'GrayscaleTransform': GrayscaleTransform(),
        'ToTensorTransform': ToTensorTransform(),
        'NormalizeTransform': NormalizeTransform()
    }
    