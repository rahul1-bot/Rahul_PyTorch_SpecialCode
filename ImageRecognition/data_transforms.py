
from __future__ import annotations
import os, torch, torchvision, random
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
    The RandomCropTransform class is a custom image transformation class that wraps around PyTorch's built-in transforms.RandomCrop 
    function. Here's a breakdown of the class:

        *   __init__ method: This method initializes the class with a single optional argument, size, which defaults to 224. 
            The size parameter specifies the size of the random crop.

        *   __call__ method: This method takes an input image (of type PIL.Image) and returns the transformed image. It applies 
            the transforms.RandomCrop function from PyTorch with the provided size parameter.

        *   __repr__ method: This method returns a string representation of the RandomCropTransform class instance, which includes 
            the class name and the size parameter. This makes it easier to understand the object when printed or logged.

'''
class RamdomCropTransform:
    def __init__(self, size: Optional[int] = 224) -> None:
        self.size = size
        
        
    def __call__(self, image: PIL.Image) -> PIL.Image:
        return transforms.RandomCrop(self.size)(image)
    
    
    def __repr__(self) -> str:
        return f"RandomCropTransform(size= {self.size})"
    


__transform_two__: str = r'''
    The RandomFlipTransform class is a custom image transformation class that applies random horizontal and/or vertical flips to an input 
    image using PyTorch's built-in transformation functions. Here's an explanation of the class components:

        *   __init__ method: This method initializes the class with two optional arguments, p_horizontal and p_vertical, both defaulting to 0.5. 
            These parameters represent the probability of applying the horizontal and vertical flip, respectively.

        *   __call__ method: This method takes an input image (of type PIL.Image) and returns the transformed image. It applies the horizontal 
            flip with probability p_horizontal using transforms.RandomHorizontalFlip and the vertical flip with probability p_vertical using 
            transforms.RandomVerticalFlip.

        *   __repr__ method: This method returns a string representation of the RandomFlipTransform class instance, which includes the class 
            name and the p_horizontal and p_vertical parameters. This makes it easier to understand the object when printed or logged.

''' 
class RandomFlipTransform:
    def __init__(self, p_horizontal: Optional[float] = 0.5, p_vertical: Optional[float] = 0.5) -> None:
        self.p_horizontal = p_horizontal
        self.p_vertical = p_vertical
        
        
        
    def __call__(self, image: PIL.Image) -> PIL.Image:
        if random.random() < self.p_horizontal:
            image: PIL.Image = transforms.RandomHorizontalFlip(1.0)(image)
        
        if random.random() < self.p_vertical:
            image: PIL.Image = transforms.RandomVerticalFlip(1.0)(image)
        
        return image
    
    
    
    def __repr__(self) -> str:
        return f"RandomFlipTransform(p_horizontal= {self.p_horizontal}, p_vertical= {self.p_vertical})"
        
            
            
__transform_three__: str = r'''
    The RandomRotationTransform class is a custom image transformation class that applies a random rotation to an input image using PyTorch's 
    built-in transformation functions. Here's an explanation of the class components:

        *    __init__ method: This method initializes the class with an optional argument degrees, defaulting to 10. The degrees parameter can 
             be a single float or a tuple of two floats, representing the range of rotation degrees.

        *   __call__ method: This method takes an input image (of type PIL.Image) and returns the transformed image. It applies a random rotation 
            within the specified degrees range using transforms.RandomRotation.

        *   __repr__ method: This method returns a string representation of the RandomRotationTransform class instance, which includes the class 
            name and the degrees parameter. This makes it easier to understand the object when printed or logged.
'''
class RandomRotationTransform:
    def __init__(self, degrees: Optional[Union[float, tuple[float, float]]] = 10) -> None:
        self.degrees = degrees
        
        
    def __call__(self, image: PIL.Image) -> PIL.Image:
        return transforms.RandomRotation(self.degrees)(image)
    
    
    def __repr__(self) -> str:
        return f"RandomRotationTransform(degrees= {self.degrees})"
    
    


__transform_four__: str = r'''
    The ColorJitterTransform class is a custom image transformation class that applies random color perturbations to an input image using PyTorch's 
    built-in transformation functions. Here's an explanation of the class components:

        *   __init__ method: This method initializes the class with optional arguments brightness, contrast, saturation, and hue, all defaulting to 0. 
            These parameters can be single integers or floats, representing the ranges for the respective color adjustments.

        *   __call__ method: This method takes an input image (of type PIL.Image) and returns the transformed image. It applies random color perturbations 
            to the input image using transforms.ColorJitter, which takes the brightness, contrast, saturation, and hue parameters as inputs.

        *   __repr__ method: This method returns a string representation of the ColorJitterTransform class instance, which includes the class name and the 
            color adjustment parameters. This makes it easier to understand the object when printed or logged.

'''
class ColorJitterTransform:
    def __init__(self, brightness: Optional[Union[int, float]] = 0, contrast: Optional[Union[int, float]] = 0, 
                                                                    saturation: Optional[Union[int, float]] = 0, 
                                                                    hue: Optional[Union[int, float]] = 0) -> None:
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        
        
        
    def __call__(self, image: PIL.Image) -> PIL.Image:
        return transforms.ColorJitter(
            self.brightness,
            self.contrast,
            self.saturation,
            self.hue
        )(image)
        
        
        
    def __repr__(self) -> str:
        return f'''ColorJitterTransform(
            brightness= {self.brightness}, contrast= {self.contrast}, 
            saturation= {self.saturation}, hue= {self.hue}
        )
        '''
        
    
__transform_five__: str = r'''
    The GaussianBlurTransform class is a custom image transformation class that applies Gaussian blur to an input image using PyTorch's built-in 
    transformation functions. Here's an explanation of the class components:

        *   __init__ method: This method initializes the class with the required kernel_size argument and an optional sigma argument, which is a tuple with default
            values (0.1, 2.0). The kernel_size parameter is an integer defining the size of the Gaussian kernel, while sigma is a tuple containing the range of sigma 
            values for the Gaussian blur.

        *   __call__ method: This method takes an input image (of type PIL.Image) and returns the transformed image. It applies the Gaussian blur to the input image
            using transforms.GaussianBlur, which takes the kernel_size and sigma parameters as inputs.

        *   __repr__ method: This method returns a string representation of the GaussianBlurTransform class instance, which includes the class name and the Gaussian 
            blur parameters (kernel size and sigma). This makes it easier to understand the object when printed or logged.

'''
class GaussianBlurTransform:
    def __init__(self, kernel_size: int, sigma: Optional[tuple[float, float]] = (0.1, 2.0)) -> None:
        self.kernel_size = kernel_size
        self.sigma = sigma
        
        
    
    def __call__(self, image: PIL.Image) -> PIL.Image:
        return transforms.GaussianBlur(
            self.kernel_size, self.sigma
        )(image)
        
        
    
    def __repr__(self) -> str:
        return f"GaussianBlurTransform(kernel_size= {self.kernel_size}, sigma= {self.sigma})"
        



__transform_six__: str = r'''
    The RandomAffineTransform class is a custom image transformation class that applies random affine transformations to an input image using PyTorch's built-in 
    transformation functions. Here's an explanation of the class components:

        *   __init__ method: This method initializes the class with the required degrees argument and optional translate, scale, and shear arguments. 
            The degrees parameter is a float defining the range of rotation angles for the affine transformation. The translate, scale, and shear parameters are 
            optional and can be set to customize the affine transformation.

        *   __call__ method: This method takes an input image (of type PIL.Image) and returns the transformed image. It applies the random affine transformation 
            to the input image using transforms.RandomAffine, which takes the degrees, translate, scale, and shear parameters as inputs.

        *   __repr__ method: This method returns a string representation of the RandomAffineTransform class instance, which includes the class name and the affine 
            transformation parameters (degrees, translate, scale, and shear). This makes it easier to understand the object when printed or logged. 

'''
class RandomAffineTransform:
    def __init__(self, degrees: float, translate: Optional[Any] = None, scale: Optional[Any], 
                                                                        shear: Optional[Any]) -> None:
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
        
        
    def __call__(self, image: PIL.Image) -> PIL.Image:
        return transforms.RandomAffine(
            self.degrees,
            self.translate,
            self.scale, 
            self.shear
        )(image)
        
    
    def __repr__(self) -> str:
        return f"RandomAffineTransform(degrees= {self.degrees}, translate= {self.translate}, scale= {self.scale}, shear= {self.shear})"
        
        
    
    
__transform_seven__: str = r'''
    The NoiseInjectionTransform class is a custom image transformation class that adds Gaussian noise to an input image. Here's an explanation of the class components:

        *   __init__ method: This method initializes the class with the optional noise_factor argument, which determines the standard deviation of the Gaussian 
            noise that will be added to the input image. The default value is 0.05.

        *   __call__ method: This method takes an input image (of type PIL.Image) and returns the transformed image with Gaussian noise added to it. It first 
            generates the noise using the np.random.normal function, with a mean of 0 and standard deviation of self.noise_factor. Then, it adds the noise to the 
            image and clips the resulting values between 0 and 1 to keep the image within a valid range.

        *   __repr__ method: This method returns a string representation of the NoiseInjectionTransform class instance, which includes the class name and the noise 
            factor. This makes it easier to understand the object when printed or logged.
'''
class NoiseInjectionTransform:
    def __init__(self, noise_factor: Optional[float] = 0.05) -> None:
        self.noise_factor = noise_factor
        
    
    def __call__(self, image: PIL.Image) -> PIL.Image:
        noise = np.random.normal(0, self.noise_factor, image.size)
        noisy_image = image + noise
        return np.clip(noisy_image, 0, 1)

    
    def __repr__(self) -> str:
        return f"NoiseInjectionTransform(noise_factor= {self.noise_factor})"
    
    
    
    
__transform_eight__: str = r'''
    The CutoutTransform class is a custom image transformation class that applies the Cutout data augmentation technique to an input image. It randomly 
    removes square patches from the image. Here's an explanation of the class components:

        *   __init__ method: This method initializes the class with two optional arguments: n_holes, the number of square patches to remove (default is 1), 
            and length, the side length of the square patches (default is 16).

        *   __call__ method: This method takes an input image (of type PIL.Image) and returns the transformed image with the square patches removed. 
            It first creates a mask filled with ones, which has the same size as the input image. Then, for each hole, it selects random coordinates (y, x) 
            within the image, computes the square's boundaries (y1, y2, x1, x2) while ensuring they are within the image, and sets the corresponding mask 
            values to zero. Afterward, it converts the mask from a NumPy array to a PyTorch tensor, expands the mask to match the image channels, and 
            multiplies the image by the mask, effectively applying the Cutout transformation.

        *   __repr__ method: This method returns a string representation of the CutoutTransform class instance, which includes the class name and the n_holes 
            and length parameters. This makes it easier to understand the object when printed or logged.

'''
class CutoutTransform:
    def __init__(self, n_holes: Optional[int] = 1, length: Optional[int] = 16) -> None:
        self.n_holes = n_holes
        self.length = length
        
    
    def __call__(self, image: PIL.Image) -> PIL.Image:
        h, w = image.size
        mask: Any = np.ones((h, w), np.float32)
        
        for _ in range(self.n_holes):
            y: Any = np.random.randint(h)
            x: Any = np.random.randint(w)
            
            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)
            
            mask[y1: y2, x1: x2] = 0
            
        mask: Any = torch.from_numpy(mask)
        mask: Any = mask.expand_as(image)
        image: PIL.Image = image * mask 
        
        return image
    
    
    
    def __repr__(self) -> str:
        return f"CutoutTransform(n_holes= {self.n_holes}, length= {self.length})"
            



__transform_nine__: str = r'''
    The ToTensorTransform class is a custom image transformation class that converts a PIL Image to a PyTorch tensor. Here's an explanation of the class components:

        *   __call__ method: This method takes an input image (of type PIL.Image) and returns the image as a PyTorch tensor. It uses the transforms.ToTensor() 
            function from the PyTorch library to perform the conversion. The output tensor has dimensions (C, H, W) and pixel values are normalized to the range [0, 1].

        *   __repr__ method: This method returns a string representation of the ToTensorTransform class instance, which includes the class name. This makes it
            easier to understand the object when printed or logged.
'''
class ToTensorTransform:
    def __call__(self, image: PIL.Image) -> torch.Tensor:
        return transforms.ToTensor()(image)
    
    
    def __repr__(self) -> str:
        return f"ToTensorTransform()"
   
   
   
   
    
__transform_ten__: str = r'''
    The NormalizeTransform class is a custom image transformation class that normalizes a PyTorch tensor using the given mean and standard deviation values for 
    each channel. Here's an explanation of the class components:

        *   __init__ method: This method initializes the class instance with the provided mean and standard deviation values (each as a tuple of three floats, 
            one for each color channel).

        *   __call__ method: This method takes an input image (of type torch.Tensor) and returns the normalized image as a PyTorch tensor. It uses the transforms.Normalize() 
            function from the PyTorch library to perform the normalization, applying the specified mean and standard deviation values to each channel. The output tensor 
            has dimensions (C, H, W).

        *   __repr__ method: This method returns a string representation of the NormalizeTransform class instance, which includes the class name and the mean and standard 
            deviation values. This makes it easier to understand the object when printed or logged.

'''
class NormalizeTransform:
    def __init__(self, mean: Tuple[float, float, float], std: Tuple[float, float, float]) -> None:
        self.mean = mean
        self.std = std


    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        return transforms.Normalize(self.mean, self.std)(image)
    


    def __repr__(self) -> str:
        return f"NormalizeTransform(mean= {self.mean}, std= {self.std})"

    
        



#@: Driver Code
if __name__.__contains__('__main__'):
    
    #@: > use case example_-
    
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