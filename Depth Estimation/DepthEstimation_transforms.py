from __future__ import annotations
import torch, os, random
import numpy as np
import pandas as pd
from PIL import Image
from torchvision.transforms import ToTensor, Normalize, Resize


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

__getitem__return__: str = r'''
    __getitem__(index: int) -> dict[str, torch.tensor]:
        {
            'independent_variable': image, 
            'dependent_variable': depth_map
        }
'''

__transform_dict_doc__: str = r'''
    transform_dict: dict[str, dict[str, Callable[..., Any]]] = {
        'independent_variable': {
            'transform_1': Image_transform_1,
            'transform_2': Image_transform_2,
            ...
        }
        'dependent_variable': {
            'transform_1': DepthMap_transform_1,
            'transform_2': DepthMap_transform_2,
            ...
        }
    }
'''

# Here's an example of how to use the DepthEstimationDataset class along with the custom transformation classes defined above.

# First, we define a transform_dict that contains the custom transformations for both independent and dependent variables. The independent_variable 
# dictionary contains the following transformations:

#     *   custom_resize: Resizes an image to (224, 224) using the CustomResize class.
#     *   random_vertical_flip: Applies a random vertical flip to the image with a probability of 0.5 using the CustomRandomVerticalFlip class.
#     *   to_tensor: Converts the image to a PyTorch tensor using the built-in ToTensor class from torchvision.
#     *   normalize: Normalizes the image using the mean and standard deviation of the ImageNet dataset using the built-in Normalize class from torchvision.

# The dependent_variable dictionary contains the following transformations:

#     *   depth_resize: Resizes the depth map to (224, 224) using the CustomResize class.
#     *   depth_to_meters: Converts the depth values to meters using the CustomDepthToMeters class.
#     *   depth_scaling: Scales the depth values between 0.0 and 1.0 using the CustomDepthScaling class.

# Now, you can instantiate the DepthEstimationDataset class using this transform_dict and pass the dataset to a DataLoader or use it in your training/evaluation pipeline.


class CustomResize:    
    # The CustomResize class is a custom transformation class that resizes an image using the provided dimensions. The constructor takes a tuple 
    # of integers representing the desired width and height. The class has a __call__ method that takes a PIL.Image object as input and returns 
    # the resized image using the BICUBIC interpolation.
    
    def __init__(self, size: tuple[int, int]) -> None:
        self.size = size
        
    
    def __call__(self, image: torch.tensor) -> PIL.Image:
        return image.resize(self.size, Image.BICUBIC)




class CustomRandomVerticalFlip:
    # The CustomRandomVerticalFlip class is a custom transformation class that applies a random vertical flip to an image. The constructor takes two optional 
    # parameters: p, a float representing the probability of applying the vertical flip (default is 0.5), and seed, an integer for setting the random seed to 
    # ensure reproducibility (default is None). The class has a __call__ method that takes a PIL.Image object as input. If the random number generated is less
    # than p, it applies the vertical flip to the image using image.transpose(Image.FLIP_TOP_BOTTOM) and returns the flipped image. Otherwise, it returns the original 
    # image.
    def __init__(self, p: Optional[float] = 0.5, seed: Optional[int] = None) -> None:
        self.p = p
        self.seed = seed
        
        
    def __call__(self, image: PIL.Image) -> PIL.Image:
        if self.seed is not None:
            random.seed(self.seed)
        
        if random.random() < self.p:
            return image.transpose(Image.FLIP_TOP_BOTTOM)
        
        return image
    



class CustomDepthToMeters: 
    # The CustomDepthToMeters class is a custom transformation class that converts depth values from an input depth map to meters. The constructor takes an optional 
    # parameter depth_scale, a float representing the scale factor to apply to the depth values (default is 0.001). The class has a __call__ method that takes a NumPy
    # array representing the depth map as input. The method multiplies the depth values by the depth_scale factor and returns the rescaled depth map as a NumPy array. 
    # This is useful for converting depth values from one unit (e.g., millimeters) to another (e.g., meters) in the context of depth estimation tasks.
    
    def __init__(self, depth_scale: Optional[float] = 0.001) -> None:
        self.depth_scale = depth_scale
    
    
    def __call__(self, depth: np.ndarray) -> np.ndarray:
        return depth * self.depth_scale
    




class CustomDepthScaling:
    # The CustomDepthScaling class is a custom transformation class that rescales depth values in an input depth map to a specified range. The constructor takes two 
    # optional parameters, min_value and max_value, both floats representing the minimum and maximum values of the new range (default is 0.0 and 1.0, respectively). 
    # The class has a __call__ method that takes a NumPy array representing the depth map as input.

    # Inside the __call__ method, the minimum and maximum depth values in the input depth map are first calculated. Then, the depth values are rescaled using a linear 
    # transformation such that the minimum and maximum values in the rescaled depth map correspond to min_value and max_value, respectively. The rescaled depth map 
    # is returned as a NumPy array. This transformation is useful for normalizing depth values for depth estimation tasks or for preparing depth data for visualization.
    
    def __init__(self, min_value: Optional[float] = 0.0, max_value: Optional[float] = 1.0) -> None:
        self.min_value = min_value
        self.max_value = max_value
    
    
    def __call__(self, depth: np.ndarray) -> np.ndarray:
        depth_min: float = depth.min()
        depth_max: float = depth.max()
        return (depth - depth_min) * (self.max_value - self.min_value) / (depth_max - depth_min) + self.min_value
    




#@: Driver Code
if __name__.__contains__('__main__'):
    #@: NOTE: Use Case Example: 
    
    # transform_dict: dict[str, dict[str, Callable[..., Any]]] = {
    #     'independent_variable': {
    #         'custom_resize': CustomResize(size= (224, 224)),
    #         'random_vertical_flip': CustomRandomVerticalFlip(p= 0.5),
    #         'to_tensor': ToTensor(),
    #         'normalize': Normalize(mean= [0.485, 0.456, 0.406], std= [0.229, 0.224, 0.225])
    #     },
    #     'dependent_variable': {
    #         'depth_resize': CustomResize(size= (224, 224)),
    #         'depth_to_meters': CustomDepthToMeters(depth_scale= 0.001),
    #         'depth_scaling': CustomDepthScaling(min_value= 0.0, max_value= 1.0)
    #     }
    # }
    
    
    
    
    