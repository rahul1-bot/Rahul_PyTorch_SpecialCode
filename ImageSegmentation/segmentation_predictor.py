from __future__ import annotations
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
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
    *   The ImageSegmentationPredictor class that is provided is a complete implementation for an image segmentation predictor using PyTorch. It includes methods for 
        applying transformations to images and masks, predicting segmentation masks, applying a colormap to the masks, and visualizing the segmentation results.

        Here's a quick summary of the class and its methods:

        *   __init__: Initializes the predictor by setting the model, transform dictionary, number of classes, and colormap.
        
        *   __repr__: Provides a string representation of the predictor object.
        
        *   _create_colormap: Generates a colormap based on the number of classes.
        
        *   _apply_colormap: Applies the colormap to a given segmentation mask.
        
        *   _apply_transforms: Applies the specified transformations to the input image and mask.
        
        *   predict_mask: Predicts the segmentation mask for a given image.
        
        *   visualize_segmentation: Visualizes the original image, segmentation mask, and an overlay of the mask on the image using Matplotlib.


'''



#@: Image Segmentation Predictor Class 
class ImageSegmentationPredictor:
    def __init__(self, model: torch.nn.Module, transform_dict: Optional[dict[str, dict[str, Callable[Any]]]] = None, num_classes: Optional[int] = 2, 
                                                                                                                     colormap: Optional[np.ndarray] = None) ->  None:
        
        device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(device)
        self.transform_dict = transform_dict
        self.num_classes = num_classes
        
        if colormap is None:
            self.colormap = self._create_colormap()
        
        else: 
            self.colormap = colormap
            
            
        
    def __repr__(self) -> str:
        return f'''
            ImageSegmentationPredictor(
                model= {self.model}, 
                transform= {self.transform}, 
                num_classes= {self.num_classes}, 
                colormap= {self.colormap}
            )
        '''
        
        

    __doc_1: str = r'''
            *   The _create_colormap() method creates a colormap based on the 'viridis' color map from Matplotlib and the number of classes specified in the 
                ImageSegmentationPredictor. This colormap is used to generate a color-coded representation of the segmentation masks.

            *   color_map is created using the plt.get_cmap() function from Matplotlib, specifying the colormap name ('viridis') and the number of classes self.num_classes. 
                This returns a colormap object.

            *   np.arange(self.num_classes) creates an array of integers from 0 to self.num_classes - 1.

            *   color_map(np.arange(self.num_classes)) applies the colormap to the array of integers, returning an Nx4 NumPy array of RGBA color values, 
                where N is the number of classes.

            *   color_map(np.arange(self.num_classes))[:, :3] slices the Nx4 array to only include the first 3 columns (RGB values), discarding the alpha channel.

            *   (color_map(np.arange(self.num_classes))[:, :3] * 255) multiplies the RGB values by 255, converting them from the range [0, 1] to [0, 255].

            *   (color_map(np.arange(self.num_classes))[:, :3] * 255).astype(np.uint8) converts the NumPy array's data type to unsigned 8-bit integers (np.uint8), 
                which is the common format for image color values.

            The method returns an Nx3 NumPy array of RGB colors for each class, where N is the number of classes.
    '''
    def _create_colormap(self) -> np.ndarray:
        color_map: np.ndarray = plt.get_cmap('viridis', self.num_classes)
        return (color_map(np.arange(self.num_classes))[:, :3] * 255).astype(np.uint8)

    
    
    __doc_2: str = r'''
            *   The _apply_colormap() method applies the colormap created in the _create_colormap() method to a given segmentation mask. The colormap is an Nx3 NumPy 
                array containing RGB colors for each class, where N is the number of classes.

            *   mask: The input segmentation mask is a 2D NumPy array where each element represents a class label (integer) for a specific pixel in the image.

            *   self.colormap: The colormap is an Nx3 NumPy array containing RGB color values for each class, where N is the number of classes.
            
            *   self.colormap[mask]: This line performs advanced NumPy indexing. For each element in the mask array, it looks up the corresponding RGB color value 
                in the colormap. This results in a new 3D NumPy array with the same height and width as the input mask and an additional dimension for the RGB color 
                channels.
    '''
    def _apply_colormap(self, mask: np.ndarray) -> np.ndarray:
        return self.colormap[mask]
    
                                  
                                  
    
    __doc_3: str = r'''
            *   The _apply_transforms() method applies specified transformations to an input image and its corresponding mask (optional). It iterates through the 
                transformation functions in self.transform_dict for both 'image' and 'mask' keys and applies them. The transformed image and mask are returned in a dictionary.
    '''
    def _apply_transforms(self, image: PIL.Image, mask: Optional[PIL.Image] = None) -> dict[str, torch.tensor]:
        if self.transform_dict is not None:
            if 'image' in self.transform_dict:
                for transform_func in self.transform_dict['image'].values():
                    image: Any = transform_func(image)
            
            if 'mask' in self.transform_dict:
                for transform_func in self.transform_dict['mask'].values():
                    mask: Any = transform_func(mask)
                                        
        return {
            'image': image, 'mask': mask
        }
        
    
    
    __doc_4: str = r'''
            *   The predict_mask() method takes an image path, loads the image, applies transformations, and feeds it to the model to get segmentation predictions. It returns the 
                segmentation mask as a 2D NumPy array with class labels for each pixel in the image.
    '''
    def predict_mask(self, image_path: str) -> np.ndarray:
        image: PIL.Image = Image.open(image_path).convert('RGB')
        transformed_dict: dict[str, torch.tensor] = self._apply_transforms(image)
        image: torch.tensor = transformed_dict['image']
        
        with torch.no_grad():
            output: torch.tensor = self.model(image.unsqueeze(0))
        
        mask: np.ndarray = torch.argmax(output.squeeze(), dim= 0).cpu().numpy()
        return mask
    
    
    

    __doc_5: str = r'''
            *   The visualize_segmentation() method visualizes segmentation results for a given image path. It displays the original image, the predicted segmentation mask, and an 
                overlay of the image with the colored mask. The method uses Matplotlib to create a side-by-side plot of these three visualizations.
    '''
    def visualize_segmentation(self, image_path: str) -> None:
        mask: np.ndarray = self.predict_mask(image_path)
        colored_mask: np.ndarray = self._apply_colormap(mask)
        
        image: PIL.Image = Image.open(image_path).convert('RGB')
        
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        ax1.imshow(image)
        ax1.axis('off')
        ax1.set_title('Original Image')
        
        ax2.imshow(mask, cmap='viridis')
        ax2.axis('off')
        ax2.set_title('Segmentation Mask')
        
        ax3.imshow(image)
        ax3.imshow(colored_mask, alpha=0.6)
        ax3.axis('off')
        ax3.set_title('Overlay')

        plt.show()
        
        
        


#@: Driver Code
if __name__.__contains__('__main__'):
    ...