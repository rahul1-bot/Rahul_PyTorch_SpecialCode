
from __future__ import annotations
import os, torch, torchvision
from PIL import Image
import pandas as pd
import numpy as np


__author_info__: dict[str, Union[str, list[str]]] = {
    'Name': 'Rahul Sawhney',
    'Education': 'Amity University, Noida : Btech CSE (Final Year)',
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


#@: Binary Segmentation Dataset Class
class BinarySegmentationDataset(torch.utils.data.Dataset):
    def __init__(self, image_directory: str, mask_directory: str, 
                                        transform_dict: Optional[dict[str, Callable[Any]]] = None) -> None:
        
        self.data_dict: pd.DataFrame = BinarySegmentationDataset.prepare_dataset(
                                        image_directory= image_directory, 
                                        mask_directory= mask_directory
                                    )
        self.transform_dict = transform_dict
    
    
    
    def __len__(self) -> int:
        return len(self.dataframe)
    
    
    
    def __repr__(self) -> str(dict[str, Any]):
        return str({
            x: y for x, y in zip(['Module', 'Name', 'ObjectID'], [self.__module__, type(self).__name__, hex(id(self))])
        })
    
    
    
    @staticmethod
    def prepare_dataset(image_directory: str, mask_directory: str) -> pd.DataFrame:
        image_filenames: list[str] = os.listdir(image_directory)
        mask_filenames: list[str] = os.listdir(mask_directory)
        
        image_paths: list[str] = [
            os.path.join(image_directory, filename) for filename in image_filenames
        ]

        mask_paths: list[str] = [
            os.path.join(image_directory, filename) for filename in mask_filenames
        ]
        
        data_dict: dict[str, list[str]] = {
            'image_path': image_paths,
            'mask_path' : mask_paths 
        }
        
        data_dict: pd.DataFrame = pd.DataFrame(data_dict)
        return data_dict
    
    
    
    def __getitem__(self, index: int) -> dict[str, torch.tensor]:
        image_path: str = self.data_dict.iloc[index]['image_path']
        mask_path: str = self.data_dict.iloc[index]['mask_path']
        
        image: PIL.Image = Image.open(image_path).convert('RGB')
        mask: PIL.Image = Image.open(mask_path).convert('L')
        
        image: np.ndarray = np.array(image)
        mask: np.ndarray = np.array(mask)
        
        if self.transform_dict is not None:
            for transform_func in self.transform_dict.values():
                image: Any = transform_func(image)
        
        image: torch.tensor = torch.from_numpy(image).permute(2, 0, 1).float()
        mask: torch.tensor = torch.from_numpy(mask).unsqueeze(0).float()
        
        image: torch.tensor = image / 255.
        mask: torch.tensor = (mask > 0).float()
        
        return {
            'independent_variable': image, 'dependent_variable': mask
        }
            



#@: Driver Code
if __name__.__contains__('__main__'):
    image_directory: str = ...
    mask_directory: str = ...
    
    transform_dict: dict[str, Callable[Any]] = {
        'h_flip': transforms.RandomHorizontalFlip(),
        'v_flip': transforms.RandomVerticalFlip(),
        'r_rotate': transforms.RandomRotation(degrees=15),
        'color_jitter': transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        'tensor': transforms.ToTensor()
    }
    
    dataset: BinarySegmentationDataset = BinarySegmentationDataset(
        image_directory= image_directory, 
        image_directory= mask_directory, 
        transform_dict= transform_dict
    )