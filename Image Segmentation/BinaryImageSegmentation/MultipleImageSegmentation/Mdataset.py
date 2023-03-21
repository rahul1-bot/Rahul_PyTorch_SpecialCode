

from __future__ import annotations
import os, torch, torchvision
from PIL import Image
import pandas as pd
import numpy as np



#@: Multiple Image Segmentation

class SegmentationDataset(torch.utils.data.Dataset):
    def __init__(self, image_directory: str, mask_directory: str, 
                                             transform_dict: Optional[dict[str, Callable[Any]]] = None) -> None:
        
        self.data_dict: pd.DataFrame = SegmentationDataset.prepare_dataset(
                                        image_directory= image_directory,
                                        mask_directory= mask_directory,
                                    )
        self.transform_dict = transform_dict
        
    
    def __len__(self) -> int:
        return len(self.data_dict)
    
    
    
    def __repr__(self) -> str(dict[str, Any]):
        return str({
            x: y for x, y in zip(['Module', 'Name', 'ObjectID'], [self.__module__, type(self).__name__, hex(id(self))])
        })
        
        
    
    @staticmethod
    def prepare_dataset(image_directory: str, data_directory: str) -> pd.DataFrame:
        image_filenames: list[str] = os.listdir(image_directory)
        mask_filenames: list[str] = os.listdir(mask_directory)
        
        image_paths: list[str] = [
            os.path.join(image_directory, filename) for filename in image_filenames
        ]
        
        mask_paths: list[str] = [
            os.path.join(mask_directory, filename) for filename in mask_filenames
        ]
        
        data_dict: dict[str, list[str]] = {
            'image_path': image_paths,
            'mask_path' : mask_paths
        }
        
        data_dict: pd.DataFrame = pd.DataFrame(data_dict)
        return data_dict
    
    
    
    def __getitem__(self, index: index) -> dict[str, torch.tensor]:
        image_path: str = self.data_dict[index]['image_path']
        mask_path: str = self.data_dict[index]['mask_path']
        
        image: PIL.Image = Image.open(image_path).convert('RGB')
        mask: PIL.Image = Image.open(mask_path).convert('L') #grayscale mask
        
        if self.transform_dict is not None:
            for transform_func in self.transform_dict.values():
                image: Any = transform_func(image)
                mask: Any = transform_func(mask)
                
        return {
            'independent_variable': image, 'dependent_variable': mask
        }


#@: driver code
if __name__.__contains__('__main__'):
    ...


