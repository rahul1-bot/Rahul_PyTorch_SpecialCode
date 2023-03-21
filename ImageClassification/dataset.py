
from __future__ import annotations
import os, torch, torchvision
from PIL import Image
import pandas as pd
import numpy as np



#@: Image Classification Dataset
class ImageClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, data_directory: str, transform_dict: Optional[dict[str, Callable[Any]]] = None) -> None:
        
        self.data_dict: pd.DataFrame = ImageClassificationDataset.prepare_dataset(
                                        data_directory= data_directory
                                    )
        
        self.transform_dict = transform_dict
        
        

    
    def __len__(self) -> int: 
        return len(self.data_dict)
    
    
    
    def __repr__(self) -> str(dict[str, Any]):
        return str({
            x: y for x, y in zip(['Module', 'Name', 'ObjectID'], [self.__module__, type(self).__name__, hex(id(self))])
        })
    
    
    
    
    @staticmethod
    def prepare_dataset(data_directory: str) -> pd.DataFrame:
        image_paths: list[str] = [
            os.path.join(data_directory, filename) 
            for filename in os.listdir(data_directory)
        ]    

        labels: list[Union[str, int]] = [
            filename.split('_')[0] for filename in os.listdir(data_directory)
        ]
        
        data_dict: dict[str, list[Union[str, int]]] = {
            'image_path': image_paths,
            'label': labels
        } 
        
        data_dict: pd.DataFrame = pd.DataFrame(data_dict)
        return data_dict
    
    
    
    
    def __getitem__(self, index: int) -> dict[str, torch.tensor]:
        image_path: str = self.data_dict.loc[index, 'image_path']
        label: Union[int, str] = self.data_dict.loc[index, 'label']
        
        image: PIL.Image = Image.open(image_path).convert('RGB')
        
        if self.transform_dict is not None:
            for transform_func in self.transform_dict.values():
                image: Any = transform_func(image)
        
        return {
            'independent_variable': image, 'dependent_variable': label
        }

        
        

#@: Driver Code
if __name__.__contains__('__main__'):
    ...