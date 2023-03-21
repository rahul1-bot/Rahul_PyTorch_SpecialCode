from __future__ import annotations
import os, torch, torchvision
from PIL import Image
import pandas as pd
import numpy as np


'''
- root_directory/
  - train/
    - class_1/
      - image_1.jpg
      - image_2.jpg
      - ...
    - class_2/
      - image_1.jpg
      - image_2.jpg
      - ...
    - ...
  
  - test/
    - class_1/
      - image_1.jpg
      - image_2.jpg
      - ...
    - class_2/
      - image_1.jpg
      - image_2.jpg
      - ...
    - ...

'''

#@: Image Recognition Dataset
class ImageRecognitionDataset(torch.utils.data.Dataset):
    def __init__(self, data_directory: str, transform_dict: Optional[dict[str, Callable[Any]]] = None) -> None:
        self.data_dict: pd.DataFrame = ImageRecognitionDataset.prepare_dataset(
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
        class_names: list[str] = os.listdir(data_directory)
        
        images_paths: list[str] = [
            os.path.join(data_directory, class_name, image_name)                                        # main list item
            for class_name in class_names                                                               # 1st loop 
            for image_name in os.listdir(os.path.join(data_directory, class_name))                      # 2nd loop
        ]
             
        labels: list[str] = [
            index                                                                                       # main list item
            for index, class_name in enumerate(class_names)                                             # 1st loop                                                                        
            for _ in os.listdir(os.path.join(data_directory, class_name))                               # 2nd loop
        ]
        
        
        data_dict: dict[str, list[str]] = {
            'image_path': image_paths,
            'label': labels
        }
        
        data_dict: pd.DataFrame = pd.DataFrame(data_dict)
        return data_dict





    def __getitem__(self, index: int) -> dict[str, torch.tensor]: 
        image_path: str = self.data_dict.iloc[index]['image_path']
        label: str = self.data_dict.iloc[index]['label']
        
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
