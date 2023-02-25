#@: "Multiple transformations: Data Augumentation"
#@: Data Preprocess pipeline 

'''
for each index: 
    return Map<int, Map<str, torch.tensor>>
    
'''

from __future__ import annotations
import torch
from torchvision import transforms, utils


class Dataset(torch.utils.data.Dataset):
    def __init__(self, x: Any, y: Any, 
                               transforms_dict: Optional[dict[str, dict[str, Callable[Any]]]] = None) -> None:
        self.x = x
        self.y = y
        self.transforms_dict = transforms_dict
        
    
    def __len__(self) -> int:
        return len(self.x)
    
    
    def __repr__(self) -> str(dict[str, Any]):
        return str({
            key : value for key, value in zip(
                ['Module', 'Name', 'ObjectID'], [self.__module__, type(self).__name__, hex(id(self))]
            )
        })
    
    
    def __getitem__(self, index: int) -> dict[int, dict[str, torch.tensor]]:
        global_mapper: dict[int, dict[str, torch.tensor]] = {}
        count: int = 0
        
        if self.transforms_dict is not None:
            for transform_dict in self.transforms_dict.values():
                count += 1
                current_x: Any = self.x[index]
                current_y: Any = self.y[index]
                
                for transform_func in transform_dict.values():
                    current_x: Any = transform_func(current_x)
                    current_y: Any = transform_func(current_y)
                
                global_mapper.update({
                    count : {
                        'independent_variable': current_x, 'dependent_variable': current_y
                    }
                })    

        return global_mapper
                    

#@: To visualize, do reverse sequential inverse of all transformation
        


if __name__.__contains__('__main__'):
    data_path: str = ...
    x, y = ..., ...
    
    transform_dict_one: dict[str, Callable[Any]] = {}
    transform_dict_two: dict[str, Callable[Any]] = {}
    transform_dict_three: dict[str, Callable[Any]] = {}
    
    transforms_dict: dict[str, dict[str, Callable[Any]]] = {
        'transform_dictOne': transform_dict_one, 
        'transform_dictTwo': transform_dict_two,
        'transform_dictThree': transform_dict_three 
    }
    
    data_obj: Dataset = Dataset(x= x, y= y, transforms_dict= transforms_dict)
    
    
    
    
    
    
    
    
    
    
    
    