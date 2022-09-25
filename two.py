from __future__ import annotations
import torch
from typing import Optional, Any, Callable, Union, Iterable, Generator

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
    Copyright (c) 2022 Rahul Sawhney
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

#@: Dataset Class : returns one datasample dict which belongs to that corresponding index
#@: all the transformation on the single sample will happen over here. 
#@: models needs large amount of data, for this we have to load the data from the disk 
#@: usually run some processing on that data and then feed it into the model.
#@: so for this, we need very efficient algorithms to load and pre-process the data sample


'''
>>> Dataset Class
        Dataset class is the class which contains the reference of all the data
        which we want or it contains methods which fetch the data one the fly 
        and then retrieve it when necessary. 

>>> Samplers Class

>>> DataLoader Class
        DataLoader Class has Dataset Class. Role is to batch all the samples together
        in a very computationally efficiently manner using parallel processing.
        i.e num_workers
        
        loader = torch.utils.data.DataLoader(dataset)
        for batch in loader:
            print(batch)
'''

class Dataset_Interface(torch.utils.data.Dataset):
    def __init__(self) -> NotImplementedError:
        raise NotImplementedError
    
    
    def __len__(self) -> NotImplementedError:
        raise NotImplementedError
    
    
    def __getitem__(self, index: int) -> NotImplementedError:
        raise NotImplementedError
        
    
    
class CustomDataset(Dataset_Interface):
    def __init__(self, data: Any, targets: Any, 
                                  transform_dict: Optional[dict[str, Callable[Any]]] = None) -> None:
        
        self.data = data
        self.tragets = targets
        self.transform_dict = transform_dict
        
    
    def __len__(self) -> int:
        return len(self.data)
    
    
    def __repr__(self) -> str(dict[str, Any]):
        return str({
            key : value for key, value in zip(
                ['Module', 'Name', 'ObjectID'], [self.__module__, type(self).__name__, hex(id(self))]
            )
        })
        
    
    def __getitem__(self, index: int) -> dict[str, torch.tensor]:
        #@: list of all transform func on single sample of data
        #@: index : { sample, specific_target }
        
        sample: Any = self.data[index]
        current_target: Any = self.targets[index]
        
        if self.transform_dict is not None:
            for transform_func in self.transform_dict.values():
                sample: Any = transform_func(sample)
                current_target: Any = transform_func(current_target)
        
        return {
            'sample': torch.tensor(sample), 'current_target': torch.tensor(current_target)
        }
        
        

#@: Samplers: every sampler subclass has to provide 
#@:     :__iter__(): to iterate over the indices of dataset elements
#@:     :__len__(): which returns the length of the iterator


class RandomSampler:
    #@: samples elements randomly
    def __init__(self, data_source: Any, replacement: Optional[bool] = False, 
                                         num_samples: Optional[int, NoneType] = None, 
                                         generator: Optional[Generator, NoneType] = None) -> None:
        self.data_source = data_source
        self.replacement = replacement
        self._num_samples = num_samples
        self.generator = generator

        if not isinstance(self.replacement, bool):
            raise TypeError

        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError



    @property
    def num_samples(self) -> int:
        if self._num_samples is None:
            return len(self.data_source)
        
        return self._num_samples



    def __iter__(self) -> Iterator[int]:
        n: int = len(self.data_source)
        
        if self.generator is None:
            seed: int = int(torch.empty((), dtype = torch.int64).random_().item())
            generator: Generator[Any] = torch.Generator()
            generator.manual_seed(seed)
        else:
            generator = self.generator

        if self.replacement:
            for _ in range(self.num_samples // 32):
                yield from torch.randint(high = n, size = (32,), dtype = torch.int64, generator = generator).tolist()
            yield from torch.randint(high = n, size = (self.num_samples % 32,), dtype = torch.int64, generator = generator).tolist()
        
        else:
            for _ in range(self.num_samples // n):
                yield from torch.randperm(n, generator = generator).tolist()
            yield from torch.randperm(n, generator = generator).tolist()[ : self.num_samples % n]



    def __len__(self) -> int:
        return self.num_samples