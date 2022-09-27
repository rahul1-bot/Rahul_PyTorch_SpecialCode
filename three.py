from __future__ import annotations
import torch
from typing import Optional, Any, Callable, Union, Iterable, Generator, Generic, Iterator, Sequence

__author_info__: dict[str, Union[str, list[str]]] = {
    'Name': 'Rahul Sawhney',
    'Education': 'Amity University, Noida : Btech CSE (Final Year)',
    'Mail': [
        'sawhney.rahulofficial@outlook.com', 
        'rahulsawhney321@gmail.com'
    ]
}

#@: Date -> 27 Sep, 2022

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
#@: Sampler: The purpose of samplers is to determine how batches should be formed when they are passed in data_loader

class Sampler_Interface:
    def __init__(self, data_source: Optional[Iterable[Any]] = None) -> None:
        pass
    
    
    def __iter__(self) -> NotImplementedError:
        raise NotImplementedError
    



class Sequential_Sampler(Sampler_Interface):
    def __init__(self, data_source: Iterable[Any]) -> None:
        if not isinstance(data_source, Iterable):
            raise TypeError
        
        self.data_source = data_source
            
    
    def __iter__(self) -> Iterator[int]:
        return [
            idx for idx in range(len(self.data_source))        
        ].__next__()
    
    
    def __len__(self) -> int:
        return len(self.data_source)    
    
    


class Random_Sampler(Sampler_Interface):
    def __init__(self, data_source: Iterable[Any], generator: Generator[Any], 
                                                   replacement: Optional[bool] = False, 
                                                   num_samples: Optional[int] = None) -> None:
        self.data_source = data_source
        self.generator = generator
        self.replacement = replacement
        
        if num_samples is None:
            self.num_samples: int = len(self.data_source)
        else:
            self.num_samples = num_samples
        
        
        
    def __iter__(self) -> Iterator[int]:
        if self.replacement:
            for _ in range(self.num_samples // 32):
                yield from torch.randint(
                    high = self.num_samples, 
                    size = (32, ), 
                    dtype = torch.int64,
                    generator = generator
                ).tolist()
            
            yield from torch.randint(
                high = self.num_samples, 
                size = (self.num_samples % 32, ),
                dtype = torch.int64, 
                generator = generator
            ).tolist()
            
        else:
            for _ in range(self.num_samples // len(self.data_source)):
                yield from torch.randperm(len(self.data_source), generator = generator).tolist()
            yield from torch.randperm(
                len(self.data_source), 
                generator = generator
            ).tolist()[ : self.num_samples % len(self.data_source)]
            
        
        
    def __len__(self) -> int:
        return len(self.data_source)
    
    
    
    
class Subset_Random_Sampler(Sampler_Interface):
    def __init__(self, indices: Sequence[int], generator: Optional[Union[Generator[Any], NoneType]] = None) -> None:
        self.indices = indices
        self.generator = generator
        
    
    def __iter__(self) -> Iterator[int]:
        for idx in torch.randperm(len(self.indices), generator= self.generator):
            yield self.indices[idx]
            
    
    def __len__(self) -> int:
        return len(self.indices)
    
    
    
    
class Batch_Sampler(Sampler_Interface):
    def __init__(self, sampler: Union[Sampler_Interface[int], Iterable[int]], batch_size: int, 
                                                                              drop_last: bool) -> None:
        
        if not isinstance(batch_size, int) or not isinstance(drop_last, bool) or batch_size <= 0:
            raise ValueError
        
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last
        
        
    
    def __iter__(self) -> Iterator[list[int]]:
        if self.drop_last:
            sampler_iter: Iterator[int] = self.sampler.__iter__()
            while True:
                try:
                    batch: list[int] = [next(sampler_iter) for _ in range(self.batch_size)]
                    yield batch
                except StopIteration: break
        else:
            batch: list[int] = [0 for _ in self.batch_size]
            idx_in_batch: int = 0
            
            for idx in self.sampler:
                batch[idx_in_batch] = idx
                idx_in_batch += 1
                
                if idx_in_batch == self.batch_size:
                    yield batch
                    idx_in_batch = 0
                    batch = [0] * self.batch_size
            
            if idx_in_batch > 0:
                yield batch[ : idx_in_batch]


    
    def __len__(self) -> int:
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size
        

if __name__ == '__main__':
    print('hemllo')