
from __future__ import annotations
import os, torch, torchvision
from PIL import Image
import numpy as np
import pandas as pd


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
    The ObjectDetectionDataset class is a custom dataset class that inherits from PyTorch's torch.utils.data.Dataset. It is designed to handle object detection tasks 
    by taking in images and their corresponding bounding box annotations, along with optional transformations to be applied to images and annotations.

    When you instantiate the class with the image_directory and annotation_csv parameters, it processes the annotations and creates a DataFrame to store the image paths 
    and corresponding annotations. The class also supports optional transformations specified in a nested dictionary format through the transform_dict parameter.

    The class implements the __len__() method to return the total number of samples in the dataset and the __getitem__() method to return a specific sample at the given index, 
    including the image and annotation (bounding box and label). The __getitem__() method also applies any specified transformations on the image and annotation.

    Here is a summary of the important methods and their functionalities:
    
    *   __init__: Initializes the dataset with the image directory, annotation CSV file, and optional transform dictionary.
    *   __len__: Returns the number of samples in the dataset.
    *   __getitem__: Returns a specific sample at the given index, including the image and annotation (bounding box and label), while also applying any specified transformations.
    *   prepare_dataset: A static method that processes the annotations and creates a DataFrame to store the image paths and corresponding annotations.
    *   __repr__: a string representation of the dataset object.
'''

__data_dict__: str = r'''
       image_path                                annotation
0  path/to/image1.jpg                       ((x1, y1, x2, y2), label1)
1  path/to/image2.jpg                       ((x3, y3, x4, y4), label2)
2  path/to/image3.jpg                       ((x5, y5, x6, y6), label3)
...                                         ...

'''

__transform_dict__: str = r'''
    transform_dict: dict[str, dict[str, dict[str, Callable[Any]]]] = {
        'image': {
            'transform_1': transform_one,
            'transform_2': transform_two 
        },
        
        'annotation': {
            'bounding_box': {
                'transform_1': transform_one, 
                'transform_2': transform_two
            },
            'label': {
                'transform_1': transform_one, 
                'transform_2': transform_two
            }
        }
    }

'''

__getitem__return__: str = r'''
    __getitem__(index: int) -> dict[str, Union[torch.tensor, dict[str, torch.tensor]]]:
        {
            'independent_variable': image, 
            'dependent_variable': {
                'bounding_box': bounding_box,
                'label': label    
            }
        }

'''

#@: Object Detection Dataset
class ObjectDetectionDataset(torch.utils.data.Dataset):
    def __init__(self, image_directory: str, annotation_csv: str, 
                                             transform_dict: Optional[dict[str, dict[str, dict[str, Callable[Any]]]]] = None) -> None:
        
        '''
        #@: Args:
                image_directory: A String representing the path to the directory containing all the images.
                annotation_csv: A String representing the path to the csv file containing the corresponding bounding box annotations.
                transform_dict: A dictionary containing the transformations to be applied to the images and annotations
            
        
        #@: Returns:
                None
                
        
        #@: Structure format of annotation CSV:
                image_path,x1,y1,x2,y2,label
                path/to/image1.jpg,10,20,30,40,label1
                path/to/image1.jpg,50,60,70,80,label2
                path/to/image2.jpg,15,25,35,45,label1
                path/to/image3.jpg,100,110,120,130,label3  
        
        
        #@: Structure format of transform_dict:
            transform_dict: dict[str, dict[str, dict[str, Callable[Any]]]] = {
                'image': {
                    'transform_1': transform_one,
                    'transform_2': transform_two 
                },
            
                'annotation': {
                    'bounding_box': {
                        'transform_1': transform_one, 
                        'transform_2': transform_two
                    },
                    'label': {
                        'transform_1': transform_one, 
                        'transform_2': transform_two
                    }
                }
            }

        '''
        self.data_dict: pd.DataFrame = ObjectDetectionDataset.prepare_dataset(
            image_directory= image_directory,
            annotation_csv= annotation_csv
        )
        
        self.transform_dict = transform_dict 
        
        
    
    def __len__(self) -> int:
        return len(self.data_dict)
    
    
    
    def __repr__(self) -> str(dict[str, Any]):
        return str({
            x: y for x, y in zip(['Module', 'Name', 'ObjectID'], [self.__module__, type(self).__name__, hex(id(self))])
        })
        
        
    
    @staticmethod
    def prepare_dataset(image_directory: str, annotation_csv: str) -> pd.DataFrame:
        '''
        #@: Args:
            image_directory: A String representing the path to the directory containing all the images.
            annotation_csv: A String representing the path to the csv file containing the corresponding bounding box annotations.
        
        #@: Returns:
            data_dict: (pd.DataFrame)
            The prepare_dataset() function processes the Annotation CSV file and converts it into a pandas DataFrame with the desired format:
                                        image_path                                annotation
                0  path/to/image1.jpg                       ((x1, y1, x2, y2), label1)
                1  path/to/image2.jpg                       ((x3, y3, x4, y4), label2)
                2  path/to/image3.jpg                       ((x5, y5, x6, y6), label3)
                ...                                         ...
        
        '''
        
        #@: listing all the image files in the image_directory 
        image_filenames: list[str] = [
            item for item in os.listdir(image_directory) if item.endswith(('.jpg', '.jpeg', '.png'))
        ]
        
        
        #@: reading the annotation csv file 
        annotation_df: pd.DataFrame = pd.read_csv(annotation_csv)
        
        #@: matching image filenames with corresponding annotations
        images_paths: list[str] = []
        annotations: list[Any] = []
        
        for filename in image_filenames:
            image_path: str = os.path.join(image_directory, filename)
            image_paths.append(image_path)

            matched_annotation = annotation_df[annotation_df['filename'] == filename]
            annotation: tuple = (
                (
                    matched_annotation['x1'].item(), matched_annotation['y1'].item(),
                    matched_annotation['x2'].item(), matched_annotation['y2'].item()
                ),
                    matched_annotation['label'].item()
            )
            
            annotations.append(annotation)

        #@: Creating a dictionary containing the image paths and the corresponding annotations
        data_dict: dict[str, Union[list, tuple]] = {
            'image_path': image_paths, 
            'annotation': annotations
        }
        
        #@: Converting the dictionary to a pandas DataFrame
        data_dict: pd.DataFrame = pd.DataFrame(data_dict)
        return data_dict
    
    
    
    
    
    def __getitem__(self, index: int) -> dict[str, Union[torch.tensor, dict[str, torch.tensor]]]:
        '''
        #@: Args:
            index: index for the sample 
        
        #@: Returns:
            sample: 
                dict[str, Union[torch.tensor, dict[str, torch.tensor]]]
        
        #@: Structure of sample:
            sample: dict[str, Union[torch.tensor, dict[str, torch.tensor]]] = {
                'independent_variable': image, 
                'dependent_variable': {
                    'bounding_box': bounding_box,
                    'label': label    
                }
            }
        
        '''
        image_path: str = self.data_dict.iloc[index]['image_path']
        annotation: tuple = self.data_dict.iloc[index]['annotation']
        
        image: PIL.Image = Image.open(image_path).convert('RGB')
        bounding_box, label = annotation
        
        if self.transform_dict is not None:
            if 'image' in self.transform_dict:
                for transform_func in self.transform_dict['image'].values():
                    image: Any = transform_func(image)

            if 'annotation' in self.transform_dict:
                if 'bounding_box' in self.transform_dict['annotation']:
                    for transform_func in self.transform_dict['annotation']['bounding_box'].values():
                        bounding_box: Any = transform_func(bounding_box)
                        
                if 'label' in self.transform_dict['annotation']:
                    for transform_func in self.transform_dict['annotation']['label'].values():
                        label: Any = transform_func(label)
                        
        
        return {
            'independent_variable': image, 
            'dependent_variable': {
                'bounding_box': bounding_box,
                'label': label
            }
        }
        
        
        
                        
        
        
#@: Driver Code
if __name__.__contains__('__main__'):
    
    #@: example use case _
    
    transform_dict: dict[str, dict[str, dict[str, Callable[Any]]]] = {
        'image': {
            'transform_1': transform_one,
            'transform_2': transform_two 
        },
        
        'annotation': {
            'bounding_box': {
                'transform_1': transform_one, 
                'transform_2': transform_two
            },
            'label': {
                'transform_1': transform_one, 
                'transform_2': transform_two
            }
        }
    }
    
    
    
    
    
    
    
    
    