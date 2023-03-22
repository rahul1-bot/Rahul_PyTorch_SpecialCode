from __future__ import annotations
import os, torch, torchvision
from PIL import Image
import pandas as pd
import numpy as np


'''
a dataset class for object detection in PyTorch, 
using Python static annotations and a Pandas DataFrame to store the image paths, bounding boxes, and labels:


The code assumes that you have a CSV file with columns
'image_path', 'x_min', 'y_min', 'x_max', 'y_max', and 'class' 
containing the paths to the images, bounding box coordinates, and class labels, respectively. 
Replace "path/to/annotations.csv" with the path to your CSV file containing the annotations.

annotation.csv:
    image_path,x_min,y_min,x_max,y_max,class
    path/to/image1.jpg,10,20,100,120,class1
    path/to/image2.jpg,15,30,110,130,class2
    path/to/image3.jpg,20,40,120,140,class1


annotation dataFrame

           image_path  x_min  y_min  x_max  y_max   class
0  path/to/image1.jpg     10     20    100    120  class1
1  path/to/image2.jpg     15     30    110    130  class2
2  path/to/image3.jpg     20     40    120    140  class1


bbox = self.annotations.iloc[index, 1:5].tolist()

The code bbox = self.annotations.iloc[index, 1:5].tolist() performs the following operations:

self.annotations.iloc[index, 1:5]: 
        This line of code uses the iloc property to access a specific row and columns of the annotations DataFrame. 
        In this case, it selects the row specified by the index variable and columns 1 to 4 (the end index 5 is exclusive). 
        The selected columns correspond to the bounding box coordinates: x_min, y_min, x_max, and y_max.

.tolist(): 
        This method is called on the result of the previous operation, which is a Pandas Series object containing 
        the bounding box coordinates. The tolist() method converts the Series object to a Python list.


So, this line of code is extracting the bounding box coordinates for a specific row (specified by index) from the



If you do not have an annotations.csv file, you can either create one or define the annotations directly in your 
code as a list of tuples. Here's an example of how to create a custom dataset class for object detection in PyTorch using static annotations defined in the code:



# Static annotations

annotations: list[tuple[Any, ...]] = [
    ('path/to/image1.jpg', 10, 20, 100, 120, 'class1'),
    ('path/to/image2.jpg', 15, 30, 110, 130, 'class2'),
    ('path/to/image3.jpg', 20, 40, 120, 140, 'class1'),
]


annotations: dict[str, dict[str, Any]] = {
                'Object_One': {
                    'image_path': ..., 
                    'x_min': ...,
                    'y_min': ...,
                    'x_max': ...,
                    'y_max': ...
                },
                'Object_Two': {
                    'image_path': ..., 
                    'x_min': ...,
                    'y_min': ...,
                    'x_max': ...,
                    'y_max': ...
                },
                'Object_Three': {
                    'image_path': ..., 
                    'x_min': ...,
                    'y_min': ...,
                    'x_max': ...,
                    'y_max': ...
                },
                ...
            }

In the context of object detection, the independent and dependent variables can be defined as follows:

Independent variables: These are the input features or data used to make predictions. In this case, the independent variable is the image itself. The model will use the pixel values and learn patterns or features from the images to predict the bounding box coordinates and class labels.

Dependent variables: These are the target values or outputs that we want to predict based on the independent variables. In the case of object detection, there are two dependent variables:
    1) Bounding box coordinates: These are the x_min, y_min, x_max, and y_max values that define the location of the object in the image.
    2) Class labels: These are the object category labels (e.g., 'dog', 'cat', 'car') that the model will predict for each detected object in the image.


'''


#@: object Detection Dataset

class ObjectDetectionDataset(torch.utils.data.Dataset):
    def __init__(self, annotation_csv_path: str, transform_dict: Optional[dict[str, Callable[Any]]] = None) -> None:
        self.data_dict: pd.DataFrame = ObjectDetectionDataset.prepare_dataset(
                                        annotation_csv_path= annotation_csv_path
                                    )
        self.transform_dict = transform_dict
        
    
    
    def __len__(self) -> int:
        return len(self.data_dict)
    
    
         
    def __repr__(self) -> str(dict[str, Any]):
        return str({
            x: y for x, y in zip(['Module', 'Name', 'ObjectID'], [self.__module__, type(self).__name__, hex(id(self))])
        })
         
    
    
    @staticmethod
    def prepare_dataset(annotation_csv_path: str) -> pd.DataFrame:
        data_dict: pd.DataFrame = pd.read_csv(annotation_csv_path)
        return data_dict
        
        
        
    def __getitem__(self, index: int) -> dict[str, Union[dict[str, torch.tensor], torch.tensor]]:
        image_path: str = self.data_dict.iloc[index, 0]
        image: PIL.Image = Image.open(image_path)
        
        #@: reading the bounding box and the class label
        
        bounding_box: list[int] = self.data_dict.iloc[index, 1: -1].tolist()
        label: str = self.data_dict.iloc[index, -1]
        
        if self.transform_dict is not None:
            for transform_func in self.transform_dict.values():
                image: Any = transform_func(image)
                
                
        bounding_box: torch.tensor = torch.tensor(bounding_box)
        label: torch.tensor = torch.tensor(label)
        
        return {
            'independent_variable': image, 
            'dependent_variable': {
                'bounding_box': bounding_box,
                'label': label
            }
        }
        
        
        


#@: Driver Code
if __name__.__contains__('__main__'):
    ...






















