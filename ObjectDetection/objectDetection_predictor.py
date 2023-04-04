
from __future__ import annotations
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


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

__NOTE__: str = r'''
    Python Type Annotations are used Interchangably. Refer your Linter. 
'''


#@: Data Frame Format { Object Detection Dataset Pipeline }

__data_dict__: str = r'''
        image_path                                annotation
    0  path/to/image1.jpg                       ((x1, y1, x2, y2), label1)
    1  path/to/image2.jpg                       ((x3, y3, x4, y4), label2)
    2  path/to/image3.jpg                       ((x5, y5, x6, y6), label3)
    ...                                         ...
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

__sample__doc__: str = r'''
    sample: dict[str, Union[torch.tensor, dict[str, torch.tensor]]] = {
        'independent_variable': image, 
        'dependent_variable': {
            'bounding_box': bounding_box,
            'label': label    
        }
    }

'''


#@: Transform dict { Object Detection Transformation Pipeline }

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


#@: Models Returns { Object Detection Models Pipeline }
__model_returns__: str = r'''

    # Forward Pass (x: torch.tensor) -> dict[str, dict[str, torch.tensor]]
    
    Model prediction: dict[str, dict[str, torch.tensor]] = {
        'bounding_box': Some Tensors,
        'class_prob': Some Tensor
    }
    
    Converted Predictions: dict[str, np.ndarray] = {
        'bounding_box': numpy array of shape (num_objects, 4),
        'class_prob': numpy array of shape (num_objects, num_classes)

    }
    
    #@: Example of Prediction 
    {
        'bounding_box': array([[ 93, 163, 231, 341],
                              [331,  64, 444, 246],
                              [  3, 155,  71, 304]]),
    
    
        'class_prob': array([0.91, 0.78, 0.65])
    }
'''

#@: Batch { Object Detection Prediction Pipeline }
__batch_doc__: str = r'''

    #@: Single Prediction
    prediction: dict[str, np.ndarray] = {
        'bounding_box': array([[ 93, 163, 231, 341],
                              [331,  64, 444, 246],
                              [  3, 155,  71, 304]]),
    
        'class_prob': array([0.91, 0.78, 0.65])
    }
    
    #@: Batch of prediction for new set of images 
    batch_dict = [
        {
            'image_path': 'image1.jpg', 
            'annotations': {
                'bounding_box': np.array([[10, 20, 100, 200], [50, 60, 150, 250]]), 
                'label': np.array([0, 1])
            }
        },
        {
            'image_path': 'image2.jpg', 
            'annotations': {
                'bounding_box': np.array([[30, 40, 120, 220]]), 
                'label': np.array([0])
            }
        }
    ...
    ]
    
  #@: Example 
  batch = [
        {'image_path': 'image1.jpg', 'annotations': {'bounding_box': np.array([[10, 20, 100, 200], [50, 60, 150, 250]]), 'label': np.array([0, 1])}},
        {'image_path': 'image2.jpg', 'annotations': {'bounding_box': np.array([[30, 40, 120, 220]]), 'label': np.array([0])}}
        ...
    ]

'''

__doc__: str = r'''
        *   The code defines a class called ObjectDetectionPredictor which contains methods for predicting object detections in images, predicting object detections in a batch of images, 
            and visualizing the detections in an image.

        *   The class constructor takes in a PyTorch neural network model, a dictionary containing the different transformations to be applied to the input data, and a device to run the model on.

        *   The __transform__ method applies a set of transformations to a given input data using a dictionary of transformation functions.

        *   The __predict__ method takes in an image path and optionally an annotations dictionary containing the bounding boxes and labels of the objects in the image. It uses the given model 
            to generate predictions on the image and returns a dictionary containing the predicted bounding boxes and class probabilities of the objects.

        *   The __visualize_sample__ method takes in an image path and optionally an annotations dictionary containing the bounding boxes and labels of the objects in the image. It displays the 
            image with bounding boxes and labels overlaid on it.
'''


#@: Object Detection Prediction Pipeline 
class ObjectDetectionPredictor:
    def __init__(self, model: torch.nn.Module, transform_dict: Dict[str, Dict[str, Dict[str, Callable[Any]]]], device: Optional[str] = 'cpu') -> None:
        self.model = model
        self.transform_dict = transform_dict
        self.device = torch.device(device)



    def __transform__(self, data: Any, transforms: Dict[str, Callable[Any]]) -> Any:
        for transform_name, transform_func in transforms.items():
            data = transform_func(data)
        return data




    def __predict__(self, image_path: str, annotations: Optional[dict[str, np.ndarray]] = None) -> dict[str, np.ndarray]:
        with Image.open(image_path) as image:
            image = np.array(image)

        if annotations:
            sample = {
                'independent_variable': self.__transform__(image, self.transform_dict['image']),
                'dependent_variable': {
                    'bounding_box': self.__transform__(annotations['bounding_box'], self.transform_dict['annotation']['bounding_box']),
                    'label': self.__transform__(annotations['label'], self.transform_dict['annotation']['label'])
                }
            }
        else:
            sample = {
                'independent_variable': self.__transform__(image, self.transform_dict['image'])

            }

        inputs: torch.tensor = sample['independent_variable'].unsqueeze(0).to(self.device)
        outputs: dict[str, dict[str, torch.tensor]] = self.model(inputs)
        outputs: dict[str, np.ndarray] = {
            'bounding_box': outputs['bounding_box'].cpu().numpy(),
            'class_prob': outputs['class_prob'].cpu().numpy()
        }
        return outputs





    def __visualize_sample__(self, image_path: str, annotations: Optional[dict[str, np.ndarray]] = None) -> None:
        with Image.open(image_path) as image:
            image = np.array(image)

        if annotations:
            sample = {
                'independent_variable': self.__transform__(image, self.transform_dict['image']),
                'dependent_variable': {
                    'bounding_box': annotations['bounding_box'],
                    'label': annotations['label']
                }
            }
        else:
            sample = {
                'independent_variable': self.__transform__(image, self.transform_dict['image'])
            }

        image = sample['independent_variable']
        image = np.transpose(image, (1, 2, 0))
        image = np.clip(image, 0, 1)
        plt.imshow(image)
        plt.axis('off')
        
        if annotations:
            for idx in range(len(annotations['label'])):
                bbox = annotations['bounding_box'][idx]
                plt.plot([bbox[0], bbox[2]], [bbox[1], bbox[1]], color='red', linewidth=2)
                plt.plot([bbox[0], bbox[2]], [bbox[3], bbox[3]], color='red', linewidth=2)
                plt.plot([bbox[0], bbox[0]], [bbox[1], bbox[3]], color='red', linewidth=2)
                plt.plot([bbox[2], bbox[2]], [bbox[1], bbox[3]], color='red', linewidth=2)
                plt.text(
                    bbox[0], bbox[1], self.transform_dict['annotation']['label']['transform_2'](annotations['label'][idx]), 
                    fontsize=10, 
                    color='red', 
                    backgroundcolor='white', 
                    fontweight='bold'
                )
        plt.show()
    
    
    

    def predict(self, image_path: str, annotations: Optional[dict[str, np.ndarray]] = None) -> dict[str, np.ndarray]:
        return self.__predict__(image_path, annotations)
    
    
    
    

    def predict_batch(self, batch: List[Tuple[str, Optional[dict[str, np.ndarray]]]]) -> List[dict[str, np.ndarray]]:
        return [
            self.__predict__(image_path, annotations) for image_path, annotations in batch 
        ]
    
    
    
    
    def visualize(self, image_path: str, annotations: Optional[Dict[str, np.ndarray]] = None) -> None:
        return self.__visualize_sample__(image_path, annotations)
    
    
    
    
    def batch_visualization(self, image_folder: str, annotations: Optional[dict[str, np.ndarray]] = None,...) -> None:
        #@: Will Update Soon 
        ...





#@: Driver Code
if __name__.__contains__('__main__'):
    ...