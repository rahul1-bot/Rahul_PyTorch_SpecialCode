from __future__ import annotations
import torch, os, json, logging
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(level=logging.INFO)

__author_info__: dict[str, Union[str, list[str]]] = {
    'Name': 'Rahul Sawhney',
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


#: NOTE: Code Structure Top to Bottom { "Check Software-Design Patterns " } 
#@: Pose Estimation Predictor Class
#@: Optimize this piece of CODE according to your needs. 

class PoseEstimationPredictor:
    def __init__(self, model: torch.nn.Module, transform_dict: dict[str, dict[str, dict[str, Callable[..., Any]]]], 
                                               device: torch.device = 'cuda' if torch.cuda.is_available() else 'cpu') -> None:
        
        # This code snippet defines the beginning of a class named PoseEstimationPredictor. The class is designed to make predictions using a pre-trained pose estimation model. 
        # The class takes three arguments when initializing:

        # model: A PyTorch neural network model that has been pre-trained to perform pose estimation tasks.
        # transform_dict: A dictionary containing data preprocessing and post-processing transformation functions for both independent and dependent variables.
        # device: A PyTorch device that specifies whether to use GPU or CPU for computations. By default, it is set to 'cuda' if a GPU is available, otherwise, it falls back to 'cpu'.
        
        # Inside the __init__ method, the class stores the input arguments as instance variables:

        #     self.model: The input pose estimation model is assigned to this instance variable.
        #     self.transform_dict: The input dictionary of transformation functions is assigned to this instance variable.
        #     self.device: The input device (either 'cuda' or 'cpu') is assigned to this instance variable.
        
        # This provided code snippet sets up the initial state of the PoseEstimationPredictor class. The complete class should include additional methods to perform tasks such as data 
        # transformation, prediction, and visualization using the model and the transformation functions.
        
        self.model = model
        self.transform_dict = transform_dict
        self.device = device
                
                
        
    
    def __transform_sample(self, data: Any, transforms: dict[str, Callable[..., Any]]) -> Any:
        # The __transform_sample method is a helper function designed to apply a series of transformation functions to the input data. The method takes two arguments:

        #     data: The input data (in this case, image or annotation data) that needs to be transformed.
            
        #     transforms: A dictionary containing the transformation functions to be applied to the input data. The keys in this dictionary are the names of the transformation functions, 
        #                 and the values are the corresponding callable functions.
        
        # Inside the method, a for loop iterates through the transformation functions in the transforms dictionary. For each iteration, the current transformation function (transform_func) 
        # is applied to the data. The transformed data is then assigned back to the data variable, allowing the next transformation function in the loop to be applied on the already transformed 
        # data.

        # After applying all the transformation functions, the method returns the transformed data.
        
        for transform_func in transforms.values():
            data: Any = transform_func(data)
        return data
        
        
        
        
    def __predict_sample(self, image_path: str, ground_truth: Optional[dict[str, np.ndarray]] = None) -> dict[str, np.ndarray]:
        # This __predict_sample method is a helper function in the PoseEstimationPredictor class designed to make predictions using the model for a single image. It takes two arguments:

        #     image_path: The file path of the input image for which predictions need to be made.
            
        #     ground_truth: An optional dictionary containing the ground truth values for keypoints, pose_label, bounding_box, and metadata. If provided, these values will be used to create 
        #                   the sample dictionary, which contains both the preprocessed input image and the transformed ground truth values.

        
        # First, the method opens the image at image_path and converts it into a NumPy array. Then, based on whether the ground_truth is provided or not, it creates a sample dictionary 
        # that contains the transformed image as the independent variable and, if available, the transformed ground truth values as the dependent variables.

        
        # The transformed image tensor is then moved to the specified device (either CPU or GPU), and the model is used to make predictions on this input. The outputs from the model are 
        # stored in a dictionary, and these outputs are then converted to NumPy arrays and returned.

        # This method is useful for making predictions on a single image using the pose estimation model and preprocessing the input image and ground truth values based on the transformation
        # functions provided in the transform_dict.
        
        with Image.open(image_path) as image:
            image: np.ndarray = np.array(image)
                        
        if ground_truth:
            sample: dict[str, torch.tensor | dict[str, torch.tensor]] = {
                'independent_variable': self.__transform_sample(image, self.transform_dict['independent_variable']),
                'dependent_variable': {
                    'keypoints': self.__transform_sample(ground_truth['keypoints'], self.transform_dict['dependent_variable']['keypoints']),
                    'pose_label': self.__transform_sample(ground_truth['pose_label'], self.transform_dict['dependent_variable']['pose_label']),
                    'bounding_box': self.__transform_sample(ground_truth['bounding_box'], self.transform_dict['dependent_variable']['bounding_box']),
                    'metadata': self.__transform_sample(ground_truth['metadata'], self.transform_dict['dependent_variable']['metadata'])
                }
            }
        
        else:
            sample: dict[str, torch.tensor | dict[str, torch.tensor]] = {
                'independent_variable': self.__transform_sample(image, self.transform_dict['independent_variable'])
            }

        inputs: torch.tensor = sample['independent_variable'].unsqueeze(0).to(self.device)
        outputs: dict[str, dict[str, torch.tensor]] = self.model(inputs)
        
        outputs: dict[str, np.ndarray] = {
            'keypoints': outputs['keypoints'].cpu().numpy(),
            'pose_label': outputs['pose_label'].cpu().numpy(),
            'bounding_box': outputs['bounding_box'].cpu().numpy(),
            'metadata': outputs['metadata'].cpu().numpy()
        }
        
        return outputs


        
        
    def __visualize_sample(self, image_path: str, ground_truth: Optional[dict[str, np.ndarray]] = None) -> None:
        # The __visualize_sample method in the PoseEstimationPredictor class visualizes a given sample image, along with its ground truth values (if provided) and the model's predicted output. 
        # It takes two arguments:

        #     image_path: The file path of the input image to visualize.
            
        #     ground_truth: An optional dictionary containing the ground truth values for keypoints, pose_label, bounding_box, and metadata.
        
        
        # First, the method opens the image at image_path and converts it into a NumPy array. Then, based on whether the ground_truth is provided or not, it creates a sample dictionary that 
        # contains the transformed image as the independent variable and, if available, the ground truth values as the dependent variables.

        # Next, the method uses __predict_sample to get the model's predictions for the input image. The original image is then preprocessed for visualization.

        # The method then plots the image and visualizes the ground truth keypoints, bounding boxes, pose labels, and metadata (if provided). The keypoints are marked with blue circles, 
        # and the bounding boxes are drawn with blue solid lines.

        # Afterward, the method visualizes the model's predicted keypoints, bounding boxes, pose labels, and metadata. The predicted keypoints are marked with red crosses, and the 
        # bounding boxes are drawn with red dashed lines.

        # This method provides a helpful way to visualize and compare the ground truth values and model predictions for a given input image in the context of a pose estimation problem.
        
        with Image.open(image_path) as image:
            image: np.ndarray = np.array(image)

        if ground_truth:
            sample: dict[str, torch.tensor | dict[str, torch.tensor]] = {
                'independent_variable': self.__transform_sample(image, self.transform_dict['independent_variable']),
                'dependent_variable': ground_truth
            }
        else:
            sample: dict[str, torch.tensor | dict[str, torch.tensor]] = {
                'independent_variable': self.__transform_sample(image, self.transform_dict['independent_variable'])
            }

        #@: Get the real output predictions
        predictions: dict[str, np.ndarray] = self.__predict_sample(image_path, ground_truth)

        image: np.ndarray = sample['independent_variable']
        image: np.ndarray = np.transpose(image, (1, 2, 0))
        image: np.ndarray = np.clip(image, 0, 1)
        plt.imshow(image)
        plt.axis('off')


        if ground_truth:
            #@: Visualizing the ground truth keypoints, bounding box, pose_label, and metadata
            for gt_keypoints in ground_truth['keypoints']:
                plt.scatter(gt_keypoints[:, 0], gt_keypoints[:, 1], marker='o', color='blue', s=30)

            for gt_bbox in ground_truth['bounding_box']:
                plt.plot([gt_bbox[0], gt_bbox[2]], [gt_bbox[1], gt_bbox[1]], color='blue', linewidth=2)
                plt.plot([gt_bbox[0], gt_bbox[2]], [gt_bbox[3], gt_bbox[3]], color='blue', linewidth=2)
                plt.plot([gt_bbox[0], gt_bbox[0]], [gt_bbox[1], gt_bbox[3]], color='blue', linewidth=2)
                plt.plot([gt_bbox[2], gt_bbox[2]], [gt_bbox[1], gt_bbox[3]], color='blue', linewidth=2)

            gt_pose_label_text = f"GT Pose Label: {ground_truth['pose_label']}"
            gt_metadata_text = f"GT Metadata: {ground_truth['metadata']}"


        #@: Visualize predicted keypoints, bounding box, pose_label, and metadata
        for pred_keypoints in predictions['keypoints']:
            plt.scatter(pred_keypoints[:, 0], pred_keypoints[:, 1], marker='x', color='red', s=30)


        for pred_bbox in predictions['bounding_box']:
            plt.plot([pred_bbox[0], pred_bbox[2]], [pred_bbox[1], pred_bbox[1]], color='red', linestyle='dashed', linewidth=2)
            plt.plot([pred_bbox[0], pred_bbox[2]], [pred_bbox[3], pred_bbox[3]], color='red', linestyle='dashed', linewidth=2)
            plt.plot([pred_bbox[0], pred_bbox[0]], [pred_bbox[1], pred_bbox[3]], color='red', linestyle='dashed', linewidth=2)
            plt.plot([pred_bbox[2], pred_bbox[2]], [pred_bbox[1], pred_bbox[3]], color='red', linestyle='dashed', linewidth=2)

        pred_pose_label_text = f"Predicted Pose Label: {predictions['pose_label']}"
        pred_metadata_text = f"Predicted Metadata: {predictions['metadata']}"
            
        
        
        
        
    def predict_sample(self, image_path: str, ground_truth: Optional[dict[str, np.ndarray]] = None) -> dict[str, np.ndarray]:
        # The predict_sample method in the PoseEstimationPredictor class is a public method that calls the private method __predict_sample. It takes two arguments:

        #     image_path: The file path of the input image for which the model predictions are needed.
            
        #     ground_truth: An optional dictionary containing the ground truth values for keypoints, pose_label, bounding_box, and metadata.

        
        # The method simply calls the private method __predict_sample with the provided image_path and ground_truth arguments, and returns the model's predictions as a dictionary 
        # containing the predicted keypoints, pose_label, bounding_box, and metadata as NumPy arrays.
        
        return self.__predict_sample(image_path, ground_truth)
        
        
        
        
    def visualize_sample(self, image_path: str, ground_truth: Optional[dict[str, np.ndarray]] = None) -> dict[str, np.ndarray]:
        # The visualize_sample method in the PoseEstimationPredictor class is a public method that calls the private method __visualize_sample. It takes two arguments:

        #     image_path: The file path of the input image for which the model predictions and ground truth values (if provided) are to be visualized.
        #     ground_truth: An optional dictionary containing the ground truth values for keypoints, pose_label, bounding_box, and metadata.
        
        # The method calls the private method __visualize_sample with the provided image_path and ground_truth arguments. 
        
        return self.__visualize_sample(image_path, ground_truth)
    
        
    
     
    def predict_batch(self, image_batch: list[dict[str, torch.tensor | dict[str, torch.tensor]]]) -> list[dict[str, np.ndarray]]:
        # NOTE: image_batch = [
        #     {
        #         'independent_variable': image_path_1st, 
        #         'dependent_variable': {
        #             'keypoints': keypoints,
        #             'pose_label': pose_label, 
        #             'bounding_box': bounding_box, 
        #             'metadata': metadata 
        #         }
        #     }, 
        #     {
        #         'independent_variable': image_path_2nd, 
        #         'dependent_variable': {
        #             'keypoints': keypoints,
        #             'pose_label': pose_label, 
        #             'bounding_box': bounding_box, 
        #             'metadata': metadata 
        #         }
        #     },
        #     ... 
        # ]
        return [
            self.__predict_sample(image_path= sample['independent_variable'], ground_truth= sample['dependent_variable'])
            for sample in image_batch
        ]
        
        
        
    def visualize_batch(self, image_batch: list[dict[str, torch.tensor | dict[str, torch.tensor]]]) -> None:
        # NOTE: image_batch = [
        #     {
        #         'independent_variable': image_path_1st, 
        #         'dependent_variable': {
        #             'keypoints': keypoints,
        #             'pose_label': pose_label, 
        #             'bounding_box': bounding_box, 
        #             'metadata': metadata 
        #         }
        #     }, 
        #     {
        #         'independent_variable': image_path_2nd, 
        #         'dependent_variable': {
        #             'keypoints': keypoints,
        #             'pose_label': pose_label, 
        #             'bounding_box': bounding_box, 
        #             'metadata': metadata 
        #         }
        #     },
        #     ... 
        # ]
        for sample in image_batch:
            self.__visualize_sample(image_path= sample['independent_variable'], ground_truth= sample['dependent_variable'])

        
        
        
        


#@: Driver Code
if __name__.__contains__('__main__'):
    # > Use CASE EXAMPLE
    # Load your pre-trained pose estimation model
    # model = torch.load("path/to/your/model.pt")

    # # Define your transformation functions
    # transform_dict = {
    #     "independent_variable": {
    #         "transform1": some_function,
    #         "transform2": another_function
    #     },
    #     "dependent_variable": {
    #         "keypoints": {
    #             "transform1": some_function,
    #             "transform2": another_function
    #         },
    #         "pose_label": {
    #             "transform1": some_function,
    #             "transform2": another_function
    #         },
    #         "bounding_box": {
    #             "transform1": some_function,
    #             "transform2": another_function
    #         },
    #         "metadata": {
    #             "transform1": some_function,
    #             "transform2": another_function
    #         }
    #     }
    # }
    # NOTE : for transform_dict : Check Data PrePrecessing and Data Loadining Pipeline 

    # # Create an instance of the PoseEstimationPredictor class
    # predictor = PoseEstimationPredictor(model=model, transform_dict=transform_dict)

    # # Provide the image path and (optionally) ground truth data
    # image_path = "path/to/image.jpg"
    # ground_truth = {
    #     "keypoints": np.array([...]),
    #     "pose_label": np.array([...]),
    #     "bounding_box": np.array([...]),
    #     "metadata": np.array([...])
    # }

    # # Make predictions for a single image
    # predictions = predictor.predict_sample(image_path, ground_truth)

    # # Visualize the predictions and ground truth (if provided) for a single image
    # predictor.visualize_sample(image_path, ground_truth)

    # # Create an image_batch
    # image_batch = [
    #     {
    #         "independent_variable": "path/to/image1.jpg",
    #         "dependent_variable": {
    #             "keypoints": np.array([...]),
    #             "pose_label": np.array([...]),
    #             "bounding_box": np.array([...]),
    #             "metadata": np.array([...])
    #         }
    #     },
    #     {
    #         "independent_variable": "path/to/image2.jpg",
    #         "dependent_variable": {
    #             "keypoints": np.array([...]),
    #             "pose_label": np.array([...]),
    #             "bounding_box": np.array([...]),
    #             "metadata": np.array([...])
    #         }
    #     }
    # ]

    # # Make predictions for a batch of images
    # batch_predictions = predictor.predict_batch(image_batch)

    # # Visualize the predictions and ground truth (if provided) for a batch of images
    # predictor.visualize_batch(image_batch)
    
    
    ...