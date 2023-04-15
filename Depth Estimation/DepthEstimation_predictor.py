
from __future__ import annotations
import torch, os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


__author_info__: dict[str, Union[str, list[str]]] = {
    'Name': 'Rahul Sawhney',
    'Mail': [
        'sawhney.rahulofficial@outlook.com', 
        'rahulsawhney321@gmail.com'
    ]
}

#@: NOTE : Depth - Estimation Predictor Pipeline 

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

__getitem__return__: str = r'''
    __getitem__(index: int) -> dict[str, torch.tensor]:
        {
            'independent_variable': image, 
            'dependent_variable': depth_map
        }
'''

__transform_dict_doc__: str = r'''
    transform_dict: dict[str, dict[str, Callable[..., Any]]] = {
        'independent_variable': {
            'transform_1': Image_transform_1,
            'transform_2': Image_transform_2,
            ...
        }
        'dependent_variable': {
            'transform_1': DepthMap_transform_1,
            'transform_2': DepthMap_transform_2,
            ...
        }
    }
'''


class DepthEstimationPredictor:
    def __init__(self, model: torch.nn.Module, transform_dict: dict[str, dict[str, Callable[..., Any]]], 
                                               device: torch.device = 'cuda' if torch.cuda.is_available() else 'cpu') -> None:
        
        # This class, DepthEstimationPredictor, is designed to perform depth estimation using a given model, which is typically a deep learning model implemented 
        # using the PyTorch framework. Depth estimation is a computer vision task that involves predicting the depth values (i.e., distances) of objects in a scene 
        # from a single image or a series of images. This class takes the following parameters:

        #     *   model (torch.nn.Module): The input model is an instance of a PyTorch neural network (nn.Module) that has been trained to perform depth estimation. 
        #                                  The model is responsible for making depth predictions when provided with an input image.

        #     *   transform_dict (dict[str, dict[str, Callable[..., Any]]]): A dictionary containing a set of data preprocessing and transformation functions that 
        #                                                                    need to be applied to the input data (image or images) before feeding them into the model. 
        #                                                                    These transformations may include resizing, normalization, and data augmentation, among others. 
        #                                                                    The dictionary keys represent the types of transforms, while the values are dictionaries 
        #                                                                    containing the actual transformation functions.

        #     *   device (torch.device, optional): Specifies the device on which to run the model, either 'cuda' for GPU or 'cpu' for CPU. By default, the code checks if 
        #                                          a GPU is available and uses it if possible; otherwise, it defaults to the CPU.

        
        # The __init__() method of the class initializes the class attributes with the provided parameters, making them available for use within other methods of the class.
                
        self.model = model
        self.transform_dict = transform_dict
        self.device = device





    def __transform_sample(self, data: Any, transforms: dict[str, Callable[..., Any]]) -> Any:
        
        # The __transform_sample method is a private method of the DepthEstimationPredictor class, which applies a series of transformations to the input data. This method 
        # is intended to preprocess and transform the input data, typically an image or a set of images, before feeding it into the depth estimation model.

        # The method takes two parameters:

        #         *   data (Any): The input data to be transformed. This could be an image or any other data type that is compatible with the transformation functions specified 
        #                         in the transforms dictionary.

        #         *   transforms (dict[str, Callable[..., Any]]): A dictionary containing transformation functions that need to be applied to the input data. The dictionary 
        #                                                         keys represent the types of transforms, while the values are the actual transformation functions (callables).

        
        # The method iterates through the values of the transforms dictionary, which are the transformation functions, and applies each of them in sequence to the input data. 
        # The transformed data is then returned.
        
        for transform_func in transforms.values():
            data = transform_func(data)
        return data





    def __predict_sample(self, sample: dict[str, torch.tensor]) -> np.ndarray:
        
        # The __predict_sample method is a private method of the DepthEstimationPredictor class, which performs depth estimation for a single input sample using the provided model. 
        # This method is intended to be used internally by the class to make depth predictions for individual samples.

        # The method takes one parameter:

        #     *   sample (dict[str, torch.tensor]): A dictionary containing a single sample's independent variable (input image) as a PyTorch tensor. The key is 'independent_variable'
        #                                           and the value is the corresponding tensor.

        # The method performs the following steps:

        #     1)  Retrieve the input tensor from the sample dictionary and add an extra dimension using unsqueeze(0). This is done to create a batch with a single sample, as 
        #         deep learning models typically expect input data in batches. Then, move the input tensor to the specified device (GPU or CPU) using the to(self.device) method.

        #     2)  Pass the input tensor through the depth estimation model using self.model(inputs) to obtain the output tensor, which represents the predicted depth map.

        #     3)  Convert the output tensor to a NumPy array using outputs.cpu().numpy(). Before the conversion, the tensor is moved back to the CPU memory using the cpu() method.

        #     4)  Return the depth map as a NumPy array.
        
        inputs: torch.tensor = sample['independent_variable'].unsqueeze(0).to(self.device)
        outputs: torch.tensor = self.model(inputs)
        depth_map: np.ndarray = outputs.cpu().numpy()
        return depth_map





    def __visualize_sample(self, sample: dict[str, torch.tensor]) -> None:
        
        # The __visualize_sample() method is a private method of the DepthEstimationPredictor class, which visualizes the input image, the predicted depth map, and the ground truth depth 
        # map (if available) using Matplotlib. This method is useful for visually assessing the performance of the depth estimation model by comparing the predicted depth maps with the 
        # ground truth depth maps.

        # The method takes one parameter:

        #     *   sample (dict[str, torch.tensor]): A dictionary containing a single sample's independent variable (input image) and, optionally, the dependent variable 
        #                                           (ground truth depth map). Both are represented as PyTorch tensors or NumPy arrays.

        # The method performs the following steps:

        #     1)  Retrieve the input image tensor from the sample dictionary and convert it to a NumPy array using numpy().

        #     2)  Obtain the predicted depth map by calling the __predict_sample method with the sample parameter.

        #     3)  Check if the ground truth depth map is available in the sample dictionary using sample.get('dependent_variable'). If it is, store it in gt_depth_map.

        #     4)  If the ground truth depth map is available, create a figure with three subplots (ax1, ax2, and ax3) using plt.subplots(1, 3). Otherwise, create a figure with two 
        #         subplots (ax1 and ax2) using plt.subplots(1, 2).

        #     5)  Display the input image on the first subplot (ax1) and set the title to 'Input Image'.

        #     6)  Display the predicted depth map on the second subplot (ax2), using the 'viridis' colormap, and set the title to 'Predicted Depth Map'.

        #     7)  If the ground truth depth map is available, display it on the third subplot (ax3), using the 'viridis' colormap, and set the title to 'Ground Truth Depth Map'.

        #     8)  Finally, display the figure using plt.show().


        # This method can be used to visually inspect the performance of the depth estimation model by comparing the predicted depth maps with the ground truth depth maps when 
        # available.
        
        image: torch.tensor = sample['independent_variable'].numpy()
        predictions: np.ndarray = self.__predict_sample(sample)
        gt_depth_map: torch.tensor | np.ndarray = sample.get('dependent_variable')


        if gt_depth_map is not None:
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
            ax3.imshow(gt_depth_map, cmap= 'viridis')
            ax3.axis('off')
            ax3.set_title('Ground Truth Depth Map')
            
        else:
            fig, (ax1, ax2) = plt.subplots(1, 2)

        ax1.imshow(image)
        ax1.axis('off')
        ax1.set_title('Input Image')

        ax2.imshow(predictions, cmap= 'viridis')
        ax2.axis('off')
        ax2.set_title('Predicted Depth Map')

        plt.show()





    def predict_sample(self, sample: dict[str, torch.tensor]) -> np.ndarray:
        
        # The predict_sample method is a public method of the DepthEstimationPredictor class, which allows users to make depth predictions for individual input samples using the provided 
        # depth estimation model. This method serves as an interface for external code to make predictions without directly accessing the private __predict_sample() method.

        # The method takes one parameter:

        #     *   sample (dict[str, torch.tensor]): A dictionary containing a single sample's independent variable (input image) as a PyTorch tensor. The key is 'independent_variable' 
        #                                           and the value is the corresponding tensor.

        # The method simply calls the private __predict_sample() method with the sample parameter and returns the resulting depth map as a NumPy array. This method can be used by 
        # external code to make depth predictions for individual input images after they have been preprocessed and transformed as needed.
        
        return self.__predict_sample(sample)






    def visualize_sample(self, sample: dict[str, torch.tensor]) -> None:
        
        # The visualize_sample method is a public method of the DepthEstimationPredictor class, which allows users to visualize the input image, the predicted depth map, and the ground 
        # truth depth map (if available) using Matplotlib. This method serves as an interface for external code to visualize the depth predictions without directly accessing the
        # private __visualize_sample() method.

        # The method takes one parameter:

        #     *   sample (dict[str, torch.tensor]): A dictionary containing a single sample's independent variable (input image) and, optionally, the dependent variable 
        #                                           (ground truth depth map). Both are represented as PyTorch tensors or NumPy arrays.

        # The method simply calls the private __visualize_sample() method with the sample parameter and displays the resulting plot. This method can be used by external code to 
        # visually assess the performance of the depth estimation model by comparing the predicted depth maps with the ground truth depth maps when available.
        
        return self.__visualize_sample(sample)





    def predict_batch(self, batch: list[dict[str, torch.tensor]]) -> list[np.ndarray]:
        
        # The predict_batch method is a public method of the DepthEstimationPredictor class, which allows users to make depth predictions for a batch of input samples using the provided 
        # depth estimation model. This method is useful when making predictions for a large number of input images.

        # The method takes one parameter:

        #     *   batch (list[dict[str, torch.tensor]]): A list of dictionaries, where each dictionary contains a single sample's independent variable (input image) as a PyTorch tensor. 
        #                                                The key is 'independent_variable' and the value is the corresponding tensor.

        
        # The method uses a list comprehension to iterate through the samples in the batch list and call the private __predict_sample() method for each sample. The resulting depth maps are 
        # then collected into a list of NumPy arrays, which is returned. This method can be used by external code to make depth predictions for a batch of input images after they have been 
        # preprocessed and transformed as needed.
        
        return [
            self.__predict_sample(sample) for sample in batch
        ]





    def visualize_batch(self, batch: list[dict[str, torch.tensor]]) -> None:
        
        # The visualize_batch method is a public method of the DepthEstimationPredictor class, which allows users to visualize the input images, the predicted depth maps, and the ground truth 
        # depth maps (if available) for a batch of input samples using Matplotlib. This method is useful for visually assessing the performance of the depth estimation model on a set of input 
        # images.

        # The method takes one parameter:

        #     *   batch (list[dict[str, torch.tensor]]): A list of dictionaries, where each dictionary contains a single sample's independent variable (input image) and, optionally, the 
        #                                                dependent variable (ground truth depth map). Both are represented as PyTorch tensors or NumPy arrays.

        
        # The method iterates through the samples in the batch list and calls the private __visualize_sample() method for each sample, displaying the resulting plot. This method can be used by
        # external code to visually assess the performance of the depth estimation model on a set of input images by comparing the predicted depth maps with the ground truth depth maps when 
        # available.
        
        for sample in batch:
            self.__visualize_sample(sample)

          

    
    
    def save_predictions(self, batch: list[dict[str, torch.tensor]], output_dir: str, type_format: Optional[str] = 'npy') -> None:
        
        # The save_predictions method is a public method of the DepthEstimationPredictor class, which allows users to save the predicted depth maps for a batch of input samples to a specified 
        # output directory. The predicted depth maps can be saved in either NumPy or PNG format.

        # The method takes three parameters:

        #     *   batch (list[dict[str, torch.tensor]]): A list of dictionaries, where each dictionary contains a single sample's independent variable (input image) as a PyTorch tensor. 
        #                                                The key is 'independent_variable' and the value is the corresponding tensor.

        #     *   output_dir (str): The path to the directory where the predicted depth maps will be saved.

        #     *   type_format (str, optional): The type of format to use when saving the predicted depth maps, either 'npy' for NumPy or 'png' for PNG. Defaults to 'npy'.
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for idx, sample in enumerate(batch):
            prediction: np.ndarray = self.predict_sample(sample)
            if type_format == 'npy':
                np.save(os.path.join(output_dir, f'prediction_{idx}.npy'), prediction)
            
            elif type_format == 'png':
                plt.imsave(os.path.join(output_dir, f'prediction_{idx}.png'), prediction, cmap= 'viridis')
            
            else:
                raise ValueError(f"Unsupported format: {type_format}")
        
        
        

    
    
    def compare_samples(self, batch: list[dict[str, torch.tensor]], output_dir: Optional[str] = None) -> None:
        
        # The compare_samples method is a public method of the DepthEstimationPredictor class, which allows users to compare the input images, predicted depth maps, and ground truth depth maps
        # (if available) for a batch of input samples using Matplotlib. This method is useful for visually assessing the performance of the depth estimation model by comparing the predicted depth 
        # maps with the ground truth depth maps when available.

        # The method takes two parameters:

        #     *   batch (list[dict[str, torch.tensor]]): A list of dictionaries, where each dictionary contains a single sample's independent variable (input image) and, optionally, 
        #                                                the dependent variable (ground truth depth map). Both are represented as PyTorch tensors or NumPy arrays.

        #     *   output_dir (str, optional): The path to the directory where the comparison plots will be saved. If not specified, the plots will be displayed on the screen instead 
        #                                     of being saved.
        
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for idx, sample in enumerate(batch):
            self.visualize_sample(sample)
            
            if output_dir:
                plt.savefig(os.path.join(output_dir, f'comparison_{idx}.png'))




#@: Driver Code 
if __name__.__contains__('__main__'):
    #@: NOTE: Example Use Case NOTE
    # Create an instance of the DepthEstimationPredictor class
    # model = MyDepthEstimationModel()
    # transform_dict = {...}
    # predictor = DepthEstimationPredictor(model, transform_dict)

    # # Load the input data
    # dataset = MyDataset(...)
    # dataloader = DataLoader(dataset, batch_size=4)

    # # Make predictions for a batch of input samples
    # batch = next(iter(dataloader))
    # predictions = predictor.predict_batch(batch)

    # # Visualize the predictions for a single sample
    # sample = dataset[0]
    # predictor.visualize_sample(sample)

    # # Save the predictions to a file
    # output_dir = 'predictions'
    # predictor.save_predictions(batch, output_dir, type_format='npy')

    # # Compare the predictions with the ground truth (if available)
    # predictor.compare_samples(batch, output_dir='comparisons')
    ...
    