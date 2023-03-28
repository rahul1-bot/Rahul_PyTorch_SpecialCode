
from __future__ import annotations
import os, torch, torchvision, shutil
from PIL import Image
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


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
    *   Image segmentation is a computer vision problem where the task is to partition an image into multiple segments, each of which represents a different object or region of the image. 
        The goal is to accurately identify and separate the objects or regions in the image. Image segmentation is a fundamental step in many computer vision tasks such as object detection, tracking, 
        and recognition.

    *   The main challenge in image segmentation is to separate the objects or regions of interest from the background in a given image. There are various techniques and algorithms used for image 
        segmentation, including thresholding, edge detection, region growing, clustering, and deep learning-based approaches such as convolutional neural networks (CNNs).

    *   CNNs have become the state-of-the-art method for image segmentation, achieving impressive results on various benchmarks. These models are trained on large datasets of annotated images and 
        can learn to identify and segment different objects or regions in images based on their features.

    *   Image segmentation has numerous applications in various fields such as medical imaging, robotics, surveillance, and autonomous driving. For example, in medical imaging, image segmentation 
        is used for identifying and segmenting different organs or tissues in the body, which is important for diagnosis and treatment planning. In robotics, image segmentation can be used for object 
        recognition and localization, which is important for robotic manipulation and control. In surveillance, image segmentation can be used for tracking and identifying objects or people in the scene.
        And in autonomous driving, image segmentation can be used for obstacle detection and avoidance, which is crucial for safe driving.
'''


#@: Image Segmentation Data Loading Pipeline 
__data_loading_pipeline_doc__: str = r'''
    The ImageSegmentationDataset class is a PyTorch dataset for image segmentation tasks. The class has several methods and attributes, some of which include:

        *   __init__(self, image_directory: str, mask_directory: str, transform_dict: Optional[dict[str, dict[str, torch.tensor]]] = None): initializes the dataset by creating a DataFrame 
            that holds the paths to the images and masks, and the optional transform dictionary.

        *   __len__(self) -> int: returns the number of samples in the dataset.

        *   __repr__(self) -> str: returns a string representation of the class instance.

        *   prepare_dataset(image_directory: str, mask_directory: str): a static method that creates a pandas DataFrame with the paths to the images and masks.

        *   __getitem__(self, index: int) -> dict[str, torch.tensor]: returns a dictionary with the image and mask tensors for a given index.

        *   visualize_sample(self, index: Optional[int] = None) -> None: displays a randomly selected sample from the dataset.

        *   split_dataset(self, train_ratio: float, valid_ratio: float) -> dict[str, ImageSegmentationDataset]: splits the dataset into training, validation, and testing datasets based on the provided ratios.

        *   save_dataset(self, output_directory: str) -> None: saves the images and masks to the provided output directory.

        *   load_dataset(cls, input_directory: str, transform_dict: Optional[dict[str, dict[str, Callable[Any]]]] = None) -> ImageSegmentationDataset: loads a previously saved dataset from the provided 
            input directory.

'''

class ImageSegmentationDataset(torch.utils.data.Dataset):
    def __init__(self, image_directory: str, mask_directory: str, 
                                             transform_dict: Optional[dict[str, dict[str, torch.tensor]]] = None) -> None:
        
        self.data_dict: pd.DataFrame = ImageSegmentationDataset.prepare_dataset(
                                        image_directory= image_directory, 
                                        mask_directory= mask_directory
                                    )
        
        self.transform_dict = transform_dict
        
        
        

    def __len__(self) -> int:
        return len(self.data_dict)
    
    
    
    
    def __repr__(self) -> str(dict[str, Any]):
        return str({
            x: y for x, y in zip(['Module', 'Name', 'ObjectID'], [self.__module__, type(self).__name__, hex(id(self))])
        })
        
    
    
    __prepare_dataset_doc: str = r'''
        *   This is a static method in the ImageSegmentationDataset class that takes the paths to the directory containing the input images and the directory containing the corresponding masks.

        *   The method first creates a list of image filenames and a list of mask filenames using the os.listdir() method. It then generates the full path of each image and mask file using 
            the os.path.join() method, and stores these paths in separate lists.

        *   Finally, the method creates a dictionary data_dict with the image and mask paths, and converts this dictionary to a pandas DataFrame, which is then returned as the output of 
            the method. This DataFrame contains two columns: image_path and mask_path, where each row represents the paths of an image and its corresponding mask.
    
    '''
    @staticmethod
    def prepare_dataset(image_directory: str, data_directory: str) -> pd.DataFrame:
        image_filenames: list[str] = [
            item for item in os.listdir(image_directory)
        ]
        
        mask_filenames: list[str] = [
            item for item in os.listdir(mask_directory)
        ]
        
        image_paths: list[str] = [
            os.path.join(image_directory, filename) for filename in image_filenames
        ]
        
        mask_paths: list[str] = [
            os.path.join(mask_directory, filename) for filename in mask_filenames
        ]
        
        data_dict: dict[str, list[str]] = {
            'image_path': image_paths, 'mask_path': mask_paths   
        }
        data_dict: pd.DataFrame = pd.DataFrame(data_dict)
        return data_dict
    
    
    
    __getitem_doc: str = r'''
        *   This is the __getitem__ method of the ImageSegmentationDataset class. It returns a dictionary that contains a sample from the dataset at the given index. The sample includes an 
            independent variable image and a dependent variable mask.

        *   The method gets the file paths of the image and mask from the data_dict using the given index. It then opens the image and mask files using the PIL library and converts the image 
            to RGB format and mask to grayscale format.

        *   If the transform_dict is not None, it applies the image and mask transforms specified in the dictionary. It loops over each transform function in the image and mask keys of the 
            dictionary and applies them to the image and mask, respectively.

        *   Finally, it returns a dictionary containing the transformed image and mask tensors, with keys 'independent_variable' and 'dependent_variable', respectively.
    '''
    def __getitem__(self, index: int) -> dict[str, torch.tensor]:
        image_path: str = self.data_dict.iloc[index]['image_path']
        mask_path: str = self.data_dict.iloc[index]['mask_path']
        
        image: PIL.Image = Image.open(image_path).convert('RGB')
        mask: PIL.Image = Image.open(mask_path).convert('L')
        
        
        if self.transform_dict is not None:
            if 'image' in self.transform_dict:
                for transform_func in self.transform_dict['image'].values():
                    image: Any = transform_func(image)
            
            if 'mask' in self.transform_dict:
                for transform_func in self.transform_dict['mask'].values():
                    mask: Any = transform_func(mask)
                    
        return {
            'independent_variable': image,
            'dependent_variable': mask
        }
                
    
    
    __visualize_sample_doc: str = r'''
       *    The visualize_sample method is used to display a sample image and its corresponding segmentation mask from the dataset.

       *    It takes an optional integer argument index, which is used to select the index of the sample to be displayed. If index is not provided, a random index is selected using np.random.randint().

       *    The method then gets the image and mask tensors of the selected sample using the __getitem__() method. It then creates a figure with two subplots using plt.subplots() and displays 
            the image and mask in each subplot using ax1.imshow() and ax2.imshow(), respectively. The title of the subplots is set using ax1.set_title() and ax2.set_title(). Finally, the plot is 
            displayed using plt.show().
    '''
    def visualize_sample(self, index: Optional[int] = None) -> None:
        if index is None:
            index: int = np.random.randint(0, len(self.data_dict))
        
        sample: dict[str, torch.tensor] = self.__getitem__(index)
        image: torch.tensor = sample['independent_variable']
        mask: torch.tensor = sample['dependent_variable']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        ax1.imshow(image)
        ax1.set_title('Image')
        ax2.imshow(mask, cmap= 'gray')
        ax2.set_title('Segmentation Mask')
        plt.show()
        
       
       
        
    __split_dataset_doc: str = r'''
       *    The split_dataset method is used to split the data_dict into train, validation, and test sets. It takes train_ratio and valid_ratio as inputs, which represent the proportion of the data that 
            should be allocated to the train and validation sets, respectively. The test set is the remaining data after the train and validation sets have been allocated.

       *    The method uses the train_test_split function from the sklearn library to split the data into train, validation, and test sets. The train_size parameter of the function is set to train_ratio 
            to allocate the proportion of the data to the train set. The valid_ratio parameter is used to calculate the proportion of the data to be allocated to the validation set. The random_state 
            parameter is set to 42 for reproducibility, and the shuffle parameter is set to True to shuffle the data before splitting.

       *    Three new instances of the ImageSegmentationDataset class are created to hold the train, validation, and test sets. The data_dict attribute of each new instance is updated with the corresponding
            split of the original data_dict.

       *    The method returns a dictionary containing the train, validation, and test sets.
    '''
    def spilt_dataset(self, train_ratio: float, valid_ratio: float) -> dict[str, ImageSegmentationDataset]:
        assert 0 < train_ratio < 1 and 0 < valid_ratio < 1 and train_ratio + valid_ratio < 1, 'Invalid Split Ratio'
        
        train_data, temp_data = train_test_split(
            self.data_dict, 
            train_size= train_ratio, 
            random_state= 42, 
            shuffle= True
        )
        
        valid_data, test_data = train_test_split(
            temp_data, 
            train_size= valid_ratio / (1 - train_ratio),
            random_state= 42, 
            shuffle= True
        )
        
        train_dataset: ImageSegmentationDataset = ImageSegmentationDataset(
            image_directory= self.image_directory,
            mask_directory= self.mask_directory,
            transform_dict= self.transform_dict
        )
        
        valid_dataset: ImageSegmentationDataset = ImageSegmentationDataset(
            image_directory= self.image_directory,
            mask_directory= self.mask_directory,
            transform_dict= self.transform_dict
        )
        
        test_dataset: ImageSegmentationDataset = ImageSegmentationDataset(
            image_directory= self.image_directory,
            mask_directory= self.mask_directory,
            transform_dict= self.transform_dict
        )
        
        #@: updating data_dict for each dataset
        train_dataset.data_dict: pd.DataFrame = train_data
        valid_dataset.data_dict: pd.DataFrame = valid_data
        test_dataset.data_dict: pd.DataFrame = test_data
        
        return {
            'train': train_dataset,
            'valid': valid_dataset,
            'test': test_dataset
        } 
    
    
    
    
    __save_dataset_doc: str = r'''
        *   The save_dataset method saves the entire dataset to disk. The method creates two subdirectories within the specified output_directory: one for the images and one for the corresponding segmentation 
            masks. It then saves each image and mask to the corresponding subdirectory using the PIL.Image.save method. The filenames are created using a loop counter, such that the nth image and mask are saved 
            with filenames image_n.png and mask_n.png, respectively.
    
    '''
    def save_dataset(self, output_directory: str) -> None:
        output_path: Any = Path(output_directory)
        output_path.mkdir(parents= True, exist_ok= True)
        
        #@: creating sub_directories for images and masks
        image_directory: str = output_path / 'images'
        mask_directory: str = output_path / 'masks'
        image_directory.mkdir(parents= True, exist_ok= True)
        mask_directory.mkdir(parents= True, exist_ok= True)
        
        #@: saving each image and mask 
        for idx in range(len(self.data_dict)):
            sample: dict[str, torch.tensor] = self.__getitem__(idx)
            image: torch.tensor = sample["independent_variable"]
            mask: torch.tensor = sample["dependent_variable"]
            image.save(image_directory / f"image_{idx}.png")
            mask.save(mask_directory / f"mask_{idx}.png")
            
    
    
    
    __load_dataset_doc: str = r'''
        *   This is a class method named load_dataset that is used to load an already saved dataset. It takes the directory where the dataset was saved and an optional dictionary of transformation functions.

        *   First, it creates a Path object for the input directory, then lists all the image and mask files in the directory using the glob() method. It then creates the images and masks directories in 
            the input directory using os.makedirs() function, with exist_ok=True to prevent errors if the directories already exist.

        *   Next, it iterates over the image and mask files, copying them to their corresponding directories in the input directory using shutil.copy() method.

        *   Finally, it returns a new instance of the ImageSegmentationDataset class, initialized with the image and mask directories and the optional transform dictionary.
    
    '''
    @classmethod
    def load_dataset(cls, input_directory: str, transform_dict: Optional[dict[str, dict[str, Callable[Any]]]] = None) -> ImageSegmentationDataset:
        input_path: str = Path(input_directory)
        image_paths: list[str] = sorted(input_path.glob('image_*.png'))
        mask_paths: list[str] = sorted(input_path.glob('mask_*.png'))
        
        image_directory: str = str(input_path / 'images')
        mask_directory: str = str(input_path / 'masks')
        
        os.makedirs(image_directory, exist_ok= True)
        os.makedirs(mask_directory, exist_ok= True)
        
        #@: copying the images and masks to the corresponding directories 
        for idx, (image_path, mask_path) in enumerate(zip(image_paths, mask_paths)):
            shutil.copy(image_path, f'{image_directory}/image_{idx}.png')
            shutil.copy(mask_path, f'{mask_directory}/mask_{idx}.png')
            
        return cls(
            image_directory= image_directory,
            mask_directory= mask_directory,
            transform_dict= transform_dict
        )
        
        


#@: Driver Code 
if __name__.__contains__('__main__'):
    #@: use case 
    transform_dict: dict[str, dict[str, torch.tensor]] = {
        'image': {
            'transform_one': ..., 
            'transform_two': ...,
            'transform_n': ...
        }, 
        
        'mask': {
            'transform_one': ...,
            'transform_two': ...,
            'transform_n': ...
        }
    }
    
    #> . 
    
    
    
    
    
    