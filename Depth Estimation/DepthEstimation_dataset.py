
from __future__ import annotations
import torch, os, pickle
from torch.utils.data import random_split, DataLoader
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt


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

# Here are some additional methods that you can consider adding to the DepthEstimationDataset class:

#         *   get_image_size(): This method can be used to return the size (width and height) of the images in the dataset.

#         *   get_depth_range(): This method can be used to return the minimum and maximum values of the depth maps in the dataset.

#         *   get_image_paths(): This method can be used to return a list of the paths to all the images in the dataset.

#         *   get_depth_paths(): This method can be used to return a list of the paths to all the depth maps in the dataset.

#         *   filter_data(): This method can be used to filter the dataset based on specific criteria, such as image size, depth range, or class labels.

#         *   shuffle_data(): This method can be used to randomly shuffle the dataset.

#         *   k_fold_cross_validation(): This method can be used to perform k-fold cross-validation on the dataset, where the dataset is split into k equal-sized subsets, 
#                                        and each subset is used as a validation set once while the rest are used as the training set.

#         *   balance_classes(): This method can be used to balance the number of samples in each class of the dataset by oversampling or undersampling.

#         *   plot_distribution(): This method can be used to plot the distribution of the data in the dataset, such as the distribution of image sizes or depth ranges.

#         *   get_statistics(): This method can be used to compute statistics about the dataset, such as the mean and standard deviation of the images or depth maps.


#@: Depth-Estimation Dataset Class
class DepthEstimationDataset(torch.utils.data.Dataset):
    def __init__(self, image_directory: str, depth_directory: str, 
                                             transform_dict: Optional[dict[str, dict[str, Callable[..., Any]]]] = None) -> None:
        
        # This code defines a class named DepthEstimationDataset that inherits from the torch.utils.data.Dataset class.

        # The constructor method __init__() takes in three arguments:

        #     *   image_directory: A string that represents the directory where the RGB images are stored.
            
        #     *   depth_directory: A string that represents the directory where the depth images are stored.

        #     *   transform_dict: An optional dictionary that contains the transformations to be applied to the independent and dependent variables.

        # Inside the __init__() method, prepare_dataset() method is called to create a Pandas dataframe that stores the paths of all the image and depth files in 
        # the respective directories. This dataframe is stored in the class variable self.data_dict.

        # The transform_dict argument is also stored in the instance variable self.transform_dict. This is useful for applying data augmentations or preprocessing to the
        # images and depth maps. If transform_dict is None, no transformations are applied to the data.
        
        
        self.data_dict: pd.DataFrame = DepthEstimationDataset.prepare_dataset(
            image_directory= image_directory,
            depth_directory= depth_directory
        )
        
        self.transform_dict = transform_dict
        
        
        
        
    
    def __len__(self) -> int:
        # This method returns the length of the dataset, which is the number of samples in the dataset. It uses the len() function on the data_dict attribute of the 
        # dataset object, which is a Pandas DataFrame that contains the paths to the images and their corresponding depth maps. The len() function returns the number of rows
        # in the DataFrame, which is equal to the number of samples in the dataset.
        return len(self.data_dict)




    def __repr__(self) -> str:
        return str({
            x: y for x, y in zip(['Module', 'Name', 'ObjectID'], [self.__module__, type(self).__name__, hex(id(self))])
        })
        
    
    
    
    
    @staticmethod
    def prepare_dataset(image_directory: str, depth_directory: str) -> pd.DataFrame:
        
        # This is a static method named prepare_dataset() which prepares the dataset by creating a pandas DataFrame consisting of two columns, "image_path" and "depth_path". It 
        # takes two arguments: image_directory and depth_directory which are the paths to the directories containing the image and depth files respectively.

        # Inside the method, os.listdir() is used to get a list of filenames in the given directories. The os.path.isfile() function is used to filter out only the file names, 
        # and os.path.join() is used to join the directory path and file name together to create the full path to each file. The resulting list of file paths is then stored in two 
        # separate variables, image_paths and depth_paths.

        # Finally, the two lists of file paths are combined into a dictionary with keys "image_path" and "depth_path". This dictionary is then converted to a pandas DataFrame and returned.
        
        image_paths: list[str] = [
            os.path.join(image_directory, file_name) for file_name in os.listdir(image_directory) if os.path.isfile(os.path.join(image_directory, file_name)) 
        ]
        
        depth_paths: list[str] = [
            os.path.join(depth_directory, file_name) for file_name in os.listdir(depth_directory) if os.path.isfile(os.path.join(depth_directory, file_name))
        ]
        
        data_dict: dict[str, list[str]] = {
            'image_path': image_paths, 'depth_path': depth_paths 
        }
        
        data_dict: pd.DataFrame = pd.DataFrame(data_dict)
        return data_dict
    
    
    
    
    
    
    def __getitem__(self, index: int) -> dict[str, torch.tensor]:
        
        # This method returns the sample of the dataset specified by the given index. It loads the corresponding RGB image and depth map paths from the data_dict attribute, opens and converts 
        # the image to RGB using the PIL library and loads the depth map using numpy. It then applies any specified transformations to the image and depth map using the transform_dict attribute, 
        # and returns a dictionary containing the transformed image and depth map tensors as the 'independent_variable' and 'dependent_variable' keys, respectively.
        
        image_path: str = self.data_dict.iloc[index]['image_path']
        depth_path: str = self.data_dict.iloc[index]['depth_path']
        
        image: PIL.Image = Image.open(image_path).convert('RGB')
        depth: np.ndarray = np.load(depth_path)
        
        
        if self.transform_dict is not None:
            if 'independent_variable' in self.transform_dict:
                for transform_func in self.transform_dict['independent_variable'].values():
                    image: Any = transform_func(image)
                
            if 'dependent_variable' in self.transform_dict:
                for transform_func in self.transform_dict['dependent_variable'].values():
                    depth: Any = transform_func(depth)
                    
        return {
            'independent_variable': image,
            'dependent_variable': depth
        }
        
    #########################################################
    # import os
    # import pandas as pd
    # from concurrent.futures import ThreadPoolExecutor
    
    # @staticmethod
    # def get_file_paths(directory: str) -> list[str]:
    #     paths = []
    #     for file_name in os.listdir(directory):
    #         full_path = os.path.join(directory, file_name)
    #         if os.path.isfile(full_path):
    #             paths.append(full_path)
    #     return paths


    # @staticmethod
    # def prepare_dataset(image_directory: str, depth_directory: str) -> pd.DataFrame:
    #     # 12 threads 
    #     with ThreadPoolExecutor(max_workers=12) as executor:
    #         image_paths, depth_paths = executor.map(get_file_paths, [image_directory, depth_directory])
        
    #      data_dict = {
    #         'image_path': image_paths,
    #         'depth_path': depth_paths
    #     }

    #     return pd.DataFrame(data_dict)
    
    #########################################################
    
    def visualize_sample(self, index: int) -> None:
        
        # This method takes an index as input and displays the corresponding sample's RGB image and its depth image. It retrieves the sample using the index and then extracts 
        # the 'independent_variable' and 'dependent_variable' tensors, which are then plotted using Matplotlib. The RGB image is plotted in the left subplot and the depth image 
        # is plotted in the right subplot. Finally, the method calls plt.show() to display the plot.
        
        sample: Dict[str, torch.Tensor] = self[index]
        image: torch.Tensor = sample['independent_variable']
        depth: torch.Tensor = sample['dependent_variable']

        plt.subplot(1, 2, 1)
        plt.imshow(image.permute(1, 2, 0))
        plt.title('RGB Image')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(depth, cmap= 'gray')
        plt.title('Depth Image')
        plt.axis('off')

        plt.show()
        
    
    
    
    
    def split_dataset(self, train_ratio: float, test_ratio: float, val_ratio: float, seed: Optional[int] = 42) -> dict[str, DepthEstimationDataset]:
        
        # This method splits the dataset into training, validation, and test sets according to the given ratios. The method takes four arguments:

        #     *   train_ratio: A float indicating the proportion of the dataset to include in the training set.
            
        #     *   test_ratio: A float indicating the proportion of the dataset to include in the test set.
            
        #     *   val_ratio: A float indicating the proportion of the dataset to include in the validation set.
        
        #     *   seed: An optional integer value used to initialize the random number generator for reproducibility.
        
        # The method first checks if the sum of ratios equals 1. It then calculates the number of samples for each split by multiplying the total number of samples by 
        # the respective ratio. The method sets the seed for the random number generator for reproducibility and then uses the random_split() function from PyTorch to split
        # the dataset into training, validation, and test sets. Finally, the method returns a dictionary with the keys "train", "val", and "test", each containing the 
        # corresponding split dataset.
        
        assert train_ratio + test_ratio + val_ratio == 1, 'The sum of the ratios must be equal to 1.'
        total_samples: int = len(self)
        
        #@: Calculate the number of samples for each split
        train_samples: int = int(total_samples * train_ratio)
        test_samples: int = int(total_samples * test_ratio)
        val_samples: int = total_samples - train_samples - test_samples
        
        #@: Setting the random seed for reproducibility
        torch.manual_seed(seed)
        
        #@: Splitting the dataset into training, validation, and test sets
        train_set, val_set, test_set = random_split(self, [train_samples, val_samples, test_samples])
        
        return {
            'train': train_set,
            'val': val_set,
            'test': test_set
        }
        
        
    
    
    def save_dataset(self, filename: str) -> None:
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
            
    
    
    
    @staticmethod
    def load_dataset(filename: str) -> DepthEstimationDataset:
        with open(filename, 'rb') as f:
            dataset = pickle.load(f)
        
        if not isinstance(dataset, DepthEstimationDataset):
            raise TypeError(f'Invalid dataset type. Expected DepthEstimationDataset, got {type(dataset)}.')
        
        return dataset
    
        
        
        
        
#@: Driver Code  
if __name__.__contains__('__main__'):
    # > NOTE: Use Case Example 
    
    # # Define the paths to the image and depth folders
    # image_dir: str = 'path/to/image/folder'
    # depth_dir: str = 'path/to/depth/folder'

    # # Create an instance of the DepthEstimationDataset
    # dataset: DepthEstimationDataset = DepthEstimationDataset(
    #     image_directory= image_dir, 
    #     depth_directory= depth_dir,
    #     transform_dict= None
    # )

    # # Visualize a sample
    # dataset.visualize_sample(4)

    # # Split the dataset
    # train_ratio = 0.8
    # test_ratio = 0.1
    # val_ratio = 0.1
    # seed = 42
    # dataset_splits = dataset.split_dataset(train_ratio=train_ratio, test_ratio=test_ratio, val_ratio=val_ratio, seed=seed)

    # # Save the dataset
    # dataset.save_dataset('my_dataset.pkl')

    # # Load the dataset
    # loaded_dataset: DepthEstimationDataset = DepthEstimationDataset.load_dataset('my_dataset.pkl')

    
    # # Create data loaders for each set
    # batch_size: int = 32

    # train_loader = DataLoader(data_splits['train'], batch_size= batch_size, shuffle= True)
    # test_loader = DataLoader(data_splits['test'], batch_size= batch_size, shuffle= False)
    # val_loader = DataLoader(data_splits['val'], batch_size= batch_size, shuffle= False)
    ...
