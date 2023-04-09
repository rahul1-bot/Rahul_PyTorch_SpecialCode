from __future__ import annotations
import os, shutil, json, torch
import pandas as pd
from PIL import Image
from torchvision.transforms import ToTensor
from torch.utils.data import random_split


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
    
    
__doc__: str = r'''
    The PoseEstimationDataset class is a custom dataset class designed for pose estimation tasks. It inherits from torch.utils.data.Dataset to enable seamless integration with PyTorch's data loading 
    and processing utilities. The class reads image and annotation data, applies specified transformations, and provides access to the data for training, validation, and testing.

    Here's a brief explanation of the methods and attributes within the class:

        *   __init__(self, dataset_dir, annotation_file, transform_dict): The constructor initializes the dataset object. It takes the dataset directory, annotation file name, and an optional transform 
                                                                          dictionary as arguments. The method loads the annotations and sets the image directory and the transform dictionary as instance variables.


        *   __len__(self): Returns the number of samples in the dataset.


        *   __repr__(self): Returns a string representation of the dataset object, including the number of samples and the transform dictionary.


        *   __getitem__(self, index): Given an index, this method returns a sample from the dataset, including the image and its associated dependent variables (keypoints, bounding box, pose label, and metadata), 
                                      after applying the specified transformations.


        *   visualize_sample(self, index): Given an index, this method visualizes a sample from the dataset, displaying the image, keypoints, and bounding box.


        *   split_dataset(self, train_ratio, test_ratio, val_ratio, seed): This method splits the dataset into training, validation, and testing sets based on the provided ratios. It returns a dictionary 
                                                                           containing the PoseEstimationDataset instances for each split.

    
        *   save_dataset(self, save_dir): This method saves the dataset, including the images and annotations, to the specified directory.


        *   load_dataset(load_dir, annotation_file, transform_dict): This static method loads a previously saved dataset from the specified directory, returning a new PoseEstimationDataset instance with the loaded 
                                                                     data and the specified transform dictionary.


    The transform_dict provided as an argument to the constructor is used to apply a series of transformations to the data when loading samples. These transformations can be applied to the independent variable (image) 
    and dependent variables (keypoints, bounding box, pose label, and metadata), allowing for data augmentation and preprocessing tailored to the specific requirements of the pose estimation task.

'''

# Some additional functionalities you can consider adding to the PoseEstimationDataset class:

    # 1) Data augmentation: You can extend the existing transform_dict functionality to include more data augmentation techniques, such as rotation, scaling, flipping, and cropping. This will help improve the 
    #    generalization capabilities of the model.

    # 2) Normalization: Add normalization functionality to preprocess the input images and keypoints according to a specified mean and standard deviation. This can help improve the training process.

    # 3) Filter dataset: Add a method that filters the dataset based on certain criteria, like pose labels, camera distance, or light intensity. This can be useful when you want to train or evaluate your 
    #    model on a specific subset of the dataset.

    # 4) Balance dataset: Implement a method to balance the dataset by oversampling underrepresented pose labels or undersampling overrepresented ones. This can help improve the performance of the 
    #    model on imbalanced datasets.

    # 5) Shuffle dataset: Add an option to shuffle the dataset when splitting into train, validation, and test sets. This can help reduce potential biases in the dataset splits.

    # 6) K-fold cross-validation: Implement a method to generate dataset splits for K-fold cross-validation. This can help you get a better estimate of your model's performance.

    # 7) Dataset statistics: Implement methods to compute various dataset statistics, such as the number of samples per pose label, the distribution of camera distances, or the distribution of light intensities. 
    #    This can help you better understand your dataset and make informed decisions about preprocessing and model selection.

    # 8) Display sample with annotations: Improve the visualize_sample method to display the keypoints, bounding box, pose label, and metadata information as text annotations on the image. This can help you 
    #    better understand the quality of the annotations in your dataset.


__annotation_file_structure__: str = r'''
    [
        {
            "image_id": 1,
            "keypoints": [x1, y1, v1, x2, y2, v2, ...],
            "bounding_box": [x, y, w, h],
            "pose_label": 1,
            "metadata": {
                "camera_distance": 1000,
                "light_intensity": 200
            }
        },
        {
            "image_id": 2,
            "keypoints": [x1, y1, v1, x2, y2, v2, ...],
            "bounding_box": [x, y, w, h],
            "pose_label": 2,
            "metadata": {
                "camera_distance": 1500,
                "light_intensity": 100
            }
        },
        ...
    ]

'''

__getitem_structure_doc__: str = r'''
    for each index, return this:
        {
            'image_id': int,
            'keypoints': List[float],
            'bounding_box': List[float],
            'pose_label': int,
            'metadata': Dict[str, Any]
        }


'''

#@: Pose Estimation Dataset Class
class PoseEstimationDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_dir: str, annotation_file: str, 
                                         transform_dict: Optional[dict[str, dict[str, dict[str, Callable[..., Any]]]]] = None) -> None:

        
        self.image_dir: str = os.path.join(dataset_dir, 'images')
        with open(os.path.join(dataset_dir, annotation_file), 'r') as f:
            self.annotations: list[dict[str, Any]] = json.load(f)
        
        self.transform_dict = transform_dict
        
        
        
                        
    def __len__(self) -> int:
        return len(self.annotations)
        
    
    
    
    def __repr__(self) -> str:
        return f'PoseEstimationDataset(num_samples= {len(self)}, transform_dict= {self.transform_dict})'
    
    
        
        
    def __getitem__(self, index: int) -> dict[str, Union[torch.tensor, dict[str, torch.tensor]]]:
        img_id: int = self.annotations[index]['image_id']
        img_path: str = os.path.join(self.image_dir, f'{img_id}.jpg')
        
        #@: getting all the dependent and independent_variables
        image: PIL.Image = Image.open(img_path).convert('RGB')
        keypoints: list[float] = self.annotations[index]['keypoints']
        bounding_box: list[float] = self.annotations[index]['bounding_box']
        pose_label: int = self.annotations[index]['pose_label']
        metadata: dict[str, Any] = self.annotations[index]['metadata']
        
        
        #@: transforms dependent and independent variable 
        if self.transform_dict is not None:
            if 'independent_variable' in self.transform_dict:
                for transform_func in self.transform_dict['independent_variable'].values():
                    image: Any = transform_func(image)
                    
            if 'dependent_variable' in self.transform_dict:
                if 'keypoints' in self.transform_dict['dependent_variable']:
                    for transform_func in self.transform_dict['dependent_variable']['keypoints'].values():
                        keypoints: list[float] = transform_func(keypoints)
            
                if 'bounding_box' in self.transform_dict['dependent_variable']:
                    for transform_func in self.transform_dict['dependent_variable']['bounding_box'].values():
                        bounding_box: list[float] = transform_func(bounding_box)
                        
                if 'pose_label' in self.transform_dict['dependent_variable']:
                    for transform_func in self.transform_dict['dependent_variable']['pose_label'].values():
                        pose_label: Any = transform_func(pose_label)
                
                if 'metadata' in self.transform_dict['dependent_variable']:
                    for transform_func in self.transform_dict['dependent_variable']['metadata'].values():
                        metadata: Any = transform_func(metadata)
                        
         
        #@: converting dependent variable to tensors       
        keypoints: torch.FloatTensor = torch.FloatTensor(keypoints)
        bounding_box: torch.FloatTensor = torch.FloatTensor(bounding_box)
        pose_label: torch.FloatTensor = torch.FloatTensor(pose_label)
        metadata: torch.FloatTensor = torch.FloatTensor(metadata)
        
        
        
        #@: returning a dict with independent_variable and dependent_variable
        return {
            'independent_variable': image,
            'dependent_variable': {
                'keypoints': keypoints,
                'bounding_box': bounding_box,
                'pose_label': pose_label,
                'metadata': metadata
            }
        }
        
        
    
    def visualize_sample(self, index: int) -> None:
        sample: dict[str, Union[torch.tensor, dict[str, torch.tensor]]] = self[index]
        img: torch.tensor = sample['independent']
        keypoints: torch.tensor = sample['dependent']['keypoints']
        bbox: torch.tensor = sample['dependent']['bounding_box']

        #@: Visualize image
        plt.imshow(img)
        plt.title('Sample Image')

        #@: Visualize keypoints
        plt.scatter(keypoints[0::3], keypoints[1::3], s= 10, marker= '.', c= 'r')
        plt.title('Sample Keypoints')

        #@: Visualize bounding box
        x, y, w, h = bbox
        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='g', facecolor='none')
        ax = plt.gca()
        ax.add_patch(rect)
        plt.title('Sample Bounding Box')
        plt.show()
        
        
        
        
    def split_dataset(self, train_ratio: float, test_ratio: float, val_ratio: float, seed: Optional[int] = 42) -> dict[str, PoseEstimationDataset]:
        assert train_ratio + test_ratio + val_ratio == 1, "The sum of the ratios must be equal to 1."
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
        
        
        
        
    def save_dataset(self, save_dir: str) -> None:
        #@: Creating the save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)

        #@: Saving the annotations as a new JSON file
        save_annotation_file = os.path.join(save_dir, 'annotations.json')
        with open(save_annotation_file, 'w') as f:
            json.dump(self.annotations, f)

        #@: Copying image files to the new directory
        save_image_dir = os.path.join(save_dir, 'images')
        os.makedirs(save_image_dir, exist_ok=True)
        for annotation in self.annotations:
            img_id = annotation['image_id']
            src_img_path = os.path.join(self.image_dir, f'{img_id}.jpg')
            dst_img_path = os.path.join(save_image_dir, f'{img_id}.jpg')
            shutil.copy(src_img_path, dst_img_path)
        
    
    
    
    @staticmethod
    def load_dataset(load_dir: str, annotation_file: Optional[str] = 'annotations.json', 
                                    transform_dict: Optional[dict[str, dict[str, dict[str, Callable[..., Any]]]]] = None) -> PoseEstimationDataset:
        
        dataset_dir = load_dir
        return PoseEstimationDataset(dataset_dir, annotation_file, transform_dict)
        
        
        




    
#@: Driver Code
if __name__.__contains__('__main__'):
    #@: dataset class for this type of transform_dict 
    
    dataset_dir: str = "/path/to/dataset"
    annotation_file: str = "annotations.json"

    #@: > use case Example:
    #@: Check Data Pre-Processing Transforms Pipeline
    transform_dict: dict[str, dict[str, dict[str, Callable[..., Any]]]] = {
        'independent_variable': {
            'transform_1': RandomCropTransform(224),
            'transform_2': RandomFlipTransform(0.5, 0.5),
            'transform_3': RandomRotationTransform(10),
            'transform_4': ColorJitterTransform(0.1, 0.1, 0.1, 0.1),
            'transform_5': GaussianBlurTransform(5, (0.1, 2.0)),
            'transform_6': RandomAffineTransform(10),
            'transform_7': NoiseInjectionTransform(0.05),
            'transform_8': CutoutTransform(1, 16),
            'transform_9': ToTensorTransform(),
            'transform_10': NormalizeTransform((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        },
        'dependent_variable': {
            'keypoints': {
                'normalize_keypoints': lambda keypoints, img_w, img_h: keypoints / np.array([img_w, img_h, 1]),
                'horizontal_flip_keypoints': lambda keypoints, img_w: keypoints * np.array([-1, 1, 1]) + np.array([img_w, 0, 0])
            },
            'bounding_box': {
                'normalize_bounding_box': lambda bbox, img_w, img_h: bbox / np.array([img_w, img_h, img_w, img_h]),
                'horizontal_flip_bounding_box': lambda bbox, img_w: np.array([img_w - bbox[2], bbox[1], img_w - bbox[0], bbox[3]])
            },
            'pose_label': {
                'label_encode_pose': lambda pose: label_encoder.fit_transform([pose])[0],
                'one_hot_encode_pose': lambda pose, num_classes: np.eye(num_classes)[label_encoder.fit_transform([pose])[0]]
            },
            'metadata': {
                'normalize_camera_distance': lambda metadata: metadata['camera_distance'] / 1000,
                'normalize_light_intensity': lambda metadata: metadata['light_intensity'] / 255
            }
        }
    }
    
    
    pose_estimation_dataset: PoseEstimationDataset = PoseEstimationDataset(dataset_dir, annotation_file, transform_dict)

    # Split the dataset
    dataset_splits = pose_estimation_dataset.split_dataset(train_ratio= 0.8, test_ratio= 0.1, val_ratio= 0.1)

    # Loaders for each dataset split
    train_loader = torch.utils.data.DataLoader(dataset_splits["train"], batch_size= 32, shuffle= True)
    val_loader = torch.utils.data.DataLoader(dataset_splits["val"], batch_size= 32, shuffle= False)
    test_loader = torch.utils.data.DataLoader(dataset_splits["test"], batch_size= 32, shuffle= False)

    
    
    