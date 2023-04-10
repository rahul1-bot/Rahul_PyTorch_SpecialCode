from __future__ import annotations
import torch
from torch import nn
from torch import Tensor
from typing import Dict
from torchvision.models import resnet50, vgg16, mobilenet_v2


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


#@: NOTE : This will be the Models Base Struture 
class MultiTaskPoseEstimationModel(nn.Module):
    # The provided code defines a MultiTaskPoseEstimationModel class that inherits from nn.Module. This class is a multi-task pose estimation model that takes a backbone 
    # network as input and predicts keypoints, bounding boxes, pose labels, and metadata.

    # The model has four task-specific heads:

    #     *   keypoints_head: A fully connected neural network that predicts keypoints.
    #     *   bbox_head: A fully connected neural network that predicts bounding box coordinates.
    #     *   pose_label_head: A fully connected neural network that predicts pose labels.
    #     *   metadata_head: A fully connected neural network that predicts metadata values.
    
    # The forward method takes an input tensor x and passes it through the backbone network to extract features. These features are then passed through each of the task-specific 
    # heads to produce the predictions for keypoints, bounding boxes, pose labels, and metadata.

    # The method returns a dictionary containing the output tensors for each task.
    
    def __init__(self, backbone: nn.Module, num_keypoints: int, num_pose_labels: int, num_metadata: int) -> None:
        super(MultiTaskPoseEstimationModel, self).__init__()
        self.backbone = backbone

        # Keypoints output
        self.keypoints_head = nn.Sequential(
            nn.Linear(backbone.output_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_keypoints * 2)
        )

        # Bounding box output
        self.bbox_head = nn.Sequential(
            nn.Linear(backbone.output_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 4)
        )

        # Pose label output
        self.pose_label_head = nn.Sequential(
            nn.Linear(backbone.output_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_pose_labels)
        )

        # Metadata output
        self.metadata_head = nn.Sequential(
            nn.Linear(backbone.output_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_metadata)
        )




    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.backbone(x)

        keypoints = self.keypoints_head(features)
        bbox = self.bbox_head(features)
        pose_label = self.pose_label_head(features)
        metadata = self.metadata_head(features)

        return {
            'keypoints': keypoints,
            'bounding_box': bbox,
            'pose_label': pose_label,
            'metadata': metadata
        }



#@: 1) ----------------------------------------------------- RESNET - 50 as the Backbone ---------------------------------------------------------

class ResNet50MultiTaskPoseEstimationModel(nn.Module):
    # The provided code defines a ResNet50MultiTaskPoseEstimationModel class that inherits from nn.Module. This class is a multi-task pose estimation model that utilizes 
    # the ResNet-50 architecture as a backbone for feature extraction. It predicts keypoints, bounding boxes, pose labels, and metadata.

    # First, the code imports the resnet50 model from torchvision, and then removes the final layer of the model to use it as a feature extractor. The extracted features are then 
    # passed through four task-specific heads:

    #     *   keypoints_head: A fully connected neural network that predicts keypoints.
    #     *   bbox_head: A fully connected neural network that predicts bounding box coordinates.
    #     *   pose_label_head: A fully connected neural network that predicts pose labels.
    #     *   metadata_head: A fully connected neural network that predicts metadata values.
    
    # The forward method takes an input tensor x and passes it through the backbone network to extract features. These features are then flattened and passed through each of the 
    # task-specific heads to produce the predictions for keypoints, bounding boxes, pose labels, and metadata.

    # The method returns a dictionary containing the output tensors for each task.
    
    def __init__(self, num_keypoints: int, num_pose_labels: int, num_metadata: int) -> None:
        super(ResNet50MultiTaskPoseEstimationModel, self).__init__()
        resnet = resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        backbone_output_dim = resnet.fc.in_features

        self.keypoints_head = nn.Sequential(
            nn.Linear(backbone_output_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_keypoints * 2)
        )

        self.bbox_head = nn.Sequential(
            nn.Linear(backbone_output_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 4)
        )

        self.pose_label_head = nn.Sequential(
            nn.Linear(backbone_output_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_pose_labels)
        )

        self.metadata_head = nn.Sequential(
            nn.Linear(backbone_output_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_metadata)
        )



    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        features = self.backbone(x)
        features = features.view(features.size(0), -1)

        keypoints = self.keypoints_head(features)
        bbox = self.bbox_head(features)
        pose_label = self.pose_label_head(features)
        metadata = self.metadata_head(features)

        return {
            'keypoints': keypoints,
            'bounding_box': bbox,
            'pose_label': pose_label,
            'metadata': metadata
        }



# 2) -------------------------------------------------------- Using VGG-16 as the backbone ------------------------------------------------------

class VGG16MultiTaskPoseEstimationModel(nn.Module):
    # The provided code defines a VGG16MultiTaskPoseEstimationModel class that inherits from nn.Module. This class is a multi-task pose estimation model that uses the VGG-16 architecture 
    # as a backbone for feature extraction. It predicts keypoints, bounding boxes, pose labels, and metadata.

    # First, the code imports the vgg16 model from torchvision, and then takes the feature extraction part of the VGG-16 model. The code also adds an AdaptiveAvgPool2d layer to the backbone
    # to ensure a consistent feature map size.

    # The extracted features are then passed through four task-specific heads:

    #     *   keypoints_head: A fully connected neural network that predicts keypoints.
    #     *   bbox_head: A fully connected neural network that predicts bounding box coordinates.
    #     *   pose_label_head: A fully connected neural network that predicts pose labels.
    #     *   metadata_head: A fully connected neural network that predicts metadata values.

    # The forward method takes an input tensor x and passes it through the backbone network to extract features. These features are then flattened and passed through each of the task-specific
    # heads to produce the predictions for keypoints, bounding boxes, pose labels, and metadata.

    # The method returns a dictionary containing the output tensors for each task.
    
    def __init__(self, num_keypoints: int, num_pose_labels: int, num_metadata: int):
        super(VGG16MultiTaskPoseEstimationModel, self).__init__()
        
        vgg = vgg16(pretrained=True)
        self.backbone = nn.Sequential(*list(vgg.features.children()), nn.AdaptiveAvgPool2d(output_size=(7, 7)))
        backbone_output_dim = vgg.classifier[0].in_features

        self.keypoints_head = nn.Sequential(
            nn.Linear(backbone_output_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_keypoints * 2)
        )

        self.bbox_head = nn.Sequential(
            nn.Linear(backbone_output_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 4)
        )

        self.pose_label_head = nn.Sequential(
            nn.Linear(backbone_output_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_pose_labels)
        )

        self.metadata_head = nn.Sequential(
            nn.Linear(backbone_output_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_metadata)
        )



    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        features = self.backbone(x)
        features = features.view(features.size(0), -1)

        keypoints = self.keypoints_head(features)
        bbox = self.bbox_head(features)
        pose_label = self.pose_label_head(features)
        metadata = self.metadata_head(features)

        return {
            'keypoints': keypoints,
            'bounding_box': bbox,
            'pose_label': pose_label,
            'metadata': metadata
        }

 

#@: 3) ------------------------------------ Using MobileNetV2 as the backbone --------------------------------------

class MobileNetV2MultiTaskPoseEstimationModel(nn.Module):
    # The provided code defines a MobileNetV2MultiTaskPoseEstimationModel class that inherits from nn.Module. This class is a multi-task pose estimation model that uses the 
    # MobileNetV2 architecture as a backbone for feature extraction. It predicts keypoints, bounding boxes, pose labels, and metadata.

    # First, the code imports the mobilenet_v2 model from torchvision and removes the final classification layer of the model to use it as a feature extractor. The extracted features
    # are then passed through four task-specific heads:

    #     *   keypoints_head: A fully connected neural network that predicts keypoints.
    #     *   bbox_head: A fully connected neural network that predicts bounding box coordinates.
    #     *   pose_label_head: A fully connected neural network that predicts pose labels.
    #     *   metadata_head: A fully connected neural network that predicts metadata values.

    # The forward method takes an input tensor x and passes it through the backbone network to extract features. These features are then flattened and passed through each of the 
    # task-specific heads to produce the predictions for keypoints, bounding boxes, pose labels, and metadata.

    # The method returns a dictionary containing the output tensors for each task.
    
    def __init__(self, num_keypoints: int, num_pose_labels: int, num_metadata: int) -> None:
        super(MobileNetV2MultiTaskPoseEstimationModel, self).__init__()
        mobilenet = mobilenet_v2(pretrained=True)
        
        self.backbone = nn.Sequential(*list(mobilenet.children())[:-1])
        backbone_output_dim = mobilenet.classifier[1].in_features

        self.keypoints_head = nn.Sequential(
            nn.Linear(backbone_output_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_keypoints * 2)
        )

        self.bbox_head = nn.Sequential(
            nn.Linear(backbone_output_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 4)
        )

        self.pose_label_head = nn.Sequential(
            nn.Linear(backbone_output_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_pose_labels)
        )

        self.metadata_head = nn.Sequential(
            nn.Linear(backbone_output_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_metadata)
        )



    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        features = self.backbone(x)
        features = features.view(features.size(0), -1)

        keypoints = self.keypoints_head(features)
        bbox = self.bbox_head(features)
        pose_label = self.pose_label_head(features)
        metadata = self.metadata_head(features)

        return {
            'keypoints': keypoints,
            'bounding_box': bbox,
            'pose_label': pose_label,
            'metadata': metadata
        }


#@: 4) -------------------- Stacked Hourglass Networks ------------------------------------

class HourglassModule(nn.Module):
    # The provided code defines an HourglassModule class that inherits from nn.Module. This class represents a single hourglass module within a stacked hourglass network, 
    # which is commonly used for pose estimation tasks. The module is designed to capture features at different scales by processing the input through a series of downsampling 
    # and upsampling layers while preserving spatial resolution.

    # In the __init__ method, three sequential blocks are defined:

    #     *   self.down: This block performs downsampling using a convolutional layer with a stride of 2, followed by ReLU activation and batch normalization.
        
    #     *   self.inner: This block contains either a recursive call to another HourglassModule with depth reduced by 1 or an identity layer if the depth is 1. 
    #                     It is followed by a convolutional layer, ReLU activation, and batch normalization.
        
    #     *   self.up: This block performs upsampling using a transposed convolutional layer with a stride of 2, followed by ReLU activation and batch normalization.
    
    # The forward method takes an input tensor x and processes it through the downsampling, inner, and upsampling blocks. The output of the upsampling block is added to the input 
    # tensor x, and the result is returned. This residual connection helps in preserving spatial information and learning fine-grained features.
    
    def __init__(self, in_channels: int, out_channels: int, depth: int) -> None:
        super(HourglassModule, self).__init__()

        self.down = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels)
        )

        self.inner = nn.Sequential(
            HourglassModule(out_channels, out_channels, depth - 1) if depth > 1 else nn.Identity(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels)
        )

        self.up = nn.Sequential(
            nn.ConvTranspose2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels)
        )


    def forward(self, x: Tensor) -> Tensor:
        down = self.down(x)
        inner = self.inner(down)
        up = self.up(inner)

        return up + x




class StackedHourglassNetwork(nn.Module):
    # The provided code defines a StackedHourglassNetwork class that inherits from nn.Module. This class represents a full stacked hourglass network used for multi-task pose 
    # estimation. The network is designed to handle four different tasks: keypoint detection, bounding box regression, pose label classification, and metadata regression.

    # In the __init__ method, the following layers are defined:

    #     *   self.initial_conv, self.initial_bn, and self.initial_relu: These layers form the initial block of the network, which processes the input image.
        
    #     *   self.hourglass_modules: A list of hourglass modules (based on the HourglassModule class previously defined) stacked sequentially.

    #     *   self.keypoints_head, self.bbox_head, self.pose_label_head, and self.metadata_head: These are the task-specific heads for keypoint detection, bounding box regression, 
    #         pose label classification, and metadata regression, respectively. Each head consists of a 1x1 convolutional layer, batch normalization, ReLU activation, and another 1x1 
    #         convolutional layer with the output channels corresponding to the task.

    # The forward method takes an input tensor x and processes it through the initial block and the stacked hourglass modules. Then, the processed features are fed into each task-specific 
    # head to produce the final outputs for keypoint detection, bounding box regression, pose label classification, and metadata regression. These outputs are returned as a dictionary.
    
    def __init__(self, num_stacks: int, num_keypoints: int, num_pose_labels: int, num_metadata: int) -> None:
        super(StackedHourglassNetwork, self).__init__()
        self.initial_conv = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.initial_bn = nn.BatchNorm2d(64)
        self.initial_relu = nn.ReLU(inplace=True)

        self.hourglass_modules = nn.ModuleList([HourglassModule(64, 64, 4) for _ in range(num_stacks)])

        self.keypoints_head = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_keypoints * 2, kernel_size=1)
        )

        self.bbox_head = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 4, kernel_size=1)
        )

        self.pose_label_head = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_pose_labels, kernel_size=1)
        )

        self.metadata_head = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_metadata, kernel_size=1)
        )



    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        x = self.initial_conv(x)
        x = self.initial_bn(x)
        x = self.initial_relu(x)

        for hg in self.hourglass_modules:
            x = hg(x)

        keypoints = self.keypoints_head(x)
        bbox = self.bbox_head(x)
        pose_label = self.pose_label_head(x)
        metadata = self.metadata_head(x)

        return {
            'keypoints': keypoints,
            'bounding_box': bbox,
            'pose_label': pose_label,
            'metadata': metadata
        }
        
        
        


#@: 5) ----------------------------------- Convolutional Pose Machines (CPMs) --------------------------------------------------------

class CPMStage(nn.Module):
    # The provided code defines a CPMStage class that inherits from nn.Module. This class represents a single stage in a Convolutional Pose Machine (CPM) network, 
    # which is designed for pose estimation tasks.

    # In the __init__ method, the following layers are defined:

    #     *   self.features: A sequential container with a 3x3 convolutional layer with the specified in_channels and out_channels, a ReLU activation applied in-place, 
    #                        and batch normalization.
    
    # The forward method takes an input tensor x and processes it through the self.features sequential container, producing the output of the CPM stage. The output tensor 
    # is then returned.
    
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(CPMStage, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.features(x)




class ConvolutionalPoseMachines(nn.Module):
    # The provided code defines a ConvolutionalPoseMachines class that inherits from nn.Module. This class represents the architecture of a Convolutional Pose Machines (CPM) 
    # network for multi-task pose estimation.

    # In the __init__ method, the following layers are defined:

    #     *   self.initial_features: A sequential container with several convolutional layers, ReLU activations, and batch normalization layers.
        
    #     *   self.stages: A ModuleList containing CPMStage instances, with the specified number of num_stages.
        
    #     *   self.keypoints_head: A sequential container with a 1x1 convolutional layer for keypoints estimation.
        
    #     *   self.bbox_head: A sequential container with a 1x1 convolutional layer for bounding box estimation.

    #     *   self.pose_label_head: A sequential container with a 1x1 convolutional layer for pose label estimation.

    #     *   self.metadata_head: A sequential container with a 1x1 convolutional layer for metadata estimation.

    # The forward method takes an input tensor x and processes it through the self.initial_features sequential container, followed by the CPM stages. The output of the final stage is then 
    # fed into the keypoints, bounding box, pose label, and metadata heads. The method returns a dictionary containing the output tensors of each task (keypoints, bounding box, pose label, 
    # and metadata).
    
    def __init__(self, num_stages: int, num_keypoints: int, num_pose_labels: int, num_metadata: int) -> None:
        super(ConvolutionalPoseMachines, self).__init__()

        self.initial_features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128)
        )

        self.stages = nn.ModuleList([CPMStage(128, 128) for _ in range(num_stages)])

        self.keypoints_head = nn.Sequential(
            nn.Conv2d(128, num_keypoints * 2, kernel_size=1)
        )

        self.bbox_head = nn.Sequential(
            nn.Conv2d(128, 4, kernel_size=1)
        )

        self.pose_label_head = nn.Sequential(
            nn.Conv2d(128, num_pose_labels, kernel_size=1)
        )

        self.metadata_head = nn.Sequential(
            nn.Conv2d(128, num_metadata, kernel_size=1)
        )



    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        x = self.initial_features(x)

        for stage in self.stages:
            x = x + stage(x)

        keypoints = self.keypoints_head(x)
        bbox = self.bbox_head(x)
        pose_label = self.pose_label_head(x)
        metadata = self.metadata_head(x)

        return {
            'keypoints': keypoints,
            'bounding_box': bbox,
            'pose_label': pose_label,
            'metadata': metadata
        }



#@: 6) ---------------------------------- HRNet (High-Resolution Network) -----------------------------------


class HighResolutionModule(nn.Module):
    # The provided code defines a HighResolutionModule class that inherits from nn.Module. This class represents the architecture of a High-Resolution Module, 
    # which is typically used in high-resolution networks (HRNet) for computer vision tasks.

    # In the __init__ method, the following layers are defined:

    #     *   self.branch1: A sequential container with a convolutional layer, batch normalization layer, and ReLU activation.
        
    #     *   self.branch2: A sequential container with two convolutional layers, each followed by a batch normalization layer and ReLU activation.
    
    # The forward method takes an input tensor x and processes it through both branches. The output tensors of the two branches are then element-wise added and returned.

    # This High-Resolution Module aims to learn both low-level and high-level features from the input tensor by employing two parallel branches with different complexities. 
    # The first branch captures low-level features, while the second branch captures high-level features. The combination of the outputs from both branches can help improve 
    # the performance of the network by allowing it to learn richer representations of the data.
    
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(HighResolutionModule, self).__init__()

        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )


    def forward(self, x: Tensor) -> Tensor:
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        return out1 + out2
    
    

class HighResolutionNetwork(nn.Module):
    # The provided code defines a HighResolutionNetwork class that inherits from nn.Module. This class represents a high-resolution network (HRNet) for multi-task pose estimation, 
    # which can be used for computer vision tasks such as human pose estimation or object detection.

    # In the __init__ method, the following layers are defined:

    #     *   self.initial_features: A sequential container with a convolutional layer, batch normalization layer, and ReLU activation.
        
    #     *   self.high_res_modules: A list of HighResolutionModule instances (defined earlier) with a specified num_modules.
        
    #     *   self.keypoints_head: A sequential container with a 1x1 convolutional layer for predicting keypoints.

    #     *   self.bbox_head: A sequential container with a 1x1 convolutional layer for predicting bounding boxes.

    #     *   self.pose_label_head: A sequential container with a 1x1 convolutional layer for predicting pose labels.
        
    #     *    self.metadata_head: A sequential container with a 1x1 convolutional layer for predicting metadata.

    
    # The forward method takes an input tensor x and processes it through the initial features layer, then iteratively processes it through the high-resolution modules. After that,
    # it processes the output through each of the four heads (keypoints, bounding box, pose label, and metadata) and returns a dictionary containing the resulting tensors.

    # This HighResolutionNetwork architecture aims to capture both low-level and high-level features of the input image, which can help improve the performance of the network by allowing 
    # it to learn richer representations of the data for multi-task pose estimation.
    
    def __init__(self, num_modules: int, num_keypoints: int, num_pose_labels: int, num_metadata: int) -> None:
        super(HighResolutionNetwork, self).__init__()

        self.initial_features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.high_res_modules = nn.ModuleList([HighResolutionModule(64, 64) for _ in range(num_modules)])

        self.keypoints_head = nn.Sequential(
            nn.Conv2d(64, num_keypoints * 2, kernel_size=1)
        )

        self.bbox_head = nn.Sequential(
            nn.Conv2d(64, 4, kernel_size=1)
        )

        self.pose_label_head = nn.Sequential(
            nn.Conv2d(64, num_pose_labels, kernel_size=1)
        )

        self.metadata_head = nn.Sequential(
            nn.Conv2d(64, num_metadata, kernel_size=1)
        )
        
        
        

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        x = self.initial_features(x)

        for hr_module in self.high_res_modules:
            x = hr_module(x)

        keypoints = self.keypoints_head(x)
        bbox = self.bbox_head(x)
        pose_label = self.pose_label_head(x)
        metadata = self.metadata_head(x)

        return {
            'keypoints': keypoints,
            'bounding_box': bbox,
            'pose_label': pose_label,
            'metadata': metadata
        }





#@: Driver Code 
if __name__.__contains__('__main__'):
    ...
    
    
    
    
    
    
    