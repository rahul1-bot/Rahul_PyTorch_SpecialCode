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

__doc__: str = r'''
    The above code defines a PoseEstimationTrainer class that trains, validates, and tests a pose estimation model. The class is designed to work with a custom model 
    and dataset, and the code is organized into various methods to perform specific tasks:

    *   __init__: The constructor initializes the class with a given model, training DataLoader, validation DataLoader, optional test DataLoader, device (default is 'cuda'), and learning rate (default is 0.001). 
                  It sets up the optimizer and creates a dictionary for storing training history.

    *   loss_function: This method calculates the total loss for the model's output and the target labels. It computes separate losses for keypoints, bounding boxes, pose labels, and metadata, and sums them up to 
                       get the total loss.

    *   metric: This method computes evaluation metrics for the model's output and target labels. It calculates the mean squared error (MSE) for keypoints and bounding boxes, accuracy for pose labels, and mean 
                absolute error (MAE) for metadata. These metrics are returned as a dictionary.

    *   __run_epoch: This private method performs a single training, validation, or testing epoch. It takes an optional mode parameter ('train', 'val', or 'test') to specify the operation. Depending on the mode, 
                     the method iterates through the corresponding DataLoader, computes the loss and evaluation metrics for each batch, and updates the model's parameters if it's in 'train' mode. The average loss 
                     and metrics for the epoch are returned as a dictionary.

    *   train: This method trains the model for a specified number of epochs. It calls the __run_epoch method for training and validation modes and stores the results (loss and metrics) in the history dictionary. 
               It also logs the epoch results.

    *   evaluate: This method evaluates the model on a given DataLoader. It calls the __run_epoch method with 'test' mode and logs the test loss and metrics.

    *   plot_metrics: This method plots the training and validation loss and metrics stored in the history dictionary using Matplotlib.

    *   save_history: This method saves the training history as a NumPy file to a specified file path.

    *   save_model: This method saves the model's state dictionary to a specified file path.

    *   load_and_evaluate: This class method loads a saved model from a file path, creates a PoseEstimationTrainer instance with the loaded model and the given test DataLoader, and evaluates the model by calling 
                           the private __run_epoch method with 'test' mode. It returns the test loss and metrics as a dictionary.

'''

__model_output__doc__: str = r'''
    for each image, predict these values and store them in dictionary: 
        {
             'keypoints': keypoints,
             'bounding_box': bbox,
             'pose_label': pose_label,
             'metadata': metadata
        }
'''

#@: Pose Estimation Trainer Class 
class PoseEstimationTrainer:
    def __init__(self, model: torch.nn.Module, train_loader: torch.utils.data.DataLoader, val_loader: torch.utils.data.DataLoader, 
                                                                                          test_loader: Optional[torch.utils.data.DataLoader] = None,
                                                                                          device: Optional[str] = 'cuda', 
                                                                                          learning_rate: Optional[float] = 0.001) -> None:
        
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.optimizer = optim.Adam(self.model.parameters(), lr= learning_rate)
        self.history: dict[str, list[float]] = {
            'train_loss': [], 'val_loss': [], 'train_metric': [], 'val_metric': []
        }
        
    
    
    
    def loss_function(self, output: dict[str, torch.tensor], target: dict[str, torch.tensor]) -> torch.tensor:
        #@: keypoints_loss
        keypoints_loss: torch.tensor = nn.MSELoss()(output['keypoints'], target['keypoints'])
            
        #@: bounding_box loss
        bounding_box_loss: torch.tensor = nn.MSELoss()(output['bounding_box'], target['bounding_box'])

        #@: pose label loss
        pose_label_loss: torch.tensor = nn.CrossEntropyLoss()(output['pose_label'], target['pose_label'])
        
        #@: meta data loss { L1 - Loss }
        meta_data_loss: torch.tensor = nn.L1Loss()(output['metadata'], target['metadata'])
        
        total_loss: torch.tensor = keypoints_loss + bounding_box_loss + pose_label_loss + meta_data_loss
        return total_loss
    
    
    
    
    
    
    def metric(self, output: dict[str, torch.tensor], target: dict[str, torch.tensor]) -> dict[str, float]:
        #@: mean square error for keypoints 
        keypoints_mse: float = nn.MSELoss()(output['keypoints'], target['keypoints']).item()
        
        #@: mean square error for bounding box
        bounding_box_mse: float = nn.MSELoss()(output['bounding_box'], target['bounding_box']).item()
        
        #@: acc for pose label
        pose_label_correct: int = (output['pose_label'].argmax(dim= 1) == target['pose_label']).sum().item()
        pose_label_acc: float = pose_label_correct / len(output['pose_label'])
        
        #@: mean absolute error for meta_data
        meta_data_mae: float = nn.L1Loss()(output['metadata'], target['metadata']).item()
        
        return {
            'keypoints_mse': keypoints_mse, 
            'bbox_mse': bounding_box_mse, 
            'pose_label_accuracy': pose_label_accuracy, 
            'metadata_mae': meta_data_mae
        }
    
    
    
    
    
    def __run_epoch(self, mode: Optional[str] = 'train') -> dict[str, float | dict[str, float]]:
        if mode == 'train':
            self.model.train()
            loader: torch.utils.data.DataLoader = self.train_loader
        elif mode == 'val':
            self.model.eval()
            loader: torch.utils.data.DataLoader = self.val_loader
        elif mode == 'test':
            self.model.eval()
            loader: torch.utils.data.DataLoader = self.test_loader
        else:
            raise ValueError('Invalid mode.')
            
        epoch_loss: float = 0.0
        epoch_metrics: dict[str, float] = {
            'keypoints_mse': 0, 
            'bbox_mse': 0,
            'pose_label_accuracy': 0, 
            'metadata_mae': 0
        }
        
        for batch in loader:
            self.optimizer.zero_grad()
            image: torch.tensor = batch['independent_variable'].to(self.device)
            target: dict[str, torch.tensor | float] = {
                key: value.to(self.device) for key, value in batch['independent_variable'].item()
            }
            
            output: dict[str, torch.tensor] = self.model(image)
            loss: torch.tensor = self.loss_function(output, target)
            
            if mode == 'train':
                loss.backward()
                self.optimizer.step()
            
            epoch_loss += loss.item()
            batch_metrics: float = self.metric(output, target)
            
            for key, value in epoch_metrics.item():
                epoch_metrics[key] += value
            
        epoch_loss /= len(loader)
        for key, value in epoch_metrics.item():
            epoch_metrics[key] /= len(loader)
        
        return {
            'loss': epoch_loss, 'metrics': epoch_metrics
        }
            
            
    
    
    def train(self, epochs: int) -> None:
        for epoch in range(epochs):
            train_results: dict[str, float | dict[str, float]] = self.__run_epoch(mode= 'train')
            val_results: dict[str, float | dict[str, float]] = self.__run_epoch(mode= 'val')
            
            self.history['train_loss'].append(train_results['loss'])
            self.history['val_loss'].append(val_results['loss'])
            self.history['train_metric'].append(train_results['metrics'])
            self.history['val_metric'].append(val_results['metrics'])
            
            #@: logging functionalities here....
            logging.info(f"Epoch {epoch + 1}/{epochs}")
            logging.info(f"Train Loss: {train_results['loss']:.4f}, Val Loss: {val_results['loss']:.4f}")
            logging.info(f"Train Metrics: {train_results['metrics']}")
            logging.info(f"Val Metrics: {val_results['metrics']}")
            
        


    
    def evaluate(self, dataloader: torch.utils.data.DataLoader) -> dict[str, float]:
        results: dict[str, float | dict[str, float]] = self.__run_epoch(mode= 'test')
        logging.info(f"Test Loss: {results['loss']:.4f}")
        logging.info(f"Test Metrics: {results['metrics']}")
        return results
    
    
    
    
    def plot_metrics(self) -> None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        ax1.plot(self.history["train_loss"], label="Train Loss")
        ax1.plot(self.history["val_loss"], label="Val Loss")
        ax1.set_title("Loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.legend()

        for metric_name in self.history["train_metric"][0].keys():
            train_metric_values = [metrics[metric_name] for metrics in self.history["train_metric"]]
            val_metric_values = [metrics[metric_name] for metrics in self.history["val_metric"]]
            ax2.plot(train_metric_values, label=f"Train {metric_name}")
            ax2.plot(val_metric_values, label=f"Val {metric_name}")

        ax2.set_title("Metrics")
        ax2.set_xlabel("Epoch")
        ax2.legend()

        plt.show()



    def save_history(self, file_path: str) -> None:
        np.save(file_path, self.history)




    def save_model(self, file_path: str) -> None:
        torch.save(self.model.state_dict(), file_path)
        
    
    
    
    
    @classmethod
    def load_and_evaluate(cls, model_path: str, test_loader: torch.utils.data.DataLoader, device: str = 'cuda') -> dict[str, float]:
        model = cls.model.to(device)
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()

        trainer: PoseEstimationTrainer = cls(
            model,
            train_loader= None,
            val_loader= None,
            test_loader= test_loader,
            device= device
        )
        results: dict[str, float] = trainer.__run_epoch(mode= 'test')
        return results
        
    

#@: To get general idea of the output from the model:
    
# class MultiTaskPoseEstimationModel(nn.Module):
#     def __init__(self, backbone: nn.Module, num_keypoints: int, num_pose_labels: int, num_metadata: int):
#         super(MultiTaskPoseEstimationModel, self).__init__()
#         self.backbone = backbone

#         # Keypoints output
#         self.keypoints_head = nn.Sequential(
#             nn.Linear(backbone.output_dim, 256),
#             nn.ReLU(),
#             nn.Dropout(0.5),
#             nn.Linear(256, num_keypoints * 2)
#         )

#         # Bounding box output
#         self.bbox_head = nn.Sequential(
#             nn.Linear(backbone.output_dim, 256),
#             nn.ReLU(),
#             nn.Dropout(0.5),
#             nn.Linear(256, 4)
#         )

#         # Pose label output
#         self.pose_label_head = nn.Sequential(
#             nn.Linear(backbone.output_dim, 256),
#             nn.ReLU(),
#             nn.Dropout(0.5),
#             nn.Linear(256, num_pose_labels)
#         )

#         # Metadata output
#         self.metadata_head = nn.Sequential(
#             nn.Linear(backbone.output_dim, 256),
#             nn.ReLU(),
#             nn.Dropout(0.5),
#             nn.Linear(256, num_metadata)
#         )

#     def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
#         features = self.backbone(x)

#         keypoints = self.keypoints_head(features)
#         bbox = self.bbox_head(features)
#         pose_label = self.pose_label_head(features)
#         metadata = self.metadata_head(features)

#         return {
#             'keypoints': keypoints,
#             'bounding_box': bbox,
#             'pose_label': pose_label,
#             'metadata': metadata
#         }
        
        
    
#@: Driver Code
if __name__.__contains__('__main__'):
    #@: > Use CASE EXAMPLE 
    
    #@: Load the dataset and create DataLoaders
    # train_dataset = YourDataset(train=True)
    # val_dataset = YourDataset(train=False)
    # train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)


    # #@: Instantiate pose estimation model
    # model = YourPoseEstimationModel()


    #@: Create an instance of the PoseEstimationTrainer class
    # trainer = PoseEstimationTrainer(model, train_loader, val_loader, device='cuda', learning_rate=0.001)


    #@: Train the model for a specified number of epochs
    # epochs = 10
    # trainer.train(epochs)


    #@: Plot the training metrics
    # trainer.plot_metrics()


    #@: Save the trained model and history
    # model_path = 'path/to/save/model.pth'
    # history_path = 'path/to/save/history.npy'
    # trainer.save_model(model_path)
    # trainer.save_history(history_path)
    
    
    #@: Load the saved model and evaluate on the test DataLoader
    # model_path = 'path/to/your/saved/model.pth'
    # test_results = PoseEstimationTrainer.load_and_evaluate(model_path, test_loader)
    # print(test_results)
    
    
    ...
    
    
    
    
    