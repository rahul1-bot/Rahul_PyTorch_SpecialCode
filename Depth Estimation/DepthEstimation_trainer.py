from __future__ import annotations
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pytorch_ssim import ssim


__author_info__: dict[str, Union[str, list[str]]] = {
    'Name': 'Rahul Sawhney',
    'Mail': [
        'sawhney.rahulofficial@outlook.com', 
        'rahulsawhney321@gmail.com'
    ]
}

#@: NOTE : Depth - Estimation Training Pipeline 

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
 

#@: NOTE: -------------------- Loss Functions and Metrics ----------------------

def berhu_loss(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # The BerHu (reverse Huber) loss function is a type of robust loss function that combines the benefits of 
    # L1 loss (mean absolute error) and L2 loss (mean squared error). It is widely used in depth estimation and 
    # depth prediction problems in computer vision.

    # Here's a step-by-step explanation of the provided function:

    #     *   c = 0.2 * torch.max(torch.abs(output - target)).item(): c is a threshold calculated as 20% of the 
    #                                                                 maximum absolute difference between the output and target tensors.
    #                                                                 It determines the point where the loss function transitions from L1 to L2.

    #     *   diff = torch.abs(output - target): Calculate the absolute difference between the output and target tensors.

    #     *   berhu = torch.where(diff <= c, diff, (diff ** 2 + c ** 2) / (2 * c)): For each element in the diff tensor, if 
    #                                                                               the difference is less than or equal to c, the element is unchanged (L1 loss). 
    #                                                                               If the difference is greater than c, the squared difference is added to the square 
    #                                                                               of c, and the result is divided by 2 times c (L2 loss).

    #     *   return torch.mean(berhu): Calculate the mean of the BerHu loss tensor and return it.

    # This function calculates the BerHu loss between two tensors, which is useful for depth estimation tasks in computer vision.
    
    c = 0.2 * torch.max(torch.abs(output - target)).item()
    diff = torch.abs(output - target)
    berhu = torch.where(diff <= c, diff, (diff ** 2 + c ** 2) / (2 * c))
    return torch.mean(berhu)





def ssim_loss(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    
    # The provided function calculates the SSIM (Structural Similarity Index Measure) loss between two tensors, which is a metric used for comparing the similarity 
    # between two images. The SSIM loss is simply the complement (1 - SSIM) of the SSIM value, so a lower SSIM loss means a higher similarity between the output 
    # and target tensors.

    # Here's a step-by-step explanation of the provided function:

    #     *   ssim(output, target): Calculate the SSIM value between the output and target tensors. Note that the ssim function is not provided here, 
    #                               and you should import it from an appropriate library like pytorch_ssim or implement it yourself.

    #     *   return 1 - ssim(output, target): Calculate the complement of the SSIM value by subtracting it from 1, and return the result as the SSIM loss.

    
    # This function calculates the SSIM loss between two tensors, which is useful for image similarity tasks in computer vision.
    
    return 1 - ssim(output, target)





def abs_relative_error(output: torch.Tensor, target: torch.Tensor) -> float:
    
    # The provided function calculates the absolute relative error between two tensors. The absolute relative error is a metric used for comparing the difference 
    # between two tensors, taking into account the magnitude of the target tensor. It is often used in depth estimation tasks in computer vision.

    # Here's a step-by-step explanation of the provided function:

    #     *   torch.abs(target - output): Calculate the absolute difference between the target and output tensors.
        
    #     *   torch.abs(target - output) / target: Divide the absolute difference by the target tensor element-wise, resulting in the element-wise relative error.
        
    #     *   torch.mean(torch.abs(target - output) / target): Calculate the mean of the element-wise relative error to obtain the overall absolute relative error.

    #     *   return abs_rel_error.item(): Convert the absolute relative error from a PyTorch scalar to a Python float and return it.


    # This function calculates the absolute relative error between two tensors, which is useful for depth estimation tasks in computer vision.
    
    abs_rel_error = torch.mean(torch.abs(target - output) / target)
    return abs_rel_error.item()




def log10_error(output: torch.Tensor, target: torch.Tensor) -> float:
    
    # This function calculates the log10 error between two tensors, which is a metric often used in depth estimation tasks in computer vision.

    # Here's a step-by-step explanation of the provided function:

    #     *   torch.log10(output + 1e-8): Calculate the element-wise log10 of the output tensor while adding a small constant (1e-8) to prevent division by zero or 
    #                                     taking the logarithm of zero.

    #     *   torch.log10(target + 1e-8): Calculate the element-wise log10 of the target tensor while adding a small constant (1e-8) to prevent division by zero or 
    #                                     taking the logarithm of zero.

    #     *   torch.abs(torch.log10(output + 1e-8) - torch.log10(target + 1e-8)): Calculate the absolute difference between the log10-transformed tensors.

    #     *   torch.mean(log10_diff): Calculate the mean of the absolute differences to obtain the overall log10 error.
        
    #     *   return torch.mean(log10_diff).item(): Convert the log10 error from a PyTorch scalar to a Python float and return it.

    # This function calculates the log10 error between two tensors, which is useful for depth estimation tasks in computer vision.
    
    log10_diff = torch.abs(torch.log10(output + 1e-8) - torch.log10(target + 1e-8))
    return torch.mean(log10_diff).item()






#@: NOTE: ------------------------------------- Depth - Estimation Trainer ----------------------------------------

class DepthEstimationTrainer:
    def __init__(self, model: torch.nn.Module, train_loader: torch.utils.data.DataLoader, 
                                               val_loader: torch.utils.data.DataLoader, 
                                               test_loader: Optional[torch.utils.data.DataLoader] = None, 
                                               device: Optional[torch.device] = 'cuda' if torch.cuda.is_available() else 'cpu',
                                               learning_rate: Optional[float] = 0.001) -> None:
        
        # The DepthEstimationTrainer class is a custom trainer for depth estimation models. It takes as input a model, train_loader, 
        # val_loader, and optionally a test_loader, device, and learning_rate.

        # Here's a summary of the class attributes:

        #     *   model: The depth estimation model you want to train.

        #     *   train_loader: A DataLoader instance for the training dataset.

        #     *   val_loader: A DataLoader instance for the validation dataset.

        #     *   test_loader: An optional DataLoader instance for the test dataset.

        #     *   device: The device to run the model on, which is set to 'cuda' if a GPU is available, or 'cpu' if not.

        #     *   optimizer: The optimizer to use during training, which is set to Adam by default with a specified learning_rate.

        #     *   history: A dictionary to store the training and validation losses and metrics during training.
        
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.optimizer: torch.optim = torch.optim.Adam(self.model.parameters(), lr= learning_rate)
        
        self.history: dict[str, list[float]] = {
            'train_loss': [], 'val_loss': [], 'train_metrics': [], 'val_metrics': []
        }
        
        
        
        
        
    
    def loss_function(self, output: torch.tensor, target: torch.tensor) -> torch.tensor:
        
        # The loss_function method for the DepthEstimationTrainer class calculates the total loss for a given output and target. It first computes the 
        # individual loss values for Mean Absolute Error (MAE), Root Mean Square Error (RMSE), BerHu loss, and SSIM loss, and then adds them together to 
        # form the total loss.

        # Here's a summary of the individual loss components:

        #     *   mae_loss: The Mean Absolute Error between the output and target depth maps.

        #     *   rmse_loss: The Root Mean Square Error between the output and target depth maps.

        #     *   berhu_loss: The BerHu loss between the output and target depth maps. It combines L1 and L2 losses, depending on the error magnitude.

        #     *   ssim_loss: The Structural Similarity Index Measure (SSIM) loss between the output and target depth maps. It measures the similarity of the 
        #                    depth maps in terms of their structural information.

        # The total loss is a sum of these individual loss components, which can be used to train the depth estimation model effectively.
        
        mae_loss: torch.tensor = nn.L1Loss()(output, target)
        rmse_loss: torch.tensor = torch.sqrt(nn.MSELoss()(output, target))
        berhu_loss: torch.tensor = berhu_loss(output, target)
        ssim_loss: torch.tensor = ssim_loss(output, target)

        total_loss: torch.tensor = mae_loss + rmse_loss + berhu_loss + ssim_loss
        return total_loss
    
    
    
    
    


    def metrics(self, output: torch.tensor, target: torch.tensor) -> dict[str, float]:
        
        # The metrics method for the DepthEstimationTrainer class computes various evaluation metrics for a given output and target. It calculates the Mean Absolute 
        # Error (MAE), Root Mean Squared Error (RMSE), Absolute Relative Error, Log10 Error, and BerHu loss, and then returns a dictionary containing these metric values.

        # Here's a summary of the individual metrics:

        #     *   mean_absolute_error: The Mean Absolute Error between the output and target depth maps.

        #     *   root_mean_squared_error: The Root Mean Square Error between the output and target depth maps.

        #     *   absolute_relative_error: The Absolute Relative Error between the output and target depth maps.

        #     *   log10_error: The Log10 Error between the output and target depth maps.

        #     *   berhu_loss: The BerHu loss between the output and target depth maps. It combines L1 and L2 losses, depending on the error magnitude.

        # These metrics can be used to evaluate the performance of the depth estimation model during training and validation.
        
        mae: float = nn.L1Loss()(output, target).item()
        rmse: float = torch.sqrt(nn.MSELoss()(output, target)).item()
        abs_rel: float = abs_relative_error(output, target)
        log10_error: float = log10_error(output, target)
        berhu: float = berhu_loss(output, target).item()

        return {
            'mean_absolute_error': mae,
            'root_mean_squared_error': rmse,
            'absolute_relative_error': abs_rel,
            'log10_error': log10_error,
            'berhu_loss': berhu
        }

        
        
        
        
    
    def __run_epoch(self, mode: Optional[str] = 'train') -> dict[str, float | dict[str, float]]:    
        
        # The __run_epoch() method in the DepthEstimationTrainer class processes one complete epoch for the model, either in 'train', 'val', or 'test' mode. 
        # Depending on the mode, it updates the model's parameters and calculates the loss and evaluation metrics for each batch.

        # Here's a summary of the main components of the __run_epoch() method:

        #     *   The method checks the given mode and sets the model and data loader accordingly. If the mode is 'train', the model is set to training mode, 
        #         and the training data loader is used. If the mode is 'val' or 'test', the model is set to evaluation mode, and the validation or test data 
        #         loader is used, respectively.

        #     *   The method initializes the epoch_loss and epoch_metrics variables for accumulating the total loss and metrics values for the epoch.

        #     *   For each batch in the data loader, the method performs the following steps:
                
        #             *   It resets the gradients of the model's parameters.

        #             *   It moves the input images and target depth maps to the appropriate device (CPU or GPU).

        #             *   It passes the input images through the model to obtain the output depth maps.

        #             *   It calculates the loss using the loss_function method and accumulates the total epoch loss.

        #             *   If the mode is 'train', it computes the gradients and updates the model's parameters using the optimizer.

        #             *   It calculates the evaluation metrics using the metrics method and accumulates the total epoch metrics.

        #     *   After processing all batches, the method calculates the average epoch loss and metrics by dividing the accumulated values by the number of batches.

        #     *   It returns a dictionary containing the average epoch loss and metrics.

        # This method allows you to monitor the training and validation/test performance of your depth estimation model, ensuring that you can track the effectiveness of your 
        # model throughout the training process.
        
        if mode == 'train':
            self.model.train()
            loader: torch.utils.data.DataLoader = self.train_loader
        
        elif mode == 'val':
            self.model.eval()
            loader: torch.utils.data.DataLoader = self.val_loader
            
        elif mode == 'train':
            self.model.eval()
            loader: torch.utils.data.DataLoader = self.test_loader
            
        else:
            raise ValueError('Invalid Mode')
        
        epoch_loss: float = 0.0
        
        epoch_metrics: dict[str, float] = {
            'mean_absolute_error': 0.0,
            'root_mean_squared_error': 0.0,
            'absolute_relative_difference': 0.0,
            'log_10_error': 0.0,
            'berhu': 0.0
        }
        
                
        for batch in loader:
            self.optimizer.zero_grad()
            image: torch.tensor = batch['independent_variable'].to(self.device)
            target: torch.tensor = batch['dependent_variable'].to(self.device)
            
            output: torch.tensor = self.model(image)
            loss: torch.tensor = self.loss_function(output, target)
            
            if mode == 'train':
                loss.backward()
                self.optimizer.step()
            
            epoch_loss += loss.item()
            batch_metrics: dict[str, float] = self.metrics(output, target)
            
            for key, value in batch_metrics.items():
                epoch_metrics[key] += value 
            
        
        epoch_loss /= len(loader)
        for key, value in epoch_metrics.items():
            epoch_metrics[key] /= len(loader)
            
        return {
            'loss': epoch_loss, 'metrics': epoch_metrics
        }                 
        
        
        
        
        
        
     
    def train(self, epochs: int) -> None:
        
        # The train method in the DepthEstimationTrainer class trains the depth estimation model for a specified number of epochs. For each epoch, it runs an iteration 
        # over the training and validation sets using the __run_epoch() method, and it logs the results.

        # Here's a summary of the main components of the train method:

        #     *   It takes an integer epochs as input, which determines the number of times the training process will iterate over the entire training dataset.

        #     *   For each epoch, it runs the __run_epoch method with mode='train' to process the training dataset and with mode='val' to process the validation dataset. 
        #         It stores the results of each run in the train_results and val_results dictionaries, respectively.

        #     *   It updates the self.history dictionary with the training and validation losses and metrics for the current epoch.

        #     *   It logs the current epoch number, training and validation losses, and training and validation metrics using the logging module.

        # This method allows you to train your depth estimation model and monitor its performance on both the training and validation sets. Logging the training progress
        # helps you track the model's performance, detect overfitting or underfitting, and decide when to stop training.
        
        for epoch in range(epochs):
            train_results: dict[str, float | dict[str, float]] = self.__run_epoch(mode= 'train')
            val_results: dict[str, float | dict[str, float]] = self.__run_epoch(mode= 'val')
            
            self.history['train_loss'].append(train_results['loss'])
            self.history['val_loss'].append(val_results['loss'])
            self.history['train_metric'].append(train_results['metrics'])
            self.history['val_metric'].append(val_results['metrics'])
        
        
            #@: logging functionalities here...
            logging.info(f"Epoch {epoch + 1}/{epochs}")
            logging.info(f"Train Loss: {train_results['loss']:.4f}, Val Loss: {val_results['loss']:.4f}")
            logging.info(f"Train Metrics: {train_results['metrics']}")
            logging.info(f"Val Metrics: {val_results['metrics']}")
            
            
            




    def evaluate(self, dataloader: torch.utils.data.DataLoader) -> dict[str, float | dict[str, float]]:
    
        # The evaluate method in the DepthEstimationTrainer class is used to evaluate the depth estimation model on a test dataset. It takes a DataLoader object dataloader 
        # as input and returns a dictionary containing the test loss and test metrics.

        # Here's a summary of the main components of the evaluate method:

        #     *   It takes a DataLoader object dataloader as input, which is the test dataset you want to evaluate the model on.

        #     *   It assigns the input dataloader to the self.test_loader attribute.

        #     *   It calls the __run_epoch() method with mode='test' to process the test dataset and stores the results in the results dictionary.

        #     *   It logs the test loss and test metrics using the logging module.

        #     *   It returns the results dictionary containing the test loss and test metrics.

        # This method allows you to evaluate your depth estimation model on a test set and obtain its performance metrics. This information can be useful to compare
        # different models, choose the best model for your application, and assess the model's generalization performance on new data.
        
        self.test_loader = dataloader
        results: dict[str, float | dict[str, float]] = self.__run_epoch(mode= 'test')
        
        logging.info(f"Test Loss: {results['loss']:.4f}")
        logging.info(f"Test Metrics: {results['metrics']}")
        
        return results
        
    
    
    
    
    
    
    def plot_metrics(self) -> None:
        
        # The plot_metrics method in the DepthEstimationTrainer class is used to plot the training and validation loss and metrics over time. This function generates two 
        # subplots within a single figure. The first subplot displays the training and validation loss, while the second subplot shows the training and validation metrics.

        # Here's a summary of the main components of the plot_metrics method:

        #     *   It creates a figure with two subplots, ax1 and ax2, using plt.subplots.
            
        #     *   On the first subplot (ax1), it plots the training and validation loss from the self.history dictionary.

        #     *   It sets the title, x-label, and y-label for the first subplot and adds a legend.

        #     *   On the second subplot (ax2), it iterates through the metric names in the train_metric key of the self.history dictionary and plots the training and 
        #         validation metric values for each metric.

        #     *   It sets the title and x-label for the second subplot and adds a legend.

        #     *   Finally, it calls plt.show() to display the figure with the two subplots.

        # This method is useful for visualizing the model's training progress and can help you identify if the model is overfitting, underfitting, or converging well. 
        # It can also be helpful for tuning hyperparameters and deciding when to stop training.
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize= (15, 5))

        ax1.plot(self.history["train_loss"], label= "Train Loss")
        ax1.plot(self.history["val_loss"], label= "Val Loss")
        ax1.set_title("Loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.legend()

        for metric_name in self.history['train_metric'][0].keys():
            train_metric_values = [metrics[metric_name] for metrics in self.history["train_metric"]]
            val_metric_values = [metrics[metric_name] for metrics in self.history["val_metric"]]
            ax2.plot(train_metric_values, label=f"Train {metric_name}")
            ax2.plot(val_metric_values, label=f"Val {metric_name}")

        ax2.set_title("Metrics")
        ax2.set_xlabel("Epoch")
        ax2.legend()

        plt.show()

        
        
        
        
        
    def save_history(self, file_path: str) -> None:
        
        # The save_history method in the DepthEstimationTrainer class is used to save the training history, which includes the training and validation loss and metrics, 
        # to a file. The method takes a file path as its argument and saves the self.history dictionary to that file in NumPy's .npy format.

        # Here's a summary of the main components of the save_history method:

        #     *   It takes a file_path argument as a string, which represents the path to the file where the training history will be saved.

        #     *   It calls np.save to save the self.history dictionary to the specified file in the .npy format.

        # This method is useful for preserving the training history for later analysis or for comparing different models or training runs. When you want to load the 
        # saved history, you can use np.load(file_path, allow_pickle=True).item() to load the dictionary back into memory.
        
        np.save(file_path, self.history)
        
    
    
    
    
    
    
    def save_model(self, file_path: str) -> None:
        
        # The save_model method in the DepthEstimationTrainer class is used to save the trained model's state_dict to a file. The method takes a file path as its argument 
        # and saves the model's state_dict to that file in PyTorch's native format.

        # Here's a summary of the main components of the save_model method:

        #     *   It takes a file_path argument as a string, which represents the path to the file where the model's state_dict will be saved.

        #     *   It calls torch.save to save the self.model.state_dict() to the specified file.

        # This method is useful for preserving the trained model for later use or for sharing the model with others. To load the saved model later, you can use the following steps:

        #     *   Instantiate the model architecture again.

        #     *   Call model.load_state_dict(torch.load(file_path)) to load the saved state_dict into the new model instance.

        #     *   If necessary, move the model to the desired device using model.to(device).

        # This way, you can resume training, fine-tune the model on a different dataset, or use the model for inference tasks.
        
        torch.save(self.model.state_dict(), file_path)
        
    
    
    
    
    
    @classmethod
    def load_and_evaluate(cls, model_path: str, test_loader: torch.utils.data.DataLoader, 
                                                device: torch.device) -> dict[str, float | dict[str, float]]:
        
        # The load_and_evaluate class method in the DepthEstimationTrainer class is designed to load a pre-trained model from a file and evaluate it on a given test dataset. The method 
        # takes the model path, test loader, and the device as arguments, and returns the evaluation results (loss and metrics).

        # Here's an overview of the main steps in the load_and_evaluate method:

        #     *   The method takes model_path (a string representing the file path to the pre-trained model), test_loader (a DataLoader object containing the test dataset), and 
        #         device (a torch.device object representing the target device, e.g., 'cpu' or 'cuda') as arguments.

        #     *   The pre-trained model's state_dict is loaded using torch.load(model_path, map_location=device) and assigned to the state_dict variable.

        #     *   The model's state_dict is updated using the model.load_state_dict(state_dict) method.

        #     *   The model is set to evaluation mode using the model.eval() method.

        #     *   A new instance of the DepthEstimationTrainer class, called trainer, is created using the class method cls. The train_loader and val_loader are set to None since 
        #         they are not needed for evaluation.

        #     *   The __run_epoch() method is called with mode='test', and the evaluation results (loss and metrics) are returned as a dictionary.

    
        # This method is useful when you want to evaluate a pre-trained model on a test dataset without training the model again. The evaluation results can be used to assess the
        # model's performance on unseen data and to compare it with other models.
        
        model = model.to(device)
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()

        trainer: DepthEstimationTrainer = cls(
            model,
            train_loader=None,
            val_loader=None,
            test_loader=test_loader,
            device=device
        )
        results: dict[str, float | dict[str, float]] = trainer.__run_epoch(mode='test')
        return results
    
    
        
        
            
        
        

#@: Driver Code
if __name__.__contains__('__main__'):
    #@: NOTE: Use Case Example 
    train_dataset = DepthEstimationDataset(...)  # Add arguments for your dataset
    val_dataset = DepthEstimationDataset(...)  # Add arguments for your dataset
    test_dataset = DepthEstimationDataset(...)  # Add arguments for your dataset

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Instantiate the depth estimation model
    model = DepthEstimationModel(...)  # Add arguments for your model

    # Create an instance of the DepthEstimationTrainer class
    trainer = DepthEstimationTrainer(model, train_loader, val_loader, test_loader)

    # Train the model
    trainer.train(epochs= 100)

    # Evaluate the model
    results = trainer.evaluate(test_loader)
    print("Evaluation results:", results)

    # Save the model and history
    trainer.save_model("depth_estimation_model.pth")
    trainer.save_history("depth_estimation_history.npy")

    # Plot the metrics
    trainer.plot_metrics()
    
    
    
    
    