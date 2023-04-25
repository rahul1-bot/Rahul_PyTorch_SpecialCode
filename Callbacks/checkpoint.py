
from __future__ import annotations
from abc import ABC, abstractmethod
import os
import torch

#@: NOTE : Check Core_Callback.py file 
#@: NOTE BONUS: Model CheckPoint 

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




#@: Genetric Callback class updated for depthEstimation Task 
#@: NOTE : Update the code according to your need 
class Callback(ABC):
    @abstractmethod
    def on_train_start(self, trainer: 'Trainer') -> None:
        pass

    @abstractmethod
    def on_train_end(self, trainer: 'Trainer') -> None:
        pass

    @abstractmethod
    def on_epoch_start(self, trainer: 'Trainer') -> None:
        pass

    @abstractmethod
    def on_epoch_end(self, trainer: 'Trainer') -> None:
        pass

    @abstractmethod
    def on_batch_start(self) -> None:
        pass

    @abstractmethod
    def on_batch_end(self) -> None:
        pass

    @abstractmethod
    def on_backward_start(self) -> None:
        pass

    @abstractmethod
    def on_backward_end(self) -> None:
        pass

    @abstractmethod
    def on_optimizer_step_start(self) -> None:
        pass

    @abstractmethod
    def on_optimizer_step_end(self) -> None:
        pass

    @abstractmethod
    def on_learning_rate_update(self) -> None:
        pass

    @abstractmethod
    def on_gradient_clipping(self) -> None:
        pass

    @abstractmethod
    def on_data_loading_start(self) -> None:
        pass

    @abstractmethod
    def on_data_loading_end(self) -> None:
        pass

    @abstractmethod
    def on_model_saving(self) -> None:
        pass

    @abstractmethod
    def on_model_loading(self) -> None:
        pass

    @abstractmethod
    def on_checkpoint_save(self, checkpoint: dict[str, Any]) -> None:
        pass

    @abstractmethod
    def on_checkpoint_load(self, checkpoint: dict[str, Any]) -> None:
        pass




__doc__: str = r'''
    In the context of artificial intelligence (AI) and machine learning, a checkpoint refers to a snapshot of the state of a model 
    during the training process. This snapshot typically includes the model's weights, optimizer state, training hyperparameters, and 
    any other relevant information that allows you to resume training from a specific point if needed.

    Checkpoints are crucial for several reasons:

            *   Fault tolerance: If training is interrupted due to issues such as hardware failure or software crashes, you can resume the 
                                 process from the last checkpoint rather than starting from scratch.

            *   Model selection: During training, a model's performance on a validation set may fluctuate. By saving checkpoints at regular intervals 
                                 or when performance improves, you can choose the best model based on its validation performance.

            *   Resource management: Training deep learning models can be computationally expensive and time-consuming. Checkpoints allow you to pause 
                                     training and release resources when needed and then resume later.

    In practice, checkpoints are often saved as files on disk, and many deep learning frameworks provide utilities to simplify the process of creating, 
    saving, and loading checkpoints.
'''

class Checkpoint(Callback):
    def __init__(self, save_dir: str, save_freq: int) -> None:
        
        # This code snippet defines a Checkpoint class that inherits from another class called Callback. The purpose of the Checkpoint class is to create 
        # a functionality that saves the training progress at certain intervals during the training process. This way, if something goes wrong or the training 
        # is interrupted, you can continue from the last saved checkpoint instead of starting over.

        # In this Checkpoint class, there is an __init__ method, which is the constructor for the class. It takes two arguments:

        #         *   save_dir: This is a string representing the directory where the checkpoint files will be saved.

        #         *   save_freq: This is an integer that determines how often a checkpoint should be saved, for example, every 10 training steps.

        # Inside the constructor, the method assigns the input arguments to instance variables self.save_dir and self.save_freq. Additionally, it uses the os.makedirs
        # function to create the directory specified by save_dir if it doesn't already exist. This ensures that the directory is available to save checkpoint files.
        
        self.save_dir = save_dir
        self.save_freq = save_freq
        os.makedirs(self.save_dir, exist_ok= True)





    def on_epoch_end(self, trainer: 'Trainer') -> None:
        
        # This code snippet defines an on_epoch_end method for the Checkpoint class, which is designed to be called when an epoch (a full iteration through 
        # the training dataset) has ended. The purpose of this method is to save a checkpoint if the conditions for saving are met.

        # The method takes one argument:

        #         *   trainer: This is an instance of the Trainer class that is responsible for handling the training process. By passing the Trainer object, 
        #                      the method has access to information about the training state, such as the current epoch.

        # Inside the method, there is an if statement that checks if the current epoch number plus one, modulo the checkpoint saving frequency (self.save_freq), 
        # is equal to zero. This condition ensures that a checkpoint is saved only when the current epoch is a multiple of the specified saving frequency.

        # If the condition is met, the method calls self.on_checkpoint_save(trainer). This is another method of the Checkpoint class that is responsible for actually 
        # saving the checkpoint file. The trainer object is passed as an argument so that the method has access to the necessary information to create the checkpoint.
        
        if (trainer.current_epoch + 1) % self.save_freq == 0:
            self.on_checkpoint_save(trainer)






    def on_checkpoint_save(self, trainer: 'Trainer') -> None:
        
        # This code snippet defines the on_checkpoint_save method for the Checkpoint class. The purpose of this method is to save a checkpoint containing the current 
        # state of the model, optimizer, and other relevant information at a specified epoch.

        # The method takes one argument:

        #         *   trainer: This is an instance of the Trainer class that is responsible for handling the training process. By passing the Trainer object, 
        #                      the method has access to information about the training state.

        
        # Inside the method, a dictionary called checkpoint is created to store various pieces of information:

        #         *   'model_state_dict': This key holds the state of the model at the current epoch. trainer.model.state_dict() is called to get the model's 
        #                                 state dictionary.

        #         *   'optimizer_state_dict': This key holds the state of the optimizer at the current epoch. trainer.optimizer.state_dict() is called to get 
        #                                     the optimizer's state dictionary.

        #         *   'epoch': This key holds the current epoch number.

        #         *   'loss': This key holds the running loss value at the current epoch.

        
        # You can also add any other data you'd like to store in the checkpoint by creating additional keys in the checkpoint dictionary.

        # Next, the method creates a string checkpoint_path which specifies the file path where the checkpoint will be saved. The path is constructed by joining 
        # self.save_dir (the directory for storing checkpoints) with the file name checkpoint_epoch_{trainer.current_epoch}.pth, where {trainer.current_epoch} is 
        # replaced with the actual epoch number.

        # The torch.save(checkpoint, checkpoint_path) function is then called to save the checkpoint dictionary to the specified file path.

        # Finally, a message is printed to the console to inform the user that the checkpoint has been saved and provide the file path.
        
        checkpoint: dict[str, Any] = {
            'model_state_dict': trainer.model.state_dict(),
            'optimizer_state_dict': trainer.optimizer.state_dict(),
            'epoch': trainer.current_epoch,
            'loss': trainer.running_loss,
            # Add any other data you want to store in the checkpoint
        }
        
        checkpoint_path: str = os.path.join(self.save_dir, f'checkpoint_epoch_{trainer.current_epoch}.pth')
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved at: {checkpoint_path}")





    def on_checkpoint_load(self, checkpoint_path: str, trainer: 'Trainer') -> None:
        
        # This code snippet defines the on_checkpoint_load method for the Checkpoint class. The purpose of this method is to load a saved checkpoint and restore 
        # the model, optimizer, and other relevant information to their saved states.

        # The method takes two arguments:

        #         *   checkpoint_path: A string representing the file path of the saved checkpoint.

        #         *   trainer: This is an instance of the Trainer class that is responsible for handling the training process. By passing the Trainer object, 
        #                      the method has access to information about the training state.

        # The method first checks if the specified checkpoint_path exists using os.path.exists(checkpoint_path). If the checkpoint file exists, the following 
        # steps are performed:

        #         *   The checkpoint is loaded using torch.load(checkpoint_path), which returns a dictionary containing the saved states of the model, optimizer,
        #             and other relevant information.

        #         *   The model state is restored by calling trainer.model.load_state_dict(checkpoint['model_state_dict']).

        #         *   The optimizer state is restored by calling trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict']).

        #         *   The current epoch number is restored by setting trainer.current_epoch to the saved value from checkpoint['epoch'].

        #         *   The running loss value is restored by setting trainer.running_loss to the saved value from checkpoint['loss'].

        
        # You can also load any other data you stored in the checkpoint by accessing the appropriate keys in the checkpoint dictionary.

        # After restoring the training state, a message is printed to the console to inform the user that the checkpoint has been loaded and provide the file path.

        # If the checkpoint file does not exist, a message is printed to the console stating that no checkpoint was found at the specified file path.
        
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            trainer.model.load_state_dict(checkpoint['model_state_dict'])
            trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            trainer.current_epoch = checkpoint['epoch']
            trainer.running_loss = checkpoint['loss']
            # Load any other data you stored in the checkpoint
            print(f"Checkpoint loaded from: {checkpoint_path}")
        else:
            print(f"No checkpoint found at: {checkpoint_path}")

    # Implement empty methods for the rest of the abstract methods in the Callback class


    def on_epoch_start(self) -> None:
        pass

    def on_batch_start(self) -> None:
        pass

    def on_batch_end(self) -> None:
        pass

    def on_backward_start(self) -> None:
        pass

    def on_backward_end(self) -> None:
        pass

    def on_optimizer_step_start(self) -> None:
        pass

    def on_optimizer_step_end(self) -> None:
        pass

    def on_learning_rate_update(self) -> None:
        pass

    def on_gradient_clipping(self) -> None:
        pass

    def on_data_loading_start(self) -> None:
        pass

    def on_data_loading_end(self) -> None:
        pass

    def on_model_saving(self) -> None:
        pass

    def on_model_loading(self) -> None:
        pass



#@: Now alter the Trainer class 
# class DepthEstimationTrainer:
#     def __init__(self, model: torch.nn.Module, train_loader: torch.utils.data.DataLoader, 
#                  val_loader: torch.utils.data.DataLoader, 
#                  test_loader: Optional[torch.utils.data.DataLoader] = None, 
#                  device: Optional[torch.device] = 'cuda' if torch.cuda.is_available() else 'cpu',
#                  learning_rate: Optional[float] = 0.001,
#                  callback: Optional[Callback] = None,
#                  checkpoint: Optional[Checkpoint] = None) -> None:

#         # Add checkpoint parameter
#         self.checkpoint = checkpoint
#         # Other attributes and initializations...

#     # Other methods...

#     def train(self, epochs: int) -> None:
#         if self.callback:
#             self.callback.on_train_start()
        
#         for epoch in range(epochs):
#             if self.callback:
#                 self.callback.on_epoch_start()
            
#             # Training and validation loops
#             # ...

#             if self.callback:
#                 self.callback.on_epoch_end()
 
 
 
 

#@: Driver Code 
if __name__.__contains__('__main__'):
    ...
    
    
    
    
    