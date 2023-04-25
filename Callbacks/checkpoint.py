from abc import ABC, abstractmethod
import os
import torch

#@: NOTE : Check Core_Callback.py file 


#@: Genetric Callback class updated for depthEstimation Task 
class Callback(ABC):
    @abstractmethod
    def on_train_start(self, trainer: 'DepthEstimationTrainer') -> None:
        pass

    @abstractmethod
    def on_train_end(self, trainer: 'DepthEstimationTrainer') -> None:
        pass

    @abstractmethod
    def on_epoch_start(self, trainer: 'DepthEstimationTrainer') -> None:
        pass

    @abstractmethod
    def on_epoch_end(self, trainer: 'DepthEstimationTrainer') -> None:
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





class Checkpoint(Callback):
    def __init__(self, save_dir: str, save_freq: int) -> None:
        self.save_dir = save_dir
        self.save_freq = save_freq
        os.makedirs(self.save_dir, exist_ok= True)



    def on_epoch_end(self, trainer: 'Trainer') -> None:
        if (trainer.current_epoch + 1) % self.save_freq == 0:
            self.on_checkpoint_save(trainer)



    def on_checkpoint_save(self, trainer: 'Trainer') -> None:
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
    
    
    
    
    