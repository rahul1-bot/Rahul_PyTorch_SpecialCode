
from __future__ import annotations
from abc import ABC, abstractmethod
from torchsummary import summary
import torch, logging, time

#@: NOTE: Timer Callback

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
    
    
    
    

class Timer(Callback):
    # The Timer class is a Callback that's designed to keep track of the total time spent in training, validation, and testing loops. 
    # The primary motivation behind creating this class is to set a time limit for the training loop, after which the loop will be 
    # interrupted. This can be very useful when you're running experiments that are time-bound or if you want to avoid overusing 
    # resources.
    
    #     *   Constructor (__init__ method): When you create a new instance of the Timer class, you must specify a time_limit_sec 
    #                                        parameter which is the maximum number of seconds you want to allow for training. 
    #                                        The __init__ method also sets up a logger that can be used for printing messages.


    #     *   on_train_start method: This method is called when the training process begins. It records the current time in 
    #                                self.start_time.


    #     *   on_batch_end method: This method is called at the end of every batch during training. If the self.start_time is 
    #                              set (which should be the case if on_train_start was called), this method checks if the elapsed 
    #                              time since self.start_time exceeds the time limit (self.time_limit_sec). If the time limit has 
    #                              been reached, it logs a message and sets a should_stop flag in the trainer to True. 
    #                              This flag signals the trainer to stop training.


    #     *   on_train_end method: This method is called at the end of the training process. It calculates the total time spent 
    #                              in training (current time - self.start_time) and logs a message with this information.
    
    def __init__(self, time_limit_sec: int | float) -> None:
        self.start_time: Optional[float] = None
        self.time_limit_sec: int | float = time_limit_sec
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(handler)



    def on_train_start(self, trainer: 'DepthEstimationTrainer' | 'Trainer') -> None:
        self.start_time: float = time.time()



    def on_train_end(self, trainer: 'DepthEstimationTrainer' | 'Trainer') -> None:
        if self.start_time is not None:
            elapsed_time: float = time.time() - self.start_time
            self.logger.info(f'Total training time: {elapsed_time} seconds')



    def on_batch_end(self, trainer: 'DepthEstimationTrainer' | 'Trainer') -> None:
        if self.start_time is not None and time.time() - self.start_time > self.time_limit_sec:
            self.logger.info(f'Time limit reached after {self.time_limit_sec} seconds. Stopping training...')
            trainer.should_stop = True
            
     
     
    #@: add remaining methods for code consistency 
    #@: otherwise this class won't work...
    #@: OR remove @abstractmethod from the callback class
    



#@: NOTE : Update the Trainer or DepthEstimationTrainer class 

# class DepthEstimationTrainer:
#     # ...

#     def train(self):
#         for callback in self.callbacks:
#             callback.on_train_start(self)

#         for epoch in range(self.epochs):
#             for batch in self.train_loader:
#                 # Perform forward pass, backward pass, and optimization
                
#                 for callback in self.callbacks:
#                     callback.on_batch_end(self)
                
#                 if self.should_stop:
#                     break

#             if self.should_stop:
#                 break

#         for callback in self.callbacks:
#             callback.on_train_end(self)
            
            
            
            
#@: Driver Code
if __name__.__contains__('__main__'):
    #@: NOTE : Use case eg:
    # trainer: DepthEstimationTrainer = DepthEstimationTrainer(model, data_module, ...)
    # trainer.add_callback(Timer(time_limit_sec=3600))  # 1 hour limit
    # trainer.train() 
    ...
    
    
    