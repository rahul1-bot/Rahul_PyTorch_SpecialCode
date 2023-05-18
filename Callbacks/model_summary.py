from __future__ import annotations
from abc import ABC, abstractmethod
from torchsummary import summary
import torch, logging


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
    
    
    

#@: To maintain Code consistency 
#@: Generate and log the summary of all the layers of the model
class ModelSummary(Callback):
    def __init__(self, input_size: int) -> None:
        self.input_size = input_size
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(handler)



    def on_train_start(self, trainer: 'DepthEstimationTrainer') -> None:
        self.logger.info("Printing model summary...")
        summary_str: str = summary(trainer.model, self.input_size, verbose=0)
        self.logger.info("\n" + summary_str)
    
    
    
    def on_train_end(self, trainer: 'DepthEstimationTrainer') -> None:
        ...

    
    def on_epoch_start(self, trainer: 'DepthEstimationTrainer') -> None:
        ...

    
    def on_epoch_end(self, trainer: 'DepthEstimationTrainer') -> None:
        ...

    
    def on_batch_start(self) -> None:
        ...

    
    def on_batch_end(self) -> None:
        ...

    
    def on_backward_start(self) -> None:
        ...

    
    def on_backward_end(self) -> None:
        ...

    
    def on_optimizer_step_start(self) -> None:
        ...

    
    def on_optimizer_step_end(self) -> None:
        ...

    
    def on_learning_rate_update(self) -> None:
        ...

    
    def on_gradient_clipping(self) -> None:
        ...

    
    def on_data_loading_start(self) -> None:
        ...

    
    def on_data_loading_end(self) -> None:
        ...

    
    def on_model_saving(self) -> None:
        ...

    
    def on_model_loading(self) -> None:
        ...

    
    def on_checkpoint_save(self, checkpoint: dict[str, Any]) -> None:
        ...

    
    def on_checkpoint_load(self, checkpoint: dict[str, Any]) -> None:
        ...
        
        


#@: Driver Code
if __name__.__contains__('__main__'):
    #@: NOTE: Use Case Ex
    
    # trainer: DepthEstimationTrainer = DepthEstimationTrainer(...)
    # trainer <- add_callback(ModelSummary(input_size=(3, 224, 224)))
    
    # trainer.train()
    ...
    
    
    
    