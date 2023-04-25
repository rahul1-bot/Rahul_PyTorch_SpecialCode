
from __future__ import annotations
import logging
import torch
from torch.optim import Optimizer
from torch.nn import Module

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

# NOTE : REFER Core_Callbacks.py file NOTE 

from Core_Callbacks import Callback

# NOTE : Don't Alter the Code 
# NOTE : Remove the @abstractmethod, if needed 

logging.basicConfig(level= logging.INFO)
logger = logging.getLogger(__name__)


__doc__: str = r'''
    Early stopping is a regularization technique used in training deep learning and machine learning models to prevent overfitting.
    Overfitting occurs when a model learns the training data too well, capturing noise and patterns that are not relevant to the 
    underlying problem, which results in poor generalization to new, unseen data.

    The basic idea of early stopping is to monitor a performance metric (e.g., validation loss) during the training process. If 
    the performance metric stops improving (or even starts to degrade) for a certain number of consecutive epochs (iterations over the entire dataset),
    the training process is stopped early. This helps to prevent the model from fitting the noise in the training data and ensures better generalization.

    The key components of early stopping are:

        *   Performance metric: A metric, such as validation loss or accuracy, is used to evaluate the model's performance during training.
        
        *   Patience: The number of consecutive epochs without improvement in the performance metric before stopping the training. It is a 
                      hyperparameter that determines how long to wait before stopping.

        *   Minimum delta: A threshold for the change in the performance metric, below which the improvement is considered insignificant. If the change
                           in the performance metric is smaller than the minimum delta, the counter for patience is incremented.

    Early stopping is typically used in conjunction with a model checkpointing mechanism to save the best model observed during training, so that 
    the best model can be used for inference, rather than the final model obtained at the time of early stopping.
'''


class EarlyStopping(Callback):
    def __init__(self, patience: int, min_delta: Optional[float] = 0.0) -> None:
        # This code snippet defines an EarlyStopping class that inherits from the Callback class. The EarlyStopping class is used to implement 
        # early stopping during model training, which helps to prevent overfitting by stopping the training process when the model's performance 
        # on the validation set stops improving.

        # In the __init__ method, the following parameters are initialized:

        #     *   patience: The number of consecutive epochs without improvement in the performance metric (e.g., validation loss) before stopping the training.

        #     *   min_delta: A threshold for the change in the performance metric, below which the improvement is considered insignificant.

        #     *   counter: A counter variable to track the number of consecutive epochs without significant improvement in the performance metric.

        #     *   best_loss: The best (lowest) loss value observed so far during the training process.

        # The __init__ method has an optional min_delta parameter, which defaults to 0.0, and a mandatory patience parameter. The class also has a type 
        # hint for the __init__ method, indicating that it returns None.
        
        self.patience = patience
        self.min_delta = min_delta
        self.counter: int = 0
        self.best_loss: float = float('inf')



    def on_train_start(self) -> None:
        self.counter: int = 0
        self.best_loss: float = float('inf')



    def on_epoch_end(self) -> None:
        #@ Assuming the loss is already computed and 
        #@ available as an attribute of the callback
        current_loss: float = self.loss
        
        if current_loss + self.min_delta < self.best_loss:
            self.best_loss = current_loss
            self.counter = 0
        
        else:
            self.counter += 1
            if self.counter >= self.patience:
                logger.info("Early stopping triggered")
                # Stop the training process by raising an exception
                raise KeyboardInterrupt



    # Other methods should be implemented as no-op for EarlyStopping
    # NOTE : Remove the @abstractmethod, if needed 
    def on_train_end(self) -> None:
        pass

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

    def on_checkpoint_save(self, checkpoint: Dict[str, Any]) -> None:
        pass

    def on_checkpoint_load(self, checkpoint: Dict[str, Any]) -> None:
        pass




#@: Driver Code 
if __name__.__contains__('__main__'):
    ...