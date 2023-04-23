
from __future__ import annotations
from abc import ABC, abstractmethod


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
    In AI, a callback is a set of functions that are executed at specific points during the training of a machine learning model. 
    These functions allow you to access and manipulate the model's internal state during training and validation, and can be used 
    to implement various tasks such as logging, saving model checkpoints, stopping training early, adjusting the learning rate, and more.

    The use of callbacks in AI is to provide a flexible and customizable framework for monitoring and controlling the training process of a model. 
    By implementing your own callbacks, you can tailor the training process to your specific needs and goals, and use advanced techniques like early stopping, 
    model checkpointing, and dynamic learning rate adjustment to improve the performance of your model.

    Callbacks can also be used to implement more complex tasks like data augmentation, adversarial training, and model ensembling, and are a crucial component 
    of many advanced AI applications.

'''

# This code defines an abstract base class called Callback, which represents a generic callback class that can be used in a training loop. 
# The class defines a set of abstract methods that correspond to different stages of the training process. These methods are intended to be 
# implemented by concrete subclasses that inherit from this Callback class.

# Here's a brief explanation of each method in the Callback class:

#     *   on_train_start: Called when the training process starts.

#     *   on_train_end: Called when the training process ends.

#     *   on_epoch_start: Called at the beginning of each training epoch.

#     *   on_epoch_end: Called at the end of each training epoch.

#     *   on_batch_start: Called at the beginning of each training batch.

#     *   on_batch_end: Called at the end of each training batch.

#     *   on_backward_start: Called before the backpropagation step.

#     *   on_backward_end: Called after the backpropagation step.

#     *   on_optimizer_step_start: Called before the optimizer step.

#     *   on_optimizer_step_end: Called after the optimizer step.

#     *   on_learning_rate_update: Called when the learning rate is updated.

#     *   on_gradient_clipping: Called when gradient clipping is performed.

#     *   on_data_loading_start: Called before data loading begins.

#     *   on_data_loading_end: Called after data loading finishes.

#     *   on_model_saving: Called when the model is being saved.

#     *   on_model_loading: Called when the model is being loaded.

#     *   on_checkpoint_save: Called when a checkpoint is being saved. It takes a dictionary containing the checkpoint data as an argument.

#     *   on_checkpoint_load: Called when a checkpoint is being loaded. It takes a dictionary containing the checkpoint data as an argument.

# All of the methods are decorated with the @abstractmethod decorator, which means they must be implemented by any concrete class that inherits from Callback. 
# This ensures that the derived classes provide a consistent interface for different stages of the training process.

# The Callback class can be used as a blueprint for creating custom callback classes for various training frameworks or specific use cases. By inheriting from the 
# Callback class and implementing the required methods, you can create custom callback classes tailored to your specific needs.



class Callback(ABC):
    @abstractmethod
    def on_train_start(self) -> None:
        '''
        This line of code defines an abstract method called on_train_start in the Callback abstract base class. The @abstractmethod 
        decorator indicates that this method must be implemented by any concrete subclass that inherits from the Callback class. 
        The on_train_start method is expected to return None.

        The on_train_start method is meant to be called at the beginning of the training process, allowing you to execute any custom code or 
        logic when training starts. When creating a custom callback class, you would provide your own implementation of this method to perform 
        any required actions at this stage of the training process.
        '''
        ...



    @abstractmethod
    def on_train_end(self) -> None:
        '''
        This line of code defines an abstract method called on_train_end in the Callback abstract base class. The @abstractmethod decorator 
        indicates that this method must be implemented by any concrete subclass that inherits from the Callback class. The on_train_end 
        method is expected to return None.

        The on_train_end method is meant to be called at the end of the training process, allowing you to execute any custom code or 
        logic when training finishes. When creating a custom callback class, you would provide your own implementation of this method to 
        perform any required actions at this stage of the training process, such as cleaning up resources or generating reports.
        '''
        ...


    @abstractmethod
    def on_epoch_start(self) -> None:
        '''
        This line of code defines an abstract method called on_epoch_start in the Callback abstract base class. The @abstractmethod decorator 
        indicates that this method must be implemented by any concrete subclass that inherits from the Callback class. The on_epoch_start 
        method is expected to return None.

        The on_epoch_start method is meant to be called at the beginning of each training epoch, allowing you to execute any custom code or 
        logic when a new epoch starts. When creating a custom callback class, you would provide your own implementation of this method to 
        perform any required actions at this stage of the training process, such as resetting counters or logging the start of a new epoch.
        '''
        ...
        
        
    @abstractmethod
    def on_epoch_end(self) -> None:
        '''
        This line of code defines an abstract method called on_epoch_end in the Callback abstract base class. The @abstractmethod decorator 
        indicates that this method must be implemented by any concrete subclass that inherits from the Callback class. The on_epoch_end 
        method is expected to return None.

        The on_epoch_end method is meant to be called at the end of each training epoch, allowing you to execute any custom code or 
        logic when an epoch finishes. When creating a custom callback class, you would provide your own implementation of this method 
        to perform any required actions at this stage of the training process, such as calculating and logging epoch-level metrics, saving 
        a model checkpoint, or updating learning rates.
        '''
        ...



    @abstractmethod
    def on_batch_start(self) -> None:
        '''
        This code snippet defines an abstract method on_batch_start in the Callback abstract base class. The purpose of this method is to 
        provide a hook for users to implement custom behavior at the start of each training batch.

        By inheriting from the Callback abstract base class, users can define their own on_batch_start method that is executed at the start 
        of each batch during training. This method can be used to implement custom behavior, such as logging batch statistics, modifying 
        learning rate, or manipulating the data.
        '''
        ...



    @abstractmethod
    def on_batch_end(self) -> None:
        '''
        This is an abstract method in the Callback abstract base class, which is called at the end of each batch during the training phase 
        of the model. The on_batch_end method can be implemented in a subclass to perform custom actions such as logging batch-level metrics, 
        updating learning rate, or modifying gradients before backpropagation.

        During the training phase, after each batch of data is fed to the model for processing and the corresponding loss and gradients are 
        computed, the on_batch_end method is called to perform any custom actions that the subclass has implemented. For example, if the 
        subclass has implemented on_batch_end to log batch-level metrics, the method will be called at the end of each batch to print or 
        save the metrics to a file for analysis.
        '''
        ...



    @abstractmethod
    def on_backward_start(self) -> None:
        '''
        This is an abstract method defined in the Callback abstract base class. It specifies that any class that inherits from the Callback 
        class must implement the on_backward_start method.

        This method is called at the start of the backward pass through the model during training. It can be used to perform any necessary 
        operations or logging before the backward pass starts.

        For example, one can use this method to start tracking the gradients of the model's parameters or to print the loss value 
        before gradients are computed.
        '''
        ...



    @abstractmethod
    def on_backward_end(self) -> None:
        '''
        The on_backward_end method is an abstract method in the Callback abstract base class, which is called at the end of the backward pass 
        during training.

        During the backward pass, the gradients of the loss with respect to the model parameters are computed and stored in the model's 
        parameter gradients. Once the gradients have been computed, this method is called, and any necessary operations can be performed. 
        For example, this method can be used to monitor the gradients and perform gradient clipping to prevent exploding gradients.

        This method can be implemented in a concrete callback class that inherits from the Callback abstract base class to customize the 
        training process according to the user's needs.
        '''
        ...
        


    @abstractmethod
    def on_optimizer_step_start(self) -> None:
        '''
        This is an abstract method of the Callback abstract base class. It is called at the beginning of each optimizer step during the 
        training process.

        During each optimizer step, the model parameters are updated based on the gradients computed during the previous backward pass. 
        This method can be used to perform any operations before the optimizer step is taken, such as logging or modifying the gradients.

        The on_optimizer_step_start method takes no arguments and returns nothing, and should be implemented by any subclasses of Callback 
        that want to perform some operation at the beginning of each optimizer step.
        '''
        ...
        
        

    @abstractmethod
    def on_optimizer_step_end(self) -> None:
        '''
        The on_optimizer_step_end() method is an abstract method of the Callback abstract base class, which is called at the end of an optimizer 
        step during training.

        During the optimizer step, the gradients are computed and the optimizer is updated using these gradients. The on_optimizer_step_end() 
        method is called at the end of this process, after the optimizer has been updated.

        This method can be used in the training phase of the model to perform any actions that should be taken at the end of an optimizer step, 
        such as logging optimizer statistics or updating the learning rate. It can also be used to modify the optimizer parameters before the next step.
        '''
        ...



    @abstractmethod
    def on_learning_rate_update(self) -> None:
        '''
        This is an abstract method in the Callback abstract base class. It represents a callback that is called when the learning rate is updated 
        by the optimizer during training.

        In deep learning models, the learning rate is an important hyperparameter that determines the step size at each iteration of the optimizer.
        During training, the optimizer adjusts the learning rate based on various factors such as the current epoch or the validation loss. 
        This method allows the user to perform any desired action when the learning rate is updated during training, such as printing the new
        learning rate or logging it for analysis.

        This method takes no input parameters and returns nothing. Any implementation of this method should perform the desired action when the
        learning rate is updated.
        '''
        ...

    
    @abstractmethod
    def on_gradient_clipping(self) -> None:
        '''
        This abstract method in the Callback abstract base class is used to define a callback method that can be executed during the training phase
        when gradient clipping is applied to the optimizer.

        Gradient clipping is a technique used to prevent gradients from exploding during training. When the gradient is too large, it can cause 
        the weights to update significantly, leading to an unstable model. To prevent this, the gradient is clipped to a maximum value, which 
        limits the magnitude of the gradient.

        The on_gradient_clipping method in the Callback class can be implemented in a derived class to perform additional operations before or
        after the gradient clipping is applied.
        '''
        ...
        


    @abstractmethod
    def on_data_loading_start(self) -> None:
        '''
        This is an abstract method defined in the Callback abstract base class. It specifies that any concrete implementation of the Callback class 
        must have a method named on_data_loading_start which takes no parameters and returns nothing.

        This method can be used in the training phase of a model to specify actions that should be taken when data loading starts. For example, 
        a ProgressBarCallback implementation of on_data_loading_start could display a loading bar to indicate to the user that the model is currently
        loading data. Another implementation might perform some preprocessing or data augmentation on the loaded data before passing it to the model.
        '''
        ...


    
    @abstractmethod
    def on_data_loading_end(self) -> None:
        '''
        The on_data_loading_end() method is an abstract method of the Callback abstract base class, which is called when the data loading is completed 
        for each epoch during the training phase of a model. This method is useful to perform any necessary post-processing on the loaded data or to 
        print some information related to the data loading.

        The method takes no arguments and returns nothing. It is the responsibility of the child class to implement this method to perform the required
        actions at the end of data loading for each epoch.
        '''
        ...


    @abstractmethod
    def on_model_saving(self) -> None:
        '''
        This method is an abstract method defined in the abstract base class Callback. It specifies the behavior to be implemented when the model is 
        saved during the training process.

        The implementation of this method will vary depending on the specific use case, but it could involve saving the model's weights or architecture, 
        or saving other related information such as the optimizer state or training history.

        The on_model_saving method can be called by the training loop at specific points during the training process, allowing the callback to customize 
        the saving behavior as needed. For example, the callback could be configured to save the model's weights after every epoch, or to only save the 
        model when a certain performance threshold has been reached.

        Concrete subclasses of Callback can override this method to implement custom behavior when the model is saved during training.
        '''
        ...

    
    @abstractmethod
    def on_model_loading(self) -> None:
        '''
        The on_model_loading method is an abstract method defined in the Callback abstract base class. It is called when a saved model is loaded during 
        training. This method is intended to be implemented by any class that inherits from Callback and needs to perform some action when a model 
        is loaded. For example, one could use this method to print a message indicating that the model has been loaded successfully or to set the 
        state of some internal variables based on the loaded model.
        '''
        ...


    @abstractmethod
    def on_checkpoint_save(self, checkpoint: dict[str, Any]) -> None:
        '''
        This is an abstract method in the Callback class that takes a dictionary checkpoint as input and returns nothing. This method is called when 
        a checkpoint is saved during training. A checkpoint is a snapshot of the model and optimizer state at a particular point in training that 
        can be used to resume training or to evaluate the model.

        Subclasses that inherit from Callback must implement this method and define what should happen when a checkpoint is saved, for example, 
        saving the model's state_dict and optimizer's state_dict to a file.
        '''
        ...

    
    @abstractmethod
    def on_checkpoint_load(self, checkpoint: dict[str, Any]) -> None:
        '''
        This method is defined in the abstract base class Callback. It is intended to be overridden in derived classes to provide functionality 
        to execute when a model checkpoint is loaded during the training process. The checkpoint argument is a dictionary containing the state 
        of the model and optimizer at the time the checkpoint was saved.

        The implementation of this method can include restoring the state of the model and optimizer from the checkpoint, as well as any other tasks 
        that need to be performed when a checkpoint is loaded.

        This method can be used in the training phase of a machine learning model to provide additional functionality when loading a checkpoint, 
        such as setting a new learning rate, resuming training from a previous checkpoint, or modifying the model architecture before continuing training.
        '''
        ...



#@: Driver Code
if __name__.__contains__('__main__'):
    ...