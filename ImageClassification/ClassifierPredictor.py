import torch
from PIL import Image


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
    The ClassifierPredictor class is designed to simplify the process of making predictions using a trained PyTorch deep learning model, 
    particularly for image classification tasks. It accepts a trained model and an optional dictionary of image transforms during initialization.

    Here's a breakdown of the class components:

    *   __init__: This is the constructor method, which initializes the predictor class with a PyTorch model (model) and an optional dictionary of 
        transforms (transform_dict). The dictionary keys are the transform names, and the values are the transform functions.

    *   __repr__: This method returns a string representation of the predictor object, which includes the model and transform dictionary. 
        It's useful for better visualization of the object when printed.

    *   predict_result: This method accepts an image file path (image_path) as input and returns the predicted class index as an integer. It starts 
        by opening and converting the image to RGB format. If the transform_dict is provided, it applies the corresponding image transformations sequentially. 
        After preprocessing the image, it feeds the image through the model and obtains the output. Finally, it returns the index of the highest predicted class probability.


    To use this class, you would first train a PyTorch image classification model and define the necessary image transforms. Then, create an instance of the 
    ClassifierPredictor class with the trained model and transform dictionary. You can then call the predict_result method with an image path to obtain predictions for new images.

'''


class ClassifierPredictor:
    def __init__(self, model: torch.nn.Module, transform_dict: Optional[dict[str, Callable[Any]]] = None) -> None:
        self.model = model
        self.transform_dict = transform_dict
        

        
        
    def __repr__(self) -> str:
        return f"Predictor(model={repr(self.model)}, transform_dict={repr(self.transform_dict)})"

        
        
    def predict_result(self, image_path: str) -> int:
        image: PIL.Image = Image.open(image_path).convert('RGB')
        if self.transform_dict is not None:
            for transform_func in self.transform_dict.values():
                image: Any = transform_func(image)
                
        preprocessed_image: torch.tensor = image.unsqueeze(0)
        
        with torch.no_grad():
            output: torch.tensor = self.model(preprocessed_image)
            prediction: int = torch.argmax(output, dim= 1).item()
            
            
        return prediction




#@: Driver Code
if __name__.__contains__('__main__'):
    
    #@: created this dict in previous module...
    transform_dict: dict[str, Callable[Any]] = {
        'RandomResizedCropTransform': RandomResizedCropTransform(),
        'RandomHorizontalFlipTransform': RandomHorizontalFlipTransform(),
        'RandomRotationTransform': RandomRotationTransform(),
        'RandomVerticalFlipTransform': RandomVerticalFlipTransform(),
        'ColorJitterTransform': ColorJitterTransform(),
        'GrayscaleTransform': GrayscaleTransform(),
        'ToTensorTransform': ToTensorTransform(),
        'NormalizeTransform': NormalizeTransform()
    }
    
    
    
    
    
    