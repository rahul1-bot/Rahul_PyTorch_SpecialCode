from __future__ import annotations
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

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
    *   The code defines a class ObjectDetectionTrainer that is used to train an object detection model. The class takes in various parameters including the 
        model, training and validation datasets, optimizer, device, batch size, number of epochs, and patience.

    *   The constructor initializes the class with these parameters and creates data loaders for the training and validation datasets. It also initializes an 
        empty dictionary to store the metrics of the training process.

    *   The loss_function method defines the loss function for the object detection task. It computes two losses - regression loss using SmoothL1Loss() and 
        classification loss using CrossEntropyLoss(). The two losses are then added to obtain the total loss.

    *   The intersection_over_union method computes the intersection over union (IoU) metric for the object detection task. It takes in two arrays of predicted 
        and true bounding boxes and returns the mean IoU value.

    *   The mean_average_precision method computes the mean average precision (mAP) metric for the object detection task. It takes in two lists of predicted and 
        true bounding boxes, and optionally the number of classes and IoU threshold. It first computes the AP score for each class separately and then takes the mean 
        of all AP scores to obtain the mAP score.

    *   The train method performs the training process for the model. It iterates over the training data loader, computes the forward pass for the inputs, computes the 
        loss using the loss_function method, and backpropagates the loss to update the model parameters. It also computes the IoU and mAP scores for the predicted and true 
        bounding boxes using the intersection_over_union and mean_average_precision methods, respectively. It then updates the metrics dictionary with the computed metrics. 
        The method also prints the epoch number and the average loss, IoU, and mAP scores for the training dataset. If a validation dataset is provided, the method calls the
        evaluate method to compute the metrics for the validation dataset.

    *   The evaluate method computes the metrics for the validation dataset. It iterates over the validation data loader, computes the forward pass for the inputs, computes 
        the loss using the loss_function method, and computes the IoU and mAP scores for the predicted and true bounding boxes using the intersection_over_union and 
        mean_average_precision methods, respectively. It then returns the average loss, IoU, and mAP scores for the validation dataset.

    *   Overall, the ObjectDetectionTrainer class provides a convenient way to train and evaluate object detection models using PyTorch.
'''


__loss_function_docs: str = r'''
    *   The loss_function method in this code is used to calculate the loss for the object detection task. This method takes two inputs, outputs and targets, which are dictionaries
        containing the predicted bounding boxes and class probabilities and the true bounding boxes and labels, respectively.

    *   First, two loss functions are defined using PyTorch's built-in loss functions, nn.SmoothL1Loss() for the regression loss and nn.CrossEntropyLoss() for the classification 
        loss.

    *   Next, the regression loss is calculated by passing the predicted bounding boxes and the true bounding boxes to the criterion_regression loss function. Similarly, 
        the classification loss is calculated by passing the predicted class probabilities and the true labels to the criterion_classification loss function.

    *   Finally, the total loss is calculated by adding the regression loss and the classification loss. The total loss is returned as a torch.tensor.
'''



__intersection_over_union_docs: str = r'''
    *   The intersection_over_union method is a static method that calculates the intersection over union (IoU) score between the predicted bounding boxes and the ground truth 
        bounding boxes. The inputs to the method are two NumPy arrays pred_boxes and true_boxes.

    *   The method first calculates the coordinates of the intersection rectangle between the predicted and true boxes by taking the maximum of the x and y coordinates of the 
        top-left corner of the two boxes and the minimum of the x and y coordinates of the bottom-right corner of the two boxes. The width and height of the intersection 
        rectangle are then calculated as the difference between the x and y coordinates of the top-left and bottom-right corners of the intersection rectangle, respectively.

    *   The area of the intersection rectangle is then calculated as the product of the width and height of the intersection rectangle, after clipping any negative values to zero. 
        The area of the predicted box and the area of the true box are also calculated as the product of their respective widths and heights.

    *   The area of the union between the predicted and true boxes is then calculated as the sum of the areas of the two boxes minus the area of the intersection rectangle.

    *   Finally, the IoU score is calculated as the ratio of the area of the intersection rectangle to the area of the union, after clipping any negative values to zero. The mean 
        IoU score is returned as a float.
'''



__mean_average_precision_docs__: str = r'''
    *   This is a static method to compute the mean average precision (mAP) score for object detection. It takes in two lists: true_boxes_list and pred_boxes_list. Each element 
        of these lists is a dictionary containing bounding_box and label keys.

    *   It also takes in two optional parameters: num_classes which is the number of object classes, and iou_threshold which is the minimum intersection over union (IoU) score to 
        consider a detection as correct.

    *   Inside the method, we first initialize an array ap of size num_classes with zeros. We will calculate the AP score separately for each class.

    *   Then we iterate over each class and extract the bounding_box arrays for both the true and predicted boxes for that class. We concatenate these arrays to get all the true
        and predicted boxes for that class.

    *   We then sort the predicted boxes in decreasing order of confidence score and initialize arrays tp and fp of size num_pred_boxes with zeros. We also initialize an empty list 
        matched_true_boxes to keep track of the true boxes that have already been matched.

    *   We then loop through each predicted box and compute the IoU scores between the predicted box and all the true boxes. We choose the true box with the highest IoU score as 
        the match for the predicted box, as long as the IoU score is greater than or equal to iou_threshold and the true box has not already been matched. If a match is found, 
        we set the corresponding element in tp to 1, otherwise we set the corresponding element in fp to 1.

    *   We then compute the cumulative sum of tp and fp, and compute the precision and recall values. We use these to compute the AP score for that class using the
        formula np.sum((recall[:-1] - recall[1:]) * precision[:-1]). We store this value in the ap array.

    *   Finally, we compute the mean of the ap array and return this value as the mAP score.
'''


__train_docs__: str = r'''
    *   This is the implementation of the train method in the ObjectDetectionTrainer class. This method is used to train the object detection model for the specified number of epochs 
        using the provided train dataset and optimizer.

    *   At the beginning of each epoch, early_stopping_counter is initialized to zero and best_val_loss is set to infinity. Then, the model is set to the training mode and total_loss, 
        total_iou, all_true_boxes, and all_pred_boxes are initialized to zero.

    *   The method then iterates over the batches in the train dataset using self.train_loader. For each batch, the optimizer is zeroed, and the input and target tensors are retrieved 
        from the batch and converted to the device (self.device) used for training. Then, the model is called with the inputs tensor to obtain the predicted outputs. The loss 
        function is then calculated using the predicted and target outputs, and the gradients are calculated using backpropagation. The optimizer is then updated using the gradients.

    *   The total loss and IoU score are accumulated for each batch, and the predicted and target outputs are appended to all_pred_boxes and all_true_boxes, respectively.

    *   At the end of each epoch, the average loss, IoU score, and mean average precision (mAP) are calculated over all batches in the train dataset using the accumulated 
        total_loss and total_iou, and all_true_boxes and all_pred_boxes, respectively. The train loss, IoU score, and mAP are then added to the self.metrics dictionary under 
        the 'train' key.

    *   If a validation dataset (self.val_dataset) has been provided, the evaluate method is called to calculate the validation loss, IoU score, and mAP. These scores are then 
        added to the self.metrics dictionary under the 'val' key.

    *   If early stopping is enabled (self.patience is not None), the current validation loss is compared to best_val_loss. If the current validation loss is better than best_val_loss, 
        best_val_loss is updated, and early_stopping_counter is reset to zero. Otherwise, early_stopping_counter is incremented by one. If early_stopping_counter reaches the 
        patience limit, the method prints a message stating that early stopping has been triggered, and the training is stopped.

    *   Finally, the self.metrics dictionary is returned.
'''




__evaluate_docs__: str = r'''
    *   This is the evaluate() method of the ObjectDetectionTrainer class, which evaluates the trained model on the validation dataset.

    *   The method first sets the model in evaluation mode using the eval() method. It then initializes variables to keep track of the total loss and IoU score, as well as two 
        lists to store the true bounding boxes and predicted bounding boxes for each sample in the validation dataset.

    *   The method then loops through each sample in the validation dataset and sends it through the model to obtain the predicted bounding boxes and labels. The loss between the 
        predicted and true bounding boxes is calculated using the loss_function() method, and added to the total loss. The IoU score between the predicted and true bounding boxes 
        is also calculated using the intersection_over_union() method, and added to the total IoU score. The true bounding boxes and predicted bounding boxes for the current sample
        are appended to their respective lists.

    *   After processing all samples in the validation dataset, the average loss, average IoU score, and mAP score are calculated using the mean() method of NumPy arrays. The 
        mean_average_precision() method is called to compute the mAP score, using the true and predicted bounding boxes for all samples in the validation dataset.
    
    *   Finally, the method returns a tuple containing the average loss, average IoU score, and mAP score.

'''



#@: Trainer Class
class ObjectDetectionTrainer:
    def __init__(self, model: torch.nn.Module, train_dataset: torch.utils.data.Dataset, optimizer: torch.optim.Optimizer, 
                                                                                        device: torch.device,
                                                                                        val_dataset: Optional[torch.utils.data.Dataset] = None, 
                                                                                        batch_size: Optional[int] = 1,
                                                                                        num_epochs: Optional[int] = 1, 
                                                                                        patience: Optional[int] = None) -> None:
         
        self.model = model.to(device)
        self.train_dataset = train_dataset
        self.optimizer = optimizer
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.device = device
        
        self.train_loader: torch.utils.data.DataLoader = torch.utils.data.DataLoader(
            train_dataset, batch_size= batch_size, shuffle= True
        )
        
        self.val_loader: torch.utils.data.DataLoader = torch.utils.data.DataLoader(
            val_dataset, batch_size= batch_size 
        ) if val_dataset else None
        
        self.patience = patience
        
        self.metrics: dict[str, dict[str, list[float]]] = {
            'train': {
                'loss': [], 'iou': [], 'mAP': []
            },
            'val': {
                'loss': [], 'iou': [], 'mAP': []
            }
        }
        
    
    
    
    def loss_function(self, outputs: dict[str, torch.tensor], targets: dict[str, torch.tensor]) -> torch.tensor:
        criterion_regression = nn.SmoothL1Loss()
        criterion_classification = nn.CrossEntropyLoss()

        loss_regression = criterion_regression(
            outputs['bounding_box'], 
            targets['bounding_box']
        )
        
        loss_classification = criterion_classification(
            outputs['class_prob'], 
            targets['label']
        )
    
        total_loss: torch.tensor = loss_regression + loss_classification
        return total_loss
    
    
    
    
    
    @staticmethod
    def intersection_over_union(pred_boxes: np.ndarray, true_boxes: np.ndarray) -> float:
        intersection_x1: np.ndarray = np.maximum(pred_boxes[:, 0], true_boxes[:, 0])
        intersection_y1: np.ndarray = np.maximum(pred_boxes[:, 1], true_boxes[:, 1])
        intersection_x2: np.ndarray = np.minimum(pred_boxes[:, 2], true_boxes[:, 2])
        intersection_y2: np.ndarray = np.minimum(pred_boxes[:, 3], true_boxes[:, 3])

        intersection_area: np.ndarray = np.clip(intersection_x2 - intersection_x1, 0, None) * np.clip(intersection_y2 - intersection_y1, 0, None)

        #@: Calculating the union
        pred_area: np.ndarray = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
        true_area: np.ndarray = (true_boxes[:, 2] - true_boxes[:, 0]) * (true_boxes[:, 3] - true_boxes[:, 1])

        union_area: np.ndarray = pred_area + true_area - intersection_area

        iou_score: np.ndarray = np.clip(intersection_area / union_area, 0, None)
        return iou_score.mean().item()
            
        
        
    
    
    
    @staticmethod
    def mean_average_precision(true_boxes_list: list[dict[str, torch.tensor]], pred_boxes_list: list[dict[str, torch.tensor]], 
                                                                               num_classes: Optional[int] = 1, 
                                                                               iou_threshold: Optional[float] = 0.5) -> float:
        
        ap: np.ndarray = np.zeros(num_classes)
    
        for class_idx in range(num_classes):
            true_boxes = [sample['bounding_box'][sample['label'] == class_idx] for sample in true_boxes_list]
            pred_boxes = [sample['bounding_box'][sample['class_prob'][:, class_idx] > 0.5] for sample in pred_boxes_list]

            true_boxes = np.concatenate(true_boxes, axis=0)
            pred_boxes = np.concatenate(pred_boxes, axis=0)

            #@: Sort predicted boxes by confidence score
            pred_boxes = pred_boxes[np.argsort(-pred_boxes[:, -1])]

            num_true_boxes = len(true_boxes)
            num_pred_boxes = len(pred_boxes)

            if num_true_boxes == 0 or num_pred_boxes == 0:
                ap[class_idx] = 0
                continue

            tp = np.zeros(num_pred_boxes)
            fp = np.zeros(num_pred_boxes)

            matched_true_boxes: list[Any] = []

            for pred_idx, pred_box in enumerate(pred_boxes):
                iou_values = ObjectDetectionTrainer.intersection_over_union(
                    pred_boxes= pred_box[np.newaxis, :4], 
                    true_boxes= true_boxes[:, :4]
                    
                )
                max_iou_idx = np.argmax(iou_values)
                max_iou = iou_values[max_iou_idx]

                if max_iou >= iou_threshold and max_iou_idx not in matched_true_boxes:
                    tp[pred_idx] = 1
                    matched_true_boxes.append(max_iou_idx)
                else:
                    fp[pred_idx] = 1

            tp_cumsum = np.cumsum(tp)
            fp_cumsum = np.cumsum(fp)
            precision = tp_cumsum / (tp_cumsum + fp_cumsum)
            recall = tp_cumsum / num_true_boxes

            ap[class_idx] = np.sum((recall[:-1] - recall[1:]) * precision[:-1])

        mAP: float = np.mean(ap)
        return mAP
        
        
        
        
        
    
    
    def train(self) -> dict[str, dict[str, list[float]]]:
        early_stopping_counter: int = 0
        best_val_loss: float = float('inf')
        
        for epoch in range(self.num_epochs):
            self.model.train()
            total_loss: float = 0.0
            total_iou: float = 0.0
            all_true_boxes: list[dict[str, torch.tensor]] = []
            all_pred_boxes: list[dict[str, torch.tensor]] = []
            
            for idx, sample in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                inputs: torch.tensor = sample['independent_variable'].to(self.device)
                
                targets: dict[str, torch.tensor] = {
                    'bounding_box': sample['dependent_variable']['bounding_box'].to(self.device),
                    'label': sample['dependent_variable']['label'].to(self.device)
                }
                
                outputs: dict[str, dict[str, torch.tensor]] = self.model(inputs)
                loss: torch.tensor = self.loss_function(
                    outputs= outputs, 
                    targets= targets
                )
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                
                iou_score: float = ObjectDetectionTrainer.intersection_over_union(
                    pred_boxes= outputs['bounding_box'].cpu(), 
                    true_boxes= targets['bounding_box'].cpu()
                )
                
                total_iou += iou_score
                
                all_true_boxes.append(targets)
                all_pred_boxes.append(outputs)
                
            avg_loss: float = total_loss / len(self.train_loader)
            avg_iou: float = total_iou / len(self.train_loader)
            
            mAP_score: float = ObjectDetectionTrainer.mean_average_precision(
                true_boxes_list= all_true_boxes, 
                pred_boxes_list= all_pred_boxes
            )
            
            
            self.metrics['train']['loss'].append(avg_loss)
            self.metrics['train']['iou'].append(avg_iou)
            self.metrics['train']['mAP'].append(mAP_score)
            
            print(f'Epoch: {epoch+1}/{self.num_epochs}, Train Loss: {avg_loss:.6f}, IoU: {avg_iou:.6f}, mAP: {mAP_score:.6f}')
            
            
            if self.val_dataset:
                val_loss, val_iou, val_mAP = self.evaluate()
                self.metrics['val']['loss'].append(val_loss)
                self.metrics['val']['iou'].append(val_iou)
                self.metrics['val']['mAP'].append(val_mAP)
                print(f'Epoch: {epoch+1}/{self.num_epochs}, Validation Loss: {val_loss:.6f}, IoU: {val_iou:.6f}, mAP: {val_mAP:.6f}')

                if self.patience:
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        early_stopping_counter: int = 0
                    
                    else:
                        early_stopping_counter += 1
                        if early_stopping_counter >= self.patience:
                            print(f"Early stopping triggered. Stopping training after {epoch + 1} epochs.")
                            break
                
        
        return self.metrics


    
    
    def evaluate(self) -> tuple[float, float, float]:
        self.model.eval()
        total_loss: float = 0.0
        total_iou: float = 0.0
        all_true_boxes: list[dict[str, torch.tensor]] = []
        all_pred_boxes: list[dict[str, torch.tensor]] = []
        
        with torch.no_grad():
            for idx, sample in enumerate(self.val_loader):
                inputs: torch.tensor = sample['independent_variable'].to(self.device)
                targets: dict[str, torch.tensor] = {
                    'bounding_box': sample['dependent_variable']['bounding_box'].to(self.device),
                    'label': sample['dependent_variable']['label'].to(self.device)
                }
                outputs: dict[str, dict[str, torch.tensor]] = self.model(inputs)
                loss: float = self.loss_function(
                    outputs= outputs, 
                    targets= targets
                )
                total_loss += loss.item()
                
                
                iou_score: float = ObjectDetectionTrainer.intersection_over_union(
                    pred_boxes= outputs['bounding_box'].cpu(),
                    true_boxes= targets['bounding_box'].cpu()
                )
                
                total_iou += iou_score
                
                all_true_boxes.append(targets)
                all_pred_boxes.append(outputs)
                
            avg_loss: float = total_loss / len(self.val_loader)
            avg_iou: float = total_iou / len(self.val_loader)
            
            mAP_score: float = ObjectDetectionTrainer.mean_average_precision(
                true_boxes_list= all_true_boxes, 
                pred_boxes_list= all_pred_boxes
            )
        
        return avg_loss, avg_iou, mAP_score
    
    
    
                


#@: Driver Code
if __name__.__contains__('__main__'):
    ...

