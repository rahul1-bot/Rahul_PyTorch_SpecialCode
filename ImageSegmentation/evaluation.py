from __future__ import annotations
from code.utils import load_segmentation
import numpy as np
import nibabel as nib

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


class SegmentationEvaluator:
    def __init__(self, case_id: str) -> None:
        self.case_id: str = case_id


    def __repr__(self) -> str:
        return f"SegmentationEvaluator(case_id={self.case_id})"


    def evaluate(self, predictions: Union[np.ndarray, nib.Nifti1Image]) -> Tuple[float, float]:
        if len(predictions.shape) == 4:
            predictions = np.argmax(predictions, axis=-1)

        if not isinstance(predictions, (np.ndarray, nib.Nifti1Image)):
            raise ValueError("Predictions must by a numpy array or Nifti1Image")

        if isinstance(predictions, nib.Nifti1Image):
            predictions = predictions.get_fdata()

        if not np.issubdtype(predictions.dtype, np.integer):
            predictions = np.round(predictions)
        
        predictions = predictions.astype(np.uint8)
        gt = load_segmentation(self.case_id).get_fdata()

        if not predictions.shape == gt.shape:
            raise ValueError(
                ("Predictions for case {} have shape {} "
                "which do not match ground truth shape of {}").format(
                    self.case_id, predictions.shape, gt.shape
                )
            )

        try:
            tk_pd = np.greater(predictions, 0)
            tk_gt = np.greater(gt, 0)
            tk_dice = 2*np.logical_and(tk_pd, tk_gt).sum()/(tk_pd.sum() + tk_gt.sum())
        except ZeroDivisionError:
            return 0.0, 0.0

        try:
            tu_pd = np.greater(predictions, 1)
            tu_gt = np.greater(gt, 1)
            tu_dice = 2*np.logical_and(tu_pd, tu_gt).sum()/(tu_pd.sum() + tu_gt.sum())
        except ZeroDivisionError:
            return tk_dice, 0.0

        return tk_dice, tu_dice





#@: Driver Code
if __name__.__contains__('__main__'):
    ...