from __future__ import annotations
from pathlib import Path
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

def get_full_case_id(cid: str) -> str:
    try:
        cid = int(cid)
        case_id = f"case_{cid:05d}"
    except ValueError:
        case_id = cid

    return case_id



def get_case_path(cid: str) -> Path:
    data_path = Path(__file__).parent.parent / "data"
    if not data_path.exists():
        raise IOError(f"Data path, {str(data_path)}, could not be resolved")

    case_id = get_full_case_id(cid)
    case_path = data_path / case_id
    if not case_path.exists():
        raise ValueError(f"Case could not be found \"{case_path.name}\"")

    return case_path



def load_volume(cid: str) -> nib.Nifti1Image:
    case_path = get_case_path(cid)
    vol = nib.load(str(case_path / "imaging.nii.gz"))
    return vol



def load_segmentation(cid: str) -> nib.Nifti1Image:
    case_path = get_case_path(cid)
    seg = nib.load(str(case_path / "segmentation.nii.gz"))
    return seg



def load_case(cid: str) -> tuple:
    vol = load_volume(cid)
    seg = load_segmentation(cid)
    return vol, seg




#@: Driver Code 
if __name__.__contains__('__main__'):
    ...