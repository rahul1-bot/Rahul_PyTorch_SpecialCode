from __future__ import annotations
from pathlib import Path
import argparse
import numpy as np
from imageio import imwrite
from PIL import Image
from code.utils import load_case


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

#@: Constants
DEFAULT_KIDNEY_COLOR = [255, 0, 0]
DEFAULT_TUMOR_COLOR = [0, 0, 255]
DEFAULT_HU_MAX = 512
DEFAULT_HU_MIN = -512
DEFAULT_OVERLAY_ALPHA = 0.3
DEFAULT_PLANE = "axial"



class Visualizer:
    def __init__(self, cid: str, destination: str, hu_min: int=-512, hu_max: int= 512, 
                                                                     k_color: list= [255, 0, 0], 
                                                                     t_color: list= [0, 0, 255], 
                                                                     alpha: float= 0.3, 
                                                                     plane: str= "axial", 
                                                                     less_ram: bool= False) -> None:
        
        self.cid = cid
        self.destination = destination
        self.hu_min = hu_min
        self.hu_max = hu_max
        self.k_color = k_color
        self.t_color = t_color
        self.alpha = alpha
        self.plane = plane
        self.less_ram = less_ram




    def __repr__(self) -> str:
        return f"Visualizer({self.cid}, {self.destination}, {self.hu_min}, {self.hu_max}, {self.k_color}, {self.t_color}, {self.alpha}, {self.plane}, {self.less_ram})"
    
    
    
    
    
    def hu_to_grayscale(self, volume: np.ndarray, hu_min: int, hu_max: int) -> np.ndarray:
        if hu_min is not None or hu_max is not None:
            volume = np.clip(volume, hu_min, hu_max)

        mxval = np.max(volume)
        mnval = np.min(volume)
        im_volume = (volume - mnval)/max(mxval - mnval, 1e-3)
        im_volume = 255*im_volume
        return np.stack((im_volume, im_volume, im_volume), axis=-1)
    
    
    


    def class_to_color(self, segmentation: np.ndarray, k_color: list, t_color: list) -> np.ndarray:
        shp = segmentation.shape
        seg_color = np.zeros((shp[0], shp[1], shp[2], 3), dtype=np.float32)
        seg_color[np.equal(segmentation,1)] = k_color
        seg_color[np.equal(segmentation,2)] = t_color
        return seg_color






    def overlay(self, volume_ims: np.ndarray, segmentation_ims: np.ndarray, segmentation: np.ndarray, alpha: float) -> np.ndarray:
        segbin = np.greater(segmentation, 0)
        repeated_segbin = np.stack((segbin, segbin, segbin), axis=-1)
        overlayed = np.where(
            repeated_segbin,
            np.round(alpha*segmentation_ims+(1-alpha)*volume_ims).astype(np.uint8),
            np.round(volume_ims).astype(np.uint8)
        )
        return overlayed






    def visualize(self, cid, destination, hu_min=DEFAULT_HU_MIN, 
                                          hu_max=DEFAULT_HU_MAX, 
                                          k_color=DEFAULT_KIDNEY_COLOR, 
                                          t_color=DEFAULT_TUMOR_COLOR,
                                          alpha=DEFAULT_OVERLAY_ALPHA, 
                                          plane=DEFAULT_PLANE, 
                                          less_ram=False) -> None:

        plane = plane.lower()

        plane_opts = ["axial", "coronal", "sagittal"]
        if plane not in plane_opts:
            raise ValueError((
                "Plane \"{}\" not understood. " 
                "Must be one of the following\n\n\t{}\n"
            ).format(plane, plane_opts))

        #@: Preparing output location
        out_path = Path(destination)
        if not out_path.exists():
            out_path.mkdir()  

        #@: Loading segmentation and volume
        vol, seg = load_case(cid)
        spacing = vol.affine
        vol = vol.get_data()
        seg = seg.get_data()
        seg = seg.astype(np.int32)
        
        vol_ims = None
        seg_ims = None
        if not less_ram:
            vol_ims = hu_to_grayscale(vol, hu_min, hu_max).astype(np.uint8)
            seg_ims = class_to_color(seg, k_color, t_color).astype(np.uint8)
        
        if plane == plane_opts[0]:
            if less_ram:
                for j in range(vol.shape[0]):
                    vol_ims = hu_to_grayscale(vol[j:j+1], hu_min, hu_max)
                    seg_ims = class_to_color(seg[j:j+1], k_color, t_color)

                    viz_ims = overlay(vol_ims, seg_ims, seg[j:j+1], alpha)
                    for i in range(viz_ims.shape[0]):
                        fpath = out_path / ("{:05d}.png".format(j))
                        imwrite(str(fpath), viz_ims[i])

            else:
                viz_ims = overlay(vol_ims, seg_ims, seg, alpha)
                for i in range(viz_ims.shape[0]):
                    fpath = out_path / ("{:05d}.png".format(i))
                    imwrite(str(fpath), viz_ims[i])

        if plane == plane_opts[1]:
            spc_ratio = np.abs(spacing[2,0])/np.abs(spacing[0,2])

            if less_ram:
                for j in range(vol.shape[1]):
                    vol_ims = hu_to_grayscale(vol[:,j:j+1], hu_min, hu_max).astype(np.uint8)
                    seg_ims = class_to_color(seg[:,j:j+1], k_color, t_color).astype(np.uint8)

                    for i in range(vol_ims.shape[1]):
                        fpath = out_path / ("{:05d}.png".format(j))
                        vol_im = np.array(Image.fromarray(
                            vol_ims[:,i,:]
                        ).resize((
                                int(vol_ims.shape[2]),
                                int(vol_ims.shape[0]*spc_ratio)
                            ), resample=Image.BICUBIC
                        ))
                        seg_im = np.array(Image.fromarray(
                            seg_ims[:,i,:]
                        ).resize((
                                int(vol_ims.shape[2]),
                                int(vol_ims.shape[0]*spc_ratio)
                            ), resample=Image.NEAREST
                        ))
                        sim = np.array(Image.fromarray(
                            seg[:,j,:]
                        ).resize((
                                int(vol_ims.shape[2]),
                                int(vol_ims.shape[0]*spc_ratio)
                            ), resample=Image.NEAREST
                        ))
                        viz_im = overlay(vol_im, seg_im, sim, alpha)
                        imwrite(str(fpath), viz_im)

            else:
                for i in range(vol_ims.shape[1]):
                    fpath = out_path / ("{:05d}.png".format(i))
                    vol_im = np.array(Image.fromarray(
                        vol_ims[:,i,:]
                    ).resize((
                            int(vol_ims.shape[2]),
                            int(vol_ims.shape[0]*spc_ratio)
                        ), resample=Image.BICUBIC
                    ))
                    seg_im = np.array(Image.fromarray(
                        seg_ims[:,i,:]
                    ).resize((
                            int(vol_ims.shape[2]),
                            int(vol_ims.shape[0]*spc_ratio)
                        ), resample=Image.NEAREST
                    ))
                    sim = np.array(Image.fromarray(
                        seg[:,i,:]
                    ).resize((
                            int(vol_ims.shape[2]),
                            int(vol_ims.shape[0]*spc_ratio)
                        ), resample=Image.NEAREST
                    ))
                    viz_im = overlay(vol_im, seg_im, sim, alpha)
                    imwrite(str(fpath), viz_im)



        if plane == plane_opts[2]:
            spc_ratio = np.abs(spacing[2,0])/np.abs(spacing[1,1])

            if less_ram:
                for j in range(vol.shape[2]):
                    vol_ims = hu_to_grayscale(vol[:,:,j:j+1], hu_min, hu_max).astype(np.uint8)
                    seg_ims = class_to_color(seg[:,:,j:j+1], k_color, t_color).astype(np.uint8)

                    for i in range(vol_ims.shape[2]):
                        fpath = out_path / ("{:05d}.png".format(j))
                        vol_im = np.array(Image.fromarray(
                            vol_ims[:,:,i]
                        ).resize((
                                int(vol_ims.shape[1]),
                                int(vol_ims.shape[0]*spc_ratio)
                            ), resample=Image.BICUBIC
                        ))
                        seg_im = np.array(Image.fromarray(
                            seg_ims[:,:,i]
                        ).resize((
                                int(vol_ims.shape[1]),
                                int(vol_ims.shape[0]*spc_ratio)
                            ), resample=Image.NEAREST
                        ))
                        sim = np.array(Image.fromarray(
                            seg[:,:,j]
                        ).resize((
                                int(vol_ims.shape[1]),
                                int(vol_ims.shape[0]*spc_ratio)
                            ), resample=Image.NEAREST
                        ))
                        viz_im = overlay(vol_im, seg_im, sim, alpha)
                        imwrite(str(fpath), viz_im)

            else:
                for i in range(vol_ims.shape[2]):
                    fpath = out_path / ("{:05d}.png".format(i))
                    vol_im = np.array(Image.fromarray(
                        vol_ims[:,:,i]
                    ).resize((
                            int(vol_ims.shape[1]),
                            int(vol_ims.shape[0]*spc_ratio)
                        ), resample=Image.BICUBIC
                    ))
                    seg_im = np.array(Image.fromarray(
                        seg_ims[:,:,i]
                    ).resize((
                            int(vol_ims.shape[1]),
                            int(vol_ims.shape[0]*spc_ratio)
                        ), resample=Image.NEAREST
                    ))
                    sim = np.array(Image.fromarray(
                        seg[:,:,i]
                    ).resize((
                            int(vol_ims.shape[1]),
                            int(vol_ims.shape[0]*spc_ratio)
                        ), resample=Image.NEAREST
                    ))
                    viz_im = overlay(vol_im, seg_im, sim, alpha)
                    imwrite(str(fpath), viz_im)

    



#@: Driver Code
if __name__.__contains__('__main__'):
    desc: str = "Overlay a case's segmentation and store it as a series of pngs"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument(
        "-c", "--case_id", required=True,
        help="The identifier for the case you would like to visualize"
    )
    parser.add_argument(
        "-d", "--destination", required=True,
        help="The location where you'd like to store the series of pngs"
    )
    parser.add_argument(
        "-u", "--upper_hu_bound", required=False, default=512,
        help="The upper bound at which to clip HU values"
    )
    parser.add_argument(
        "-l", "--lower_hu_bound", required=False, default=-512,
        help="The lower bound at which to clip HU values"
    )
    parser.add_argument(
        "-p", "--plane", required=False, default="axial",
        help="The plane in which to visualize the data (axial, coronal, or sagittal)"
    )
    args = parser.parse_args()

    visualizer = Visualizer(args.case_id, args.destination, 
                            hu_min=args.lower_hu_bound, hu_max=args.upper_hu_bound,
                            k_color=[255, 0, 0], t_color=[0, 0, 255],
                            alpha=0.3, plane=args.plane,
                            less_ram=False)
    visualizer.visualize()


