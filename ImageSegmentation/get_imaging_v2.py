from __future__ import annotations
from pathlib import Path
import shutil
import os
import sys
import time
import requests

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

class ImageDownloader:
    def __init__(self) -> None:
        self.imaging_url = "https://kits19.sfo2.digitaloceanspaces.com/"
        self.imaging_name_tmplt = "master_{:05d}.nii.gz"
        self.temp_f = Path(__file__).parent / "temp.tmp"
    
    
    
    def __repr__(self) -> None:
        return f"ImageDownloader(imaging_url={self.imaging_url}, imaging_name_tmplt={self.imaging_name_tmplt})"



    @staticmethod
    def get_destination(i: int, create: bool) -> Path:
        destination = Path(__file__).parent.parent / "data" / f"case_{i:05d}" / "imaging.nii.gz"
        if create and not destination.parent.exists():
            destination.parent.mkdir()
        return destination


    def cleanup(self, msg: str) -> None:
        if self.temp_f.exists():
            self.temp_f.unlink()
        print(msg)
        sys.exit()


    def download(self, cid: int) -> None:
        remote_name = self.imaging_name_tmplt.format(cid)
        url = self.imaging_url + remote_name
        try:
            with requests.get(url, stream=True) as r:
                with self.temp_f.open('wb') as f:
                    shutil.copyfileobj(r.raw, f)
            shutil.move(str(self.temp_f), str(self.get_destination(cid, True)))
        except KeyboardInterrupt:
            self.cleanup("KeyboardInterrupt")
        except Exception as e:
            self.cleanup(str(e))
            
            

#@: Driver Code
if __name__ == "__main__":
    downloader = ImageDownloader()

    if not Path("data").exists():
        Path("data").mkdir()
    left_to_download = []
    for i in range(300):
        dst = downloader.get_destination(i, False)
        if not dst.exists():
            left_to_download = left_to_download + [i]

    print("{} cases to download...".format(len(left_to_download)))
    for i, cid in enumerate(left_to_download):
        print("{}/{}... ".format(i+1, len(left_to_download)))
        downloader.download(cid)
        
        
        