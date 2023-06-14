from __future__ import annotations
from pathlib import Path
from shutil import move
import os
import sys
import time
from tqdm import tqdm
import requests
import numpy as np

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
    def __init__(self):
        self.imaging_url = "https://kits19.sfo2.digitaloceanspaces.com/"
        self.imaging_name_tmplt = "master_{:05d}.nii.gz"
        self.temp_f = Path(__file__).parent / "temp.tmp"



    def get_destination(self, i: int) -> Path:
        destination = Path(__file__).parent.parent / "data" / f"case_{i:05d}" / "imaging.nii.gz"
        if not destination.parent.exists():
            destination.parent.mkdir()
        return destination



    def cleanup(self, bar: 'tqdm._tqdm.tqdm', msg: str) -> None:
        bar.close()
        if self.temp_f.exists():
            self.temp_f.unlink()
        print(msg)
        sys.exit()



    def download(self, i: int, cid: int, left_to_download: List[int]) -> None:
        print("Download {}/{}: ".format(i+1, len(left_to_download)))
        destination = self.get_destination(cid)
        remote_name = self.imaging_name_tmplt.format(cid)
        uri = self.imaging_url + remote_name 

        chnksz = 1000
        tries = 0
        while True:
            try:
                tries = tries + 1
                response = requests.get(uri, stream=True)
                break
            except Exception as e:
                print("Failed to establish connection with server:\n")
                print(str(e) + "\n")
                if tries < 1000:
                    print("Retrying in 30s")
                    time.sleep(30)
                else:
                    print("Max retries exceeded")
                    sys.exit()

        try:
            with self.temp_f.open("wb") as f:
                bar = tqdm(unit="KB", desc=f"case_{cid:05d}", total=int(np.ceil(int(response.headers["content-length"])/chnksz)))
                for pkg in response.iter_content(chunk_size=chnksz):
                    f.write(pkg)
                    bar.update(int(len(pkg)/chnksz))
                bar.close()
            move(str(self.temp_f), str(destination))
        except KeyboardInterrupt:
            self.cleanup(bar, "KeyboardInterrupt")
        except Exception as e:
            self.cleanup(bar, str(e))



    def main(self) -> None:
        left_to_download = []
        for i in range(300):
            if not self.get_destination(i).exists():
                left_to_download = left_to_download + [i]

        print("{} cases to download...".format(len(left_to_download)))
        for i, cid in enumerate(left_to_download):
            self.download(i, cid, left_to_download)




#@: Driver Code
if __name__.__contains__('__main__'):
    downloader = ImageDownloader()
    downloader.main()
    
    
    
