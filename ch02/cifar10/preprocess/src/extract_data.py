import os
import pickle
from typing import List

import numpy as np
from PIL import Image


def unpickle(file):
    with open(file, "rb") as fo:
        dict = pickle.load(fo, encoding="bytes")
    return dict


def parse_pickle(rawdata, rootdir) -> List[List[str]]:
    for i in range(10):
        directory = f"{rootdir}/{i}"
        os.makedirs(directory, exist_ok=True)
    class_to_filename_list = []
    
    for i in range(len(rawdata[b"filenames"])):
        filename = rawdata[b"filenames"][i].decode("utf-8")
        label = rawdata[b"labels"][i]
        data = rawdata[b"data"][i]
        data = data.reshape(3, 32, 32)
        data = np.swapaxes(data, 0, 2) # (32, 32, 3)
        data = np.swapaxes(data, 0, 1) # (32, 32, 3)
        with Image.fromarray(data) as img:
            image_path = f"{rootdir}/{label}/{filename}"
            img.save(image_path)
        class_to_filename_list.append([label, filename]) # 클래스별 파일 한장씩 담은 리스트의 리스트
    return class_to_filename_list