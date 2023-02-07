import os
import gc

import pandas as pd
import cv2
import tifffile
import tqdm
import matplotlib.pyplot as plt
from openslide import OpenSlide
import numpy as np

if __name__ == "__main__":
    INPUT_DIR = '/kaggle/input/mayo-clinic-strip-ai'
    OUTPUT = '/kaggle/working'

    test_csv = pd.read_csv(f"{INPUT_DIR}/test.csv")
    train_csv = pd.read_csv(f"{INPUT_DIR}/train.csv")

    os.mkdir(f"{OUTPUT}/test")
    os.mkdir(f"{OUTPUT}/train")

    IMAGE_SIZE = 512
    for i in tqdm.tqdm(range(len(test_csv))):
        slides = OpenSlide(f"{INPUT_DIR}/test/{test_csv.iloc[i].image_id}.tif")
        rows = int(slides.properties["openslide.level[0].height"]) // IMAGE_SIZE
        cols = int(slides.properties["openslide.level[0].width"]) // IMAGE_SIZE
        
        for j in range(rows):
            for k in range(cols):
                tile = slides.read_region((IMAGE_SIZE*j,IMAGE_SIZE*k), 0, (IMAGE_SIZE, IMAGE_SIZE))
                arr = np.asarray(tile)
                mean = arr.mean()
                std = arr.std()
                
                if mean < 200 and std > 50:
                    tile.save(f"{OUTPUT}/test/{test_csv.iloc[i].image_id}_{j}_{k}.png", compress_level=0)
                break    

    train = pd.DataFrame(columns = ["image_id", "label"])
    lines = []
    #for i in tqdm.tqdm(range(len(train_csv))):
    for i in range(1):
        slides = OpenSlide(f"{INPUT_DIR}/train/{train_csv.iloc[i].image_id}.tif")
        rows = int(slides.properties["openslide.level[0].height"]) // IMAGE_SIZE
        cols = int(slides.properties["openslide.level[0].width"]) // IMAGE_SIZE
        
        for j in tqdm.tqdm(range(rows)):
            for k in range(cols):
                tile = slides.read_region((IMAGE_SIZE*j,IMAGE_SIZE*k), 0, (IMAGE_SIZE, IMAGE_SIZE))
                arr = np.asarray(tile)
                mean = arr.mean()
                std = arr.std()
                
                if mean < 200 and std > 50:
                    tile.save(f"{OUTPUT}/train/{train_csv.iloc[i].image_id}_{j}_{k}.png", compress_level=0)
                    
                lines.append({"image_id":f"{OUTPUT}/train/{train_csv.iloc[i].image_id}_{j}_{k}.png", 
                            "label": train_csv.iloc[i].label})
                
        train = train.append(lines)
        lines = []
    train.to_csv(f"{OUTPUT}/train/train.csv")
