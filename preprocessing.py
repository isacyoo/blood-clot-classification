import os
import gc

import pandas as pd
import cv2
import tifffile
import tqdm
import matplotlib.pyplot as plt


if __name__ == "__main__":
    INPUT_DIR = '/kaggle/input/mayo-clinic-strip-ai'
    OUTPUT = '/kaggle/working'

    test_csv = pd.read_csv(f"{INPUT_DIR}/test.csv")
    train_csv = pd.read_csv(f"{INPUT_DIR}/train.csv")

    os.mkdir(f"{OUTPUT}/test")
    os.mkdir(f"{OUTPUT}/train")

    for i in tqdm.tqdm(range(len(test_csv))):
        img = tifffile.imread(f"{dir}/test/{test_csv.iloc[i].image_id}.tif")
        img = cv2.resize(img, (1024,1024))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)    
        cv2.imwrite(f"{OUTPUT}/test/{test_csv.iloc[i].image_id}.jpg", img)
        
        del img
        gc.collect()
        
    for i in tqdm.tqdm(range(len(test_csv))):
        img = tifffile.imread(f"{INPUT_DIR}/test/{test_csv.iloc[i].image_id}.tif")
        img = cv2.resize(img, (1024,1024))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)    
        cv2.imwrite(f"{OUTPUT}/test/{test_csv.iloc[i].image_id}.jpg", img)
        
        del img
        gc.collect()
        
    aa = cv2.imread(f"{OUTPUT}/test/{test_csv.iloc[0].image_id}.jpg")
    plt.imshow(aa)
    plt.show()

    os.listdir('test')
    gc.collect()