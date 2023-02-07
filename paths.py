import os

# Paths
INPUT_DIR = '/kaggle/input/mayo-clinic-strip-ai'
OUTPUT_DIR = '/kaggle/working'
RESIZED_DIR = '/kaggle/input/resizing'  
    
TRAIN_CSV = f"{INPUT_DIR}/train.csv"    # path to the csv file that contains metadata of the train images
TEST_CSV = f"{INPUT_DIR}/test.csv"      # path to the csv file that contains metadata of the test images
TRAIN_DIR = f"{RESIZED_DIR}/train"      # path to the directory that contains train images
TEST_DIR = f"{RESIZED_DIR}/test"        # path to the directory that contains test images