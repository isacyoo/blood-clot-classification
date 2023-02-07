from typing import Union, Sequence, Tuple, Optional
import pathlib

import torch
import pandas as pd
import numpy as np
import PIL
import albumentations as A
from sklearn.model_selection import train_test_split
import cv2
import pytorch_lightning as pl

from config import Config

# Pytorch Dataset class
class BloodClotDataset(torch.utils.data.Dataset):
    def __init__(self, df: pd.DataFrame, train: bool, image_dir: Union[pathlib.Path, str], transforms: A.Compose, config: Config):
        super().__init__()
        self.train = train
        self.df = df 
        self.dir = image_dir
        self.transforms = transforms
        self.device = config.data.device
        
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Union[Tuple[torch.Tensor,torch.Tensor], torch.Tensor]:
        img_path = f"{self.dir}/{self.df.image_id[idx]}.jpg"
        image = np.asarray(PIL.Image.open(img_path))
        if self.transforms is not None:
            image = self.transforms(image=image)["image"]
        image = torch.from_numpy(image.transpose(2,0,1)).to(self.device)
        
        if self.train:
            return image, torch.tensor([self.df["label"][idx] == "CE"], dtype=torch.long).to(self.device)
        else:
            return image, None
        
        
# Pytorch-Lightning Datamodule class
class BloodClotDataModule(pl.LightningDataModule):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.train_dir = config.path.train_image_dir
        self.train_csv = pd.read_csv(config.path.train_csv)
        self.test_dir = config.path.test_image_dir
        
        
        self.device = config.data.device
        self.batch_size = config.train.batch_size
        self.num_workers = config.data.num_workers
        image_ids = self.train_csv["image_id"]
        labels = self.train_csv["label"]
        
        X_train, X_val, y_train, y_val = train_test_split(image_ids, labels, test_size= config.data.val_ratio+config.data.test_ratio, random_state=78)
        X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=config.data.test_ratio/(config.data.test_ratio+config.data.val_ratio), random_state=78)
        self.train_df = pd.DataFrame({"image_id":X_train, "label":y_train}).reset_index(drop = True)
        self.val_df = pd.DataFrame({"image_id":X_val, "label":y_val}).reset_index(drop = True)
        self.test_df = pd.DataFrame({"image_id": X_test, "label":y_test}).reset_index(drop=True)
        self.train_transforms = None
        
        self.train_transforms = A.Compose([A.Resize(self.config.data.image_size,self.config.data.image_size,always_apply=True),
                                           A.Normalize(),
           A.RandomRotate90(),
           A.Flip(),
           A.ShiftScaleRotate(shift_limit = config.data.shift_limit, scale_limit = config.data.scale_limit, rotate_limit=config.data.rotate_limit, border_mode=cv2.BORDER_CONSTANT, value=[255,255,255], p=config.data.aug_p)
        ])
        
        self.val_transforms = A.Compose([A.Resize(self.config.data.image_size,self.config.data.image_size,always_apply=True),
                                        A.Normalize(),])
        
        
        self.train_dataset = BloodClotDataset(self.train_df, True, self.train_dir, self.train_transforms, self.config)
        self.val_dataset = BloodClotDataset(self.val_df, True, self.train_dir, self.val_transforms, self.config)
        self.test_dataset = BloodClotDataset(self.test_df, True, self.train_dir, None, self.config)

  
    
    def train_dataloader(self) -> torch.utils.data.Dataloader:
        return torch.utils.data.DataLoader(self.train_dataset, shuffle = True, batch_size = self.batch_size, num_workers = self.num_workers, collate_fn=self.collate_fn)
    
    def val_dataloader(self) -> torch.utils.data.Dataloader:
        return torch.utils.data.DataLoader(self.val_dataset, shuffle = False, batch_size = self.batch_size, num_workers = self.num_workers, collate_fn=self.collate_fn)
    
    def test_dataloader(self) -> torch.utils.data.Dataloader:
        return torch.utils.data.DataLoader(self.test_dataset, shuffle = False, batch_size = self.batch_size, num_workers = self.num_workers, collate_fn=self.collate_fn)
    

    def collate_fn(self, batch: Sequence[Tuple[torch.Tensor, torch.Tensor]]) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        batch_size = len(batch)
        imgs, labels = zip(*batch)
        if labels is not None:
            return torch.stack(imgs, dim = 0), torch.stack(labels)
        return torch.stack(imgs, dim = 0)
        