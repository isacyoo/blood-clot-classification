from dataclasses import dataclass

import torch

import paths

# Hierachical dataclasses for experiment configuration

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

@dataclass 
class ModelConfig:
    device = device
    dropout = 0.25
    num_stages = 4
    first_hidden_size = 64
    num_layers = 2
    act = "relu"
    
@dataclass
class TrainConfig:
    lr : float = 1e-4
    momentum : float = 0
    weight_decay : float = 1e-8
    batch_size : int = 128

@dataclass
class DataConfig:
    image_size : int = 224
    aug_p : float = 0.8
    shift_limit : float = 0.4
    scale_limit : float = 0.4
    rotate_limit : int = 20
    num_workers : int = 0
    train_ratio : float = 0.7
    val_ratio : float = 0.2
    test_ratio : float = 0.1
    device = device
    
@dataclass
class PathConfig:
    train_image_dir : str = paths.TRAIN_DIR
    test_image_dir : str = paths.TEST_DIR
    train_csv : str = paths.TRAIN_CSV
    test_csv : str = paths.TEST_CSV
    output_dir : str = paths.OUTPUT_DIR
        
        
        
@dataclass
class Config:
    model : ModelConfig = ModelConfig()
    data : DataConfig = DataConfig()
    train : TrainConfig = TrainConfig()
    path : PathConfig = PathConfig()
    
    def update_model_config(self, dropout: float, num_stages: int, first_hidden_size: int, num_layers: int, act: str):
        self.model.dropout = dropout
        self.model.num_stages = num_stages
        self.model.first_hidden_size = first_hidden_size
        self.model.num_layers = num_layers
        self.model.act = act
        
    def update_train_config(self, lr: float, weight_decay: float, momentum: float):
        self.train.lr = lr
        self.train.weight_decay = weight_decay
        self.train.momentum = momentum
        
    def update_data_config(self, aug_p: float, shift_limit: float, scale_limit: float, rotate_limit: int):
        self.data.aug_p = aug_p
        self.data.shift_limit = shift_limit
        self.data.scale_limit = scale_limit
        self.data.rotate_limit = rotate_limit
