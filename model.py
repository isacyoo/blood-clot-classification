from typing import Tuple, Sequence

import torch.nn as nn
from transformers import ResNetConfig, ResNetForImageClassification
from torchmetrics import AUROC, Accuracy
import torch
import pytorch_lightning as pl


from config import Config

# Resnet-based classifier
class BloodClotClassifier(pl.LightningModule):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.crit = nn.CrossEntropyLoss()
        
        self.model = ResNetForImageClassification(ResNetConfig(hidden_sizes=[config.model.first_hidden_size*(2**i) for i in range(config.model.num_stages)],
                                                              depths=[config.model.num_layers for i in range(config.model.num_stages)],
                                                              layer_type="basic", hidden_act=config.model.act, num_labels=2))
        
        self.acc = Accuracy(num_classes=2)
        self.AUROC = AUROC(num_classes=2)

    def forward(self, x) -> torch.Tensor:
        return self.model(x).logits
    
    
    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, labels = batch
        logits = self(x)
        labels = torch.squeeze(labels)
        loss = self.crit(logits, labels)
        self.log("train_loss", loss, on_epoch=True)
        
        return loss
    
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x, labels = batch
        logits = self(x)
        labels = torch.squeeze(labels)
        
        return logits, labels
    
    # Collects all validation results and evaluate them at once
    def validation_epoch_end(self, validation_step_outputs: Sequence[Tuple[torch.Tensor, torch.Tensor]]):
        logits, labels = zip(*validation_step_outputs)
        logits = torch.cat(logits)
        labels = torch.cat(labels)
        loss = self.crit(logits, labels)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        val_result = {"acc":self.acc(logits, labels), "auroc":self.AUROC(logits, labels)}
        self.log_dict({f"val_{k}":v for k,v in val_result.items()}, prog_bar=True)
        
        
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.model.classifier.parameters(), lr=self.config.train.lr, momentum=self.config.train.momentum)
        return {"optimizer":optimizer, "lr_scheduler":{"scheduler":torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, verbose=False),"monitor":"train_loss", "interval": "step"}}
    
    
# Backbone trained by metric learning, inspired by https://doi.org/10.1148/radiol.212482
# Utilises supervised contrastive loss to facilitate domain adaptation

class BloodClotBackbone(pl.LightningModule):
    def __init__(self, config: Config):
        super().__init__()
        self.weight_decay = config.train.weight_decay
        self.lr = config.train.lr
        self.crit = nn.CrossEntropyLoss()
        losses = "Please refer to https://github.com/KevinMusgrave/pytorch-metric-learning/blob/master/src/pytorch_metric_learning/losses/supcon_loss.py"
        self.crit = losses.SupConLoss(temperature = 0.07)
        
        self.model = ResNetForImageClassification(ResNetConfig(hidden_sizes=[config.model.first_hidden_size*(2**i) for i in range(config.model.num_stage)],
                                                              depths=[config.model.num_layers for i in range(config.model.num_stage)],
                                                              layer_type="basic", hidden_act=config.model.act, num_labels=2))
        
        
        AccuracyCalculator = "Please refer to https://github.com/KevinMusgrave/pytorch-metric-learning/blob/master/src/pytorch_metric_learning/utils/accuracy_calculator.py"
        self.acc = AccuracyCalculator(device=config.model.device)
        
    def forward(self, x):
        return torch.squeeze(self.model.resnet(x).pooler_output)
    
    
    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        x, labels = batch
        batch_size = labels.shape[0]
        features = self(x)
        features = nn.functional.normalize(features, dim=1)
        labels = torch.squeeze(labels)
        loss = self.crit(features, labels)
        self.log("train_loss", loss, on_epoch=True)
        return loss
    
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        x, labels = batch
        features = self(x)
        features = nn.functional.normalize(features, dim=1)
        labels = torch.squeeze(labels)
        return features, labels
    
    def validation_epoch_end(self, validation_step_outputs):
        features, labels = zip(*validation_step_outputs)
        features = torch.cat(features)
        labels = torch.cat(labels)
        loss = self.crit(features, labels)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        features = features.detach()
        val_result = self.acc.get_accuracy(features,labels,features,labels, True,exclude=("mean_reciprocal_rank", "mean_average_precision_at_r", "r_precision"))
        self.log_dict({f"val_{k}":v for k,v in val_result.items()}, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), weight_decay=1e-6)
        return {"optimizer":optimizer, "lr_scheduler":{"scheduler":torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, verbose=False),"monitor":"train_loss", "interval": "step"}}