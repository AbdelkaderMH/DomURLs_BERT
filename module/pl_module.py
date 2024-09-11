import torch 
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
import torchmetrics
from .report import classification_report_from_cm, compute_score

class BaseModel(pl.LightningModule):
    """
    Lightning module
    Args:
        classifier (nn.module): classifier
        num_classes (int): number of classes in the training data
        criterion (nn.Module): loss function
        config (EasyDict): training hyperparameters (lr and weight_decay are required)
        names: classes names:
        We may have data that are labeled malicious without fine-grained class label.
    """
    def __init__(self, classifier, num_classes, criterion, config, names= None):
        super().__init__()
        self.classifier =  classifier
        self.config = config
        self.num_classes = num_classes
        self.class_names = names
        self.valid_accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=self.num_classes)
        self.valid_f1_score =  torchmetrics.F1Score(task='multiclass', average='macro', num_classes=self.num_classes)
        self.valid_f1_weighted =  torchmetrics.F1Score(task='multiclass', average='weighted', num_classes=self.num_classes)

        self.test_accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=self.num_classes)
        self.test_f1_score =  torchmetrics.F1Score(task='multiclass', average='macro', num_classes=self.num_classes)
        self.test_f1_weighted =  torchmetrics.F1Score(task='multiclass', average='weighted', num_classes=self.num_classes)
        self.test_confusion_matrix = torchmetrics.ConfusionMatrix(task='multiclass', num_classes=self.num_classes)
        self.classification_report = ""

        self.criterion = criterion

        self.prefix = '_mc'
        if self.num_classes == 2:
            self.prefix = '_bin'



    def forward(self, x):
        if isinstance(x, dict):
            logits = self.classifier(**x)
        else:
            logits = self.classifier(x)
        return logits
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)

        loss = self.criterion(logits, y.long())
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
  
        logits = self(x)
        loss = self.criterion(logits, y.long())
        
        
        preds = torch.softmax(logits, dim=1)
        preds = torch.argmax(preds, dim=1)

        
        self.valid_accuracy.update(preds, y)
        self.valid_f1_score.update(preds, y)

        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return  loss

    def test_step(self, batch, batch_idx):
        x, y = batch
  
        logits = self(x)
        loss = self.criterion(logits, y.long())
        
        preds = torch.softmax(logits, dim=1)
        preds = torch.argmax(preds, dim=1)

        
        self.test_accuracy.update(preds, y)
        self.test_f1_score.update(preds, y)
        self.test_f1_weighted.update(preds, y)
        self.test_confusion_matrix.update(preds, y)

        self.log("test_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return  loss
        
    def predict_step(self, batch, batch_idx):
        y = None
        if len(batch) > 1:
            x, y = batch
        else:
            x = batch
        logits = self(x)
        probs = torch.softmax(logits, dim=1)
        classes = torch.argmax(probs, dim=1)
        if y is None:
            return {'classes': classes, 'probs': probs}
        else:
            return {'classes': classes, 'probs': probs, 'gt_labels': y}
        
    def on_validation_epoch_end(self):
        acc = self.valid_accuracy.compute()
        f1 = self.valid_f1_score.compute()  
        
        self.log(f'val_acc{self.prefix}', acc)
        self.log(f'val_F1{self.prefix}', f1, prog_bar=True)

        self.valid_accuracy.reset()
        self.valid_f1_score.reset()


    def on_test_epoch_end(self):
        acc = self.test_accuracy.compute()
        f1 = self.test_f1_score.compute()
        f1_wtd = self.test_f1_weighted.compute()
        
        self.log(f'test_acc{self.prefix}', acc)
        self.log(f'test_F1{self.prefix}', f1)
        self.log(f'test_F1_wted{self.prefix}', f1_wtd)
        metrics = compute_score(self.test_confusion_matrix.compute())
        for key, value in metrics.items():
            self.log(f'test_{key}{self.prefix}', value)
        
        self.classification_report = classification_report_from_cm(self.test_confusion_matrix.compute(), class_names=self.class_names, digits=4)
        self.test_confusion_matrix.reset()
        self.test_accuracy.reset()
        self.test_f1_score.reset()
        self.test_f1_weighted.reset()

        
    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.classifier.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": self.config.weight_decay,
                },
                {
                    "params": [p for n, p in self.classifier.named_parameters() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                },
            ]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.config.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.2, patience=20, min_lr=5e-8)
        
        return  {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}
    
    def get_hyperparams(self):
        return {'lr': self.config.lr, 
                'weight_decay': self.config.weight_decay
                } 
    def get_model(self):
        return self.classifier
