import mlflow
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch import seed_everything
from lightning.pytorch.loggers import MLFlowLogger
from torch.utils.data import DataLoader
from data_utils.bertdataset import BERTDataset
from data_utils.load_data import load_dataset
from models.urlbert import URLBertForSequenceClassification
from models.plm import PLMEncoder
from module.pl_module import BaseModel
import yaml
import argparse
import pickle
import warnings
import os
from collections import OrderedDict
from datetime import datetime
from argparse import Namespace
from utils import trace_jit_model
from pathlib import Path


def main(args):
    # Set random seed
    seed_everything(seed=args.seed)
    # Set dataset path
    path= f'data/{args.experiment_type}_datasets/{args.dataset}'
    # Set the max sequence length
    if args.experiment_type == 'domain':
        max_length = 64
    else:
        max_length = 128

    # Load the data
    data_dict = load_dataset(path, args.label_column)
    df_train, df_dev, df_test, label_encoder = data_dict['train'], data_dict['dev'], data_dict['test'], data_dict['label_encoder']
    num_classes = len(label_encoder.classes_)
    
    training_params = {"batch_size": args.batch_size,
                        "shuffle": True,
                        "num_workers": args.num_workers,
                        "pin_memory": True,
                        "drop_last":True}
    test_params = {"batch_size": args.batch_size,
                    "shuffle": False,
                    "num_workers": args.num_workers,
                    "pin_memory": True,
                    "drop_last":False}

    model_params = {
        'output_size': num_classes,
        'drop_prob': args.dropout_prob,
        'pretrained_path': args.pretrained_path
    }

    experiment_params = {
    "lr": args.lr,
    "weight_decay": args.weight_decay,
    "epochs": args.epochs,
    "gradient_clip_val": 0.9,
    "dataset" : args.dataset,
    'pretrained_path': args.pretrained_path, 
    "max_length": max_length,
    "num_workers" : args.num_workers,
    "drop_prob": args.dropout_prob,
    "batch_size": args.batch_size,
    "experiment_type": args.experiment_type,
    "label_column" : args.label_column,
    "num_classes": num_classes
    }

    config = Namespace(**experiment_params)

    train_dataset = BERTDataset(texts=df_train['input'].values, labels=df_train[args.label_column].values, max_length=max_length, pretrained_path=args.pretrained_path)
    dev_dataset = BERTDataset(texts=df_dev['input'].values, labels=df_dev[args.label_column].values, max_length=max_length, pretrained_path=args.pretrained_path)
    test_dataset = BERTDataset(texts=df_test['input'].values, labels=df_test[args.label_column].values, max_length=max_length, pretrained_path=args.pretrained_path)
    
    train_loader = DataLoader(train_dataset, **training_params)
    dev_loader = DataLoader(dev_dataset, **test_params)
    test_loader = DataLoader(test_dataset, **test_params)

    #MLFlow Experiment tracking setup

    experiment_tags = OrderedDict({
    "project_name": "DomURLs_BERT",
    "Domains_labels": ", ".join(label_encoder.classes_),
    "mlflow.note.content":f"Deep learning for malicious {args.experiment_type.upper()} detection and classification using {args.pretrained_path} model",
    })

    mlflow.set_tracking_uri(Path.cwd().joinpath("mlruns/experiments").as_uri())
    exp_name = f'{args.dataset}_binary_cls_{num_classes==2}_{args.experiment_type}'
    print('experiment', exp_name)
    exp_id = mlflow.set_experiment(exp_name)
    mlflow.pytorch.autolog()
    run_name = f"{args.pretrained_path.split('/')[1]}" if '/' in args.pretrained_path else args.pretrained_path
    mlflow.start_run(run_name=run_name)
    mlflow.log_params(experiment_params)
    mlflow.set_tags(experiment_tags)


    mlf_logger = MLFlowLogger(
        experiment_name=mlflow.get_experiment(mlflow.active_run().info.experiment_id).name,
        tracking_uri=mlflow.get_tracking_uri(),
        run_id=mlflow.active_run().info.run_id,
        log_model = False
        )
    experiment_id = mlflow.active_run().info.experiment_id
    run_id = mlflow.active_run().info.run_id

    ckpts_path = './mlruns/ckpts'
    checkpoint_path = f"{ckpts_path}/{run_id}/"
    checkpoint_callback = ModelCheckpoint(dirpath=checkpoint_path, monitor='val_loss',  
                                            filename=f'best_checkpoint')
    early_stopping = EarlyStopping(monitor="val_loss", min_delta=0.001, patience=5)

    # Instanciate the model, loss, and the Lightning module
    
    model = None
    if 'urlbert' in args.pretrained_path.lower():
        classifier = URLBertForSequenceClassification(**model_params)
    else:
        classifier = PLMEncoder(**model_params)

    criterion = torch.nn.CrossEntropyLoss()

    lit_model = BaseModel(classifier=classifier, num_classes=num_classes, criterion=criterion, config=config, names = label_encoder.classes_)

    # Instanciate model trainer
    trainer = Trainer(max_epochs=args.epochs, devices=[args.device], accelerator="gpu", logger=mlf_logger, callbacks=[checkpoint_callback, early_stopping],  benchmark=False,
                  deterministic=True, precision="16", gradient_clip_val=1)
    # Model training
    trainer.fit(lit_model, train_dataloaders=train_loader, val_dataloaders=dev_loader)
    # Evaluate the model on the dev set
    dev_results =  trainer.validate(ckpt_path='best', dataloaders=dev_loader)
    # Evaluate the model on the test set
    test_results = trainer.test(ckpt_path='best', dataloaders=test_loader)

    # Logging experiment metadata artifact
    path_metadata = f"{checkpoint_path}/model_metadata.p" 
    experiment_params['label_encoder'] = label_encoder
    with open(path_metadata, 'wb') as f:
        pickle.dump(experiment_params, f)
    mlflow.log_artifact(path_metadata, artifact_path=f"model_metadata.p")

    # Logging classification report artifact
    test_report_path = f"{checkpoint_path}/test_classification_report.txt"
    with open(test_report_path, 'w') as f:
        f.write("Classification report----------------------------------\n\n")
        f.write(lit_model.classification_report)
    mlflow.log_artifact(test_report_path, artifact_path=f"classification_report")
    

if __name__ == '__main__':
        
    parser = argparse.ArgumentParser(description="Train PLM-based models for malicious URL and domain names detection.")
    parser.add_argument('--dataset', type=str, default='Mendeley_AK_Singh_2020_phish', help='Dataset name')
    parser.add_argument('--pretrained_path', type=str, default='amahdaouy/DomURLs_BERT', help='Pretrained model path')
    parser.add_argument('--num_workers', type=int, default=1, help='Number of workers for data loading')
    parser.add_argument('--dropout_prob', type=float, default=0.2, help='Dropout probability')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-3, help='Weight decay')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--experiment_type', type=str, default='url', choices=["url", "domain"], help='Type of experiment')
    parser.add_argument('--label_column', type=str, default='label', help='Name of the label column')
    parser.add_argument('--seed', type=int, default=3407, help='Random seed')
    parser.add_argument('--device', type=int, default=0, help='GPU device id')
    args = parser.parse_args()

    main(args=args)