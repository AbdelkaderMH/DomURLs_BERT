import mlflow
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch import seed_everything
from lightning.pytorch.loggers import MLFlowLogger
from torch.utils.data import DataLoader
from data_utils.cdataset import CDataset
from data_utils.load_data import load_dataset
from models.rnn import CharLSTM, CharGRU, CharBiLSTM, CharBiGRU
from models.cnn import CharCNN
from models.rnn_cnn import CharCNNBiLSTM
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

def get_class_from_name(name):
    """
    Return the Module's class with the given name.

    Args:
        name (str): The name of the class to retrieve.

    Returns:
        class: The class object with the given name.

    Raises:
        NotImplementedError: If the module with the given name is not found in the globals().
    """
    if name not in globals():
        raise NotImplementedError("Module not found: {}".format(name))
    return globals()[name]

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
        'vocab_size' : 96,
        'output_size': num_classes,
        'embedding_dim' : 128,
        'hidden_dim': 128,
        'n_layers': 3
    }

    experiment_params = {
    "lr": args.lr,
    "weight_decay": args.weight_decay,
    "epochs": args.epochs,
    "gradient_clip_val": 0.9,
    "dataset" : args.dataset,
    'pretrained_path': args.model_name, 
    "max_length": max_length,
    "num_workers" : args.num_workers,
    "drop_prob": args.dropout_prob,
    "batch_size": args.batch_size,
    "experiment_type": args.experiment_type,
    "label_column" : args.label_column,
    "num_classes": num_classes
    }

    config = Namespace(**experiment_params)

    train_dataset = CDataset(texts=df_train['input'].values, labels=df_train[args.label_column].values, max_length=max_length)
    dev_dataset = CDataset(texts=df_dev['input'].values, labels=df_dev[args.label_column].values, max_length=max_length)
    test_dataset = CDataset(texts=df_test['input'].values, labels=df_test[args.label_column].values, max_length=max_length)    
    train_loader = DataLoader(train_dataset, **training_params)
    dev_loader = DataLoader(dev_dataset, **test_params)
    test_loader = DataLoader(test_dataset, **test_params)

    #MLFlow Experiment tracking setup

    experiment_tags = OrderedDict({
    "project_name": "DomURLs_BERT",
    "Domains_labels": ", ".join(label_encoder.classes_),
    "mlflow.note.content":f"Deep learning for malicious {args.experiment_type.upper()} detection and classification using {args.model_name} model",
    })

    mlflow.set_tracking_uri(Path.cwd().joinpath("mlruns/experiments").as_uri())
    exp_name = f'{args.dataset}_binary_cls_{num_classes==2}_{args.experiment_type}'
    print('experiment', exp_name)
    exp_id = mlflow.set_experiment(exp_name)
    mlflow.pytorch.autolog()
    run_name = args.model_name
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
    
    classifier = get_class_from_name(args.model_name)(**model_params)

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

    # Logging best checkpoint artifact
    mlflow.log_artifact(trainer.checkpoint_callback.best_model_path, artifact_path=f"checkpoints")

    # Logging classification report artifact
    test_report_path = f"{checkpoint_path}/test_classification_report.txt"
    with open(test_report_path, 'w') as f:
        f.write("Classification report----------------------------------\n\n")
        f.write(lit_model.classification_report)
    mlflow.log_artifact(test_report_path, artifact_path=f"classification_report")
    

if __name__ == '__main__':
        
    parser = argparse.ArgumentParser(description="Train character-based deep learning models for malicious URLs and domain names detection.")
    parser.add_argument('--dataset', type=str, default='Mendeley_AK_Singh_2020_phish', help='Dataset name')
    parser.add_argument('--model_name', type=str, default='CharCNN', choices=['CharCNN', "CharLSTM", "CharGRU", "CharBiLSTM", 
                                                                              "CharBiGRU", "CharCNNBiLSTM"], help='Model')
    parser.add_argument('--num_workers', type=int, default=1, help='Number of workers for data loading')
    parser.add_argument('--dropout_prob', type=float, default=0.2, help='Dropout probability')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-3, help='Weight decay')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--experiment_type', type=str, default='url', choices=["url", "domain"], help='Type of experiment')
    parser.add_argument('--label_column', type=str, default='label', help='Name of the label column')
    parser.add_argument('--seed', type=int, default=3407, help='Random seed')
    parser.add_argument('--device', type=str, default=0, help='GPU device id')
    args = parser.parse_args()

    main(args=args)