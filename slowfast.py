import torch
from datetime import datetime
import os
import pandas as pd
from pytorchvideo.data.encoded_video import EncodedVideo
import csv
import numpy as np
from PIL import Image

import torch.optim as optim
import torch.nn as nn
import glob
from tqdm.auto import tqdm
import wandb
import argparse

from typing import Dict
import json
import urllib

import torch
import yaml

from dataset import KineticsDataset2

#Create an argument parser to allow for a dynamic batch size
parser = argparse.ArgumentParser(description="Training script with tunable batch size")
parser.add_argument(
    "--config",
    type=int,
    help="configfile"
)
args = parser.parse_args()

with open(f"config/{args.config:04}.yaml", "r") as f:
    CONFIG = yaml.safe_load(f)

CONFIG['device'] = "cuda" if torch.cuda.is_available() else "cpu"

# 1. load the model 
my_model = torch.hub.load('facebookresearch/pytorchvideo', 'slowfast_r50', pretrained=False)
# 2. Move to device
my_model = my_model.to(CONFIG['device'])
# 3. Wrap with DataParallel if multiple GPUs
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
    my_model = nn.DataParallel(my_model)

#Initiate the Wandb
run = wandb.init(
    project="Slowfast_Kinetics",
    config=CONFIG,
)

run.define_metric("train loss", step_metric="train_step")
run.define_metric("train accuracy", step_metric="train_step")
run.define_metric("test loss", step_metric="test_step")
run.define_metric("test accuracy", step_metric="test_step")

run.define_metric("train loss (epoch avg)", step_metric="epoch")
run.define_metric("train accuracy (epoch avg)", step_metric="epoch")
run.define_metric("test loss (epoch avg)", step_metric="epoch")
run.define_metric("test accuracy (epoch avg)", step_metric="epoch")


train_step = 0
test_step = 0


wandb.watch(my_model, log='all', log_freq = 100)

print(torch.version.cuda)  # Should print a CUDA version, not None
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print("Using device:", CONFIG['device'])
print("Started making dataset")

# Create a dataset instance for training set
train_dataset = KineticsDataset2(
    csv_path=os.path.join(CONFIG['metadata_dir'], "clean_train.csv"),
    split="train",
    max_videos=None
)

#create a dataset instance for validation set
validation_dataset = KineticsDataset2(
     csv_path = os.path.join(CONFIG['metadata_dir'], "clean_validate.csv"),
     split="validate",
     max_videos=None
)

train_len = len(train_dataset)
validation_len = len(validation_dataset)

print("Made dataset. Length of training dataset is ", train_len)
print("Made dataset. Length of validation dataset is ", validation_len)


#my_train_dataloader = torch.utils.data.DataLoader(train_dataset, CONFIG['batch_size'], shuffle=True)
my_train_dataloader = torch.utils.data.DataLoader(train_dataset, CONFIG['batch_size'], 
                                                  num_workers=8, pin_memory=True, shuffle=True)

my_validation_dataloader = torch.utils.data.DataLoader(validation_dataset, CONFIG['batch_size'], 
                                                       num_workers=8, pin_memory=True, shuffle=False)

print("Made dataloaders")

my_optimizer = optim.SGD(my_model.parameters(), lr=CONFIG['learning_rate'])
my_loss_fn = nn.CrossEntropyLoss(reduction='mean')

def train_epoch(model, epoch, optimizer, loss_fn, dataloader):
    global train_step
    train_loss = 0.0
    correct = 0
    total = 0

    for i, batch in enumerate(dataloader):
            batch_size = batch["inputs"][0].shape[0]  # Number of samples in the batch                                         
            inputs = [x.to(CONFIG['device']) for x in batch["inputs"]]
            labels = batch["label"].to(CONFIG['device'])

            #print("         Starting batch ", i, " with batch size ", batch_size)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels) #the avg of the losses for the samples in the batch
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()

            predicted = torch.argmax(outputs, dim=1)
            correct += (predicted == labels).sum().item() #add the number of correct predictions in this batch
            total += batch_size #add the number of samples in this batch
            accuracy = float(correct)/total

            #Log performance every 50 batch
            if i % 20 == 0:
                now = datetime.now()
                run.log({
                     "train loss": loss, 
                     "train accuracy": accuracy,
                     "train_step": train_step
                })
                train_step += 1

                print("Epoch {}, Batch {}, Train Loss {:.4f}, Train Accuracy {:.3f}, Time {}".
                    format(epoch, i, train_loss/(i+1), accuracy, now.strftime("%Y-%m-%d %H:%M:%S")))
                
    run.log({
        "train loss (epoch avg)": train_loss/len(dataloader),
        "train accuracy (epoch avg)": float(correct)/total,
        "epoch": epoch
    })
                

def test_epoch(model, epoch, loss_fn, dataloader):
    global test_step
    test_loss = 0.0
    correct = 0
    total = 0

    for i, batch in enumerate(dataloader):
            batch_size = batch["inputs"][0].shape[0]  # Number of samples in the batch                                         
            inputs = [x.to(CONFIG['device']) for x in batch["inputs"]]
            labels = batch["label"].to(CONFIG['device'])

            #print("         Starting batch ", i, " with batch size ", batch_size)
            outputs = model(inputs)
            loss = loss_fn(outputs, labels) #the avg of the losses for the samples in the batch
            
            test_loss += loss.item()

            predicted = torch.argmax(outputs, dim=1)
            correct += (predicted == labels).sum().item() #add the number of correct predictions in this batch
            total += batch_size #add the number of samples in this batch
            accuracy = float(correct)/total

            #Log performance every 50 batch
            if i % 20 == 0:
                now = datetime.now()
                run.log({
                     "test loss": loss, 
                     "test accuracy": accuracy,
                     "test_step": test_step
                })
                test_step += 1

                print("Epoch {}, Batch {}, Test Loss {:.4f}, Test Accuracy {:.3f}, Time {}".
                    format(epoch, i, test_loss/(i+1), accuracy, now.strftime("%Y-%m-%d %H:%M:%S")))

    run.log({
        "test loss (epoch avg)": test_loss/len(dataloader),
        "test accuracy (epoch avg)": float(correct)/total,
        "epoch": epoch
    })

# ========== TRAINING PIPELINE ==========
def train_model():
    batches_needed = train_len // CONFIG['batch_size']
    print(f"Training with {batches_needed} batches per epoch")

    for epoch in range(CONFIG['num_epochs']):
        print("Starting epoch ", epoch)

        my_model.train()
        train_epoch(my_model, epoch, my_optimizer, my_loss_fn, my_train_dataloader)

        print("Finish epoch ", epoch, "training. Starting validation")

        my_model.eval()
        test_epoch(my_model, epoch, my_loss_fn, my_validation_dataloader)
        
        file_path = f"/n/fs/visualai-scr/temp_LLP/ellie/slowfast_kinetics/saved_weights/slowfast_kinetics400/weights_{epoch:06d}.pth"
        torch.save(my_model.state_dict(), file_path)
            

if __name__ == "__main__":
    train_model()


