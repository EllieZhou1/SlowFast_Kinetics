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

with open(f"{args.config:04}.yaml", "r") as f:
    CONFIG = yaml.safe_load(f)

CONFIG['device'] = "cuda" if torch.cuda.is_available() else "cpu"

# 1. load the model 
model = torch.hub.load('facebookresearch/pytorchvideo', 'slowfast_r50', pretrained=False)
# 2. Move to device
model = model.to(CONFIG['device'])
# 3. Wrap with DataParallel if multiple GPUs
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
    model = nn.DataParallel(model)

#Initiate the Wandb
run = wandb.init(
    project="Slowfast_Kinetics",
    config=CONFIG
)

wandb.watch(model, log='all', log_freq = 100)

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
# validation_dataset = KineticsDataset2(
#     csv_path = os.path.join(CONFIG['metadata_dir'], "clean_validate.csv"),
#     split="validate",
#     max_videos=None
# )

print("Made dataset. Length of training dataset is ", len(train_dataset))

#my_train_dataloader = torch.utils.data.DataLoader(train_dataset, CONFIG['batch_size'], shuffle=True)
my_train_dataloader = torch.utils.data.DataLoader(train_dataset, args.batch_size, shuffle=True)
# my_validation_dataloader = torch.utils.data.DataLoader(validation_dataset, CONFIG['batch_size'], shuffle=False)

print("Made dataloader")


# ========== TRAINING PIPELINE ==========
def train_model():
    # Initialize components
    optimizer = optim.SGD(model.parameters(), lr=CONFIG['learning_rate'])
    loss_fn = nn.CrossEntropyLoss(reduction='mean')
    
    # Training loop
    for epoch in range(CONFIG['num_epochs']):
        print("Starting epoch ", epoch)
        model.train()
        train_loss = 0.0

        # Training phase
        for i, batch in enumerate(my_train_dataloader):
            batch_size = batch["inputs"][0].shape[0]  # Number of samples in the batch                                         
            inputs = [torch.tensor(x) for x in batch["inputs"]]
            labels = batch["label"].to(CONFIG['device'])

            print("         Starting batch ", i, " with batch size ", batch_size)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels) #the avg of the losses for the samples in the batch
            loss.backward()
            optimizer.step()
        
            train_loss += loss.item()

            predicted = torch.argmax(outputs, dim=1)
            correct = (predicted == labels).sum().item()
            accuracy = float(correct)/batch_size


            #Log performance every 20 batches
            if True:
                now = datetime.now()
                run.log({"epoch": epoch, "loss": loss, "accuracy": accuracy, "correct": correct})
                print("Epoch {}, Batch {}, Loss {:.4f}, Correct {}, Total {}, Time {}".
                      format(epoch, i, loss, correct, batch_size, now.strftime("%Y-%m-%d %H:%M:%S")))
  

        file_path = f"/n/fs/visualai-scr/temp_LLP/ellie/slowfast_kinetics/weights_{epoch:06d}.pth"
        torch.save(model.state_dict(), file_path)

if __name__ == "__main__":
    train_model()


