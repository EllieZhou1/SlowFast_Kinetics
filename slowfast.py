#this is a copy of slowfast.ipynb


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

from typing import Dict
import json
import urllib
from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo,
)
from pytorchvideo.data.encoded_video import EncodedVideo
from pytorchvideo.transforms import (
    
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample,
    UniformCropVideo
) 

# Choose the `slowfast_r50` model 
model = torch.hub.load('facebookresearch/pytorchvideo', 'slowfast_r50', pretrained=False)

CONFIG = {
    "temp_folder": "/n/fs/visualai-scr/temp_LLP/ellie",
    "data_root": "/n/fs/visualai-scr/Data/Kinetics_cvf/frames",  # Root directory for video frames
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "metadata_dir": "/n/fs/visualai-scr/Data/Kinetics_cvf/raw",  # Directory containing CSV files
    "batch_size": 5, #according to paper
    "learning_rate": 0.01,
    "num_epochs": 1,
}

# Set to GPU or CPU
model = model.train()
model = model.to(CONFIG['device'])

if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
    model = nn.DataParallel(model)

#Initiate the Wandb
#run = wandb.init(
    #project="Slowfast_Kinetics"
#)

#wandb.watch(model, log='all', log_freq = 100)

json_filename = os.path.join(CONFIG["temp_folder"], "slowfast_kinetics", "kinetics_classnames.json")

with open(json_filename, "r") as f:
    kinetics_classnames = json.load(f)

# Create an id --> label name mapping
kinetics_id_to_classname = {}
for k, v in kinetics_classnames.items():
    kinetics_id_to_classname[v] = str(k).replace('"', "")

#Create a classname --> id mapping
kinetics_classname_to_id = {v: k for k, v in kinetics_id_to_classname.items()}

#Define input transforms
side_size = 256
mean = [0.45, 0.45, 0.45]
std = [0.225, 0.225, 0.225]
crop_size = 256

num_frames_fast = 32 #how many frames the model expects (for the fast branch)
num_frames_slow = 8
#sampling_rate = 2 #how fast we sample frame for the fast branch (every two frames)
frames_per_second = 30 #original fps of the video
#slowfast_alpha = 4

transform = Compose(
        [
            Lambda(lambda x: x/255.0), #Scale values to [0, 1]
            NormalizeVideo(mean, std), #Normalize each channel
            ShortSideScale( #Scale the short side of the video to be "side_size", preserve the aspect ratio
                size=side_size
            ),
            CenterCropVideo(crop_size), #Take the center crop of [256x256]
        ]
)

# ========== DATA PIPELINE ==========
#Given a row in csv, get the folder to the frames
#Example: /n/fs/visualai-scr/Data/Kinetics_cvf/frames/train/milking cow/

def path_to_frames_fn(row):
    """Map CSV row to video file path"""
    return os.path.join(
        CONFIG["data_root"], row['split'],row['label'],
        f"{row['youtube_id']}_{int(row['time_start']):06d}_{int(row['time_end']):06d}"
    )
    
class KineticsDataset2(torch.utils.data.Dataset):
    def __init__(self, csv_path, split, stride=2.0, max_videos=None):
        self.df = pd.read_csv(csv_path)
        self.max_videos = max_videos


    def __len__(self):
        if self.max_videos is None:
            return len(self.df)
        else:
            return min(len(self.df), self.max_videos)

    
    # Compute indices for 8 and 32 evenly spaced frames
    #Num frames = total number of frames, n = # of frames to sample (8 or 32)
    #Returns a list of indices to sample from the total number of frames
    def sample_indices(self, n, num_frames):
        if num_frames < n:
            raise ValueError(f"Requested {n} frames, but only {num_frames} available.")
        return [int(round(i * (num_frames - 1) / (n - 1) + 1)) for i in range(n)]
    
    #Given the path to the frames directory and a list of indicies, load the video frames
    #Returns a tensor of the video frames
    def load_video_frames(self, frames_path, indices):
        frames = []
        for i in indices:
            image_path = os.path.join(frames_path, f"{i:06d}.jpg")  # Assuming frames are named as 000001.jpg, 000002.jpg, etc.
            img = Image.open(image_path).convert('RGB')  # Load as RGB
            img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1)  # [C, H, W]
            frames.append(img_tensor)
    
        video_tensor = torch.stack(frames, dim=1)  # [3, num_frames, H, W]
        video_tensor = transform(video_tensor)  # Apply transformations
        return video_tensor
            

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        label = row['label']
        frames_path = f"/n/fs/visualai-scr/Data/Kinetics_cvf/frames/{row['split']}/{row['label']}/{row['youtube_id']}_{row['time_start']:06d}_{row['time_end']:06d}"


        total_frames = sum(1 for entry in os.scandir(frames_path) if entry.is_file())

        idx_fast = self.sample_indices(num_frames_fast, total_frames)
        idx_slow = self.sample_indices(num_frames_slow, total_frames)

        #Shape should be [3, 32, 256, 256] for the fast tensor
        #Shape should be [3, 8, 256, 256] for the slow tensor
        fast_tensor = self.load_video_frames(frames_path, idx_fast)
        slow_tensor = self.load_video_frames(frames_path, idx_slow)

        inputs=[slow_tensor, fast_tensor]

        result = {
            "inputs": inputs, #a list of 2 tensors
            "label": kinetics_classname_to_id[label],
        }

        #print("Get Item Result:", result)
        return result
    

print("Started loading the dataset")

# Create a dataset instance for training set
train_dataset = KineticsDataset2(
    csv_path=os.path.join(CONFIG['metadata_dir'], "train.csv"),
    split="train",
    max_videos=5
)

print("Loaded the dataset. Length of training dataset is ", len(train_dataset))

my_train_dataloader = torch.utils.data.DataLoader(train_dataset, CONFIG['batch_size'], shuffle=True)


# ========== TRAINING PIPELINE ==========

def train_model():
    # Initialize components
    optimizer = optim.SGD(model.parameters(), lr=CONFIG['learning_rate'])
    loss_fn = nn.CrossEntropyLoss(reduction='mean')

    count = 0
    
    # Training loop
    for epoch in range(CONFIG['num_epochs']):
        model.train()
        train_loss = 0.0

        # Training phase
        for i, batch in enumerate(my_train_dataloader):
            batch_size = batch["inputs"][0].shape[0]  # Number of samples in the batch                                         
            inputs = [torch.tensor(x).to(CONFIG['device']) for x in batch["inputs"]]
            labels = batch["label"].to(CONFIG['device'])

            
            #print("Input shape", inputs[0].shape, inputs[1].shape)
            #print("Labels shape", labels.shape)

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
            #if i % 20 == 0:
            if True:
                now = datetime.now()
                count += 1
                print("Epoch {}, Batch {}, Loss {:.4f}, Correct {}, Total {}, Time {}".format(epoch, i, loss, correct, batch_size, now.strftime("%Y-%m-%d %H:%M:%S")))
                file_path = f"/n/fs/visualai-scr/temp_LLP/ellie/slowfast_kinetics/weights_{count:06d}.pth"
                torch.save(model.state_dict(), file_path)
                #run.log({"epoch": epoch, "loss": loss, "accuracy": accuracy, "correct": correct})

if __name__ == "__main__":
    train_model()


