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

# ========== DATA PIPELINE ==========
class KineticsDataset2(torch.utils.data.Dataset):
    def __init__(self, csv_path, split, stride=2.0, max_videos=None):
        self.max_videos = max_videos
        self.df = pd.read_csv(csv_path)
        #TODO: reduce df to the max_videos if specified


    def __len__(self):
        if self.max_videos is None:
            return len(self.df)
        else:
            return min(len(self.df), self.max_videos)

    
    # Compute indices for 8 and 32 evenly spaced frames
    def sample_indices(self, n, total_frames):
        return [int(round(i * (total_frames - 1) / (n - 1) + 1)) for i in range(n)]
    
    #Given the path to the frames directory and a list of indicies, load the video frames
    #Returns a tensor of the video frames
    def load_video_frames(self, frames_path, indices):
        frames = []
        for i in indices:
            image_path = os.path.join(frames_path, f"{i:06d}.jpg")  # Assuming frames are named as 000001.jpg, 000002.jpg, etc.
            img = Image.open(image_path).convert('RGB')  # Load as RGB
            img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1)  # [C, H, W]
            img_tensor = img_tensor.to(CONFIG['device'])
            frames.append(img_tensor)
    
        video_tensor = torch.stack(frames, dim=1)  # [3, num_frames, H, W]
        video_tensor = transform(video_tensor)  # Apply transformations
        return video_tensor
            

    def __getitem__(self, idx):
        if(idx % 10 == 0):
            print("Get item ", idx)

        row = self.df.iloc[idx]
        label = row['label']

        total_frames = row ['num_files']

        idx_fast = self.sample_indices(num_frames_fast, total_frames)
        idx_slow = self.sample_indices(num_frames_slow, total_frames)

        #Shape should be [3, 32, 256, 256] for the fast tensor
        #Shape should be [3, 8, 256, 256] for the slow tensor
        fast_tensor = self.load_video_frames(row['full_path'], idx_fast)
        slow_tensor = self.load_video_frames(row['full_path'], idx_slow)

        inputs=[slow_tensor, fast_tensor]

        result = {
            "inputs": inputs, #a list of 2 tensors
            "label": kinetics_classname_to_id[label],
        }
        return result
    
