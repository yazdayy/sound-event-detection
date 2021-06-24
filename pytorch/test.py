import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../utils'))
import numpy as np
import argparse
import h5py
import math
import time
import pickle
import logging
import librosa

import config
from glob import glob
import soundfile as sf
from random import randint
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.utils.data
import torch.optim as optim
import torch.nn.functional as F
 
from evaluate import Evaluator
from config import (sample_rate, classes_num, mel_bins, fmin, fmax, 
    window_size, hop_size, window, pad_mode, center, device, ref, amin, top_db)
from losses import get_loss_func
from pytorch_utils import move_data_to_device, do_mixup, do_mixup_timeshift, forward
from utilities import (create_folder, get_filename, create_logging, 
    StatisticsContainer, Mixup, frame_prediction_to_event_prediction, write_submission, pad_truncate_sequence, float32_to_int16, int16_to_float32)
from data_generator import (TrainSampler, TestSampler,
    collate_fn)
from models import *

dataset_dir = '../../../../../storage/leey0204/fsd50k_audioset/dcase2017'
data_type = 'long_predict'
audios_dir = os.path.join(dataset_dir, data_type)
audio_files = sorted(glob('{}/*.wav'.format(audios_dir)))

start_time = time.time()
librosa.core.load(audio_files[0], sr=32000, mono=True)
end_time = time.time()
print('Time taken to load entire audio file: {} s'. format(end_time-start_time))

sample_duration = 5
audios_num = len(audio_files)
for n in range(audios_num):
    audio_name = audio_files[n]
    audio_duration = librosa.get_duration(filename=audio_name)
    start = 0
    start_time = time.time()
    while start < audio_duration:
        librosa.core.load(audio_name, sr=32000, offset=start, duration=sample_duration, mono=True)
        start += sample_duration
    end_time = time.time()
    print('Time taken to load audio file 5s at a time: {} s'.format(end_time-start_time))
