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
from dicttoxml import dicttoxml
import matplotlib.pyplot as plt
from collections import defaultdict

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

def sample_segment(sample, sample_duration, sample_rate):
    start = randint(0, len(sample) - (sample_duration * sample_rate))
    end = start + (sample_duration * sample_rate)
    segment = sample[start:end]

def trim_silent(dir_path, folder_name, file_path, non_silent_timestamps, original_duration, trimmed_duration):
    sample, sample_rate = librosa.load(file_path)
    non_silent = librosa.effects.split(y=sample, frame_length=sample_rate, top_db=20)
    original_duration.append(round(len(sample)/sample_rate,2))
    timestamp = []
    non_silent_sample = []
    for interval in non_silent:
        start = interval[0]
        end = interval[1]
        non_silent_sample.extend(sample[start:end])
        timestamp.append((start/sample_rate, end/sample_rate))
        
    non_silent_timestamps.append(timestamp)
    trimmed_duration.append((round(len(non_silent_sample)/sample_rate,2)))
    save_path = '{}/{}/{}'.format(dir_path, folder_name, file_path.split('/')[-1])
    sf.write(save_path, non_silent_sample, sample_rate)

def pack_audio_files_to_hdf5(n, start, sample_duration, dataset_dir, workspace, data_type, audio_name):
    """Pack waveform to hdf5 file.

    Args:
      dataset_dir: str, directory of dataset
      workspace: str, Directory of your workspace
      data_type: 'training' | 'testing' | 'evaluation'
    """

    # Arguments & parameters
    sample_rate = config.sample_rate
    classes_num = config.classes_num
    frames_per_second = config.frames_per_second
    frames_num = frames_per_second * config.audio_duration
    """The +1 frame comes from the 'center=True' argument when extracting spectrogram."""
    
#    # Trim silent segments
#    dir_path = os.path.dirname(audio_path)
#    os.makedirs('{}/{}'.format(dir_path, folder_name), exist_ok=True)
#    non_silent_timestamps = []
#    original_duration = []
#    trimmed_duration = []
#    audio_files = sorted(glob('{}/*.wav'.format(audio_path)))
#    for file_path in audio_files:
#        trim_silent(dir_path, folder_name, file_path, non_silent_timestamps, original_duration, trimmed_duration)

    audio_samples = sample_rate * sample_duration
    packed_hdf5_path = os.path.join(workspace, 'hdf5s', '{}.h5'.format(data_type))
    create_folder(os.path.dirname(packed_hdf5_path))

    # Read metadata
    
    audios_num = 1
    feature_time = time.time()
    with h5py.File(packed_hdf5_path, 'w') as hf:
        hf.create_dataset(
            name='audio_name',
            shape=(audios_num,),
            dtype='S80')

        hf.create_dataset(
            name='waveform',
            shape=(audios_num, audio_samples),
            dtype=np.int16)

        (audio, fs) = librosa.core.load(audio_name, sr=sample_rate, offset=start, duration=sample_duration, mono=True)
        audio = pad_truncate_sequence(audio, audio_samples)

        hf['audio_name'][n] = audio_name.encode()
        hf['waveform'][n] = float32_to_int16(audio)

class TestSampler(object):
    def __init__(self, hdf5_path, batch_size):
        """Testing data sampler.
        
        Args:
          hdf5_path, str
          batch_size: int
        """
        super(TestSampler, self).__init__()
        self.hdf5_path = hdf5_path
        self.batch_size = batch_size

        with h5py.File(self.hdf5_path, 'r') as hf:
            self.audios_num = len(hf['audio_name'])
            
        logging.info('Test audio num: {}'.format(self.audios_num))
        self.audio_indexes = np.arange(self.audios_num)

    def __iter__(self):
        """Generate batch meta for test.
        
        Returns:
          batch_meta: [{'hdf5_path':, xxx.h5, 'index_in_hdf5': 34},
                       {'hdf5_path':, xxx.h5, 'index_in_hdf5': 12},
                       ...]
        """
        batch_size = self.batch_size
        pointer = 0

        while pointer < self.audios_num:
            batch_indexes = np.arange(pointer,
                min(pointer + batch_size, self.audios_num))

            batch_meta = []

            for index in batch_indexes:
                batch_meta.append({
                    'hdf5_path': self.hdf5_path,
                    'index_in_hdf5': self.audio_indexes[index]})

            pointer += batch_size
            yield batch_meta


class Predictor(object):
    def __init__(self, model):
        """Evaluator.

        Args:
          model: object, model to be used for inference
        """
        self.model = model
        
        self.labels = config.labels
        self.idx_to_lb = config.idx_to_lb

        # Default parameters for SED
        self.sed_params_dict = {
            'audio_tagging_threshold': 0.5,
            'sed_high_threshold': 0.5,
            'sed_low_threshold': 0.2,
            'n_smooth': 10,
            'n_salt': 10}
    def predict(self, data_loader, submission_path):
        """SED prediction.

        Args:
          data_loader: object
          submission: str, path to write out submission file

        Returns:
          statistics: dict
          output_dict: dict
        """
        output_dict = forward(
            model=self.model,
            data_loader=data_loader,
            return_input=False,
            return_target=False)
        
        # Framewise predictions to eventwise predictions
        predict_event_list = frame_prediction_to_event_prediction(output_dict,
            self.sed_params_dict)

        # Write eventwise predictions to submission file
        write_submission(predict_event_list, submission_path)

        return output_dict, predict_event_list

class AudioDataset(object):
   def __init__(self):
       """DCASE 2017 Task 4 dataset."""
       pass

   def __getitem__(self, ):
       """Get input and target data of an audio clip.

       Args:
         meta: dict, e.g., {'hdf5_path':, xxx.h5, 'index_in_hdf5': 34}

       Returns:
         data_dict: {'audio_name': str,
                     'waveform': (audio_samples,),
                     'target': (classes_num,),
                     (ifexist) 'strong_target': (frames_num, classes_num)}
       """
       data_dict = {'audio_name': audio_name, 'waveform': waveform}

       return data_dict

def append_to_dict(dict, key, value):
    if key in dict.keys():
        dict[key].append(value)
    else:
        dict[key] = value

def predict(self):
    """Inference test and evaluate data and dump predicted probabilites to 
    pickle files.

    Args:
      dataset_dir: str
      workspace: str
      holdout_fold: '1'
      model_type: str, e.g., 'Cnn_9layers_Gru_FrameAtt'
      loss_type: str, e.g., 'clip_bce'
      augmentation: str, e.g., 'mixup'
      batch_size: int
      device: 'cuda' | 'cpu'
    
    Returns results in xml format:
      <AudioDoc name="this-is-the-file-name-of-audio-or-video">
      <SoundCaptionList>
      <SoundSegment stime="42.01" dur="3.01">loud noise</SoundSegment>
      <SoundSegment stime="48.02" dur="3.07">explosion</SoundSegment>
      <SoundSegment stime="53.03" dur="5.00">Mupliple explosion</SoundSegment>
      </SoundSegment>
      </AudioDoc>
    """
    
    # Arugments & parameters
    dataset_dir = args.dataset_dir
    workspace = args.workspace
    holdout_fold = args.holdout_fold
    model_type = args.model_type
    loss_type = args.loss_type
    augmentation = args.augmentation
    batch_size = args.batch_size
    device = 'cuda' if (args.cuda and torch.cuda.is_available()) else 'cpu'
    filename = args.filename
    feature_type = args.feature_type

    num_workers = 8
    os.makedirs('{}/long_predict_results'.format(workspace), exist_ok=True)
    data_type = 'long_predict'
    
    # Paths
    #predict_hdf5_path = os.path.join(workspace, 'hdf5s', 'long_predict.h5')

    checkpoint_path = os.path.join(workspace, 'checkpoints', 
        filename, 'holdout_fold={}'.format(holdout_fold),
        'model_type={}'.format(model_type), 'loss_type={}'.format(loss_type), 
        'augmentation={}'.format(augmentation), 'batch_size={}'.format(batch_size),
        'best.pth')

    predictions_dir = os.path.join(workspace, 'predictions', 
        filename, 'holdout_fold={}'.format(holdout_fold),
        'model_type={}'.format(model_type), 'loss_type={}'.format(loss_type), 
        'augmentation={}'.format(augmentation), 'batch_size={}'.format(batch_size))
    create_folder(predictions_dir)

    tmp_submission_path = os.path.join(workspace, '_tmp_submission', 
        filename, 'holdout_fold={}'.format(holdout_fold),
        'model_type={}'.format(model_type), 'loss_type={}'.format(loss_type), 
        'augmentation={}'.format(augmentation), 'batch_size={}'.format(batch_size),
        '_submission.csv')
    create_folder(os.path.dirname(tmp_submission_path))

    # Load model
    assert model_type, 'Please specify model_type!'
    start_time = time.time()
    Model = eval(model_type)
    model = Model(sample_rate, window_size, hop_size, mel_bins, fmin, fmax, classes_num, feature_type)

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model'])
    
    end_time = time.time()
    print('Time taken to initialise model and load weights: {} s'.format(end_time-start_time))
    # Parallel
    print('GPU number: {}'.format(torch.cuda.device_count()))
    model = torch.nn.DataParallel(model)

    if 'cuda' in device:
        model.to(device)

    # Dataset
    dataset = AudioDataset()
    
    # Predictor
    predictor = Predictor(model=model)
    
    sed_params_dict = {
    'audio_tagging_threshold': 0.099,
    'sed_high_threshold': 0.5,
    'sed_low_threshold': 0.2,
    'n_smooth': 10,
    'n_salt': 10}
    
    sample_duration = 10
    audios_dir = os.path.join(dataset_dir, data_type)
    audio_files = sorted(glob('{}/*.wav'.format(audios_dir)))
    audios_num = len(audio_files)
    audio_samples = sample_rate * sample_duration
    for n in range(audios_num):
        audio_name = audio_files[n]
        xml_string_list = []
        xml_string_list.append('<AudioDoc name="{}">\n'.format(audio_name.split('/')[-1]))
        xml_string_list.append('\t<SoundCaptionList>\n')
        last_onset = 0
        last_offset = 0
        last_event = ''
        tracker_dict = {}
        print('Predicting on {}'.format(audio_name))
        audio_path = os.path.join(audios_dir, audio_name)
        audio_duration = librosa.get_duration(filename=audio_name)
        print('Total audio duration: {} s'.format(audio_duration))
        num_segment = 1
        start = 0
        start_time = time.time()
        while start < audio_duration:
            
            # Load audio sample
            (audio, fs) = librosa.core.load(audio_name, sr=sample_rate, offset=start, duration=sample_duration, mono=True)
            audio = pad_truncate_sequence(audio, audio_samples)
            audio = torch.Tensor(audio)
            audio = torch.reshape(audio, (1,audio.size()[0])).to(device)

            output_dict = {}
            with torch.no_grad():
                model.eval()
                batch_output = model(audio)
                
            append_to_dict(output_dict, 'audio_name', audio_name)
            append_to_dict(output_dict, 'clipwise_output',
                batch_output['clipwise_output'].data.cpu().numpy())

            if 'framewise_output' in batch_output.keys():
                append_to_dict(output_dict, 'framewise_output',
                    batch_output['framewise_output'].data.cpu().numpy())
            
            predict_event_list = frame_prediction_to_event_prediction(output_dict,
            sed_params_dict)
            
            if audio_duration < start + sample_duration:
                end = audio_duration
            else:
                end = start + sample_duration

            print('Segment {}: {} s to {} s'.format(num_segment, start, end))
            print('---------------------------------------------------------------')
            if len(predict_event_list) >= 1:
                for event in predict_event_list:
                    onset = event['onset']
                    offset = event['offset']
                    event_label = event['event_label']
                    if onset == 0 and (event_label in list(tracker_dict.keys())):
                        if onset + start == tracker_dict[event_label][1]:
                            # Remove previous tracked segment from list
                            xml_string_list.pop(tracker_dict[event_label][3])
                            current_duration = event['offset']-event['onset']
                            xml_string_list.append('\t\t<SoundSegment stime="{}" dur="{}">{}</SoundSegment>\n'.format(tracker_dict[event_label][0], tracker_dict[event_label][2] + current_duration, event_label))
                            for k, v in tracker_dict.items():
                                if v[-1] - 1 == tracker_dict[event_label][3]:
                                    v[-1] = tracker_dict[event_label][3]
                            tracker_dict.pop(event_label)
                    else:
                        xml_string_list.append('\t\t<SoundSegment stime="{}" dur="{}">{}</SoundSegment>\n'.format(start+onset, offset-onset, event_label))
                    last_onset = start + onset
                    last_offset = start + offset
                    last_duration = offset - onset
                    if offset == 10:
                        tracker_dict[event['event_label']] = [last_onset, last_offset, last_duration, len(xml_string_list)-1]
                    
                    print('onset: {}, offset: {}, event_label: {}\n'.format(onset+start, offset+start, event_label))
            else:
                print('Others\n')
                xml_string_list.append('\t\t<SoundSegment stime="{}" dur="{}">Others</SoundSegment>\n'.format(start, end-start))
                #print(output_dict['clipwise_output'])
            start += sample_duration
            num_segment += 1
        end_time = time.time()
        print('Time taken to process {}: {} s\n'.format(audio_name, end_time-start_time))

        xml_string_list.append('\t</SoundCaptionList>\n')
        xml_string_list.append('</AudioDoc>')
        xml_string = ''.join(xml_string_list)
        xml_file = open("{}/long_predict_results/{}.xml".format(workspace, audio_name.split('/')[-1].split('.wav')[0]), "w")
        xml_file.write(xml_string)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Example of parser. ')
    subparsers = parser.add_subparsers(dest='mode')

    # Inference
    parser_inference_prob = subparsers.add_parser('predict')
    parser_inference_prob.add_argument('--dataset_dir', type=str, required=True, help='Directory of dataset.')
    parser_inference_prob.add_argument('--workspace', type=str, required=True, help='Directory of your workspace.')
    parser_inference_prob.add_argument('--filename', type=str, required=True)
    parser_inference_prob.add_argument('--holdout_fold', type=str, choices=['1'], required=True)
    parser_inference_prob.add_argument('--model_type', type=str, required=True)
    parser_inference_prob.add_argument('--loss_type', type=str, required=True)
    parser_inference_prob.add_argument('--augmentation', type=str, choices=['none', 'mixup', 'timeshift_mixup'], required=True)
    parser_inference_prob.add_argument('--batch_size', type=int, required=True)
    parser_inference_prob.add_argument('--feature_type', type=str, required=True)
    parser_inference_prob.add_argument('--cuda', action='store_true', default=False)
    
    # Parse arguments
    args = parser.parse_args()
    #args.filename = get_filename(__file__)

    if args.mode == 'predict':
        predict(args)

    else:
        raise Exception('Error argument!')
