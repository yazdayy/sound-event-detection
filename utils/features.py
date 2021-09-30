import os
import sys
import numpy as np
import argparse
import h5py
import librosa
import matplotlib.pyplot as plt
import time
import csv
import math
import re
import random
from glob import glob

import config
import gammatone.fftweight
from utilities import create_folder, pad_truncate_sequence, float32_to_int16


def get_weak_csv_filename(data_type):
    """Prepare weakly labelled csv path. 

    Args:
      data_type: 'training' | 'testing' | 'evaluation'

    Returns:
      str, weakly labelled csv path
    """
    if data_type in ['training', 'testing']:
        return '{}_set.csv'.format(data_type)
    elif data_type in ['weak_training', 'strong_training', 'strong_fsd50k', 'strong_validation']:
        return 'strong/{}_set.csv'.format(data_type)
    elif data_type in ['evaluation']:
        return 'groundtruth_weak_label_evaluation_set.csv'
        
    else:
        raise Exception('Incorrect argument!')


def read_weak_csv(weak_label_csv_path, data_type):
    """Read weakly labelled ground truth csv file. There can be multiple labels
    for each audio clip.

    Args:
      weak_label_csv_path: str
      data_type: 'training' | 'testing' | 'evaluation'

    Returns:
      meta_list: [{'audio_name': 'a.wav', 'labels': ['Train', 'Bus']},
                  ...]
    """
    assert data_type in ['training', 'testing', 'evaluation', 'weak_training', 'strong_training', 'strong_validation', 'strong_fsd50k']
    
    if data_type in ['training', 'testing', 'weak_training', 'strong_training', 'strong_validation', 'strong_fsd50k']:
        with open(weak_label_csv_path, 'r') as f:
            reader = csv.reader(f, delimiter=',')
            rows = list(reader)
            
    elif data_type in ['evaluation']:
        with open(weak_label_csv_path, 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            rows = list(reader)
            
    meta_list = []
            
    for row in rows:
        if data_type in ['training', 'testing', 'weak_training', 'strong_training', 'strong_validation']:
            """row: ['-5QrBL6MzLg', '60.000', '70.000', 'Train horn,Train', '/m/0284vy3,/m/07jdr']"""
            meta = {
                'audio_name': row[0] + '_' + str(int(float(row[1]))) + '.wav',
                'labels': re.split(',(?! )', row[3])}
            meta_list.append(meta)
        elif data_type in ['strong_fsd50k']:
            meta = {
                'audio_name': row[0] + '.wav',
                'labels': re.split(',(?! )', row[3])}
            meta_list.append(meta)
        elif data_type in ['evaluation']:
            """row: ['-JMT0mK0Dbg_30.000_40.000.wav', '30.000', '40.000', 'Train horn']"""
            audio_name = row[0]
            name_list = [meta['audio_name'] for meta in meta_list]

            if audio_name in name_list:
                n = name_list.index(audio_name)
                meta_list[n]['labels'].append(row[3])
            else:
                meta = {
                    'audio_name': '{}'.format(row[0]),
                    'labels': [row[3]]}
                meta_list.append(meta)

    return meta_list


def read_strong_csv(strong_meta_csv_path):
    """Read strongly labelled ground truth csv file. 
    
    Args:
      strong_meta_csv_path: str

    Returns: 
      meta_dict: {'a.wav': [{'onset': 3.0, 'offset': 5.0, 'label': 'Bus'},
                            {'onset': 4.0, 'offset': 7.0, 'label': 'Train'}
                            ...],
                  ...}
    """
    with open(strong_meta_csv_path, 'r') as fr:
        reader = csv.reader(fr, delimiter=',')
        lines = list(reader)
        
    meta_dict = {}
    for line in lines:
        """line: ['-5QrBL6MzLg_60.000_70.000.wav', '0.917', '2.029', 'Train horn']"""
        [audio_name, onset, offset, label] = line
        meta = {'onset': onset, 'offset': offset, 'label': label}
        if audio_name in meta_dict:
            meta_dict[audio_name].append(meta)
        else:
            meta_dict[audio_name] = [meta]
        
    return meta_dict


def get_weak_target(labels, lb_to_idx):
    """Labels to vector. 

    Args:
      labels: list of str
      lb_to_idx: dict

    Returns:
      target: (classes_num,)
    """
    classes_num = len(lb_to_idx)
    target = np.zeros(classes_num, dtype=np.bool)
    
    for label in labels: 
        target[lb_to_idx[label]] = 1.
        
    return target 


def get_strong_target(audio_name, strong_meta_dict, frames_num, 
    frames_per_second, lb_to_idx):
    """Reformat strongly labelled target to matrix format. 

    Args:
      audio_name: str
      strong_meta_dict: dict, e.g., 
          {'a.wav': [{'onset': 3.0, 'offset': 5.0, 'label': 'Bus'},
                     {'onset': 4.0, 'offset': 7.0, 'label': 'Train'}
                      ...],
           ...}
      frames_num: int
      frames_per_second: int
      lb_to_idx: dict

    Returns:
      target: (frames_num, classes_num)
    """
    
    meta_list = strong_meta_dict[audio_name]
    
    target = np.zeros((frames_num, len(lb_to_idx)), dtype=np.bool)
    
    for meta in meta_list:
        onset = float(meta['onset']) 
        bgn_frame = int(round(onset * frames_per_second))
        offset = float(meta['offset'])
        end_frame = int(round(offset * frames_per_second)) + 1
        label = meta['label']
        idx = lb_to_idx[label]
        
        target[bgn_frame : end_frame, idx] = 1
    
    return target


def pack_audio_files_to_hdf5(args):
    """Pack waveform to hdf5 file. 

    Args:
      dataset_dir: str, directory of dataset
      workspace: str, Directory of your workspace
      data_type: 'training' | 'testing' | 'evaluation'
      mini_data: bool, set True for debugging on a small part of data
    """

    # Arguments & parameters
    dataset_dir = args.dataset_dir
    workspace = args.workspace
    data_type = args.data_type
    mini_data = args.mini_data
    feature_type = args.feature_type
    audio_8k = args.audio_8k
    audio_16k = args.audio_16k
    
    window_size = config.window_size
    hop_size = config.hop_size
    mel_bins = config.mel_bins
    fmin = config.fmin
    sample_rate = config.sample_rate
    audio_samples = config.audio_samples
    classes_num = config.classes_num
    lb_to_idx = config.lb_to_idx
    frames_per_second = config.frames_per_second
    frames_num = frames_per_second * config.audio_duration
    """The +1 frame comes from the 'center=True' argument when extracting spectrogram."""
    
    if audio_8k:
        quality = '8k'
        sample_rate = 8000
        window_size = 256
        hop_size = 80
        mel_bins = 64
        fmin = 12
        fmax = 3500
    elif audio_16k:
        quality = '16k'
        sample_rate = 16000
        window_size = 512
        hop_size = 160
        mel_bins = 64
        fmin = 25
        fmax = 7000
    else:
        quality = '32k'
        sample_rate = 32000
        window_size = 1024
        hop_size = 320
        mel_bins = 64
        fmin = 50
        fmax = 14000
    
    audio_samples = sample_rate * 10
        
    codec_labels = ['amr475', 'ogg450','ogg550','ogg770','g726_1600','g726_2400', 'amr590', 'amr740','amr1020','ogg950','g726_3200','g723_630', 'amr1220','ogg1250','ogg1600','ogg3200','g722']
    has_strong_target = data_type in ['testing', 'evaluation', 'strong_training', 'strong_validation', 'strong_fsd50k']

    # Paths
    if data_type == 'strong_training' or data_type == 'weak_training' or data_type == 'strong_validation':
        if audio_8k:
            audios_dir = os.path.join(dataset_dir, 'training', '8k')
        else:
            audios_dir = os.path.join(dataset_dir, 'training')
    else:
        if audio_8k:
            audios_dir = os.path.join(dataset_dir, data_type, '8k')
        else:
            audios_dir = os.path.join(dataset_dir, data_type)
    
    all_files = glob('{}/*.wav'.format(audios_dir))
    all_files = [x.split('/')[-1] for x in all_files]
    
    weak_label_csv_path = os.path.join(dataset_dir, 'metadata', 
        get_weak_csv_filename(data_type))

    if data_type == 'testing':
        strong_label_csv_path = os.path.join(dataset_dir, 'metadata', 
            'groundtruth_strong_label_testing_set.csv')
    elif data_type == 'evaluation':
        strong_label_csv_path = os.path.join(dataset_dir, 'metadata', 
            'groundtruth_strong_label_evaluation_set.csv')
    elif data_type == 'strong_training':
        strong_label_csv_path = os.path.join(dataset_dir, 'metadata', 'strong',
        'groundtruth_strong_label_strong_training_set.csv')
    elif data_type == 'strong_validation':
        strong_label_csv_path = os.path.join(dataset_dir, 'metadata', 'strong',
        'groundtruth_strong_label_strong_validation_set.csv')
    elif data_type == 'strong_fsd50k':
        strong_label_csv_path = os.path.join(dataset_dir, 'metadata', 'strong',
        'groundtruth_strong_label_strong_fsd50k_set.csv')

    if mini_data:
        packed_hdf5_path = os.path.join(workspace, 'hdf5s', 'minidata_{}.h5'.format(data_type))
        
    packed_hdf5_path = os.path.join(workspace, 'hdf5s', '{}_{}_{}.h5'.format(data_type, feature_type, quality))
    
#    elif feature_type == 'logmel' and not audio_8k and not audio_16k:
#        packed_hdf5_path = os.path.join(workspace, 'hdf5s', '{}.h5'.format(data_type))
#    elif feature_type == 'logmel' and audio_8k:
#        packed_hdf5_path = os.path.join(workspace, 'hdf5s', '{}_8k.h5'.format(data_type))
#    elif feature_type == 'logmel' and audio_16k:
#        packed_hdf5_path = os.path.join(workspace, 'hdf5s', '{}_16k.h5'.format(data_type))
#    elif feature_type == 'gamma'  and not audio_8k and not audio_16k:
#        packed_hdf5_path = os.path.join(workspace, 'hdf5s', '{}_{}.h5'.format(data_type, feature_type))
#    elif feature_type == 'gamma' and audio_8k:
#        packed_hdf5_path = os.path.join(workspace, 'hdf5s', '{}_{}_8k.h5'.format(data_type, feature_type))
#    elif feature_type == 'gamma' and audio_16k:
#        packed_hdf5_path = os.path.join(workspace, 'hdf5s', '{}_{}_16k.h5'.format(data_type, feature_type))
    create_folder(os.path.dirname(packed_hdf5_path))

    # Read metadata
    weak_meta_list = read_weak_csv(weak_label_csv_path, data_type)
    """e.g., [{'audio_name': 'a.wav', 'labels': ['Train', 'Bus']},
              ...]"""

    # Use a small amount of data for debugging
    if mini_data:
        random.seed(1234)
        random.shuffle(weak_meta_list)
        weak_meta_list = weak_meta_list[0 : 100]

    audios_num = len(weak_meta_list)

    feature_time = time.time()
    with h5py.File(packed_hdf5_path, 'w') as hf:
        hf.create_dataset(
            name='audio_name', 
            shape=(audios_num,), 
            dtype='S80')
        
        if feature_type == 'logmel':
            hf.create_dataset(
                name='waveform',
                shape=(audios_num, audio_samples),
                dtype=np.int16)
        elif feature_type == 'gamma':
            hf.create_dataset(
                name='waveform',
                shape=(audios_num, mel_bins, 994),
                dtype=np.int16)

        hf.create_dataset(
            name='target', 
            shape=(audios_num, classes_num), 
            dtype=np.float32)

        if has_strong_target:
            strong_meta_dict = read_strong_csv(strong_label_csv_path)
            """e.g., {'a.wav': [{'onset': 3.0, 'offset': 5.0, 'label': 'Bus'},
                                {'onset': 4.0, 'offset': 7.0, 'label': 'Train'}
                                ...],
                      ...}
            """       
            
            hf.create_dataset(
                name='strong_target', 
                shape=(0, frames_num, classes_num), 
                maxshape=(None, frames_num, classes_num), 
                dtype=np.bool)
        
        for n in range(audios_num):
            print(n)
            weak_meta_dict = weak_meta_list[n]
            audio_name = weak_meta_dict['audio_name']
            if audio_8k:
                for filename in all_files:
                    if audio_name.split('.wav')[0] in filename:
                        codec_audio_name = filename
                        audio_path = os.path.join(audios_dir, codec_audio_name)
                        break
            else:
                audio_path = os.path.join(audios_dir, audio_name)
            try:
                (audio, fs) = librosa.core.load(audio_path, sr=sample_rate, mono=True)
            except ValueError:
                print(audio_path)
            audio = pad_truncate_sequence(audio, audio_samples)
            
            if feature_type == 'gamma':
                audio = gammatone.fftweight.fft_gtgram(audio, sample_rate, window_size/sample_rate, hop_size/sample_rate, mel_bins, fmin)
                audio = librosa.core.power_to_db(audio)
                # print(sample_rate)
                # print(audio.shape)

            # dasdsa = args.dadsa 

            hf['audio_name'][n] = audio_name.encode()
            hf['waveform'][n] = float32_to_int16(audio)
            hf['target'][n] = get_weak_target(
                weak_meta_dict['labels'], lb_to_idx)

            if has_strong_target:
                strong_target = get_strong_target(
                    weak_meta_dict['audio_name'], strong_meta_dict, 
                    frames_num, frames_per_second, lb_to_idx)
                
                hf['strong_target'].resize((n + 1, frames_num, classes_num))
                hf['strong_target'][n] = strong_target

    print('Write hdf5 to {}'.format(packed_hdf5_path))
    print('Time: {:.3f} s'.format(time.time() - feature_time))


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='')
    subparsers = parser.add_subparsers(dest='mode')

    # Pack waveform to hdf5 file
    parser_pack_audio = subparsers.add_parser('pack_audio_files_to_hdf5')
    parser_pack_audio.add_argument('--dataset_dir', type=str, required=True, help='Directory of dataset.')
    parser_pack_audio.add_argument('--workspace', type=str, required=True, help='Directory of your workspace.')
    parser_pack_audio.add_argument('--data_type', type=str, choices=['training', 'testing', 'evaluation', 'weak_training', 'strong_training', 'strong_validation', 'strong_fsd50k'], required=True, help='Directory of your workspace.')
    parser_pack_audio.add_argument('--feature_type', type=str, required=True)
    parser_pack_audio.add_argument('--mini_data', action='store_true', default=False, help='Set True for debugging on a small part of data.')
    parser_pack_audio.add_argument('--audio_8k', action='store_true', default=False)
    parser_pack_audio.add_argument('--audio_16k', action='store_true', default=False)
    
    # Parse arguments
    args = parser.parse_args()
    
    if args.mode == 'pack_audio_files_to_hdf5':
        pack_audio_files_to_hdf5(args)
        
    else:
        raise Exception('Incorrect arguments!')
