import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../utils'))
import numpy as np
import pandas as pd
import argparse
import librosa
import h5py
import math
import time
import pickle
import logging
from glob import glob
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
 
import config
from evaluate import Evaluator
from config import (sample_rate, classes_num, mel_bins, fmin, fmax, 
    window_size, hop_size, window, pad_mode, center, device, ref, amin, top_db)
from losses import get_loss_func
from pytorch_utils import move_data_to_device, do_mixup, do_mixup_timeshift
from utilities import (create_folder, frame_prediction_to_event_prediction_v2, frame_prediction_to_event_prediction, get_filename, create_logging, official_evaluate, frame_binary_prediction_to_event_prediction,
    StatisticsContainer, pad_truncate_sequence, write_submission, Mixup, count_parameters, merge, avg_merge, append_to_dict)
from calculate_metrics import get_metric
from data_generator import (AudiosetDataset, TrainSampler, TestSampler,
    collate_fn)
from models import *

def cycle_iteration(iterable):
    while True:
        for i in iterable:
            yield i

def train(args):
    """Train and evaluate.

    Args:
      dataset_dir: str
      workspace: str
      holdout_fold: '1'
      model_type: str, e.g., 'Cnn_9layers_Gru_FrameAtt'
      loss_type: str, e.g., 'clip_bce'
      augmentation: str, e.g., 'mixup'
      learning_rate, float
      batch_size: int
      resume_iteration: int
      stop_iteration: int
      device: 'cuda' | 'cpu'
      mini_data: bool
    """
    
    # Arugments & parameters
    dataset_dir = args.dataset_dir
    workspace = args.workspace
    holdout_fold = args.holdout_fold
    model_type = args.model_type
    loss_type = args.loss_type
    augmentation = args.augmentation
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    resume_iteration = args.resume_iteration
    stop_iteration = args.stop_iteration
    feature_type = args.feature_type
    
    device = 'cuda' if (args.cuda and torch.cuda.is_available()) else 'cpu'
    
    mini_data = args.mini_data
    filename = args.filename
    audio_8k = args.audio_8k
    audio_16k = args.audio_16k
    vggish = args.vggish
    fsd50k = args.fsd50k
    timeshift = False
    spec_augment = False

    weak_loss_func = get_loss_func(loss_type)
    if loss_type == 'clip_bce':
        strong_loss_func = get_loss_func('frame_bce')
    elif loss_type == 'clip_bce_logits':
        strong_loss_func = get_loss_func('frame_bce_logits')
    
    num_workers = 8
    
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
    
    frames_per_second = sample_rate // hop_size
    file_header = ''
    
    # Paths
    if mini_data:
        prefix = 'minidata_'
    else:
        prefix = ''
    
    # HDF5 file paths
    weak_train_hdf5_path = os.path.join(workspace, 'hdf5s',
        '{}weak_training_{}_{}.h5'.format(prefix, feature_type, quality))
    
    strong_train_hdf5_path = os.path.join(workspace, 'hdf5s',
    '{}strong_training_{}_{}.h5'.format(prefix, feature_type, quality))
    
    strong_valid_hdf5_path = os.path.join(workspace, 'hdf5s',
    '{}strong_validation_{}_{}.h5'.format(prefix, feature_type, quality))

    test_hdf5_path = os.path.join(workspace, 'hdf5s',
        '{}testing_{}_{}.h5'.format(prefix, feature_type, quality))
    
    if fsd50k:
        strong_fsd50k_hdf5_path = os.path.join(workspace, 'hdf5s',
        '{}strong_fsd50k.h5'.format(prefix))
        file_header = 'fsd50k'
        
    # Groundtruth file paths
    strong_valid_reference_csv_path = os.path.join(dataset_dir, 'metadata', 'strong',
    'groundtruth_strong_label_strong_validation_set.csv')
    
    test_reference_csv_path = os.path.join(dataset_dir, 'metadata', 
        'groundtruth_strong_label_testing_set.csv')
        
#    if vggish:
#        quality = 'vggish'
#        window_size = int(0.025 * sample_rate)
#        hop_size = int(0.015 * sample_rate)
#        mel_bins = 64
#        fmin = 125
#        fmax = 7500
    
    # Miscellaneous file paths
    submission_name = '_submission_{}_{}.csv'.format(feature_type, quality)
    statistics_name = 'statistics_{}_{}.pickle'.format(feature_type, quality)
    checkpoint_name = 'best_{}_{}.pth'.format(feature_type, quality)

    # Create the relevant directories
    checkpoints_dir = os.path.join(workspace, 'checkpoints', file_header,
           '{}{}'.format(prefix, filename), 'holdout_fold={}'.format(holdout_fold),
           'model_type={}'.format(model_type), 'loss_type={}'.format(loss_type),
           'augmentation={}'.format(augmentation), 'batch_size={}'.format(batch_size))
    
    tmp_submission_path = os.path.join(workspace, '_tmp_submission', file_header,
        '{}{}'.format(prefix, filename), 'holdout_fold={}'.format(holdout_fold),
        'model_type={}'.format(model_type), 'loss_type={}'.format(loss_type),
        'augmentation={}'.format(augmentation), 'batch_size={}'.format(batch_size),
        submission_name)

    statistics_path = os.path.join(workspace, 'statistics', file_header,
        '{}{}'.format(prefix, filename), 'holdout_fold={}'.format(holdout_fold),
        'model_type={}'.format(model_type), 'loss_type={}'.format(loss_type),
        'augmentation={}'.format(augmentation), 'batch_size={}'.format(batch_size),
        statistics_name)
    
    logs_dir = os.path.join(workspace, 'logs', file_header, '{}{}'.format(prefix, filename),
        'holdout_fold={}'.format(holdout_fold), 'model_type={}'.format(model_type),
        'loss_type={}'.format(loss_type), 'augmentation={}'.format(augmentation),
        'batch_size={}'.format(batch_size))
            
    create_folder(checkpoints_dir)
    create_folder(os.path.dirname(tmp_submission_path))
    create_folder(os.path.dirname(statistics_path))
    create_logging(logs_dir, 'w')
    logging.info(args)

    if 'cuda' in device:
        logging.info('Using GPU.')
    else:
        logging.info('Using CPU. Set --cuda flag to use GPU.')
        
    audio_samples = sample_rate * 10
    
    # Model
    assert model_type, 'Please specify model_type!'
    Model = eval(model_type)
    if vggish:
        vggish_path = os.path.join(workspace, 'checkpoints', 'vggish', 'pytorch_vggish.pth')
        model = Model(sample_rate, window_size, hop_size, mel_bins, fmin, fmax, classes_num, feature_type, vggish_path)
    else:
        model = Model(sample_rate, window_size, hop_size, mel_bins, fmin, fmax, classes_num, feature_type)
    
    if resume_iteration:
        resume_checkpoint_path = os.path.join(checkpoints_dir, checkpoint_name)
        logging.info('Load resume model from {}'.format(resume_checkpoint_path))
        resume_checkpoint = torch.load(resume_checkpoint_path)
        model.load_state_dict(resume_checkpoint['model'])
        statistics_container.load_state_dict(resume_iteration)
        iteration = resume_checkpoint['iteration']
    else:
        iteration = 0
    
    # Parallel
    print('GPU number: {}'.format(torch.cuda.device_count()))
    model = torch.nn.DataParallel(model)

    if 'cuda' in device:
        model.to(device)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, 
        betas=(0.9, 0.999), eps=1e-08, weight_decay=0., amsgrad=True)

    # Dataset
    dataset = AudiosetDataset()
    
    # Sampler
    weak_train_sampler = TrainSampler(
        hdf5_path=weak_train_hdf5_path,
        batch_size=(batch_size * 3) * 2 if 'mixup' in augmentation else batch_size)
    
    strong_train_sampler = TrainSampler(
        hdf5_path=strong_train_hdf5_path,
        batch_size=batch_size * 2 if 'mixup' in augmentation else batch_size)
    
    if fsd50k:
        strong_fsd50k_sampler = TrainSampler(
            hdf5_path=strong_fsd50k_hdf5_path,
            batch_size=batch_size * 2 if 'mixup' in augmentation else batch_size)
        
        strong_fsd50k_loader = torch.utils.data.DataLoader(dataset=dataset,
            batch_sampler=strong_fsd50k_sampler, collate_fn=collate_fn,
            num_workers=num_workers, pin_memory=True)
            
        strong_fsd50k_iter = cycle_iteration(strong_fsd50k_loader)
    
    strong_valid_sampler = TestSampler(hdf5_path=strong_valid_hdf5_path, batch_size=batch_size)
    
    test_sampler = TestSampler(hdf5_path=test_hdf5_path, batch_size=batch_size)

    # Data loader
    weak_train_loader = torch.utils.data.DataLoader(dataset=dataset,
        batch_sampler=weak_train_sampler, collate_fn=collate_fn,
        num_workers=num_workers, pin_memory=True)
    
    strong_train_loader = torch.utils.data.DataLoader(dataset=dataset,
        batch_sampler=strong_train_sampler, collate_fn=collate_fn,
        num_workers=num_workers, pin_memory=True)
    
    strong_valid_loader = torch.utils.data.DataLoader(dataset=dataset,
        batch_sampler=strong_valid_sampler, collate_fn=collate_fn,
        num_workers=num_workers, pin_memory=True)
    
    test_loader = torch.utils.data.DataLoader(dataset=dataset, 
        batch_sampler=test_sampler, collate_fn=collate_fn, 
        num_workers=num_workers, pin_memory=True)
    
    weak_iter = cycle_iteration(weak_train_loader)
    strong_iter = cycle_iteration(strong_train_loader)
    
    # Determine augmentation techniques to use
    if 'mixup' in augmentation:
        mixup_augmenter = Mixup(mixup_alpha=1.)
    if 'timeshift' in augmentation:
        timeshift = True
    if 'specaugment' in augmentation:
        spec_augment = True
    
    # Evaluator
    evaluator = Evaluator(model=model)

    # Statistics
    statistics_container = StatisticsContainer(statistics_path)

    train_bgn_time = time.time()
    
    error_rates = []
    framewise_maps = []
    best_error_rate = 10000
    best_framewise_map = 0
    best_iteration = 0
    # Train on mini batches
    
    # Load samples
    while iteration != stop_iteration:
        # Evaluate
        if (iteration % 1000 == 0 and iteration > resume_iteration):# or (iteration == 0):
        
            logging.info('------------------------------------')
            logging.info('Iteration: {}'.format(iteration))

            train_fin_time = time.time()

            for (data_type, data_loader, reference_csv_path) in [('valid', strong_valid_loader, strong_valid_reference_csv_path), ('test', test_loader, test_reference_csv_path)]:

                # Calculate tatistics
                (statistics, _) = evaluator.evaluate(
                    data_loader, reference_csv_path, tmp_submission_path, frames_per_second)

                logging.info('{} statistics:'.format(data_type))
                logging.info('    Clipwise mAP: {:.3f}'.format(np.nanmean(statistics['clipwise_ap'])))
                logging.info('    Framewise mAP: {:.3f}'.format(np.nanmean(statistics['framewise_ap'])))
                logging.info('    {}'.format(statistics['sed_metrics']['overall']['error_rate']))

                statistics_container.append(data_type, iteration, statistics)
                
                if data_type == 'valid':
                    if np.nanmean(statistics['framewise_ap']) >= best_framewise_map and statistics['sed_metrics']['overall']['error_rate']['error_rate'] < best_error_rate <= best_error_rate:
                    
                        best_framewise_map = np.nanmean(statistics['framewise_ap'])
                        best_error_rate = statistics['sed_metrics']['overall']['error_rate']['error_rate']
                        best_iteration = iteration
                        
                        checkpoint = {
                            'iteration': iteration,
                            'model': model.module.state_dict(),
                            'optimizer': optimizer.state_dict()}
                        
                        checkpoint_path = os.path.join(checkpoints_dir, checkpoint_name)
                            
                        torch.save(checkpoint, checkpoint_path)
                        logging.info('Model saved to {} for iteration {}'.format(checkpoint_path, iteration))
                    
            statistics_container.dump()

            train_time = train_fin_time - train_bgn_time
            validate_time = time.time() - train_fin_time

            logging.info(
                'Train time: {:.3f} s, validate time: {:.3f} s'
                ''.format(train_time, validate_time))

            train_bgn_time = time.time()
        
        # Load samples
        weak_batch_data_dict = next(weak_iter)
        strong_batch_data_dict = next(strong_iter)
        if fsd50k:
            strong_fsd50k_batch_data_dict = next(strong_fsd50k_iter)
        
        if 'mixup' in augmentation:
            weak_batch_data_dict['mixup_lambda'] = mixup_augmenter.get_lambda(
                batch_size=len(weak_batch_data_dict['waveform']))
            strong_batch_data_dict['mixup_lambda'] = mixup_augmenter.get_lambda(
                batch_size=len(strong_batch_data_dict['waveform']))
            if fsd50k:
                strong_fsd50k_batch_data_dict['mixup_lambda'] = mixup_augmenter.get_lambda(
                    batch_size=len(strong_fsd50k_batch_data_dict['waveform']))

        # Move data to GPU
        for key in weak_batch_data_dict.keys():
            weak_batch_data_dict[key] = move_data_to_device(weak_batch_data_dict[key], device)
        for key in strong_batch_data_dict.keys():
            strong_batch_data_dict[key] = move_data_to_device(strong_batch_data_dict[key], device)
        if fsd50k:
            for key in strong_fsd50k_batch_data_dict.keys():
                strong_fsd50k_batch_data_dict[key] = move_data_to_device(strong_fsd50k_batch_data_dict[key], device)
        
        # Train
        model.train()
        
        if 'mixup' in augmentation:
            weak_batch_output_dict = model(weak_batch_data_dict['waveform'], weak_batch_data_dict['mixup_lambda'], timeshift=timeshift, spec_augment=spec_augment)
            weak_batch_target_dict = {'target': do_mixup(weak_batch_data_dict['target'], weak_batch_data_dict['mixup_lambda'])}
            strong_batch_output_dict = model(strong_batch_data_dict['waveform'], strong_batch_data_dict['mixup_lambda'], timeshift=timeshift, spec_augment=spec_augment)
            strong_batch_target_dict = {'strong_target': do_mixup(strong_batch_data_dict['strong_target'], strong_batch_data_dict['mixup_lambda'])}
            if fsd50k:
                strong_fsd50k_batch_output_dict = model(strong_fsd50k_batch_data_dict['waveform'], strong_fsd50k_batch_data_dict['mixup_lambda'], timeshift=timeshift, spec_augment=spec_augment)
                strong_fsd50k_batch_target_dict = {'strong_target': do_mixup(strong_fsd50k_batch_data_dict['strong_target'], strong_fsd50k_batch_data_dict['mixup_lambda'])}

        else:
            weak_batch_output_dict = model(weak_batch_data_dict['waveform'], None, timeshift=timeshift, spec_augment=spec_augment)
            weak_batch_target_dict = {'target': weak_batch_data_dict['target']}
            strong_batch_output_dict = model(strong_batch_data_dict['waveform'], None, timeshift=timeshift, spec_augment=spec_augment)
            strong_batch_target_dict = {'strong_target': strong_batch_data_dict['strong_target']}
            if fsd50k:
                strong_fsd50k_batch_output_dict = model(strong_fsd50k_batch_data_dict['waveform'], None)
                strong_fsd50k_batch_target_dict = {'strong_target': strong_fsd50k_batch_data_dict['strong_target']}

        # loss
        weak_loss = weak_loss_func(weak_batch_output_dict, weak_batch_target_dict)
        strong_loss = strong_loss_func(strong_batch_output_dict, strong_batch_target_dict)
        if fsd50k:
            strong_fsd50k_loss = strong_loss_func(strong_fsd50k_batch_output_dict, strong_fsd50k_batch_target_dict)
            loss = weak_loss + strong_loss + strong_fsd50k_loss
            print('{} iteration - weak: {}, strong: {}, total: {}'.format(iteration, weak_loss, strong_loss, strong_fsd50k_loss, loss))
        else:
            loss = weak_loss + strong_loss
            print('{} iteration - weak: {}, strong: {}, total: {}'.format(iteration, weak_loss, strong_loss, loss))

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Stop learning
        if iteration == stop_iteration:
            logging.info('    Best iteration: {}'.format(best_iteration))
            break 
            
        iteration += 1
        

def inference_prob(self):
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
    """
    
    # Arugments & parameters
    dataset_dir = args.dataset_dir
    workspace = args.workspace
    holdout_fold = args.holdout_fold
    model_type = args.model_type
    loss_type = args.loss_type
    augmentation = args.augmentation
    batch_size = args.batch_size
    feature_type = args.feature_type
    fsd50k = args.fsd50k
    audio_8k = args.audio_8k
    audio_16k = args.audio_16k
    vggish = args.vggish
    device = 'cuda' if (args.cuda and torch.cuda.is_available()) else 'cpu'
    sed_thresholds = args.sed_thresholds
    filename = args.filename

    num_workers = 8
    data_type = 'testing'
    
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
        sample_rate = config.sample_rate
        window_size = window_size = config.window_size
        hop_size = config.hop_size
        mel_bins = config.mel_bins
        fmin = config.fmin
        fmax = config.fmax
    
    audio_samples = sample_rate * 10
    frames_per_second = sample_rate // hop_size
    
    # Paths
    test_hdf5_path = os.path.join(workspace, 'hdf5s', '{}_{}_{}.h5'.format(data_type, feature_type, quality))

    checkpoint_name = 'best_{}_{}.pth'.format(feature_type, quality)
    submission_name = '_submission_{}_{}.csv'.format(feature_type, quality)
    
    if fsd50k:
        pre_dir = 'fsd50k'
    else:
        pre_dir = ''

    test_reference_csv_path = os.path.join(dataset_dir, 'metadata', 
        'groundtruth_strong_label_{}_set.csv'.format(data_type))
    
    checkpoint_path = os.path.join(workspace, 'checkpoints', pre_dir,
        '{}'.format(filename), 'holdout_fold={}'.format(holdout_fold), 
        'model_type={}'.format(model_type), 'loss_type={}'.format(loss_type), 
        'augmentation={}'.format(augmentation), 'batch_size={}'.format(batch_size),
        checkpoint_name)

    predictions_dir = os.path.join(workspace, 'predictions', pre_dir,
        '{}'.format(filename), 'holdout_fold={}'.format(holdout_fold), 
        'model_type={}'.format(model_type), 'loss_type={}'.format(loss_type), 
        'augmentation={}'.format(augmentation), 'batch_size={}'.format(batch_size))
    create_folder(predictions_dir)

    tmp_submission_path = os.path.join(workspace, '_tmp_submission', pre_dir,
        '{}'.format(filename), 'holdout_fold={}'.format(holdout_fold), 
        'model_type={}'.format(model_type), 'loss_type={}'.format(loss_type), 
        'augmentation={}'.format(augmentation), 'batch_size={}'.format(batch_size),
        submission_name)
    create_folder(os.path.dirname(tmp_submission_path))

    if sed_thresholds:
        sed_thresholds_path = os.path.join(workspace, 'opt_thresholds', pre_dir,
            '{}'.format(filename), 'holdout_fold={}'.format(holdout_fold),
            'model_type={}'.format(model_type), 'loss_type={}'.format(loss_type),
            'augmentation={}'.format(augmentation), 'batch_size={}'.format(batch_size),
            'best_{}_{}.sed.valid.pkl'.format(feature_type, quality))
        sed_params_dict = pickle.load(open(sed_thresholds_path, 'rb'))
    else:
        sed_params_dict = {
            'audio_tagging_threshold': 0.099,
            'sed_high_threshold': 0.5,
            'sed_low_threshold': 0.2,
            'n_smooth': 10,
            'n_salt': 10}

    # Load model
    assert model_type, 'Please specify model_type!'
    Model = eval(model_type)
    if vggish:
        vggish_path = os.path.join(workspace, 'checkpoints', 'vggish', 'pytorch_vggish.pth')
        model = Model(sample_rate, window_size, hop_size, mel_bins, fmin, fmax, classes_num, feature_type, vggish_path)
    else:
        model = Model(sample_rate, window_size, hop_size, mel_bins, fmin, fmax, classes_num, feature_type)

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model'])

    # Parallel
    print('GPU number: {}'.format(torch.cuda.device_count()))
    model = torch.nn.DataParallel(model)
    
    if 'cuda' in device:
        model.to(device)
    
    print('Model size: {}\n'.format(count_parameters(model)))
    
    # Dataset
    dataset = AudiosetDataset()

    # Sampler
    test_sampler = TestSampler(hdf5_path=test_hdf5_path, batch_size=batch_size)

    # Data loader
    test_loader = torch.utils.data.DataLoader(dataset=dataset,
        batch_sampler=test_sampler, collate_fn=collate_fn,
        num_workers=num_workers, pin_memory=True)

    # Evaluator
    evaluator = Evaluator(model=model)
    
    for (data_type, data_loader, reference_csv_path) in [
        ('test', test_loader, test_reference_csv_path)]:
        
        start_time = time.time()
        print('Inferencing {} data in about 1 min ...'.format(data_type))

        (statistics, output_dict) = evaluator.evaluate(
            data_loader, reference_csv_path, tmp_submission_path, frames_per_second)
    
        predict_event_list = frame_prediction_to_event_prediction(output_dict, 
        sed_params_dict, frames_per_second)
        
        end_time = time.time()
        
        
        
        # Write predicted events to submission file
        write_submission(predict_event_list, tmp_submission_path)

        # SED with official tool
        results = official_evaluate(reference_csv_path, tmp_submission_path)
        
        sed_precision = get_metric(results, 'precision')
        sed_recall = get_metric(results, 'recall')
        sed_f1 = get_metric(results, 'f1')
        sed_er = get_metric(results, 'er')
        
        print('Processing time: {} s\n'.format(end_time-start_time))
        print('Micro precision: {:.3f}'.format(sed_precision))
        print('Micro recall: {:.3f}'.format(sed_recall))
        print('Micro F1: {:.3f}'.format(sed_f1))
        print('Micro ER: {:.3f}'.format(sed_er))


def inference_prob_overlap(self):
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
    """

    # Arugments & parameters
    dataset_dir = args.dataset_dir
    workspace = args.workspace
    holdout_fold = args.holdout_fold
    model_type = args.model_type
    loss_type = args.loss_type
    augmentation = args.augmentation
    batch_size = args.batch_size
    feature_type = args.feature_type
    device = 'cuda' if (args.cuda and torch.cuda.is_available()) else 'cpu'
    filename = args.filename
    sample_duration = args.sample_duration
    sed_thresholds = args.sed_thresholds
    audio_8k = args.audio_8k
    audio_16k = args.audio_16k
    data_type = args.data_type
    fsd50k = args.fsd50k
    
    #data_type = 'testing'

    num_workers = 8

    # Paths
    
#    if feature_type == 'logmel' and not audio_8k and not audio_16k:
#        test_hdf5_path = os.path.join(workspace, 'hdf5s', '{}.h5'.format(data_type))
#    elif feature_type == 'logmel' and audio_8k:
#        test_hdf5_path = os.path.join(workspace, 'hdf5s', '{}_8k.h5'.format(data_type))
#    elif feature_type == 'logmel' and audio_16k:
#        test_hdf5_path = os.path.join(workspace, 'hdf5s', '{}_16k.h5'.format(data_type))
#    elif feature_type == 'gamma':
#        test_hdf5_path = os.path.join(workspace, 'hdf5s', '{}_{}.h5'.format(data_type, feature_type))

    #test_reference_csv_path = os.path.join(dataset_dir, 'metadata', 'groundtruth_strong_label_testing_set.csv')
    
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
    frames_per_second = sample_rate // hop_size
    
    checkpoint_name = 'best_{}_{}.pth'.format(feature_type, quality)
    tmp_submission_name = '_overlap_submission_{}.csv'.format(quality)
    test_hdf5_path = os.path.join(workspace, 'hdf5s', '{}_{}_{}.h5'.format(data_type, feature_type, quality))
    
    if fsd50k:
        pre_dir = 'fsd50k'
    else:
        pre_dir = ''
    
    checkpoint_path = os.path.join(workspace, 'checkpoints', pre_dir,
        '{}'.format(filename), 'holdout_fold={}'.format(holdout_fold),
        'model_type={}'.format(model_type), 'loss_type={}'.format(loss_type),
        'augmentation={}'.format(augmentation), 'batch_size={}'.format(batch_size),
        checkpoint_name)
    
    predictions_dir = os.path.join(workspace, 'predictions', pre_dir,
    '{}'.format(filename), 'holdout_fold={}'.format(holdout_fold),
    'model_type={}'.format(model_type), 'loss_type={}'.format(loss_type),
    'augmentation={}'.format(augmentation), 'batch_size={}'.format(batch_size))
    
    tmp_submission_path = os.path.join(workspace, '_tmp_submission', pre_dir,
    '{}'.format(filename), 'holdout_fold={}'.format(holdout_fold),
    'model_type={}'.format(model_type), 'loss_type={}'.format(loss_type),
    'augmentation={}'.format(augmentation), 'batch_size={}'.format(batch_size),
    tmp_submission_name)
    
    reference_csv_path = os.path.join(dataset_dir, 'metadata', 'groundtruth_strong_label_{}_set.csv'.format(data_type))
        
    create_folder(predictions_dir)
    create_folder(os.path.dirname(tmp_submission_path))
    
    # Load model
    assert model_type, 'Please specify model_type!'
    Model = eval(model_type)
    model = Model(sample_rate, window_size, hop_size, mel_bins, fmin, fmax,
        classes_num, feature_type)

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model'])

    count_parameters(model)

    # Parallel
    print('GPU number: {}'.format(torch.cuda.device_count()))
    model = torch.nn.DataParallel(model)

    if 'cuda' in device:
        model.to(device)
        
    if sed_thresholds:
        sed_thresholds_path = os.path.join(workspace, 'opt_thresholds', pre_dir,
            '{}'.format(filename), 'holdout_fold={}'.format(holdout_fold),
            'model_type={}'.format(model_type), 'loss_type={}'.format(loss_type),
            'augmentation={}'.format(augmentation), 'batch_size={}'.format(batch_size),
            'best_{}_{}.sed.valid.pkl'.format(feature_type, quality))
        sed_params_dict = pickle.load(open(sed_thresholds_path, 'rb'))
    else:
        sed_params_dict = {
            'audio_tagging_threshold': 0.099,
            'sed_high_threshold': 0.5,
            'sed_low_threshold': 0.2,
            'n_smooth': 10,
            'n_salt': 10}
    
#    overlap_values = np.arange(0.5,3,0.5)
#    segment_lengths = np.arange(3,10,0.5)
#
#    param_combinations = []
#    for value in overlap_values:
#        for length in segment_lengths:
#            param_combinations.append([value, length])

    param_combinations = [[0.5, 5], [0.5,6], [0.5,7], [0.5, 8], [1,5], [1,6], [1,7], [1, 8], [10,10]]
#    param_combinations = [[0.5,6], [10,10]]
    
    if audio_8k:
        audios_dir = os.path.join(dataset_dir, data_type, '8k')
    else:
        audios_dir = os.path.join(dataset_dir, data_type)
    gt_csv = pd.read_csv(reference_csv_path, header=None)
    audio_files = gt_csv[0].unique()
    audio_files = ['{}/{}'.format(audios_dir, x) for x in audio_files]
    full_audio_files = glob('{}/*.wav'.format(audios_dir))
    
    #audio_files = sorted(glob('{}/*.wav'.format(audios_dir)))
    audios_num = len(audio_files)
    print('NUMBER OF AUDIO FILES:', audios_num)
    audio_samples = sample_rate * 10 #sample_duration
    for param in param_combinations:
        overlap_value = param[0]
        sample_duration = param[1]
        predict_length = 0
        full_predict_event_list = []
        start_time = time.time()
        for n in range(audios_num):
            audio_name = audio_files[n]
            #print('Predicting on {}'.format(audio_name))
            if audio_8k:
               for filename in full_audio_files:
                   if audio_name.split('/')[-1].split('.wav')[0] in filename:
                       full_audio_name = filename
                       break
            else:
                full_audio_name = audio_name
            audio_duration = librosa.get_duration(filename=full_audio_name)
            predict_length += audio_duration
            #print('Total audio duration: {} s'.format(audio_duration))
            num_segment = 1
            index = 0
            start = 0
            end = 0
            merged = None
            try:
                (audio_full, fs) = librosa.core.load(full_audio_name, sr=sample_rate, mono=True)
            except ValueError:
                print(full_audio_name)
            audio_full = pad_truncate_sequence(audio_full, audio_samples)
            while end <= audio_duration:
                # Load audio sample
    #            (audio, fs) = librosa.core.load(audio_name, sr=sample_rate, offset=start, duration=sample_duration, mono=True)
    #            audio = pad_truncate_sequence(audio, audio_samples)
                start_index = int(start * sample_rate)
                end_index = int((sample_duration * sample_rate) + start_index)
                audio = audio_full[start_index:end_index]
                audio = torch.Tensor(audio)
                audio = torch.reshape(audio, (1, audio.size()[0])).to(device)

                output_dict = {}
                merged_output_dict = {}
                with torch.no_grad():
                    model.eval()
                    batch_output = model(audio)
                    
                append_to_dict(output_dict, 'audio_name', audio_name)
                append_to_dict(output_dict, 'clipwise_output',
                    batch_output['clipwise_output'].data.cpu().numpy())

                if 'framewise_output' in batch_output.keys():
                    append_to_dict(output_dict, 'framewise_output',
                        batch_output['framewise_output'].data.cpu().numpy())
                
                curr_clipwise = output_dict['clipwise_output']
                curr_preds = output_dict['framewise_output']
                if num_segment == 2:
                    merged = merge(prev_preds, curr_preds, sample_duration, num_segment, overlap_value)
                elif num_segment > 2:
                    merged = merge(merged, curr_preds, sample_duration, num_segment, overlap_value)
                else:
                    merged = curr_preds
    #            predict_event_list = frame_prediction_to_event_prediction(output_dict,
    #            sed_params_dict)
                prev_clipwise = output_dict['clipwise_output']
                prev_preds = output_dict['framewise_output']
                
                #start += 1
                start += overlap_value
                end = start + sample_duration
                num_segment += 1
            merged = avg_merge(merged, sample_duration, overlap_value)
            #np.set_printoptions(threshold=sys.maxsize)
            predict_event_list = frame_prediction_to_event_prediction_v2(merged, audio_name.split('/')[-1], sed_params_dict, frames_per_second)
            full_predict_event_list.extend(predict_event_list)
        
        #print(full_predict_event_list)
        end_time = time.time()
        time_taken = end_time - start_time
        print('Processing time for {}: {} s'.format(param, time_taken))
        print('Total audio duration: {} s'.format(predict_length))
    
        # Write predicted events to submission file
        write_submission(full_predict_event_list, tmp_submission_path)

        # SED with official tool
        results = official_evaluate(reference_csv_path, tmp_submission_path)
        
        sed_precision = get_metric(results, 'precision')
        sed_recall = get_metric(results, 'recall')
        sed_f1 = get_metric(results, 'f1')
        sed_er = get_metric(results, 'er')

        print('Micro precision: {:.3f}'.format(sed_precision))
        print('Micro recall: {:.3f}'.format(sed_recall))
        print('Micro F1: {:.3f}'.format(sed_f1))
        print('Micro ER: {:.3f} \n'.format(sed_er))
       
       
#def binarize_pred(pred, sed_thresholds):
#    pred_binary = np.zeros(pred.shape)
#    for i in range(len(pred)):
#        for j in range(len(pred[i])):
#            for k in range(len(pred[i][j])):
#                if pred[i][j][k] > sed_thresholds[k]:
#                    pred_binary[i][j][k] = 1
#
#    return pred_binary

def binarize_pred(pred, sed_high_threshold, sed_thresholds=False):
    pred_binary = np.zeros(pred.shape)

    for i in range(len(pred)):
        for j in range(len(pred[i])):
            for k in range(len(pred[i][j])):
                if sed_thresholds:
                    if pred[i][j][k] > sed_high_threshold[k]:
                        pred_binary[i][j][k] = 1
                else:
#                    print('CHECK1', sed_high_threshold)
#                    print('CHECK2', pred[i][j][k])
                    if isinstance(sed_high_threshold, list):
                        sed_high_threshold = sed_high_threshold[0]
                    if pred[i][j][k] > sed_high_threshold:
                        pred_binary[i][j][k] = 1

    return pred_binary
    
def inference_prob_vote(self):
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
    """

    # Arugments & parameters
    dataset_dir = args.dataset_dir
    workspace = args.workspace
    holdout_fold = args.holdout_fold
    model_type = args.model_type
    loss_type = args.loss_type
    augmentation = args.augmentation
    batch_size = args.batch_size
    feature_type = args.feature_type
    device = 'cuda' if (args.cuda and torch.cuda.is_available()) else 'cpu'
    filename = args.filename
    sample_duration = args.sample_duration
    sed_thresholds = args.sed_thresholds
    audio_8k = args.audio_8k
    audio_16k = args.audio_16k
    data_type = args.data_type
    fsd50k = args.fsd50k

    num_workers = 8

    # Paths
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
    checkpoint_name = 'best_{}_{}.pth'.format(feature_type, quality)
    tmp_submission_name = '_overlap_submission_{}.csv'.format(quality)
    test_hdf5_path = os.path.join(workspace, 'hdf5s', '{}_{}_{}.h5'.format(data_type, feature_type, quality))

    if fsd50k:
        pre_dir = 'fsd50k'
    else:
        pre_dir = ''

    checkpoint_path = os.path.join(workspace, 'checkpoints', pre_dir,
        '{}'.format(filename), 'holdout_fold={}'.format(holdout_fold),
        'model_type={}'.format(model_type), 'loss_type={}'.format(loss_type),
        'augmentation={}'.format(augmentation), 'batch_size={}'.format(batch_size),
        checkpoint_name)

    predictions_dir = os.path.join(workspace, 'predictions', pre_dir,
    '{}'.format(filename), 'holdout_fold={}'.format(holdout_fold),
    'model_type={}'.format(model_type), 'loss_type={}'.format(loss_type),
    'augmentation={}'.format(augmentation), 'batch_size={}'.format(batch_size))

    tmp_submission_path = os.path.join(workspace, '_tmp_submission', pre_dir,
    '{}'.format(filename), 'holdout_fold={}'.format(holdout_fold),
    'model_type={}'.format(model_type), 'loss_type={}'.format(loss_type),
    'augmentation={}'.format(augmentation), 'batch_size={}'.format(batch_size),
    tmp_submission_name)

    reference_csv_path = os.path.join(dataset_dir, 'metadata', 'groundtruth_strong_label_{}_set.csv'.format(data_type))

    create_folder(predictions_dir)
    create_folder(os.path.dirname(tmp_submission_path))

    # Load model
    assert model_type, 'Please specify model_type!'
    Model = eval(model_type)
    model = Model(sample_rate, window_size, hop_size, mel_bins, fmin, fmax,
        classes_num, feature_type)

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model'])

    count_parameters(model)

    # Parallel
    print('GPU number: {}'.format(torch.cuda.device_count()))
    model = torch.nn.DataParallel(model)

    if 'cuda' in device:
        model.to(device)
        
    if sed_thresholds:
        sed_thresholds_path = os.path.join(workspace, 'opt_thresholds',
            '{}'.format(filename), 'holdout_fold={}'.format(holdout_fold),
            'model_type={}'.format(model_type), 'loss_type={}'.format(loss_type),
            'augmentation={}'.format(augmentation), 'batch_size={}'.format(batch_size),
            'best_{}_{}.sed.valid.pkl'.format(feature_type, quality))
        sed_params_dict = pickle.load(open(sed_thresholds_path, 'rb'))
    else:
        sed_params_dict = {
            'audio_tagging_threshold': 0.099,
            'sed_high_threshold': 0.5,
            'sed_low_threshold': 0.2,
            'n_smooth': 10,
            'n_salt': 10}

#    param_combinations = [[0.5,6], [0.5,7], [1,5], [1,6], [1,7]]
    param_combinations = [[0.5, 5], [1,5]]

    if audio_8k:
        audios_dir = os.path.join(dataset_dir, data_type, '8k')
    else:
        audios_dir = os.path.join(dataset_dir, data_type)
        
    gt_csv = pd.read_csv(reference_csv_path, header=None)
    audio_files = gt_csv[0].unique()
    audio_files = ['{}/{}'.format(audios_dir, x) for x in audio_files]
    full_audio_files = glob('{}/*.wav'.format(audios_dir))

    #audio_files = sorted(glob('{}/*.wav'.format(audios_dir)))
    audios_num = len(audio_files)
    print('NUMBER OF AUDIO FILES:', audios_num)
    audio_samples = sample_rate * 10 #sample_duration
    for param in param_combinations:
        overlap_value = param[0]
        sample_duration = param[1]
        predict_length = 0
        full_predict_event_list = []
        start_time = time.time()
        for n in range(audios_num):
            audio_name = audio_files[n]
            if audio_8k:
               for filename in full_audio_files:
                   if audio_name.split('/')[-1].split('.wav')[0] in filename:
                       full_audio_name = filename
                       break
            else:
                full_audio_name = audio_name
                
            audio_duration = librosa.get_duration(filename=full_audio_name)
            predict_length += audio_duration
            num_segment = 1
            index = 0
            start = 0
            end = 0
            merged = None
            # Load audio sample
            try:
                (audio_full, fs) = librosa.core.load(full_audio_name, sr=sample_rate, mono=True)
            except ValueError:
                print(full_audio_name)
                
            audio_full = pad_truncate_sequence(audio_full, audio_samples)
            while end <= audio_duration:
                start_index = int(start * sample_rate)
                end_index = int((sample_duration * sample_rate) + start_index)
                audio = audio_full[start_index:end_index]
                audio = torch.Tensor(audio)
                audio = torch.reshape(audio, (1, audio.size()[0])).to(device)

                output_dict = {}
                merged_output_dict = {}
                with torch.no_grad():
                    model.eval()
                    batch_output = model(audio)
                    
                append_to_dict(output_dict, 'audio_name', audio_name)
                append_to_dict(output_dict, 'clipwise_output',
                    batch_output['clipwise_output'].data.cpu().numpy())

                if 'framewise_output' in batch_output.keys():
                    append_to_dict(output_dict, 'framewise_output',
                        batch_output['framewise_output'].data.cpu().numpy())
                
                curr_clipwise = output_dict['clipwise_output']
                curr_preds = output_dict['framewise_output']
                
                curr_preds = binarize_pred(curr_preds, sed_params_dict['sed_low_threshold'], sed_thresholds)
                if num_segment == 2:
                    merged = merge(prev_preds, curr_preds, sample_duration, num_segment, overlap_value)
                elif num_segment > 2:
                    merged = merge(merged, curr_preds, sample_duration, num_segment, overlap_value)
                else:
                    merged = curr_preds
                prev_clipwise = output_dict['clipwise_output']
                prev_preds = output_dict['framewise_output']
                prev_preds = binarize_pred(prev_preds, sed_params_dict['sed_low_threshold'], sed_thresholds)
                
                #start += 1
                start += overlap_value
                end = start + sample_duration
                num_segment += 1
            #merged = avg_merge(merged, sample_duration, overlap_value)
            #np.set_printoptions(threshold=sys.maxsize)
            predict_event_list = frame_binary_prediction_to_event_prediction(merged, overlap_value, sample_duration, audio_name.split('/')[-1], sed_params_dict)
            full_predict_event_list.extend(predict_event_list)
            
        #print(full_predict_event_list)
        end_time = time.time()
        time_taken = end_time - start_time
        print('Processing time for {}: {} s'.format(param, time_taken))
        print('Total audio duration: {} s'.format(predict_length))

        # Write predicted events to submission file
        write_submission(full_predict_event_list, tmp_submission_path)

        # SED with official tool
        results = official_evaluate(reference_csv_path, tmp_submission_path)
        
        sed_precision = get_metric(results, 'precision')
        sed_recall = get_metric(results, 'recall')
        sed_f1 = get_metric(results, 'f1')
        sed_er = get_metric(results, 'er')

        print('Micro precision: {:.3f}'.format(sed_precision))
        print('Micro recall: {:.3f}'.format(sed_recall))
        print('Micro F1: {:.3f}'.format(sed_f1))
        print('Micro ER: {:.3f} \n'.format(sed_er))


def trim_silent(file_path, sample_rate, window_size, hop_size, db_limit):
    (sample, fs) = librosa.load(file_path, sr=sample_rate, mono=True)
    non_silent = librosa.effects.split(y=sample, frame_length=window_size, hop_length=hop_size, top_db=db_limit)

    timestamps = []
    intervals = []
    non_silent_sample = []
    for i, interval in enumerate(non_silent):
        start = interval[0]
        end = interval[1]
#        if i == 0 and start > 0:
#                non_silent_sample.extend(np.zeros(sample[0:start].shape))
#        elif i >= 1 and prev_end + 1 != start:
#                non_silent_sample.extend(np.zeros(sample[prev_end:start].shape))
#
#        prev_end = end
        if end == len(sample):
            non_silent_sample.extend(sample[start:])
        else:
            non_silent_sample.extend(sample[start:end+1])
        timestamps.append((start/sample_rate, end/sample_rate))
        intervals.append((start, end))
    
    return np.array(non_silent_sample), timestamps, intervals

# def inference_prob_trim_silent(self):
#     """Inference test and evaluate data and dump predicted probabilites to
#     pickle files.

#     Args:
#       dataset_dir: str
#       workspace: str
#       holdout_fold: '1'
#       model_type: str, e.g., 'Cnn_9layers_Gru_FrameAtt'
#       loss_type: str, e.g., 'clip_bce'
#       augmentation: str, e.g., 'mixup'
#       batch_size: int
#       device: 'cuda' | 'cpu'
#     """

#     # Arugments & parameters
#     dataset_dir = args.dataset_dir
#     workspace = args.workspace
#     holdout_fold = args.holdout_fold
#     model_type = args.model_type
#     loss_type = args.loss_type
#     augmentation = args.augmentation
#     batch_size = args.batch_size
#     feature_type = args.feature_type
#     device = 'cuda' if (args.cuda and torch.cuda.is_available()) else 'cpu'
#     filename = args.filename
#     sample_duration = args.sample_duration
#     sed_thresholds = args.sed_thresholds
#     audio_8k = args.audio_8k
#     audio_16k = args.audio_16k
#     data_type = args.data_type
#     fsd50k = args.fsd50k

#     #data_type = 'testing'

#     num_workers = 8

#     # Paths
#     if audio_8k:
#         quality = '8k'
#         sample_rate = 8000
#         window_size = 256
#         hop_size = 80
#         mel_bins = 64
#         fmin = 12
#         fmax = 3500
#     elif audio_16k:
#         quality = '16k'
#         sample_rate = 16000
#         window_size = 512
#         hop_size = 160
#         mel_bins = 64
#         fmin = 25
#         fmax = 7000
#     else:
#         quality = '32k'
#         sample_rate = 32000
#         window_size = 1024
#         hop_size = 320
#         mel_bins = 64
#         fmin = 50
#         fmax = 14000

#     audio_samples = sample_rate * 10
#     frames_per_second = sample_rate // hop_size

#     checkpoint_name = 'best_{}_{}.pth'.format(feature_type, quality)
#     tmp_submission_name = '_overlap_submission_{}.csv'.format(quality)
#     test_hdf5_path = os.path.join(workspace, 'hdf5s', '{}_{}_{}.h5'.format(data_type, feature_type, quality))

#     if fsd50k:
#         pre_dir = 'fsd50k'
#     else:
#         pre_dir = ''

#     checkpoint_path = os.path.join(workspace, 'checkpoints', pre_dir,
#         '{}'.format(filename), 'holdout_fold={}'.format(holdout_fold),
#         'model_type={}'.format(model_type), 'loss_type={}'.format(loss_type),
#         'augmentation={}'.format(augmentation), 'batch_size={}'.format(batch_size),
#         checkpoint_name)

#     predictions_dir = os.path.join(workspace, 'predictions', pre_dir,
#     '{}'.format(filename), 'holdout_fold={}'.format(holdout_fold),
#     'model_type={}'.format(model_type), 'loss_type={}'.format(loss_type),
#     'augmentation={}'.format(augmentation), 'batch_size={}'.format(batch_size))

#     tmp_submission_path = os.path.join(workspace, '_tmp_submission', pre_dir,
#     '{}'.format(filename), 'holdout_fold={}'.format(holdout_fold),
#     'model_type={}'.format(model_type), 'loss_type={}'.format(loss_type),
#     'augmentation={}'.format(augmentation), 'batch_size={}'.format(batch_size),
#     tmp_submission_name)

#     reference_csv_path = os.path.join(dataset_dir, 'metadata', 'groundtruth_strong_label_{}_set.csv'.format(data_type))

#     create_folder(predictions_dir)
#     create_folder(os.path.dirname(tmp_submission_path))

#     # Load model
#     assert model_type, 'Please specify model_type!'
#     Model = eval(model_type)
#     model = Model(sample_rate, window_size, hop_size, mel_bins, fmin, fmax,
#         classes_num, feature_type)

#     checkpoint = torch.load(checkpoint_path)
#     model.load_state_dict(checkpoint['model'])

#     count_parameters(model)

#     # Parallel
#     print('GPU number: {}'.format(torch.cuda.device_count()))
#     model = torch.nn.DataParallel(model)

#     if 'cuda' in device:
#         model.to(device)

#     if sed_thresholds:
#         sed_thresholds_path = os.path.join(workspace, 'opt_thresholds', pre_dir,
#             '{}'.format(filename), 'holdout_fold={}'.format(holdout_fold),
#             'model_type={}'.format(model_type), 'loss_type={}'.format(loss_type),
#             'augmentation={}'.format(augmentation), 'batch_size={}'.format(batch_size),
#             'best_{}_{}.sed.valid.pkl'.format(feature_type, quality))
#         sed_params_dict = pickle.load(open(sed_thresholds_path, 'rb'))
#     else:
#         sed_params_dict = {
#             'audio_tagging_threshold': 0.099,
#             'sed_high_threshold': 0.5,
#             'sed_low_threshold': 0.2,
#             'n_smooth': 10,
#             'n_salt': 10}

# #    param_combinations = [[0.5, 5], [0.5,6], [0.5,7], [0.5, 8], [1,5], [1,6], [1,7], [1, 8], [10,10]]
#     param_combinations = [[10,10]]

#     if audio_8k:
#         audios_dir = os.path.join(dataset_dir, data_type, '8k')
#     else:
#         audios_dir = os.path.join(dataset_dir, data_type)
#     gt_csv = pd.read_csv(reference_csv_path, header=None)
#     audio_files = gt_csv[0].unique()
#     audio_files = ['{}/{}'.format(audios_dir, x) for x in audio_files]
#     full_audio_files = glob('{}/*.wav'.format(audios_dir))

#     #audio_files = sorted(glob('{}/*.wav'.format(audios_dir)))
#     audios_num = len(audio_files)
#     print('NUMBER OF AUDIO FILES:', audios_num)
#     audio_samples = sample_rate * 10 #sample_duration
#     for param in param_combinations:
#         overlap_value = param[0]
#         sample_duration = param[1]
#         predict_length = 0
#         full_predict_event_list = []
#         start_time = time.time()
#         for n in range(audios_num):
#             audio_name = audio_files[n]
#             #print('Predicting on {}'.format(audio_name))
#             if audio_8k:
#                for filename in full_audio_files:
#                    if audio_name.split('/')[-1].split('.wav')[0] in filename:
#                        full_audio_name = filename
#                        break
#             else:
#                 full_audio_name = audio_name
#             audio_duration = librosa.get_duration(filename=full_audio_name)
#             predict_length += audio_duration

#             num_segment = 1
#             index = 0
#             start = 0
#             end = 0
#             merged = None
# #            try:
# #                (audio_full, fs) = librosa.core.load(full_audio_name, sr=sample_rate, mono=True)
# #            except ValueError:
# #                print(full_audio_name)

#             audio_full, timestamps, intervals = trim_silent(audio_name, sample_rate, window_size, hop_size, 20)
#             print(timestamps)
# #            print(audio_full.shape)
#             audio_full = pad_truncate_sequence(audio_full, audio_samples)
# #            audio_duration = librosa.get_duration(np.array(audio_full))
# #            print('Total audio duration: {} s'.format(audio_duration))
#             while end <= audio_duration:
#                 start_index = int(start * sample_rate)
#                 end_index = int((sample_duration * sample_rate) + start_index)
#                 audio = audio_full[start_index:end_index]
#                 audio = torch.Tensor(audio)
#                 audio = torch.reshape(audio, (1, audio.size()[0])).to(device)

#                 output_dict = {}
#                 merged_output_dict = {}
#                 with torch.no_grad():
#                     model.eval()
#                     batch_output = model(audio)

#                 append_to_dict(output_dict, 'audio_name', audio_name)
#                 append_to_dict(output_dict, 'clipwise_output',
#                     batch_output['clipwise_output'].data.cpu().numpy())

#                 if 'framewise_output' in batch_output.keys():
#                     append_to_dict(output_dict, 'framewise_output',
#                         batch_output['framewise_output'].data.cpu().numpy())

#                 curr_clipwise = output_dict['clipwise_output']
#                 curr_preds = output_dict['framewise_output']
#                 if num_segment == 2:
#                     merged = merge(prev_preds, curr_preds, sample_duration, num_segment, overlap_value)
#                 elif num_segment > 2:
#                     merged = merge(merged, curr_preds, sample_duration, num_segment, overlap_value)
#                 else:
#                     merged = curr_preds
# #                predict_event_list = frame_prediction_to_event_prediction(output_dict,
# #                sed_params_dict)
#                 prev_clipwise = output_dict['clipwise_output']
#                 prev_preds = output_dict['framewise_output']

#                 start += overlap_value
#                 end = start + sample_duration
#             merged = avg_merge(merged, sample_duration, overlap_value)
#             #np.set_printoptions(threshold=sys.maxsize)
#             predict_event_list = frame_prediction_to_event_prediction_v2(merged, audio_name.split('/')[-1], sed_params_dict, frames_per_second)
#             print(predict_event_list)
            
#             for k, result in enumerate(predict_event_list):
#                 for non_silent_ts in timestamps:
#                     if result['onset'] <= non_silent_ts[0]:
#                         add_back = non_silent_ts[0]
#                         break
#                 predict_event_list[k]['onset'] += add_back
#                 predict_event_list[k]['offset'] += add_back
                
#             print(predict_event_list)
            
#             full_predict_event_list.extend(predict_event_list)

#         #print(full_predict_event_list)
#         end_time = time.time()
#         time_taken = end_time - start_time
#         print('Processing time for {}: {} s'.format(param, time_taken))
#         print('Total audio duration: {} s'.format(predict_length))

#         # Write predicted events to submission file
#         write_submission(full_predict_event_list, tmp_submission_path)

#         # SED with official tool
#         results = official_evaluate(reference_csv_path, tmp_submission_path)

#         sed_precision = get_metric(results, 'precision')
#         sed_recall = get_metric(results, 'recall')
#         sed_f1 = get_metric(results, 'f1')
#         sed_er = get_metric(results, 'er')

#         print('Micro precision: {:.3f}'.format(sed_precision))
#         print('Micro recall: {:.3f}'.format(sed_recall))
#         print('Micro F1: {:.3f}'.format(sed_f1))
#         print('Micro ER: {:.3f} \n'.format(sed_er))

def inference_prob_trim_silent(self):
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
   """

   # Arugments & parameters
   dataset_dir = args.dataset_dir
   workspace = args.workspace
   holdout_fold = args.holdout_fold
   model_type = args.model_type
   loss_type = args.loss_type
   augmentation = args.augmentation
   batch_size = args.batch_size
   feature_type = args.feature_type
   device = 'cuda' if (args.cuda and torch.cuda.is_available()) else 'cpu'
   filename = args.filename
   sample_duration = args.sample_duration
   sed_thresholds = args.sed_thresholds
   audio_8k = args.audio_8k
   audio_16k = args.audio_16k
   data_type = args.data_type
   fsd50k = args.fsd50k

   #data_type = 'testing'

   num_workers = 8

   # Paths
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
   frames_per_second = sample_rate // hop_size

   checkpoint_name = 'best_{}_{}.pth'.format(feature_type, quality)
   tmp_submission_name = '_overlap_submission_{}.csv'.format(quality)
   test_hdf5_path = os.path.join(workspace, 'hdf5s', '{}_{}_{}.h5'.format(data_type, feature_type, quality))

   if fsd50k:
       pre_dir = 'fsd50k'
   else:
       pre_dir = ''

   checkpoint_path = os.path.join(workspace, 'checkpoints', pre_dir,
       '{}'.format(filename), 'holdout_fold={}'.format(holdout_fold),
       'model_type={}'.format(model_type), 'loss_type={}'.format(loss_type),
       'augmentation={}'.format(augmentation), 'batch_size={}'.format(batch_size),
       checkpoint_name)

   predictions_dir = os.path.join(workspace, 'predictions', pre_dir,
   '{}'.format(filename), 'holdout_fold={}'.format(holdout_fold),
   'model_type={}'.format(model_type), 'loss_type={}'.format(loss_type),
   'augmentation={}'.format(augmentation), 'batch_size={}'.format(batch_size))

   tmp_submission_path = os.path.join(workspace, '_tmp_submission', pre_dir,
   '{}'.format(filename), 'holdout_fold={}'.format(holdout_fold),
   'model_type={}'.format(model_type), 'loss_type={}'.format(loss_type),
   'augmentation={}'.format(augmentation), 'batch_size={}'.format(batch_size),
   tmp_submission_name)

   reference_csv_path = os.path.join(dataset_dir, 'metadata', 'groundtruth_strong_label_{}_set.csv'.format(data_type))

   create_folder(predictions_dir)
   create_folder(os.path.dirname(tmp_submission_path))

   # Load model
   assert model_type, 'Please specify model_type!'
   Model = eval(model_type)
   model = Model(sample_rate, window_size, hop_size, mel_bins, fmin, fmax,
       classes_num, feature_type)

   checkpoint = torch.load(checkpoint_path)
   model.load_state_dict(checkpoint['model'])

   count_parameters(model)

   # Parallel
   print('GPU number: {}'.format(torch.cuda.device_count()))
   model = torch.nn.DataParallel(model)

   if 'cuda' in device:
       model.to(device)

   if sed_thresholds:
       sed_thresholds_path = os.path.join(workspace, 'opt_thresholds', pre_dir,
           '{}'.format(filename), 'holdout_fold={}'.format(holdout_fold),
           'model_type={}'.format(model_type), 'loss_type={}'.format(loss_type),
           'augmentation={}'.format(augmentation), 'batch_size={}'.format(batch_size),
           'best_{}_{}.sed.valid.pkl'.format(feature_type, quality))
       sed_params_dict = pickle.load(open(sed_thresholds_path, 'rb'))
   else:
       sed_params_dict = {
           'audio_tagging_threshold': 0.099,
           'sed_high_threshold': 0.5,
           'sed_low_threshold': 0.2,
           'n_smooth': 10,
           'n_salt': 10}

   db_limit = [110, 120]#10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

   if audio_8k:
       audios_dir = os.path.join(dataset_dir, data_type, '8k')
   else:
       audios_dir = os.path.join(dataset_dir, data_type)
   gt_csv = pd.read_csv(reference_csv_path, header=None)
   audio_files = gt_csv[0].unique()
   audio_files = ['{}/{}'.format(audios_dir, x) for x in audio_files]
   full_audio_files = glob('{}/*.wav'.format(audios_dir))

   #audio_files = sorted(glob('{}/*.wav'.format(audios_dir)))
   audios_num = len(audio_files)
   print('NUMBER OF AUDIO FILES:', audios_num)
   audio_samples = sample_rate * 10 #sample_duration
   for param in db_limit:
       predict_length = 0
       full_predict_event_list = []
       start_time = time.time()
       for n in range(audios_num):
           audio_name = audio_files[n]
           #print('Predicting on {}'.format(audio_name))
           if audio_8k:
              for filename in full_audio_files:
                  if audio_name.split('/')[-1].split('.wav')[0] in filename:
                      full_audio_name = filename
                      break
           else:
               full_audio_name = audio_name
           audio_duration = librosa.get_duration(filename=full_audio_name)
           predict_length += audio_duration

           num_segment = 1
           index = 0
           start = 0
           end = 0
           merged = None
           try:
               (audio_full, fs) = librosa.core.load(full_audio_name, sr=sample_rate, mono=True)
           except ValueError:
               print(full_audio_name)
           non_silent = librosa.effects.split(y=audio_full, top_db=param) #frame_length=window_size, hop_length=hop_size, top_db=param)
           #print(non_silent)
           #print(non_silent.shape)
#            audio_full = pad_truncate_sequence(audio_full, audio_samples)
           for i, audio_chunk in enumerate(non_silent):
               start_index = audio_chunk[0]
               end_index = audio_chunk[1]
#            while end <= audio_duration:
#                start_index = int(start * sample_rate)
#                end_index = int((sample_duration * sample_rate) + start_index)
               audio = audio_full[start_index:end_index]
               audio = pad_truncate_sequence(audio, audio_samples)

               audio = torch.Tensor(audio)
               audio = torch.reshape(audio, (1, audio.size()[0])).to(device)

               output_dict = {}
               merged_output_dict = {}
               with torch.no_grad():
                   model.eval()
                   batch_output = model(audio)

               append_to_dict(output_dict, 'audio_name', audio_name)
               append_to_dict(output_dict, 'clipwise_output',
                   batch_output['clipwise_output'].data.cpu().numpy())

               if 'framewise_output' in batch_output.keys():
                   append_to_dict(output_dict, 'framewise_output',
                       batch_output['framewise_output'].data.cpu().numpy())

               curr_clipwise = output_dict['clipwise_output']
               curr_preds = output_dict['framewise_output']
               if start_index > 0 and i == 0:
                   merged = np.concatenate((np.zeros(shape=(curr_preds.shape[0], start_index-1, curr_preds.shape[2])), curr_preds), axis=1)
               elif i >= 1:
                   if prev_end + 1 != start_index:
                       merged = np.concatenate((merged, np.zeros(shape=(curr_preds.shape[0], start_index-prev_end, curr_preds.shape[2]))), axis=1)
                   merged = np.concatenate((merged, curr_preds), axis=1)
               else:
                   merged = curr_preds
#                if num_segment == 2:
#                    merged = merge(prev_preds, curr_preds, sample_duration, num_segment, overlap_value)
#                elif num_segment > 2:
#                    merged = merge(merged, curr_preds, sample_duration, num_segment, overlap_value)
#                else:
#                    merged = curr_preds
   #            predict_event_list = frame_prediction_to_event_prediction(output_dict,
   #            sed_params_dict)
               prev_clipwise = output_dict['clipwise_output']
               prev_preds = output_dict['framewise_output']
               prev_end = end_index

               #start += 1
           predict_event_list = frame_prediction_to_event_prediction_v2(merged,
               audio_name.split('/')[-1], sed_params_dict, frames_per_second)
#            merged = avg_merge(merged, sample_duration, overlap_value)
           #np.set_printoptions(threshold=sys.maxsize)
#            predict_event_list = frame_prediction_to_event_prediction_v2(merged, audio_name.split('/')[-1], sed_params_dict, frames_per_second)
           full_predict_event_list.extend(predict_event_list)

       #print(full_predict_event_list)
       end_time = time.time()
       time_taken = end_time - start_time
       print('Processing time for {}: {} s'.format(param, time_taken))
       print('Total audio duration: {} s'.format(predict_length))

       # Write predicted events to submission file
       write_submission(full_predict_event_list, tmp_submission_path)

       # SED with official tool
       results = official_evaluate(reference_csv_path, tmp_submission_path)

       sed_precision = get_metric(results, 'precision')
       sed_recall = get_metric(results, 'recall')
       sed_f1 = get_metric(results, 'f1')
       sed_er = get_metric(results, 'er')

       print('Micro precision: {:.3f}'.format(sed_precision))
       print('Micro recall: {:.3f}'.format(sed_recall))
       print('Micro F1: {:.3f}'.format(sed_f1))
       print('Micro ER: {:.3f} \n'.format(sed_er))


#def inference_prob_vote(self):
#    """Inference test and evaluate data and dump predicted probabilites to
#    pickle files.
#
#    Args:
#      dataset_dir: str
#      workspace: str
#      holdout_fold: '1'
#      model_type: str, e.g., 'Cnn_9layers_Gru_FrameAtt'
#      loss_type: str, e.g., 'clip_bce'
#      augmentation: str, e.g., 'mixup'
#      batch_size: int
#      device: 'cuda' | 'cpu'
#    """
#
#    # Arugments & parameters
#    dataset_dir = args.dataset_dir
#    workspace = args.workspace
#    holdout_fold = args.holdout_fold
#    model_type = args.model_type
#    loss_type = args.loss_type
#    augmentation = args.augmentation
#    batch_size = args.batch_size
#    feature_type = args.feature_type
#    device = 'cuda' if (args.cuda and torch.cuda.is_available()) else 'cpu'
#    filename = args.filename
#    sample_duration = args.sample_duration
#    sed_thresholds = args.sed_thresholds
#    data_type = 'testing'
#
#    num_workers = 8
#
#    # Paths
#    if feature_type == 'logmel':
#        test_hdf5_path = os.path.join(workspace, 'hdf5s', 'testing.h5')
#    elif feature_type == 'gamma':
#        test_hdf5_path = os.path.join(workspace, 'hdf5s', 'testing_{}.h5'.format(feature_type))
#
#    test_reference_csv_path = os.path.join(dataset_dir, 'metadata',
#        'groundtruth_strong_label_testing_set.csv')
#
#    checkpoint_path = os.path.join(workspace, 'checkpoints',
#        '{}'.format(filename), 'holdout_fold={}'.format(holdout_fold),
#        'model_type={}'.format(model_type), 'loss_type={}'.format(loss_type),
#        'augmentation={}'.format(augmentation), 'batch_size={}'.format(batch_size),
#        'best.pth'.format(feature_type))
#
#    predictions_dir = os.path.join(workspace, 'predictions',
#        '{}'.format(filename), 'holdout_fold={}'.format(holdout_fold),
#        'model_type={}'.format(model_type), 'loss_type={}'.format(loss_type),
#        'augmentation={}'.format(augmentation), 'batch_size={}'.format(batch_size))
#    create_folder(predictions_dir)
#
#    tmp_submission_path = os.path.join(workspace, '_tmp_submission',
#        '{}'.format(filename), 'holdout_fold={}'.format(holdout_fold),
#        'model_type={}'.format(model_type), 'loss_type={}'.format(loss_type),
#        'augmentation={}'.format(augmentation), 'batch_size={}'.format(batch_size),
#        '_overlap_submission.csv')
#    create_folder(os.path.dirname(tmp_submission_path))
#
#    reference_csv_path = os.path.join(dataset_dir, 'metadata', 'groundtruth_strong_label_testing_set.csv')
#
#    # Load model
#    assert model_type, 'Please specify model_type!'
#    Model = eval(model_type)
#    model = Model(sample_rate, window_size, hop_size, mel_bins, fmin, fmax,
#        classes_num, feature_type)
#
#    checkpoint = torch.load(checkpoint_path)
#    model.load_state_dict(checkpoint['model'])
#
#    # Parallel
#    print('GPU number: {}'.format(torch.cuda.device_count()))
#    model = torch.nn.DataParallel(model)
#
#    if 'cuda' in device:
#        model.to(device)
#
#    if sed_thresholds:
#        sed_thresholds_path = os.path.join(workspace, 'opt_thresholds',
#            '{}'.format(filename), 'holdout_fold={}'.format(holdout_fold),
#            'model_type={}'.format(model_type), 'loss_type={}'.format(loss_type),
#            'augmentation={}'.format(augmentation), 'batch_size={}'.format(batch_size),
#            'best.sed.test.pkl')
#        sed_params_dict = pickle.load(open(sed_thresholds_path, 'rb'))
#    else:
#        sed_params_dict = {
#            'audio_tagging_threshold': 0.099,
#            'sed_high_threshold': 0.5,
#            'sed_low_threshold': 0.2,
#            'n_smooth': 10,
#            'n_salt': 10}
#
#    param_combinations = [[0.5,6], [0.5,7], [1,5], [1,6], [1,7]]
#    #param_combinations = [[1,5]]
#
#    audios_dir = os.path.join(dataset_dir, data_type)
#    gt_csv = pd.read_csv(reference_csv_path, header=None)
#    audio_files = gt_csv[0].unique()
#    audio_files = ['{}/{}'.format(audios_dir, x) for x in audio_files]
#
#    #audio_files = sorted(glob('{}/*.wav'.format(audios_dir)))
#    audios_num = len(audio_files)
#    print('NUMBER OF AUDIO FILES:', audios_num)
#    audio_samples = sample_rate * 10 #sample_duration
#    for param in param_combinations:
#        overlap_value = param[0]
#        sample_duration = param[1]
#        predict_length = 0
#        full_predict_event_list = []
#        start_time = time.time()
#        for n in range(audios_num):
#            audio_name = audio_files[n]
#            audio_duration = librosa.get_duration(filename=audio_name)
#            predict_length += audio_duration
#            num_segment = 1
#            index = 0
#            start = 0
#            end = 0
#            merged = None
#            # Load audio sample
#            (audio_full, fs) = librosa.core.load(audio_name, sr=sample_rate, mono=True)
#            audio_full = pad_truncate_sequence(audio_full, audio_samples)
#            while end <= audio_duration:
#                start_index = int(start * sample_rate)
#                end_index = int((sample_duration * sample_rate) + start_index)
#                audio = audio_full[start_index:end_index]
#                audio = torch.Tensor(audio)
#                audio = torch.reshape(audio, (1, audio.size()[0])).to(device)
#
#                output_dict = {}
#                merged_output_dict = {}
#                with torch.no_grad():
#                    model.eval()
#                    batch_output = model(audio)
#
#                append_to_dict(output_dict, 'audio_name', audio_name)
#                append_to_dict(output_dict, 'clipwise_output',
#                    batch_output['clipwise_output'].data.cpu().numpy())
#
#                if 'framewise_output' in batch_output.keys():
#                    append_to_dict(output_dict, 'framewise_output',
#                        batch_output['framewise_output'].data.cpu().numpy())
#
#                curr_clipwise = output_dict['clipwise_output']
#                curr_preds = output_dict['framewise_output']
#                curr_preds = binarize_pred(curr_preds, sed_params_dict['sed_high_threshold'])
#                if num_segment == 2:
#                    merged = merge(prev_preds, curr_preds, sample_duration, num_segment, overlap_value)
#                elif num_segment > 2:
#                    merged = merge(merged, curr_preds, sample_duration, num_segment, overlap_value)
#                else:
#                    merged = curr_preds
#                prev_clipwise = output_dict['clipwise_output']
#                prev_preds = output_dict['framewise_output']
#                prev_preds = binarize_pred(prev_preds, sed_params_dict['sed_high_threshold'])
#
#                #start += 1
#                start += overlap_value
#                end = start + sample_duration
#                num_segment += 1
#            #merged = avg_merge(merged, sample_duration, overlap_value)
#            #np.set_printoptions(threshold=sys.maxsize)
#            predict_event_list = frame_binary_prediction_to_event_prediction(merged, overlap_value, sample_duration, audio_name.split('/')[-1], sed_params_dict)
#            full_predict_event_list.extend(predict_event_list)
#
#        print(full_predict_event_list)
#        end_time = time.time()
#        time_taken = end_time - start_time
#        print('Processing time for {}: {} s'.format(param, time_taken))
#        print('Total audio duration: {} s'.format(predict_length))
#
#        # Write predicted events to submission file
#        write_submission(full_predict_event_list, tmp_submission_path)
#
#        # SED with official tool
#        results = official_evaluate(reference_csv_path, tmp_submission_path)
#
#        sed_precision = get_metric(results, 'precision')
#        sed_recall = get_metric(results, 'recall')
#        sed_f1 = get_metric(results, 'f1')
#        sed_er = get_metric(results, 'er')
#
#        print('Micro precision: {:.3f}'.format(sed_precision))
#        print('Micro recall: {:.3f}'.format(sed_recall))
#        print('Micro F1: {:.3f}'.format(sed_f1))
#        print('Micro ER: {:.3f} \n'.format(sed_er))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Example of parser. ')
    subparsers = parser.add_subparsers(dest='mode')

    # Train
    parser_train = subparsers.add_parser('train')
    parser_train.add_argument('--dataset_dir', type=str, required=True, help='Directory of dataset.')
    parser_train.add_argument('--workspace', type=str, required=True, help='Directory of your workspace.')
    parser_train.add_argument('--holdout_fold', type=str, choices=['1'], required=True)
    parser_train.add_argument('--model_type', type=str, required=True)
    parser_train.add_argument('--loss_type', type=str, required=True)
    parser_train.add_argument('--augmentation', type=str, choices=['none', 'spec_augment', 'timeshift', 'mixup', 'timeshift_mixup', 'specaugment_timeshift_mixup', 'specaugment_mixup', 'specaugment_timeshift'], required=True)
    parser_train.add_argument('--learning_rate', type=float, required=True)
    parser_train.add_argument('--batch_size', type=int, required=True)
    parser_train.add_argument('--resume_iteration', type=int)
    parser_train.add_argument('--stop_iteration', type=int, required=True)
    parser_train.add_argument('--feature_type', type=str, required=True)
    parser_train.add_argument('--cuda', action='store_true', default=False)
    parser_train.add_argument('--audio_8k', action='store_true', default=False)
    parser_train.add_argument('--audio_16k', action='store_true', default=False)
    parser_train.add_argument('--vggish', action='store_true', default=False)
    parser_train.add_argument('--fsd50k', action='store_true', default=False)
    parser_train.add_argument('--mini_data', action='store_true', default=False)

    # Inference
    parser_inference_prob = subparsers.add_parser('inference_prob')
    parser_inference_prob.add_argument('--dataset_dir', type=str, required=True, help='Directory of dataset.')
    parser_inference_prob.add_argument('--workspace', type=str, required=True, help='Directory of your workspace.')
    parser_inference_prob.add_argument('--holdout_fold', type=str, choices=['1'], required=True)
    parser_inference_prob.add_argument('--model_type', type=str, required=True)
    parser_inference_prob.add_argument('--loss_type', type=str, required=True)
    parser_inference_prob.add_argument('--augmentation', type=str, choices=['none', 'spec_augment', 'timeshift', 'mixup', 'timeshift_mixup', 'specaugment_timeshift_mixup', 'specaugment_mixup', 'specaugment_timeshift'], required=True)
    parser_inference_prob.add_argument('--feature_type', type=str, required=True)
    parser_inference_prob.add_argument('--batch_size', type=int, required=True)
    parser_inference_prob.add_argument('--cuda', action='store_true', default=False)
    parser_inference_prob.add_argument('--fsd50k', action='store_true', default=False)
    parser_inference_prob.add_argument('--audio_8k', action='store_true', default=False)
    parser_inference_prob.add_argument('--audio_16k', action='store_true', default=False)
    parser_inference_prob.add_argument('--vggish', action='store_true', default=False)
    parser_inference_prob.add_argument('--sed_thresholds', action='store_true', default=False)
    
    # Inference (overlap + avg)
    parser_inference_prob_overlap = subparsers.add_parser('inference_prob_overlap')
    parser_inference_prob_overlap.add_argument('--dataset_dir', type=str, required=True, help='Directory of dataset.')
    parser_inference_prob_overlap.add_argument('--workspace', type=str, required=True, help='Directory of your workspace.')
    parser_inference_prob_overlap.add_argument('--holdout_fold', type=str, choices=['1'], required=True)
    parser_inference_prob_overlap.add_argument('--model_type', type=str, required=True)
    parser_inference_prob_overlap.add_argument('--loss_type', type=str, required=True)
    parser_inference_prob_overlap.add_argument('--augmentation', type=str, choices=['none', 'spec_augment', 'timeshift', 'mixup', 'timeshift_mixup', 'specaugment_timeshift_mixup', 'specaugment_mixup', 'specaugment_timeshift'], required=True)
    parser_inference_prob_overlap.add_argument('--feature_type', type=str, required=True)
    parser_inference_prob_overlap.add_argument('--batch_size', type=int, required=True)
    parser_inference_prob_overlap.add_argument('--sample_duration', type=int, default=2)
    parser_inference_prob_overlap.add_argument('--cuda', action='store_true', default=False)
    parser_inference_prob_overlap.add_argument('--sed_thresholds', action='store_true', default=False)
    parser_inference_prob_overlap.add_argument('--audio_8k', action='store_true', default=False)
    parser_inference_prob_overlap.add_argument('--audio_16k', action='store_true', default=False)
    parser_inference_prob_overlap.add_argument('--data_type', type=str, required=True)
    parser_inference_prob_overlap.add_argument('--fsd50k', action='store_true', default=False)
    
    # Inference (overlap + vote)
    parser_inference_prob_vote = subparsers.add_parser('inference_prob_vote')
    parser_inference_prob_vote.add_argument('--dataset_dir', type=str, required=True, help='Directory of dataset.')
    parser_inference_prob_vote.add_argument('--workspace', type=str, required=True, help='Directory of your workspace.')
    parser_inference_prob_vote.add_argument('--holdout_fold', type=str, choices=['1'], required=True)
    parser_inference_prob_vote.add_argument('--model_type', type=str, required=True)
    parser_inference_prob_vote.add_argument('--loss_type', type=str, required=True)
    parser_inference_prob_vote.add_argument('--augmentation', type=str, choices=['none', 'spec_augment', 'timeshift', 'mixup', 'timeshift_mixup', 'specaugment_timeshift_mixup', 'specaugment_mixup', 'specaugment_timeshift'], required=True)
    parser_inference_prob_vote.add_argument('--feature_type', type=str, required=True)
    parser_inference_prob_vote.add_argument('--batch_size', type=int, required=True)
    parser_inference_prob_vote.add_argument('--sample_duration', type=int, default=2)
    parser_inference_prob_vote.add_argument('--cuda', action='store_true', default=False)
    parser_inference_prob_vote.add_argument('--sed_thresholds', action='store_true', default=False)
    parser_inference_prob_vote.add_argument('--audio_8k', action='store_true', default=False)
    parser_inference_prob_vote.add_argument('--audio_16k', action='store_true', default=False)
    parser_inference_prob_vote.add_argument('--data_type', type=str, required=True)
    parser_inference_prob_vote.add_argument('--fsd50k', action='store_true', default=False)
    
    # Inference (trim silent)
    parser_inference_prob_trim_silent = subparsers.add_parser('inference_prob_trim_silent')
    parser_inference_prob_trim_silent.add_argument('--dataset_dir', type=str, required=True, help='Directory of dataset.')
    parser_inference_prob_trim_silent.add_argument('--workspace', type=str, required=True, help='Directory of your workspace.')
    parser_inference_prob_trim_silent.add_argument('--holdout_fold', type=str, choices=['1'], required=True)
    parser_inference_prob_trim_silent.add_argument('--model_type', type=str, required=True)
    parser_inference_prob_trim_silent.add_argument('--loss_type', type=str, required=True)
    parser_inference_prob_trim_silent.add_argument('--augmentation', type=str, choices=['none', 'spec_augment', 'timeshift', 'mixup', 'timeshift_mixup', 'specaugment_timeshift_mixup', 'specaugment_mixup', 'specaugment_timeshift'], required=True)
    parser_inference_prob_trim_silent.add_argument('--feature_type', type=str, required=True)
    parser_inference_prob_trim_silent.add_argument('--batch_size', type=int, required=True)
    parser_inference_prob_trim_silent.add_argument('--sample_duration', type=int, default=2)
    parser_inference_prob_trim_silent.add_argument('--cuda', action='store_true', default=False)
    parser_inference_prob_trim_silent.add_argument('--sed_thresholds', action='store_true', default=False)
    parser_inference_prob_trim_silent.add_argument('--audio_8k', action='store_true', default=False)
    parser_inference_prob_trim_silent.add_argument('--audio_16k', action='store_true', default=False)
    parser_inference_prob_trim_silent.add_argument('--data_type', type=str, required=True)
    parser_inference_prob_trim_silent.add_argument('--fsd50k', action='store_true', default=False)
    parser_inference_prob_trim_silent.add_argument('--filename', type=str, default=get_filename(__file__))
    
    # Parse arguments
    args = parser.parse_args()
    #args.filename = get_filename(__file__)

    if args.mode == 'train':
        train(args)

    elif args.mode == 'inference_prob':
        inference_prob(args)
        
    elif args.mode == 'inference_prob_overlap':
        inference_prob_overlap(args)
    
    elif args.mode == 'inference_prob_vote':
        inference_prob_vote(args)
        
    elif args.mode == 'inference_prob_trim_silent':
        inference_prob_trim_silent(args)
    
    else:
        raise Exception('Error argument!')
