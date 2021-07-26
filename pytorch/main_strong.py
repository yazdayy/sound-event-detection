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
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
 
from evaluate import Evaluator
from config import (sample_rate, classes_num, mel_bins, fmin, fmax, 
    window_size, hop_size, window, pad_mode, center, device, ref, amin, top_db)
from losses import get_loss_func
from pytorch_utils import move_data_to_device, do_mixup, do_mixup_timeshift
from utilities import (create_folder, frame_prediction_to_event_prediction_v2, get_filename, create_logging, official_evaluate, frame_binary_prediction_to_event_prediction,
    StatisticsContainer, pad_truncate_sequence, write_submission, Mixup)
from calculate_metrics import get_metric
from predict_v2 import merge, avg_merge, append_to_dict
from data_generator import (DCASE2017Task4Dataset, TrainSampler, TestSampler, 
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

    weak_loss_func = get_loss_func(loss_type)
    if loss_type == 'clip_bce':
        strong_loss_func = get_loss_func('frame_bce')
    elif loss_type == 'clip_bce_logits':
        strong_loss_func = get_loss_func('frame_bce_logits')
    
    num_workers = 8
    
    # Paths
    if mini_data:
        prefix = 'minidata_'
    else:
        prefix = ''
    
    if feature_type == 'logmel':
        weak_train_hdf5_path = os.path.join(workspace, 'hdf5s',
            '{}weak_training.h5'.format(prefix))
        
        strong_train_hdf5_path = os.path.join(workspace, 'hdf5s',
        '{}strong_training.h5'.format(prefix))

        test_hdf5_path = os.path.join(workspace, 'hdf5s',
            '{}testing.h5'.format(prefix))
    elif feature_type == 'gamma':
        weak_train_hdf5_path = os.path.join(workspace, 'hdf5s',
            '{}weak_training_{}.h5'.format(prefix, feature_type))
        
        strong_train_hdf5_path = os.path.join(workspace, 'hdf5s',
        '{}strong_training_{}.h5'.format(prefix, feature_type))

        test_hdf5_path = os.path.join(workspace, 'hdf5s',
            '{}testing_{}.h5'.format(prefix, feature_type))

    test_reference_csv_path = os.path.join(dataset_dir, 'metadata', 
        'groundtruth_strong_label_testing_set.csv')

    checkpoints_dir = os.path.join(workspace, 'checkpoints', 
        '{}{}'.format(prefix, filename), 'holdout_fold={}'.format(holdout_fold), 
        'model_type={}'.format(model_type), 'loss_type={}'.format(loss_type), 
        'augmentation={}'.format(augmentation), 'batch_size={}'.format(batch_size))
    create_folder(checkpoints_dir)

    tmp_submission_path = os.path.join(workspace, '_tmp_submission', 
        '{}{}'.format(prefix, filename), 'holdout_fold={}'.format(holdout_fold), 
        'model_type={}'.format(model_type), 'loss_type={}'.format(loss_type), 
        'augmentation={}'.format(augmentation), 'batch_size={}'.format(batch_size),
        '_submission.csv')
    create_folder(os.path.dirname(tmp_submission_path))

    statistics_path = os.path.join(workspace, 'statistics', 
        '{}{}'.format(prefix, filename), 'holdout_fold={}'.format(holdout_fold), 
        'model_type={}'.format(model_type), 'loss_type={}'.format(loss_type), 
        'augmentation={}'.format(augmentation), 'batch_size={}'.format(batch_size), 
        'statistics.pickle')
    create_folder(os.path.dirname(statistics_path))

    logs_dir = os.path.join(workspace, 'logs', '{}{}'.format(prefix, filename), 
        'holdout_fold={}'.format(holdout_fold), 'model_type={}'.format(model_type), 
        'loss_type={}'.format(loss_type), 'augmentation={}'.format(augmentation), 
        'batch_size={}'.format(batch_size))
    create_logging(logs_dir, 'w')
    logging.info(args)

    if 'cuda' in device:
        logging.info('Using GPU.')
    else:
        logging.info('Using CPU. Set --cuda flag to use GPU.')
    
    # Model
    assert model_type, 'Please specify model_type!'
    Model = eval(model_type)
    model = Model(sample_rate, window_size, hop_size, mel_bins, fmin, fmax, 
        classes_num, feature_type)

    if resume_iteration:
        resume_checkpoint_path = os.path.join(checkpoints_dir, 
            'best_{}.pth'.format(feature_type))
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
    dataset = DCASE2017Task4Dataset()
    
    
    # Sampler
#    weak_batch_size = batch_size * (3/4)
#    strong_batch_size = batch_size * (1/4)
    weak_train_sampler = TrainSampler(
        hdf5_path=weak_train_hdf5_path,
        batch_size=(batch_size * 3) * 2 if 'mixup' in augmentation else batch_size)
    
    strong_train_sampler = TrainSampler(
        hdf5_path=strong_train_hdf5_path,
        batch_size=batch_size * 2 if 'mixup' in augmentation else batch_size)
    
    test_sampler = TestSampler(hdf5_path=test_hdf5_path, batch_size=batch_size)

    # Data loader
    weak_train_loader = torch.utils.data.DataLoader(dataset=dataset,
        batch_sampler=weak_train_sampler, collate_fn=collate_fn,
        num_workers=num_workers, pin_memory=True)
    
    strong_train_loader = torch.utils.data.DataLoader(dataset=dataset,
        batch_sampler=strong_train_sampler, collate_fn=collate_fn,
        num_workers=num_workers, pin_memory=True)
    
    test_loader = torch.utils.data.DataLoader(dataset=dataset, 
        batch_sampler=test_sampler, collate_fn=collate_fn, 
        num_workers=num_workers, pin_memory=True)
    
    weak_iter = cycle_iteration(weak_train_loader)
    strong_iter = cycle_iteration(strong_train_loader)
    
    if 'mixup' in augmentation:
        mixup_augmenter = Mixup(mixup_alpha=1.)
    
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

            for (data_type, data_loader, reference_csv_path) in [
                ('test', test_loader, test_reference_csv_path)]:

                # Calculate tatistics
                (statistics, _) = evaluator.evaluate(
                    data_loader, reference_csv_path, tmp_submission_path)

                logging.info('{} statistics:'.format(data_type))
                logging.info('    Clipwise mAP: {:.3f}'.format(np.mean(statistics['clipwise_ap'])))
                logging.info('    Framewise mAP: {:.3f}'.format(np.nanmean(statistics['framewise_ap'])))
                logging.info('    {}'.format(statistics['sed_metrics']['overall']['error_rate']))

                statistics_container.append(data_type, iteration, statistics)
                
                if data_type == 'test':
                    if np.nanmean(statistics['framewise_ap']) >= best_framewise_map and statistics['sed_metrics']['overall']['error_rate']['error_rate'] < best_error_rate <= best_error_rate:
                    
                        best_framewise_map = np.nanmean(statistics['framewise_ap'])
                        best_error_rate = statistics['sed_metrics']['overall']['error_rate']['error_rate']
#                            error_rates.append(statistics['sed_metrics']['overall']['error_rate']['error_rate'])
#                    framewise_maps.append(np.nanmean(statistics['framewise_ap']))
#                    if np.nanmean(statistics['framewise_ap']) == max(framewise_maps) and statistics['sed_metrics']['overall']['error_rate']['error_rate'] == min(error_rates):
                        best_iteration = iteration
                        
                        checkpoint = {
                            'iteration': iteration,
                            'model': model.module.state_dict(),
                            'optimizer': optimizer.state_dict()}

                        checkpoint_path = os.path.join(
                            checkpoints_dir, 'best_{}.pth'.format(feature_type))
                            
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
        
        if 'mixup' in augmentation:
            weak_batch_data_dict['mixup_lambda'] = mixup_augmenter.get_lambda(
                batch_size=len(weak_batch_data_dict['waveform']))
            strong_batch_data_dict['mixup_lambda'] = mixup_augmenter.get_lambda(
                batch_size=len(strong_batch_data_dict['waveform']))

        # Move data to GPU
        for key in weak_batch_data_dict.keys():
            weak_batch_data_dict[key] = move_data_to_device(weak_batch_data_dict[key], device)
        for key in strong_batch_data_dict.keys():
            strong_batch_data_dict[key] = move_data_to_device(strong_batch_data_dict[key], device)
        
        # Train
        model.train()
        
        if 'mixup' in augmentation:
            if 'timeshift' not in augmentation:
                weak_batch_output_dict = model(weak_batch_data_dict['waveform'], weak_batch_data_dict['mixup_lambda'])
                weak_batch_target_dict = {'target': do_mixup(weak_batch_data_dict['target'], weak_batch_data_dict['mixup_lambda'])}
                strong_batch_output_dict = model(strong_batch_data_dict['waveform'], strong_batch_data_dict['mixup_lambda'])
                strong_batch_target_dict = {'strong_target': do_mixup(strong_batch_data_dict['strong_target'], strong_batch_data_dict['mixup_lambda'])}
            elif 'timeshift' in augmentation:
                weak_batch_output_dict = model(weak_batch_data_dict['waveform'], mixup_lambda=weak_batch_data_dict['mixup_lambda'], timeshift=True)
                weak_batch_target_dict = {'target': do_mixup(weak_batch_data_dict['target'], weak_batch_data_dict['mixup_lambda'])}
                strong_batch_output_dict = model(strong_batch_data_dict['waveform'], strong_batch_data_dict['mixup_lambda'], timeshift=True)
                strong_batch_target_dict = {'strong_target': do_mixup(strong_batch_data_dict['strong_target'], strong_batch_data_dict['mixup_lambda'])}
        else:
            weak_batch_output_dict = model(weak_batch_data_dict['waveform'], None)
            weak_batch_target_dict = {'target': weak_batch_data_dict['target']}
            strong_batch_output_dict = model(strong_batch_data_dict['waveform'], None)
            strong_batch_target_dict = {'strong_target': strong_batch_data_dict['strong_target']}

        # loss
        weak_loss = weak_loss_func(weak_batch_output_dict, weak_batch_target_dict)
        strong_loss = strong_loss_func(strong_batch_output_dict, strong_batch_target_dict)
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
    device = 'cuda' if (args.cuda and torch.cuda.is_available()) else 'cpu'
    filename = args.filename

    num_workers = 8

    # Paths
    if feature_type == 'logmel':
        test_hdf5_path = os.path.join(workspace, 'hdf5s', 'testing.h5')
    elif feature_type == 'gamma':
        test_hdf5_path = os.path.join(workspace, 'hdf5s', 'testing_{}.h5'.format(feature_type))

    test_reference_csv_path = os.path.join(dataset_dir, 'metadata', 
        'groundtruth_strong_label_testing_set.csv')

    checkpoint_path = os.path.join(workspace, 'checkpoints', 
        '{}'.format(filename), 'holdout_fold={}'.format(holdout_fold), 
        'model_type={}'.format(model_type), 'loss_type={}'.format(loss_type), 
        'augmentation={}'.format(augmentation), 'batch_size={}'.format(batch_size),
        'best_{}.pth'.format(feature_type))

    predictions_dir = os.path.join(workspace, 'predictions', 
        '{}'.format(filename), 'holdout_fold={}'.format(holdout_fold), 
        'model_type={}'.format(model_type), 'loss_type={}'.format(loss_type), 
        'augmentation={}'.format(augmentation), 'batch_size={}'.format(batch_size))
    create_folder(predictions_dir)

    tmp_submission_path = os.path.join(workspace, '_tmp_submission', 
        '{}'.format(filename), 'holdout_fold={}'.format(holdout_fold), 
        'model_type={}'.format(model_type), 'loss_type={}'.format(loss_type), 
        'augmentation={}'.format(augmentation), 'batch_size={}'.format(batch_size),
        '_submission.csv')
    create_folder(os.path.dirname(tmp_submission_path))

    # Load model
    assert model_type, 'Please specify model_type!'
    Model = eval(model_type)
    model = Model(sample_rate, window_size, hop_size, mel_bins, fmin, fmax, 
        classes_num, feature_type)

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model'])

    # Parallel
    print('GPU number: {}'.format(torch.cuda.device_count()))
    model = torch.nn.DataParallel(model)

    if 'cuda' in device:
        model.to(device)

    # Dataset
    dataset = DCASE2017Task4Dataset()

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

        print('Inferencing {} data in about 1 min ...'.format(data_type))

        (statistics, output_dict) = evaluator.evaluate(
            data_loader, reference_csv_path, tmp_submission_path)

        prediction_path = os.path.join(predictions_dir, 
            'best_{}.prediction.{}.pkl'.format(feature_type, data_type))

        # write_out_prediction(output_dict, prediction_path)
        pickle.dump(output_dict, open(prediction_path, 'wb'))
        print('Write out to {}'.format(prediction_path))

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
    data_type = 'testing'

    num_workers = 8

    # Paths
    if feature_type == 'logmel':
        test_hdf5_path = os.path.join(workspace, 'hdf5s', 'testing.h5')
    elif feature_type == 'gamma':
        test_hdf5_path = os.path.join(workspace, 'hdf5s', 'testing_{}.h5'.format(feature_type))

    test_reference_csv_path = os.path.join(dataset_dir, 'metadata',
        'groundtruth_strong_label_testing_set.csv')

    checkpoint_path = os.path.join(workspace, 'checkpoints',
        '{}'.format(filename), 'holdout_fold={}'.format(holdout_fold),
        'model_type={}'.format(model_type), 'loss_type={}'.format(loss_type),
        'augmentation={}'.format(augmentation), 'batch_size={}'.format(batch_size),
        'best.pth'.format(feature_type))

    predictions_dir = os.path.join(workspace, 'predictions',
        '{}'.format(filename), 'holdout_fold={}'.format(holdout_fold),
        'model_type={}'.format(model_type), 'loss_type={}'.format(loss_type),
        'augmentation={}'.format(augmentation), 'batch_size={}'.format(batch_size))
    create_folder(predictions_dir)

    tmp_submission_path = os.path.join(workspace, '_tmp_submission',
        '{}'.format(filename), 'holdout_fold={}'.format(holdout_fold),
        'model_type={}'.format(model_type), 'loss_type={}'.format(loss_type),
        'augmentation={}'.format(augmentation), 'batch_size={}'.format(batch_size),
        '_overlap_submission.csv')
    create_folder(os.path.dirname(tmp_submission_path))
    
    reference_csv_path = os.path.join(dataset_dir, 'metadata', 'groundtruth_strong_label_testing_set.csv')

    # Load model
    assert model_type, 'Please specify model_type!'
    Model = eval(model_type)
    model = Model(sample_rate, window_size, hop_size, mel_bins, fmin, fmax,
        classes_num, feature_type)

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model'])

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
            'best.sed.test.pkl')
        sed_params_dict = pickle.load(open(sed_thresholds_path, 'rb'))
    else:
        sed_params_dict = {
            'audio_tagging_threshold': 0.099,
            'sed_high_threshold': 0.5,
            'sed_low_threshold': 0.2,
            'n_smooth': 10,
            'n_salt': 10}
    
    # Testing different segment length and stride combinations
#    overlap_values = np.arange(0.5,2,0.5)
#    segment_lengths = np.arange(2,10,0.5)
#
#    param_combinations = []
#    for value in overlap_values:
#        for length in segment_lengths:
#            param_combinations.append([value, length])

    param_combinations = [[0.5,6], [0.5,7], [1,5], [1,6], [1,7]]
    
    audios_dir = os.path.join(dataset_dir, data_type)
    gt_csv = pd.read_csv(reference_csv_path, header=None)
    audio_files = gt_csv[0].unique()
    audio_files = ['{}/{}'.format(audios_dir, x) for x in audio_files]
    
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
            audio_duration = librosa.get_duration(filename=audio_name)
            predict_length += audio_duration
            #print('Total audio duration: {} s'.format(audio_duration))
            num_segment = 1
            index = 0
            start = 0
            end = 0
            merged = None
            (audio_full, fs) = librosa.core.load(audio_name, sr=sample_rate, mono=True)
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
            predict_event_list = frame_prediction_to_event_prediction_v2(merged, audio_name.split('/')[-1], sed_params_dict)
            full_predict_event_list.extend(predict_event_list)
        
        print(full_predict_event_list)
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
        
        print('Segment length: {} s, Stride: {} s'.format(sample_duration, overlap_value))
        print('Micro precision: {:.3f}'.format(sed_precision))
        print('Micro recall: {:.3f}'.format(sed_recall))
        print('Micro F1: {:.3f}'.format(sed_f1))
        print('Micro ER: {:.3f} \n'.format(sed_er))
        
def binarize_pred(pred, sed_thresholds):
    pred_binary = np.zeros(pred.shape)
    for i in range(len(pred)):
        for j in range(len(pred[i])):
            for k in range(len(pred[i][j])):
                if pred[i][j][k] > sed_thresholds[k]:
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
    data_type = 'testing'

    num_workers = 8

    # Paths
    if feature_type == 'logmel':
        test_hdf5_path = os.path.join(workspace, 'hdf5s', 'testing.h5')
    elif feature_type == 'gamma':
        test_hdf5_path = os.path.join(workspace, 'hdf5s', 'testing_{}.h5'.format(feature_type))

    test_reference_csv_path = os.path.join(dataset_dir, 'metadata',
        'groundtruth_strong_label_testing_set.csv')

    checkpoint_path = os.path.join(workspace, 'checkpoints',
        '{}'.format(filename), 'holdout_fold={}'.format(holdout_fold),
        'model_type={}'.format(model_type), 'loss_type={}'.format(loss_type),
        'augmentation={}'.format(augmentation), 'batch_size={}'.format(batch_size),
        'best.pth'.format(feature_type))

    predictions_dir = os.path.join(workspace, 'predictions',
        '{}'.format(filename), 'holdout_fold={}'.format(holdout_fold),
        'model_type={}'.format(model_type), 'loss_type={}'.format(loss_type),
        'augmentation={}'.format(augmentation), 'batch_size={}'.format(batch_size))
    create_folder(predictions_dir)

    tmp_submission_path = os.path.join(workspace, '_tmp_submission',
        '{}'.format(filename), 'holdout_fold={}'.format(holdout_fold),
        'model_type={}'.format(model_type), 'loss_type={}'.format(loss_type),
        'augmentation={}'.format(augmentation), 'batch_size={}'.format(batch_size),
        '_overlap_submission.csv')
    create_folder(os.path.dirname(tmp_submission_path))
    
    reference_csv_path = os.path.join(dataset_dir, 'metadata', 'groundtruth_strong_label_testing_set.csv')

    # Load model
    assert model_type, 'Please specify model_type!'
    Model = eval(model_type)
    model = Model(sample_rate, window_size, hop_size, mel_bins, fmin, fmax,
        classes_num, feature_type)

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model'])

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
            'best.sed.test.pkl')
        sed_params_dict = pickle.load(open(sed_thresholds_path, 'rb'))
    else:
        sed_params_dict = {
            'audio_tagging_threshold': 0.099,
            'sed_high_threshold': 0.5,
            'sed_low_threshold': 0.2,
            'n_smooth': 10,
            'n_salt': 10}

    param_combinations = [[0.5,6], [0.5,7], [1,5], [1,6], [1,7]]
    #param_combinations = [[1,5]]
    
    audios_dir = os.path.join(dataset_dir, data_type)
    gt_csv = pd.read_csv(reference_csv_path, header=None)
    audio_files = gt_csv[0].unique()
    audio_files = ['{}/{}'.format(audios_dir, x) for x in audio_files]
    
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
            audio_duration = librosa.get_duration(filename=audio_name)
            predict_length += audio_duration
            num_segment = 1
            index = 0
            start = 0
            end = 0
            merged = None
            # Load audio sample
            (audio_full, fs) = librosa.core.load(audio_name, sr=sample_rate, mono=True)
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
                curr_preds = binarize_pred(curr_preds, sed_params_dict['sed_high_threshold'])
                if num_segment == 2:
                    merged = merge(prev_preds, curr_preds, sample_duration, num_segment, overlap_value)
                elif num_segment > 2:
                    merged = merge(merged, curr_preds, sample_duration, num_segment, overlap_value)
                else:
                    merged = curr_preds
                prev_clipwise = output_dict['clipwise_output']
                prev_preds = output_dict['framewise_output']
                prev_preds = binarize_pred(prev_preds, sed_params_dict['sed_high_threshold'])
                
                #start += 1
                start += overlap_value
                end = start + sample_duration
                num_segment += 1
            #merged = avg_merge(merged, sample_duration, overlap_value)
            #np.set_printoptions(threshold=sys.maxsize)
            predict_event_list = frame_binary_prediction_to_event_prediction(merged, overlap_value, sample_duration, audio_name.split('/')[-1], sed_params_dict)
            full_predict_event_list.extend(predict_event_list)
            
        print(full_predict_event_list)
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
    parser_train.add_argument('--augmentation', type=str, choices=['none', 'mixup', 'timeshift_mixup'], required=True)
    parser_train.add_argument('--learning_rate', type=float, required=True)
    parser_train.add_argument('--batch_size', type=int, required=True)
    parser_train.add_argument('--resume_iteration', type=int)
    parser_train.add_argument('--stop_iteration', type=int, required=True)
    parser_train.add_argument('--feature_type', type=str, required=True)
    parser_train.add_argument('--cuda', action='store_true', default=False)
    parser_train.add_argument('--mini_data', action='store_true', default=False)

    # Inference
    parser_inference_prob = subparsers.add_parser('inference_prob')
    parser_inference_prob.add_argument('--dataset_dir', type=str, required=True, help='Directory of dataset.')
    parser_inference_prob.add_argument('--workspace', type=str, required=True, help='Directory of your workspace.')
    parser_inference_prob.add_argument('--holdout_fold', type=str, choices=['1'], required=True)
    parser_inference_prob.add_argument('--model_type', type=str, required=True)
    parser_inference_prob.add_argument('--loss_type', type=str, required=True)
    parser_inference_prob.add_argument('--augmentation', type=str, choices=['none', 'mixup', 'timeshift_mixup'], required=True)
    parser_inference_prob.add_argument('--feature_type', type=str, required=True)
    parser_inference_prob.add_argument('--batch_size', type=int, required=True)
    parser_inference_prob.add_argument('--cuda', action='store_true', default=False)
    
    # Inference (overlap + avg)
    parser_inference_prob_overlap = subparsers.add_parser('inference_prob_overlap')
    parser_inference_prob_overlap.add_argument('--dataset_dir', type=str, required=True, help='Directory of dataset.')
    parser_inference_prob_overlap.add_argument('--workspace', type=str, required=True, help='Directory of your workspace.')
    parser_inference_prob_overlap.add_argument('--holdout_fold', type=str, choices=['1'], required=True)
    parser_inference_prob_overlap.add_argument('--model_type', type=str, required=True)
    parser_inference_prob_overlap.add_argument('--loss_type', type=str, required=True)
    parser_inference_prob_overlap.add_argument('--augmentation', type=str, choices=['none', 'mixup', 'timeshift_mixup'], required=True)
    parser_inference_prob_overlap.add_argument('--feature_type', type=str, required=True)
    parser_inference_prob_overlap.add_argument('--batch_size', type=int, required=True)
    parser_inference_prob_overlap.add_argument('--sample_duration', type=int, default=2)
    parser_inference_prob_overlap.add_argument('--cuda', action='store_true', default=False)
    parser_inference_prob_overlap.add_argument('--sed_thresholds', action='store_true', default=False)
    
    # Inference (overlap + vote)
    parser_inference_prob_vote = subparsers.add_parser('inference_prob_vote')
    parser_inference_prob_vote.add_argument('--dataset_dir', type=str, required=True, help='Directory of dataset.')
    parser_inference_prob_vote.add_argument('--workspace', type=str, required=True, help='Directory of your workspace.')
    parser_inference_prob_vote.add_argument('--holdout_fold', type=str, choices=['1'], required=True)
    parser_inference_prob_vote.add_argument('--model_type', type=str, required=True)
    parser_inference_prob_vote.add_argument('--loss_type', type=str, required=True)
    parser_inference_prob_vote.add_argument('--augmentation', type=str, choices=['none', 'mixup', 'timeshift_mixup'], required=True)
    parser_inference_prob_vote.add_argument('--feature_type', type=str, required=True)
    parser_inference_prob_vote.add_argument('--batch_size', type=int, required=True)
    parser_inference_prob_vote.add_argument('--sample_duration', type=int, default=2)
    parser_inference_prob_vote.add_argument('--cuda', action='store_true', default=False)
    parser_inference_prob_vote.add_argument('--sed_thresholds', action='store_true', default=False)
    
    # Parse arguments
    args = parser.parse_args()
    args.filename = get_filename(__file__)

    if args.mode == 'train':
        train(args)

    elif args.mode == 'inference_prob':
        inference_prob(args)
        
    elif args.mode == 'inference_prob_overlap':
        inference_prob_overlap(args)
    
    elif args.mode == 'inference_prob_vote':
        inference_prob_vote(args)
    
    else:
        raise Exception('Error argument!')
