import os
import sys
import librosa
import logging
import matplotlib.pyplot as plt
import datetime
import pickle
import numpy as np
import csv
import sed_eval
import h5py
from prettytable import PrettyTable

import config
from vad import activity_detection, activity_detection_binary

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params+=param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params

def create_folder(fd):
    if not os.path.exists(fd):
        os.makedirs(fd)


def get_filename(path):
    path = os.path.realpath(path)
    name_ext = path.split('/')[-1]
    name = os.path.splitext(name_ext)[0]
    return name


def create_logging(log_dir, filemode):
    create_folder(log_dir)
    i1 = 0

    while os.path.isfile(os.path.join(log_dir, '{:04d}.log'.format(i1))):
        i1 += 1
        
    log_path = os.path.join(log_dir, '{:04d}.log'.format(i1))
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
        datefmt='%a, %d %b %Y %H:%M:%S',
        filename=log_path,
        filemode=filemode)

    # Print to console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    
    return logging


def pad_truncate_sequence(x, max_len):
    if len(x) < max_len:
        return np.concatenate((x, np.zeros(max_len - len(x))))
    else:
        return x[0 : max_len]


def float32_to_int16(x):
    if np.max(np.abs(x)) > 1.:
        x /= np.max(np.abs(x))
    return (x * 32767.).astype(np.int16)

def int16_to_float32(x):
    return (x / 32767.).astype(np.float32)


def frame_prediction_to_event_prediction(output_dict, sed_params_dict, frames_per_second):
    """Write output to submission file. 
    
    Args:
      output_dict: {
          'audio_name': (audios_num), 
          'clipwise_output': (audios_num, classes_num), 
          'framewise_output': (audios_num, frames_num, classes_num)}
      sed_params_dict: {
          'audio_tagging_threshold': float between 0 and 1, 
          'sed_high_threshold': : float between 0 and 1, 
          'sed_low_threshold': : float between 0 and 1, 
          'n_smooth': int, silence between the same sound event shorter than 
              this number will be filled with the sound event
          'n_salt': int, sound event shorter than this number will be removed}
    """
    (audios_num, frames_num, classes_num) = output_dict['framewise_output'].shape
    #frames_per_second = config.frames_per_second
    labels = config.labels
    
    event_list = []
    
    def _float_to_list(x):
        if 'list' in str(type(x)):
            return x
        else:
            return [x] * classes_num

    sed_params_dict['audio_tagging_threshold'] = _float_to_list(sed_params_dict['audio_tagging_threshold'])
    sed_params_dict['sed_high_threshold'] = _float_to_list(sed_params_dict['sed_high_threshold'])
    sed_params_dict['sed_low_threshold'] = _float_to_list(sed_params_dict['sed_low_threshold'])
    sed_params_dict['n_smooth'] = _float_to_list(sed_params_dict['n_smooth'])
    sed_params_dict['n_salt'] = _float_to_list(sed_params_dict['n_salt'])
    
    count1 = 0
    count2 = 0
    for n in range(audios_num):
        check = 0
        for k in range(classes_num):
#            if output_dict['clipwise_output'][n, k] \
#                > sed_params_dict['audio_tagging_threshold'][k]:
                #print('CLASS INDEX:', k)
                check += 1
                count1 += 1
                bgn_fin_pairs = activity_detection(
                    x=output_dict['framewise_output'][n, :, k], 
                    thres=sed_params_dict['sed_high_threshold'][k], 
                    low_thres=sed_params_dict['sed_low_threshold'][k], 
                    n_smooth=sed_params_dict['n_smooth'][k], 
                    n_salt=sed_params_dict['n_salt'][k])
                
                if len(bgn_fin_pairs) >= 1:
                    count2 += 1
                for pair in bgn_fin_pairs:
                    event = {
                        'filename': output_dict['audio_name'][n], 
                        'onset': pair[0] / float(frames_per_second), 
                        'offset': pair[1] / float(frames_per_second), 
                        'event_label': labels[k]}
                    event_list.append(event)
#        if check == 0:
#            print(output_dict['audio_name'][n])
#            print(output_dict['clipwise_output'][n])
    all_filenames = list(set([x['filename'] for x in event_list]))
    
#    print('AUDIO NUM', audios_num)
#    print('COUNT1', count1)
#    print('COUNT2', count2)
#    print('UNIQUE', len(all_filenames))
#    print('TOTAL', len(event_list))
    
    return event_list

def frame_prediction_to_event_prediction_v2(framewise_output, audio_name, sed_params_dict, frames_per_second):
    """Write output to submission file.
    
    Args:
      output_dict: {
          'audio_name': (audios_num),
          'clipwise_output': (audios_num, classes_num),
          'framewise_output': (audios_num, frames_num, classes_num)}
      sed_params_dict: {
          'audio_tagging_threshold': float between 0 and 1,
          'sed_high_threshold': : float between 0 and 1,
          'sed_low_threshold': : float between 0 and 1,
          'n_smooth': int, silence between the same sound event shorter than
              this number will be filled with the sound event
          'n_salt': int, sound event shorter than this number will be removed}
    """
    (audios_num, frames_num, classes_num) = framewise_output.shape
    #frames_per_second = config.frames_per_second
    labels = config.labels
    
    event_list = []
    
    def _float_to_list(x):
        if 'list' in str(type(x)):
            return x
        else:
            return [x] * classes_num

    sed_params_dict['audio_tagging_threshold'] = _float_to_list(sed_params_dict['audio_tagging_threshold'])
    sed_params_dict['sed_high_threshold'] = _float_to_list(sed_params_dict['sed_high_threshold'])
    sed_params_dict['sed_low_threshold'] = _float_to_list(sed_params_dict['sed_low_threshold'])
    sed_params_dict['n_smooth'] = _float_to_list(sed_params_dict['n_smooth'])
    sed_params_dict['n_salt'] = _float_to_list(sed_params_dict['n_salt'])
    
    count1 = 0
    count2 = 0
    for n in range(audios_num):
        check = 0
        for k in range(classes_num):
            check += 1
            count1 += 1
            bgn_fin_pairs = activity_detection(
            x=framewise_output[n, :, k],
            thres=sed_params_dict['sed_high_threshold'][k],
            low_thres=sed_params_dict['sed_low_threshold'][k],
            n_smooth=sed_params_dict['n_smooth'][k],
            n_salt=sed_params_dict['n_salt'][k])
            
            if len(bgn_fin_pairs) >= 1:
                count2 += 1
            for pair in bgn_fin_pairs:
                event = {
                    'filename': audio_name,
                    'onset': pair[0] / float(frames_per_second),
                    'offset': pair[1] / float(frames_per_second),
                    'event_label': labels[k]}
                event_list.append(event)
    all_filenames = list(set([x['filename'] for x in event_list]))
    
    return event_list
    
def frame_binary_prediction_to_event_prediction(framewise_output, overlap_value, sample_duration,  audio_name, sed_params_dict):
    """Write output to submission file.

    Args:
      output_dict: {
          'audio_name': (audios_num),
          'clipwise_output': (audios_num, classes_num),
          'framewise_output': (audios_num, frames_num, classes_num)}
      sed_params_dict: {
          'audio_tagging_threshold': float between 0 and 1,
          'sed_high_threshold': : float between 0 and 1,
          'sed_low_threshold': : float between 0 and 1,
          'n_smooth': int, silence between the same sound event shorter than
              this number will be filled with the sound event
          'n_salt': int, sound event shorter than this number will be removed}
    """
    (audios_num, frames_num, classes_num) = framewise_output.shape
    frames_per_second = config.frames_per_second
    labels = config.labels

    event_list = []

    def _float_to_list(x):
        if 'list' in str(type(x)):
            return x
        else:
            return [x] * classes_num

    sed_params_dict['audio_tagging_threshold'] = _float_to_list(sed_params_dict['audio_tagging_threshold'])
    sed_params_dict['sed_high_threshold'] = _float_to_list(sed_params_dict['sed_high_threshold'])
    sed_params_dict['sed_low_threshold'] = _float_to_list(sed_params_dict['sed_low_threshold'])
    sed_params_dict['n_smooth'] = _float_to_list(sed_params_dict['n_smooth'])
    sed_params_dict['n_salt'] = _float_to_list(sed_params_dict['n_salt'])

    count1 = 0
    count2 = 0
    for n in range(audios_num):
        check = 0
        for k in range(classes_num):
            check += 1
            count1 += 1
            bgn_fin_pairs = activity_detection_binary(
            x=framewise_output[n, :, k], overlap_value=overlap_value,
            sample_duration=sample_duration,
            thres=sed_params_dict['sed_high_threshold'][k],
            low_thres=sed_params_dict['sed_low_threshold'][k],
            n_smooth=sed_params_dict['n_smooth'][k],
            n_salt=sed_params_dict['n_salt'][k])
            
            if len(bgn_fin_pairs) >= 1:
                count2 += 1
            for pair in bgn_fin_pairs:
                event = {
                    'filename': audio_name,
                    'onset': pair[0] / float(frames_per_second),
                    'offset': pair[1] / float(frames_per_second),
                    'event_label': labels[k]}
                event_list.append(event)
    all_filenames = list(set([x['filename'] for x in event_list]))

    return event_list

def write_submission(event_list, submission_path):
    """Write prediction event list to submission file for later evaluation.

    Args:
      event_list: list of events
      submission_path: str
    """
    f = open(submission_path, 'w')
            
    for event in event_list:
        f.write('{}\t{}\t{}\t{}\n'.format(
            event['filename'], event['onset'], event['offset'], event['event_label']))
            
    logging.info('    Write submission file to {}'.format(submission_path))


def official_evaluate(reference_csv_path, prediction_csv_path):
    """Evaluate metrics with official SED toolbox. 

    Args:
      reference_csv_path: str
      prediction_csv_path: str
    """
    reference_event_list = sed_eval.io.load_event_list(reference_csv_path, 
        delimiter=',', csv_header=False,
        fields=['filename','onset','offset','event_label'])
    
    #print('REFERENCE', reference_event_list)
    estimated_event_list = sed_eval.io.load_event_list(prediction_csv_path, 
        delimiter='\t', csv_header=False, 
        fields=['filename','onset','offset','event_label'])
    
    #print('ESTIMATED', estimated_event_list)
    labels = ['Applause', 'Chatter', 'Cheering', 'Child_speech_kid_speaking', 'Clapping', 'Conversation', 'Cough', 'Crowd', 'Crying_sobbing', 'Female_speech_woman_speaking', 'Laughter', 'Male_speech_man_speaking', 'Screaming', 'Shout', 'Whispering', 'Air_horn_truck_horn', 'Car_alarm', 'Emergency_vehicle', 'Explosion', 'Gunshot_gunfire', 'Siren', 'Music']
    evaluated_event_labels = labels#reference_event_list.unique_event_labels
    files={}
    for event in reference_event_list:
        files[event['filename']] = event['filename']

    evaluated_files = sorted(list(files.keys()))
    
    segment_based_metrics = sed_eval.sound_event.SegmentBasedMetrics(
        event_label_list=evaluated_event_labels,
        time_resolution=1.0
    )

    for file in evaluated_files:
        reference_event_list_for_current_file = []
        for event in reference_event_list:
            if event['filename'] == file:
                reference_event_list_for_current_file.append(event)
                estimated_event_list_for_current_file = []
        for event in estimated_event_list:
            if event['filename'] == file:
                estimated_event_list_for_current_file.append(event)

        segment_based_metrics.evaluate(
            reference_event_list=reference_event_list_for_current_file,
            estimated_event_list=estimated_event_list_for_current_file
        )
    results = segment_based_metrics.results()

    return results


class StatisticsContainer(object):
    def __init__(self, statistics_path):
        self.statistics_path = statistics_path

        self.backup_statistics_path = '{}_{}.pkl'.format(
            os.path.splitext(self.statistics_path)[0], datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

        self.statistics_dict = {'train': [], 'test': [], 'valid': []}

    def append(self, data_type, iteration, statistics):
        statistics['iteration'] = iteration
        self.statistics_dict[data_type].append(statistics)
        
    def dump(self):
        pickle.dump(self.statistics_dict, open(self.statistics_path, 'wb'))
        pickle.dump(self.statistics_dict, open(self.backup_statistics_path, 'wb'))
        logging.info('    Dump statistics to {}'.format(self.statistics_path))
        logging.info('    Dump statistics to {}'.format(self.backup_statistics_path))

    def load_state_dict(self, resume_iteration):
        self.statistics_dict = pickle.load(open(self.statistics_path, 'rb'))

        resume_statistics_dict = {'train': [], 'test': [], 'evaluate': []}
        
        for key in self.statistics_dict.keys():
            for statistics in self.statistics_dict[key]:
                if statistics['iteration'] <= resume_iteration:
                    resume_statistics_dict[key].append(statistics)
                
        self.statistics_dict = resume_statistics_dict
 

class Mixup(object):
    def __init__(self, mixup_alpha, random_seed=1234):
        """Mixup coefficient generator.
        """
        self.mixup_alpha = mixup_alpha
        self.random_state = np.random.RandomState(random_seed)

    def get_lambda(self, batch_size):
        """Get mixup random coefficients.

        Args:
          batch_size: int

        Returns:
          mixup_lambdas: (batch_size,)
        """
        mixup_lambdas = []
        for n in range(0, batch_size, 2):
            lam = self.random_state.beta(self.mixup_alpha, self.mixup_alpha, 1)[0]
            mixup_lambdas.append(lam)
            mixup_lambdas.append(1. - lam)

        return np.array(mixup_lambdas)
        
def append_to_dict(dict, key, value):
    if key in dict.keys():
        dict[key].append(value)
    else:
        dict[key] = value

def merge(prev, curr, sample_duration, num_segment, overlap_value=1):
    overlap_interval = int(100 * overlap_value)
    front_cutoff = (num_segment-1) * overlap_interval
    back_cutoff = prev.shape[1] - front_cutoff
    prev_overlap = prev[:, front_cutoff:]
    curr_overlap = curr[:, :back_cutoff]
    merged = prev_overlap + curr_overlap
    add_front = np.concatenate((prev[:, :front_cutoff], merged), axis=1)
    add_back = np.concatenate((add_front, curr[:, back_cutoff:]), axis=1)
    
#    front_cutoff = (num_segment-1) * 100
#    back_cutoff = prev.shape[1] - front_cutoff
#    prev_overlap = prev[:, front_cutoff:]
#    curr_overlap = curr[:, :back_cutoff]
#    merged = prev_overlap + curr_overlap
#    add_front = np.concatenate((prev[:, :front_cutoff], merged), axis=1)
#    add_back = np.concatenate((add_front, curr[:, back_cutoff:]), axis=1)

    return add_back
    
def avg_merge(merged, sample_duration, overlap_value=1):
    overlap_interval = int(100 * overlap_value)
    interval = (sample_duration * 100) - overlap_interval
    for i in range(overlap_interval, merged.shape[1]-overlap_interval, overlap_interval):
        if i < interval:
            num_overlaps = i//overlap_interval + 1
        elif i >= merged.shape[1] - interval:
            num_overlaps = ((merged.shape[1] - i) // overlap_interval) + 1
        else:
            num_overlaps = sample_duration
        merged[:, i:i+overlap_interval] /= num_overlaps
#    interval = (sample_duration * 100) - 100
#    for i in range(100, merged.shape[1]-100, 100):
#        if i < interval:
#            num_overlaps = i//100 + 1
#        elif i >= merged.shape[1] - interval:
#            num_overlaps = ((merged.shape[1] - i) // 100) + 1
#        else:
#            num_overlaps = sample_duration
#        merged[:, i:i+100] /= num_overlaps
    
    return merged
