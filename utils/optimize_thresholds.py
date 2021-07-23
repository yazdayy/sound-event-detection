import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../utils'))
import numpy as np
import argparse
import h5py
import math
import time
import logging
import sklearn
import pickle
from sklearn import metrics
import matplotlib.pyplot as plt
#from autoth.core import HyperParamsOptimizer

from utilities import (get_filename, create_folder, 
    frame_prediction_to_event_prediction, write_submission, official_evaluate)
from calculate_metrics import calculate_precision_recall_f1, calculate_metrics
import config

class HyperParamsOptimizer(object):
    def __init__(self, score_calculator, save_dict, learning_rate=1e-2, epochs=100,
        step=0.01, max_search=5):
        """Hyper parameters optimizer. Parameters are optimized using gradient
        descend methods by using the numerically calculated graident:
        gradient: f(x + h) - f(x) / (h)
        Args:
          score_calculator: object. See ScoreCalculatorExample in example.py as
              an example.
          learning_rate: float
          epochs: int
          step: float, equals h for calculating gradients
          max_search: int, if plateaued, then search for at most max_search times
        """
        
        self.score_calculator = score_calculator
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.optimizer = Adam()
        self.optimizer.alpha = learning_rate
        self.step = step
        self.max_search = max_search
        self.save_dict = save_dict

    def do_optimize(self, init_params):
        print('Optimizing hyper parameters ...')
        print('learning rate: {:.3f}, total epochs: {}'.format(
            self.learning_rate, self.epochs))

        params = init_params.copy()

        for i in range(self.epochs):
            t1 = time.time()
            (score, grads) = self.calculate_gradients(params)
            grads = [-e for e in grads]
            params = self.optimizer.GetNewParams(params, grads)
            self.save_dict[i] = {'thresholds': params, 'score': score}
            print('    Hyper parameters: {}, score: {:.4f}'.format([round(param, 4) for param in params], score))
            print('    Epoch: {}, Time: {:.4f} s'.format(i, time.time() - t1))
        
        return score, params, self.save_dict

    def calculate_gradients(self, params):
        """Calculate gradient of thresholds numerically.
        Args:
          y_true: (N, (optional)frames_num], classes_num)
          output: (N, (optional)[frames_num], classes_num)
          thresholds: (classes_num,), initial thresholds
          average: 'micro' | 'macro'
        Returns:
          grads: vector
        """
        score = self.score_calculator(params)
        step = self.step
        grads = []

        for k, param in enumerate(params):
            new_params = params.copy()
            cnt = 0
            while cnt < self.max_search:
                cnt += 1
                new_params[k] += self.step
                new_score = self.score_calculator(new_params)

                if new_score != score:
                    break

            grad = (new_score - score) / (step * cnt)
            grads.append(grad)

        return score, grads


class Base(object):
    def _reset_memory(self, memory):
        for i1 in range(len(memory)):
            memory[i1] = np.zeros(memory[i1].shape)


class Adam(Base):
    def __init__(self):
        self.ms = []
        self.vs = []
        self.alpha = 1e-3
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.eps = 1e-8
        self.iter = 0
        
    def GetNewParams(self, params, gparams):
        if not self.ms:
            for param in params:
                self.ms += [np.zeros_like(param)]
                self.vs += [np.zeros_like(param)]
          
        # fast adam, faster than origin adam
        self.iter += 1
        new_params = []
        alpha_t = self.alpha * np.sqrt(1 - np.power(self.beta2, self.iter)) / (1 - np.power(self.beta1, self.iter))
        for i1 in range(len(params)):
            self.ms[i1] = self.beta1 * self.ms[i1] + (1 - self.beta1) * gparams[i1]
            self.vs[i1] = self.beta2 * self.vs[i1] + (1 - self.beta2) * np.square(gparams[i1])
            new_params += [params[i1] - alpha_t * self.ms[i1] / (np.sqrt(self.vs[i1] + self.eps))]
            
        return new_params
        
    def reset(self):
        self._reset_memory(self.ms)
        self._reset_memory(self.vs)
        self.epoch = 1


class AudioTaggingScoreCalculator(object):
    def __init__(self, prediction_path):
        """Used to calculate score (such as F1) given prediction, target and hyper parameters. 
        """
        self.output_dict = pickle.load(open(prediction_path, 'rb'))

    def __call__(self, params):
        """Use hyper parameters to threshold prediction to obtain output.
        Then, the scores are calculated between output and target.
        """
        (precision, recall, f1) = calculate_precision_recall_f1(
            self.output_dict['target'], self.output_dict['clipwise_output'], 
            thresholds=params)

        return f1


class SoundEventDetectionScoreCalculator(object):
    def __init__(self, prediction_path, reference_csv_path, submission_path, classes_num):
        """Used to calculate score (such as F1) given prediction, target and hyper parameters. 
        """
        self.output_dict = pickle.load(open(prediction_path, 'rb'))
        self.reference_csv_path = reference_csv_path
        self.submission_path = submission_path
        self.classes_num = classes_num

    def params_dict_to_params_list(self, sed_params_dict):
        params = sed_params_dict['audio_tagging_threshold'] + \
            sed_params_dict['sed_high_threshold'] + \
            sed_params_dict['sed_low_threshold']

        return params

    def params_list_to_params_dict(self, params):
        sed_params_dict = {
            'audio_tagging_threshold': params[0 : self.classes_num], 
            'sed_high_threshold': params[self.classes_num : 2 * self.classes_num],
            'sed_low_threshold': params[2 * self.classes_num :],
            'n_smooth': 10,
            'n_salt': 10
        }
        return sed_params_dict


    def __call__(self, params):
        """Use hyper parameters to threshold prediction to obtain output.
        Then, the scores are calculated between output and target.
        """
        params_dict = self.params_list_to_params_dict(params)
        # params_dict['n_smooth'] = 1
        # params_dict['n_salt'] = 1

        predict_event_list = frame_prediction_to_event_prediction(
            self.output_dict, params_dict)

        # Write predicted events to submission file
        write_submission(predict_event_list, self.submission_path)

        # SED with official tool
        results = official_evaluate(self.reference_csv_path, self.submission_path)
        
        f1 = results['overall']['f_measure']['f_measure']

        return f1


def optimize_at_thresholds(args):
    """Calculate audio tagging metrics with optimized thresholds.

    Args:
      dataset_dir: str
      workspace: str
      filename: str
      holdout_fold: '1'
      model_type: str, e.g., 'Cnn_9layers_Gru_FrameAtt'
      loss_type: str, e.g., 'clip_bce'
      augmentation: str, e.g., 'mixup'
      batch_size: int
      iteration: int
    """
    
    # Arugments & parameters
    dataset_dir = args.dataset_dir
    workspace = args.workspace
    filename = args.filename
    holdout_fold = args.holdout_fold
    model_type = args.model_type
    loss_type = args.loss_type
    augmentation = args.augmentation
    batch_size = args.batch_size
    iteration = args.iteration
    data_type = 'test'
    
    classes_num = config.classes_num
    
    # Paths
    if data_type == 'test':
        reference_csv_path = os.path.join(dataset_dir, 'metadata', 
            'groundtruth_strong_label_testing_set.csv')
    
    prediction_path = os.path.join(workspace, 'predictions', 
        '{}'.format(filename), 'holdout_fold={}'.format(holdout_fold), 
        'model_type={}'.format(model_type), 'loss_type={}'.format(loss_type), 
        'augmentation={}'.format(augmentation), 'batch_size={}'.format(batch_size),
        '{}_iterations.prediction.{}.pkl'.format(iteration, data_type))
    
    tmp_submission_path = os.path.join(workspace, '_tmp_submission', 
        '{}'.format(filename), 'holdout_fold={}'.format(holdout_fold), 
        'model_type={}'.format(model_type), 'loss_type={}'.format(loss_type), 
        'augmentation={}'.format(augmentation), 'batch_size={}'.format(batch_size),
        '_submission.csv')

    opt_thresholds_path = os.path.join(workspace, 'opt_thresholds', 
        '{}'.format(filename), 'holdout_fold={}'.format(holdout_fold), 
        'model_type={}'.format(model_type), 'loss_type={}'.format(loss_type), 
        'augmentation={}'.format(augmentation), 'batch_size={}'.format(batch_size),
        '{}_iterations.at.{}.pkl'.format(iteration, data_type))
    create_folder(os.path.dirname(opt_thresholds_path))

    # Score calculator
    score_calculator = AudioTaggingScoreCalculator(prediction_path)

    # Thresholds optimizer
    hyper_params_opt = HyperParamsOptimizer(score_calculator, learning_rate=1e-2, epochs=100)

    # Initialize thresholds
    init_params = [0.3] * classes_num
    score_no_opt = score_calculator(init_params)

    # Optimize thresholds
    (opt_score, opt_params) = hyper_params_opt.do_optimize(init_params=init_params)

    print('\n------ Optimized thresholds ------')
    print(np.around(opt_params, decimals=4))

    print('\n------ Without optimized thresholds ------')
    print('Score: {:.3f}'.format(score_no_opt))

    print('\n------ With optimized thresholds ------')
    print('Score: {:.3f}'.format(opt_score))

    # Write out optimized thresholds
    pickle.dump(opt_params, open(opt_thresholds_path, 'wb'))
    print('\nSave optimized thresholds to {}'.format(opt_thresholds_path))


def optimize_sed_thresholds(args):
    """Calculate sound event detection metrics with optimized thresholds.

    Args:
      dataset_dir: str
      workspace: str
      filename: str
      holdout_fold: '1'
      model_type: str, e.g., 'Cnn_9layers_Gru_FrameAtt'
      loss_type: str, e.g., 'clip_bce'
      augmentation: str, e.g., 'mixup'
      batch_size: int
      iteration: int
    """
    
    # Arugments & parameters
    dataset_dir = args.dataset_dir
    workspace = args.workspace
    filename = args.filename
    holdout_fold = args.holdout_fold
    model_type = args.model_type
    loss_type = args.loss_type
    augmentation = args.augmentation
    batch_size = args.batch_size
    data_type = 'test'
    save_dict = {}
    
    classes_num = config.classes_num
    
    # Paths
    if data_type == 'test':
        reference_csv_path = os.path.join(dataset_dir, 'metadata', 
            'groundtruth_strong_label_testing_set.csv')
    
    prediction_path = os.path.join(workspace, 'predictions', 
        '{}'.format(filename), 'holdout_fold={}'.format(holdout_fold), 
        'model_type={}'.format(model_type), 'loss_type={}'.format(loss_type), 
        'augmentation={}'.format(augmentation), 'batch_size={}'.format(batch_size),
        'best.prediction.{}.pkl'.format(data_type))
    
    tmp_submission_path = os.path.join(workspace, '_tmp_submission', 
        '{}'.format(filename), 'holdout_fold={}'.format(holdout_fold), 
        'model_type={}'.format(model_type), 'loss_type={}'.format(loss_type), 
        'augmentation={}'.format(augmentation), 'batch_size={}'.format(batch_size),
        '_submission.csv')

    opt_thresholds_path = os.path.join(workspace, 'opt_thresholds', 
        '{}'.format(filename), 'holdout_fold={}'.format(holdout_fold), 
        'model_type={}'.format(model_type), 'loss_type={}'.format(loss_type), 
        'augmentation={}'.format(augmentation), 'batch_size={}'.format(batch_size),
        'best.sed.{}.pkl'.format(data_type))
    create_folder(os.path.dirname(opt_thresholds_path))
    
    save_record_path = os.path.join(workspace, 'opt_thresholds',
        '{}'.format(filename), 'holdout_fold={}'.format(holdout_fold),
        'model_type={}'.format(model_type), 'loss_type={}'.format(loss_type),
        'augmentation={}'.format(augmentation), 'batch_size={}'.format(batch_size),
        'record.sed.{}.pkl'.format(data_type))
    create_folder(os.path.dirname(save_record_path))

    # Score calculator
    score_calculator = SoundEventDetectionScoreCalculator(
        prediction_path=prediction_path, reference_csv_path=reference_csv_path, 
        submission_path=tmp_submission_path, classes_num=classes_num)

    # Thresholds optimizer
    hyper_params_opt = HyperParamsOptimizer(score_calculator, save_dict,
        learning_rate=1e-2, epochs=70, step=0.02, max_search=5)

    # Initialize thresholds
    sed_params_dict = {
        'audio_tagging_threshold': [0.5] * classes_num, 
        'sed_high_threshold': [0.3] * classes_num, 
        'sed_low_threshold': [0.1] * classes_num}

    init_params = score_calculator.params_dict_to_params_list(sed_params_dict)
    score_no_opt = score_calculator(init_params)

    # Optimize thresholds
    (opt_score, opt_params, save_dict) = hyper_params_opt.do_optimize(init_params=init_params)
    opt_params = score_calculator.params_list_to_params_dict(opt_params)

    print('\n------ Optimized thresholds ------')
    print(opt_params)

    print('\n------ Without optimized thresholds ------')
    print('Score: {:.3f}'.format(score_no_opt))

    print('\n------ With optimized thresholds ------')
    print('Score: {:.3f}'.format(opt_score))

    # Write out optimized thresholds
    pickle.dump(opt_params, open(opt_thresholds_path, 'wb'))
    print('\nSave optimized thresholds to {}'.format(opt_thresholds_path))
    
    pickle.dump(save_dict, open(save_record_path, 'wb'))
    print('\nSave records to {}'.format(save_record_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Example of parser. ')
    subparsers = parser.add_subparsers(dest='mode')

    parser_optimize_at_thresholds = subparsers.add_parser('optimize_at_thresholds')
    parser_optimize_at_thresholds.add_argument('--dataset_dir', type=str, required=True)
    parser_optimize_at_thresholds.add_argument('--workspace', type=str, required=True)
    parser_optimize_at_thresholds.add_argument('--filename', type=str, required=True)
    parser_optimize_at_thresholds.add_argument('--holdout_fold', type=str, choices=['1', 'none'], required=True)
    parser_optimize_at_thresholds.add_argument('--model_type', type=str, required=True)
    parser_optimize_at_thresholds.add_argument('--loss_type', type=str, required=True)
    parser_optimize_at_thresholds.add_argument('--augmentation', type=str, choices=['none', 'mixup', 'timeshift_mixup'], required=True)
    parser_optimize_at_thresholds.add_argument('--batch_size', type=int, required=True)
    parser_optimize_at_thresholds.add_argument('--iteration', type=int, required=True)

    parser_optimize_sed_thresholds = subparsers.add_parser('optimize_sed_thresholds')
    parser_optimize_sed_thresholds.add_argument('--dataset_dir', type=str, required=True)
    parser_optimize_sed_thresholds.add_argument('--workspace', type=str, required=True)
    parser_optimize_sed_thresholds.add_argument('--filename', type=str, required=True)
    parser_optimize_sed_thresholds.add_argument('--holdout_fold', type=str, choices=['1', 'none'], required=True)
    parser_optimize_sed_thresholds.add_argument('--model_type', type=str, required=True)
    parser_optimize_sed_thresholds.add_argument('--loss_type', type=str, required=True)
    parser_optimize_sed_thresholds.add_argument('--augmentation', type=str, choices=['none', 'mixup', 'timeshift_mixup'], required=True)
    parser_optimize_sed_thresholds.add_argument('--batch_size', type=int, required=True)

    args = parser.parse_args()

    if args.mode == 'optimize_at_thresholds':
        optimize_at_thresholds(args)

    elif args.mode == 'optimize_sed_thresholds':
        optimize_sed_thresholds(args)

    else:
        raise Exception('Error argument!')
