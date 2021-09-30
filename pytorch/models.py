import os
import sys
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from pytorch_utils import do_mixup, do_mixup_timeshift, do_timeshift
from augmentation import SpecAugmentation
from stft import Spectrogram, LogmelFilterBank, STFT, CQTFilterBank, GammaFilterBank

from models_2020.baseline_model import CNN
from models_2020.conformer.conformer_encoder import ConformerEncoder
from models_2020.transformer.encoder import Encoder as TransformerEncoder


def init_layer(layer):
    """Initialize a Linear or Convolutional layer. """
    nn.init.xavier_uniform_(layer.weight)
 
    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)
            
    
def init_bn(bn):
    """Initialize a Batchnorm layer. """
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)


def init_gru(rnn):
    """Initialize a GRU layer. """
    
    def _concat_init(tensor, init_funcs):
        (length, fan_out) = tensor.shape
        fan_in = length // len(init_funcs)
    
        for (i, init_func) in enumerate(init_funcs):
            init_func(tensor[i * fan_in : (i + 1) * fan_in, :])
        
    def _inner_uniform(tensor):
        fan_in = nn.init._calculate_correct_fan(tensor, 'fan_in')
        nn.init.uniform_(tensor, -math.sqrt(3 / fan_in), math.sqrt(3 / fan_in))
    
    for i in range(rnn.num_layers):
        _concat_init(
            getattr(rnn, 'weight_ih_l{}'.format(i)),
            [_inner_uniform, _inner_uniform, _inner_uniform]
        )
        torch.nn.init.constant_(getattr(rnn, 'bias_ih_l{}'.format(i)), 0)

        _concat_init(
            getattr(rnn, 'weight_hh_l{}'.format(i)),
            [_inner_uniform, _inner_uniform, nn.init.orthogonal_]
        )
        torch.nn.init.constant_(getattr(rnn, 'bias_hh_l{}'.format(i)), 0)

def roundup(x):
    return x if x % 100 == 0 else x + 100 - x % 100

def pad_framewise_output(framewise_output: torch.Tensor, frames_num: int):
    """Pad framewise_output to the same length as input frames. The pad value
    is the same as the value of the last frame.
    Args:
      framewise_output: (batch_size, frames_num, classes_num)
      frames_num: int, number of frames to pad
    Outputs:
      output: (batch_size, frames_num, classes_num)
    """
    pad = framewise_output[:, -1:, :].repeat(
        1, frames_num - framewise_output.shape[1], 1)
    """tensor for padding"""

    output = torch.cat((framewise_output, pad), dim=1)
    """(batch_size, frames_num, classes_num)"""

    return output


def interpolate(x, ratio):
    '''Interpolate the prediction to have the same time_steps as the target. 
    The time_steps mismatch is caused by maxpooling in CNN. 
    
    Args:
      x: (batch_size, time_steps, classes_num)
      ratio: int, ratio to upsample
    '''
    (batch_size, time_steps, classes_num) = x.shape
    upsampled = x[:, :, None, :].repeat(1, 1, ratio, 1)
    upsampled = upsampled.reshape(batch_size, time_steps * ratio, classes_num)
    return upsampled


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        
        super(ConvBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=in_channels, 
                              out_channels=out_channels,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1), bias=False)
                              
        self.conv2 = nn.Conv2d(in_channels=out_channels, 
                              out_channels=out_channels,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1), bias=False)
                              
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.init_weight()
        
    def init_weight(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

        
    def forward(self, input, pool_size=(2, 2), pool_type='avg'):
        
        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg+max':
            x1 = F.avg_pool2d(x, kernel_size=pool_size)
            x2 = F.max_pool2d(x, kernel_size=pool_size)
            x = x1 + x2
        else:
            raise Exception('Incorrect argument!')
        
        return x


class AttBlock(nn.Module):
    def __init__(self, n_in, n_out, activation='linear', temperature=1.):
        super(AttBlock, self).__init__()
        
        self.activation = activation
        self.temperature = temperature
        self.att = nn.Conv1d(in_channels=n_in, out_channels=n_out, kernel_size=1, stride=1, padding=0, bias=True)
        self.cla = nn.Conv1d(in_channels=n_in, out_channels=n_out, kernel_size=1, stride=1, padding=0, bias=True)
        
        self.bn_att = nn.BatchNorm1d(n_out)
        self.init_weights()
        
    def init_weights(self):
        init_layer(self.att)
        init_layer(self.cla)
        init_bn(self.bn_att)
        
    def forward(self, x):
        # x: (n_samples, n_in, n_time)
        tmp = self.att(x)
        tmp = torch.clamp(tmp, -10, 10)
        att = torch.exp(tmp / self.temperature) + 1e-6
        norm_att = att / torch.sum(att, dim=2)[:, :, None]
        cla = self.nonlinear_transform(self.cla(x))
        x = torch.sum(norm_att * cla, dim=2)
        return x, norm_att, cla

    def nonlinear_transform(self, x):
        if self.activation == 'linear':
            return x
        elif self.activation == 'sigmoid':
            return torch.sigmoid(x)

class RegBlock(nn.Module):
    def __init__(self, n_in, n_out, activation='linear', temperature=1.):
        super(AttBlock, self).__init__()
        
        self.activation = activation
        self.temperature = temperature
        self.reg = nn.Linear(n_in, num_classes, bias=True)
        self.att = nn.Conv1d(in_channels=n_in, out_channels=n_out, kernel_size=1, stride=1, padding=0, bias=True)
        self.cla = nn.Conv1d(in_channels=n_in, out_channels=n_out, kernel_size=1, stride=1, padding=0, bias=True)
        
        self.bn_att = nn.BatchNorm1d(n_out)
        self.init_weights()
        
    def init_weights(self):
        init_layer(self.att)
        init_layer(self.cla)
        init_bn(self.bn_att)
        
    def forward(self, x):
        # x: (n_samples, n_in, n_time)
        tmp = self.att(x)
        tmp = torch.clamp(tmp, -10, 10)
        att = torch.exp(tmp / self.temperature) + 1e-6
        norm_att = att / torch.sum(att, dim=2)[:, :, None]
        cla = self.nonlinear_transform(self.cla(x))
        x = torch.sum(norm_att * cla, dim=2)
        return x, norm_att, cla

    def nonlinear_transform(self, x):
        if self.activation == 'linear':
            return x
        elif self.activation == 'sigmoid':
            return torch.sigmoid(x)
            
# CNN

class Cnn_9layers_FrameMax(nn.Module):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, 
        fmax, classes_num):
        
        super(Cnn_9layers_FrameMax, self).__init__()

        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None

        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(n_fft=window_size, hop_length=hop_size, 
            win_length=window_size, window=window, center=center, pad_mode=pad_mode, 
            freeze_parameters=True)

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=window_size, 
            n_mels=mel_bins, fmin=fmin, fmax=fmax, ref=ref, amin=amin, top_db=top_db, 
            freeze_parameters=True)

        # Spec augmenter
        self.spec_augmenter = SpecAugmentation(time_drop_width=64, time_stripes_num=2, 
            freq_drop_width=8, freq_stripes_num=2)

        self.bn0 = nn.BatchNorm2d(64)

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)

        self.fc = nn.Linear(512, classes_num, bias=True)

        self.init_weights()

    def init_weights(self):
        init_bn(self.bn0)
        init_layer(self.fc)

    def forward(self, input, mixup_lambda=None):
        """Input: (batch_size, times_steps, freq_bins)"""
        
        interpolate_ratio = 8

        x = self.spectrogram_extractor(input)   # (batch_size, 1, time_steps, freq_bins)
        x = self.logmel_extractor(x)    # (batch_size, 1, time_steps, mel_bins)
        
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        
        if self.training:
            x = self.spec_augmenter(x)

        # Mixup on spectrogram
        if self.training and mixup_lambda is not None:
            x = do_mixup(x, mixup_lambda)
        
        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block4(x, pool_size=(1, 1), pool_type='avg')

        x = torch.mean(x, dim=3)
        embedding = x   # (batch_size, feature_maps, time_steps)

        # Framewise output
        x = x.transpose(1, 2)
        framewise_output = torch.sigmoid(self.fc(x))
        framewise_output = interpolate(framewise_output, interpolate_ratio)
        
        # Clipwise output
        (clipwise_output, _) = torch.max(framewise_output, dim=1)
        
        output_dict = {
            'framewise_output': framewise_output, 
            'clipwise_output': clipwise_output, 
            'embedding': embedding}
            
        return output_dict


class Cnn_9layers_FrameAvg(nn.Module):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, 
        fmax, classes_num):
        
        super(Cnn_9layers_FrameAvg, self).__init__()

        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None

        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(n_fft=window_size, hop_length=hop_size, 
            win_length=window_size, window=window, center=center, pad_mode=pad_mode, 
            freeze_parameters=True)

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=window_size, 
            n_mels=mel_bins, fmin=fmin, fmax=fmax, ref=ref, amin=amin, top_db=top_db, 
            freeze_parameters=True)

        # Spec augmenter
        self.spec_augmenter = SpecAugmentation(time_drop_width=64, time_stripes_num=2, 
            freq_drop_width=8, freq_stripes_num=2)

        self.bn0 = nn.BatchNorm2d(64)

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)

        self.fc = nn.Linear(512, classes_num, bias=True)

        self.init_weights()

    def init_weights(self):
        init_bn(self.bn0)
        init_layer(self.fc)

    def forward(self, input, mixup_lambda=None):
        """Input: (batch_size, times_steps, freq_bins)"""
        
        interpolate_ratio = 8

        x = self.spectrogram_extractor(input)   # (batch_size, 1, time_steps, freq_bins)
        x = self.logmel_extractor(x)    # (batch_size, 1, time_steps, mel_bins)
        
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        
        if self.training:
            x = self.spec_augmenter(x)

        # Mixup on spectrogram
        if self.training and mixup_lambda is not None:
            x = do_mixup(x, mixup_lambda)
        
        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block4(x, pool_size=(1, 1), pool_type='avg')

        x = torch.mean(x, dim=3)
        embedding = x   # (batch_size, feature_maps, time_steps)

        # Framewise output
        x = x.transpose(1, 2)
        framewise_output = torch.sigmoid(self.fc(x))
        framewise_output = interpolate(framewise_output, interpolate_ratio)
        
        # Clipwise output
        clipwise_output = torch.mean(framewise_output, dim=1)
        
        output_dict = {
            'framewise_output': framewise_output, 
            'clipwise_output': clipwise_output, 
            'embedding': embedding}
            
        return output_dict


class Cnn_9layers_FrameAtt(nn.Module):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, 
        fmax, classes_num):
        
        super(Cnn_9layers_FrameAtt, self).__init__()

        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None

        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(n_fft=window_size, hop_length=hop_size, 
            win_length=window_size, window=window, center=center, pad_mode=pad_mode, 
            freeze_parameters=True)

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=window_size, 
            n_mels=mel_bins, fmin=fmin, fmax=fmax, ref=ref, amin=amin, top_db=top_db, 
            freeze_parameters=True)

        # Spec augmenter
        self.spec_augmenter = SpecAugmentation(time_drop_width=64, time_stripes_num=2, 
            freq_drop_width=8, freq_stripes_num=2)

        self.bn0 = nn.BatchNorm2d(64)

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.att_block = AttBlock(n_in=512, n_out=25, activation='sigmoid')

        self.init_weights()

    def init_weights(self):
        init_bn(self.bn0)

    def forward(self, input, mixup_lambda=None):
        """Input: (batch_size, times_steps, freq_bins)"""
        
        interpolate_ratio = 8

        x = self.spectrogram_extractor(input)   # (batch_size, 1, time_steps, freq_bins)
        x = self.logmel_extractor(x)    # (batch_size, 1, time_steps, mel_bins)
        
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        
        if self.training:
            x = self.spec_augmenter(x)

        # Mixup on spectrogram
        if self.training and mixup_lambda is not None:
            x = do_mixup(x, mixup_lambda)
        
        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block4(x, pool_size=(1, 1), pool_type='avg')

        x = torch.mean(x, dim=3)

        (clipwise_output, norm_att, cla) = self.att_block(x)
        """cla: (batch_size, classes_num, time_stpes)"""

        # Framewise output
        framewise_output = cla.transpose(1, 2)
        framewise_output = interpolate(framewise_output, interpolate_ratio)
        
        output_dict = {
            'framewise_output': framewise_output, 
            'clipwise_output': clipwise_output, 
            'embedding': cla}
            
        return output_dict


# CNN-GRU

class Cnn_9layers_Gru_FrameAvg(nn.Module):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, 
        fmax, classes_num, feature_type):
        
        super(Cnn_9layers_Gru_FrameAvg, self).__init__()

        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None

        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(n_fft=window_size, hop_length=hop_size, 
            win_length=window_size, window=window, center=center, pad_mode=pad_mode, 
            freeze_parameters=True)

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=window_size, 
            n_mels=mel_bins, fmin=fmin, fmax=fmax, ref=ref, amin=amin, top_db=top_db, 
            freeze_parameters=True)

        # Spec augmenter
        self.spec_augmenter = SpecAugmentation(time_drop_width=64, time_stripes_num=2, 
            freq_drop_width=8, freq_stripes_num=2)

        self.bn0 = nn.BatchNorm2d(64)

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)

        self.gru = nn.GRU(input_size=512, hidden_size=256, num_layers=1, 
            bias=True, batch_first=True, bidirectional=True)

        self.fc = nn.Linear(512, classes_num, bias=True)

        self.init_weights()

    def init_weights(self):
        init_bn(self.bn0)
        init_gru(self.gru)
        init_layer(self.fc)

    def forward(self, input, mixup_lambda=None, timeshift=False, spec_augment=True):
        """Input: (batch_size, times_steps, freq_bins)"""
        
        interpolate_ratio = 8

        x = self.spectrogram_extractor(input)   # (batch_size, 1, time_steps, freq_bins)
        x = self.logmel_extractor(x)    # (batch_size, 1, time_steps, mel_bins)
        
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        
        if self.training and spec_augment:
            x = self.spec_augmenter(x)

        # Mixup on spectrogram
        if self.training and mixup_lambda is not None:
            if not timeshift:
                x = do_mixup(x, mixup_lambda)
            elif timeshift:
                x = do_mixup_timeshift(x, mixup_lambda)
                
        if self.training and mixup_lambda is None and timeshift:
            x = do_timeshift(x)
        
        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block4(x, pool_size=(1, 1), pool_type='avg')

        x = torch.mean(x, dim=3)
        x = x.transpose(1, 2)   # (batch_size, time_steps, channels)
        (x, _) = self.gru(x)
        x = x.transpose(1, 2)
        embedding = x   # (batch_size, feature_maps, time_steps)

        # Framewise output
        x = x.transpose(1, 2)
        framewise_output = torch.sigmoid(self.fc(x))
        framewise_output = interpolate(framewise_output, interpolate_ratio)
        
        # Clipwise output
        clipwise_output = torch.mean(framewise_output, dim=1)
        
        output_dict = {
            'framewise_output': framewise_output, 
            'clipwise_output': clipwise_output, 
            'embedding': embedding}
            
        return output_dict


class Cnn_9layers_Gru_FrameAtt(nn.Module):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, 
        fmax, classes_num, feature_type):
        
        super(Cnn_9layers_Gru_FrameAtt, self).__init__()

        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None
        num_bins = 80
        self.feature_type = feature_type
        
        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(n_fft=window_size, hop_length=hop_size, 
            win_length=window_size, window=window, center=center, pad_mode=pad_mode, 
            freeze_parameters=True)

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=window_size, 
            n_mels=mel_bins, fmin=fmin, fmax=fmax, ref=ref, amin=amin, top_db=top_db, 
            freeze_parameters=True)
        
        # STFT extractor
#        self.stft_extractor = STFT(n_fft=window_size, hop_length=hop_size, win_length=window_size,
#               window=window, center=True, pad_mode=pad_mode, freeze_parameters=True)
        
        # CQT feature extractor
#        self.cqt_extractor = CQTFilterBank(sr=sample_rate, n_bins=num_bins,
#            fmin=fmin, fmax=fmax, ref=ref, amin=amin, top_db=top_db,
#            freeze_parameters=True)
            
        # Gammatone feature extractor
#        self.gamma_extractor = GammaFilterBank(sr=sample_rate, n_fft=window_size,
#        n_mels=mel_bins, fmin=fmin, fmax=fmax, ref=ref, amin=amin, top_db=top_db,
#        freeze_parameters=True)
        
        # Spec augmenter
        self.spec_augmenter = SpecAugmentation(time_drop_width=64, time_stripes_num=2, 
            freq_drop_width=8, freq_stripes_num=2)

        self.bn0 = nn.BatchNorm2d(64)

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)

        self.gru = nn.GRU(input_size=512, hidden_size=256, num_layers=1, 
            bias=True, batch_first=True, bidirectional=True)

        self.att_block = AttBlock(n_in=512, n_out=25, activation='sigmoid')

        self.init_weights()

    def init_weights(self):
        init_bn(self.bn0)
        init_gru(self.gru)

    def forward(self, input, mixup_lambda=None, timeshift=False, spec_augment=True):
        """Input: (batch_size, times_steps, freq_bins)"""
        
        interpolate_ratio = 8
        
        if self.feature_type == 'logmel':
            x = self.spectrogram_extractor(input)   # (batch_size, 1, time_steps, freq_bins)
            x = self.logmel_extractor(x)    # (batch_size, 1, time_steps, mel_bins)
        elif self.feature_type == 'cqt':
            x = self.stft_extractor(input)
            x = self.cqt_extractor(x)
        elif self.feature_type == 'gamma':
            #x = self.gamma_extractor(input)
            x = torch.unsqueeze(input, 1)
            x = x.transpose(2,3)
            x = x.to('cuda')
        
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        
        # SpecAugmnet on spectrogram
        if self.training and spec_augment:
            x = self.spec_augmenter(x)
        
        # Mixup on spectrogram
        if self.training and mixup_lambda is not None:
            if not timeshift:
                x = do_mixup(x, mixup_lambda)
            elif timeshift:
                x = do_mixup_timeshift(x, mixup_lambda)
            
#        if self.training and mixup_lambda is not None and timeshift:
#            x = do_mixup_timeshift(x, mixup_lambda)
        
        if self.training and mixup_lambda is None and timeshift:
            x = do_timeshift(x)
            
        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block4(x, pool_size=(1, 1), pool_type='avg')

        x = torch.mean(x, dim=3)
        x = x.transpose(1, 2)   # (batch_size, time_steps, channels)
        (x, _) = self.gru(x)
        x = x.transpose(1, 2)

        (clipwise_output, norm_att, cla) = self.att_block(x)
        """cla: (batch_size, classes_num, time_stpes)"""

        # Framewise output
        framewise_output = cla.transpose(1, 2)
        framewise_output = interpolate(framewise_output, interpolate_ratio)
        #framewise_output = pad_framewise_output(framewise_output, 1000)
        if framewise_output.size()[1] != 1000:
            framewise_output = pad_framewise_output(framewise_output, roundup(framewise_output.size()[1]))
        
        output_dict = {
            'framewise_output': framewise_output, 
            'clipwise_output': clipwise_output, 
            'embedding': cla}
            
        return output_dict


class Cnn_14layers_Gru_FrameAtt(nn.Module):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin,
        fmax, classes_num, feature_type):
        
        super(Cnn_14layers_Gru_FrameAtt, self).__init__()

        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None

        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(n_fft=window_size, hop_length=hop_size,
            win_length=window_size, window=window, center=center, pad_mode=pad_mode,
            freeze_parameters=True)

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=window_size,
            n_mels=mel_bins, fmin=fmin, fmax=fmax, ref=ref, amin=amin, top_db=top_db,
            freeze_parameters=True)

        # Spec augmenter
        self.spec_augmenter = SpecAugmentation(time_drop_width=64, time_stripes_num=2,
            freq_drop_width=8, freq_stripes_num=2)

        self.bn0 = nn.BatchNorm2d(64)
        
        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.conv_block5 = ConvBlock(in_channels=512, out_channels=1024)
        self.conv_block6 = ConvBlock(in_channels=1024, out_channels=2048)

        self.gru = nn.GRU(input_size=2048, hidden_size=1024, num_layers=1,
            bias=True, batch_first=True, bidirectional=True)

        self.att_block = AttBlock(n_in=2048, n_out=25, activation='sigmoid')

        self.init_weights()

    def init_weights(self):
        init_bn(self.bn0)
        init_gru(self.gru)

    def forward(self, input, mixup_lambda=None, timeshift=False, spec_augment=True):
        """Input: (batch_size, times_steps, freq_bins)"""
        
        interpolate_ratio = 32

        x = self.spectrogram_extractor(input)   # (batch_size, 1, time_steps, freq_bins)
        x = self.logmel_extractor(x)    # (batch_size, 1, time_steps, mel_bins)
        
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        
        if self.training and spec_augment:
            x = self.spec_augmenter(x)

        # Mixup on spectrogram
        if self.training and mixup_lambda is not None:
            if not timeshift:
                x = do_mixup(x, mixup_lambda)
            elif timeshift:
                x = do_mixup_timeshift(x, mixup_lambda)
                
        if self.training and mixup_lambda is None and timeshift:
            x = do_timeshift(x)
        
        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block5(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block6(x, pool_size=(1, 1), pool_type='avg')
        

        x = torch.mean(x, dim=3)
        x = x.transpose(1, 2)   # (batch_size, time_steps, channels)
        (x, _) = self.gru(x)
        x = x.transpose(1, 2)

        (clipwise_output, norm_att, cla) = self.att_block(x)
        """cla: (batch_size, classes_num, time_stpes)"""

        # Framewise output
        framewise_output = cla.transpose(1, 2)
        framewise_output = interpolate(framewise_output, interpolate_ratio)
        #framewise_output = pad_framewise_output(framewise_output, 1000)
        if framewise_output.size()[1] != 1000:
            framewise_output = pad_framewise_output(framewise_output, roundup(framewise_output.size()[1]))
        
        output_dict = {
            'framewise_output': framewise_output,
            'clipwise_output': clipwise_output,
            'embedding': cla}
            
        return output_dict


# CNN-Transformer

####################################
# The following Transformer modules are modified from Yu-Hsiang Huang's code:
# https://github.com/jadore801120/attention-is-all-you-need-pytorch
class ScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention"""

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):

        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)

        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output, attn


class MultiHead(nn.Module):
    """Multi-Head Attention module."""

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))
        self.w_qs.bias.data.fill_(0)
        self.w_ks.bias.data.fill_(0)
        self.w_vs.bias.data.fill_(0)

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)
        self.fc.bias.data.fill_(0)

        self.dropout = nn.Dropout(dropout)


    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()   # (batch_size, 80, 512)
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k) # (batch_size, T, 8, 64)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k) # (n*b) x lq x dk, (batch_size*8, T, 64)
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k) # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v) # (n*b) x lv x dv

        # mask = mask.repeat(n_head, 1, 1) # (n*b) x .. x ..
        output, attn = self.attention(q, k, v, mask=mask)   # (n_head * batch_size, T, 64), (n_head * batch_size, T, T)
        
        output = output.view(n_head, sz_b, len_q, d_v)  # (n_head, batch_size, T, 64)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1) # b x lq x (n*dv), (batch_size, T, 512)
        output = F.relu_(self.dropout(self.fc(output)))
        return output


class Cnn_9layers_Transformer_FrameAvg(nn.Module):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, 
        fmax, classes_num, feature_type):
        
        super(Cnn_9layers_Transformer_FrameAvg, self).__init__()

        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None

        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(n_fft=window_size, hop_length=hop_size, 
            win_length=window_size, window=window, center=center, pad_mode=pad_mode, 
            freeze_parameters=True)

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=window_size, 
            n_mels=mel_bins, fmin=fmin, fmax=fmax, ref=ref, amin=amin, top_db=top_db, 
            freeze_parameters=True)

        # Spec augmenter
        self.spec_augmenter = SpecAugmentation(time_drop_width=64, time_stripes_num=2, 
            freq_drop_width=8, freq_stripes_num=2)

        self.bn0 = nn.BatchNorm2d(64)

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)

        n_head = 8
        n_hid = 512
        d_k = 64
        d_v = 64
        dropout = 0.2
        self.multihead = MultiHead(n_head, n_hid, d_k, d_v, dropout)

        self.fc = nn.Linear(512, classes_num, bias=True)

        self.init_weights()

    def init_weights(self):
        init_bn(self.bn0)
        init_layer(self.fc)

    def forward(self, input, mixup_lambda=None, timeshift=False, spec_augment=True):
        """Input: (batch_size, times_steps, freq_bins)"""
        
        interpolate_ratio = 8

        x = self.spectrogram_extractor(input)   # (batch_size, 1, time_steps, freq_bins)
        x = self.logmel_extractor(x)    # (batch_size, 1, time_steps, mel_bins)
        
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        
        if self.training and spec_augment:
            x = self.spec_augmenter(x)

        # Mixup on spectrogram
        if self.training and mixup_lambda is not None:
            if not timeshift:
                x = do_mixup(x, mixup_lambda)
            elif timeshift:
                x = do_mixup_timeshift(x, mixup_lambda)
                
        if self.training and mixup_lambda is None and timeshift:
            x = do_timeshift(x)
        
        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block4(x, pool_size=(1, 1), pool_type='avg')
        
        x = torch.mean(x, dim=3)
        x = x.transpose(1, 2)   # (batch_size, time_steps, channels)
        x = self.multihead(x, x, x)
        x = x.transpose(1, 2)
        embedding = x   # (batch_size, feature_maps, time_steps)

        # Framewise output
        x = x.transpose(1, 2)
        framewise_output = torch.sigmoid(self.fc(x))
        framewise_output = interpolate(framewise_output, interpolate_ratio)
        
        # Clipwise output
        clipwise_output = torch.mean(framewise_output, dim=1)
        
        output_dict = {
            'framewise_output': framewise_output, 
            'clipwise_output': clipwise_output, 
            'embedding': embedding}
            
        return output_dict


class Cnn_9layers_Transformer_FrameAtt(nn.Module):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, 
        fmax, classes_num, feature_type):
        
        super(Cnn_9layers_Transformer_FrameAtt, self).__init__()

        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None

        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(n_fft=window_size, hop_length=hop_size, 
            win_length=window_size, window=window, center=center, pad_mode=pad_mode, 
            freeze_parameters=True)

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=window_size, 
            n_mels=mel_bins, fmin=fmin, fmax=fmax, ref=ref, amin=amin, top_db=top_db, 
            freeze_parameters=True)

        # Spec augmenter
        self.spec_augmenter = SpecAugmentation(time_drop_width=64, time_stripes_num=2, 
            freq_drop_width=8, freq_stripes_num=2)

        self.bn0 = nn.BatchNorm2d(64)

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)

        n_head = 8
        n_hid = 512
        d_k = 64
        d_v = 64
        dropout = 0.2
        self.multihead = MultiHead(n_head, n_hid, d_k, d_v, dropout)

        self.att_block = AttBlock(n_in=512, n_out=25, activation='sigmoid')

        self.init_weights()

    def init_weights(self):
        init_bn(self.bn0)

    def forward(self, input, mixup_lambda=None, timeshift=False, spec_augment=True):
        """Input: (batch_size, times_steps, freq_bins)"""
        
        interpolate_ratio = 8

        x = self.spectrogram_extractor(input)   # (batch_size, 1, time_steps, freq_bins)
        x = self.logmel_extractor(x)    # (batch_size, 1, time_steps, mel_bins)
        
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        
        if self.training and spec_augment:
            x = self.spec_augmenter(x)

        # Mixup on spectrogram
        if self.training and mixup_lambda is not None:
            if not timeshift:
                x = do_mixup(x, mixup_lambda)
            elif timeshift:
                x = do_mixup_timeshift(x, mixup_lambda)
                
        if self.training and mixup_lambda is None and timeshift:
            x = do_timeshift(x)
        
        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block4(x, pool_size=(1, 1), pool_type='avg')

        x = torch.mean(x, dim=3)
        x = x.transpose(1, 2)   # (batch_size, time_steps, channels)
        x = self.multihead(x, x, x)
        x = x.transpose(1, 2)
        embedding = x   # (batch_size, feature_maps, time_steps)

        (clipwise_output, norm_att, cla) = self.att_block(x)
        """cla: (batch_size, classes_num, time_stpes)"""

        # Framewise output
        framewise_output = cla.transpose(1, 2)
        framewise_output = interpolate(framewise_output, interpolate_ratio)
        
        output_dict = {
            'framewise_output': framewise_output, 
            'clipwise_output': clipwise_output, 
            'embedding': embedding}
            
        return output_dict


class Cnn_14layers_Transformer_FrameAtt(nn.Module):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin,
        fmax, classes_num, feature_type):
        
        super(Cnn_14layers_Transformer_FrameAtt, self).__init__()

        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None

        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(n_fft=window_size, hop_length=hop_size,
            win_length=window_size, window=window, center=center, pad_mode=pad_mode,
            freeze_parameters=True)

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=window_size,
            n_mels=mel_bins, fmin=fmin, fmax=fmax, ref=ref, amin=amin, top_db=top_db,
            freeze_parameters=True)

        # Spec augmenter
        self.spec_augmenter = SpecAugmentation(time_drop_width=64, time_stripes_num=2,
            freq_drop_width=8, freq_stripes_num=2)

        self.bn0 = nn.BatchNorm2d(64)

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.conv_block5 = ConvBlock(in_channels=512, out_channels=1024)
        self.conv_block6 = ConvBlock(in_channels=1024, out_channels=2048)

        n_head = 8
        n_hid = 2048
        d_k = 64
        d_v = 64
        dropout = 0.2
        self.multihead = MultiHead(n_head, n_hid, d_k, d_v, dropout)

        self.att_block = AttBlock(n_in=2048, n_out=25, activation='sigmoid')

        self.init_weights()

    def init_weights(self):
        init_bn(self.bn0)

    def forward(self, input, mixup_lambda=None, timeshift=False, spec_augment=True):
        """Input: (batch_size, times_steps, freq_bins)"""
        
        interpolate_ratio = 32

        x = self.spectrogram_extractor(input)   # (batch_size, 1, time_steps, freq_bins)
        x = self.logmel_extractor(x)    # (batch_size, 1, time_steps, mel_bins)
        
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        
        if self.training and spec_augment:
            x = self.spec_augmenter(x)

        # Mixup on spectrogram
        if self.training and mixup_lambda is not None:
            if not timeshift:
                x = do_mixup(x, mixup_lambda)
            elif timeshift:
                x = do_mixup_timeshift(x, mixup_lambda)
                
        if self.training and mixup_lambda is None and timeshift:
            x = do_timeshift(x)
        
        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block5(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block6(x, pool_size=(1, 1), pool_type='avg')
        

        x = torch.mean(x, dim=3)
        x = x.transpose(1, 2)   # (batch_size, time_steps, channels)
        x = self.multihead(x, x, x)
        x = x.transpose(1, 2)
        embedding = x   # (batch_size, feature_maps, time_steps)

        (clipwise_output, norm_att, cla) = self.att_block(x)
        """cla: (batch_size, classes_num, time_stpes)"""

        # Framewise output
        framewise_output = cla.transpose(1, 2)
        framewise_output = interpolate(framewise_output, interpolate_ratio)
        #framewise_output = pad_framewise_output(framewise_output, 1000)
        if framewise_output.size()[1] != 1000:
            framewise_output = pad_framewise_output(framewise_output, roundup(framewise_output.size()[1]))
        
        output_dict = {
            'framewise_output': framewise_output,
            'clipwise_output': clipwise_output,
            'embedding': embedding}
            
        return output_dict


# CNN-Conformer

class Cnn_9layers_Conformer_FrameAtt(torch.nn.Module):
    def __init__(
        self,
        sample_rate: int, window_size: int, hop_size: int,
        mel_bins: int, fmin: int, fmax: int, classes_num,
        cnn_kwargs=None,
        encoder_kwargs=None,
        encoder_type="Conformer",
        pooling="token",
        layer_init="pytorch",
    ):
        super(Cnn_9layers_Conformer_FrameAtt, self).__init__()
        
        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None
        #self.interpolate_ratio = 32  # Downsampled ratio
        
        self.cnn_kwargs = {
            'activation': "Relu",
            'conv_dropout': 0.1,
            'kernel_size': [3, 3, 3, 3, 3, 3, 3],
            'padding': [1, 1, 1, 1, 1, 1, 1],
            'stride': [1, 1, 1, 1, 1, 1, 1],
            'nb_filters': [16, 32, 64, 128, 128, 128, 128],
            'pooling': [[2, 2], [2, 2], [2, 2], [1, 2], [1, 2], [1, 2], [1, 1]]}
            
        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(
           n_fft=window_size,
           hop_length=hop_size,
           win_length=window_size,
           window=window,
           center=center,
           pad_mode=pad_mode,
           freeze_parameters=True)

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(
           sr=sample_rate,
           n_fft=window_size,
           n_mels=mel_bins,
           fmin=fmin,
           fmax=fmax,
           ref=ref,
           amin=amin,
           top_db=top_db,
           freeze_parameters=True)

        # Spec augmenter
        self.spec_augmenter = SpecAugmentation(
           time_drop_width=64,
           time_stripes_num=2,
           freq_drop_width=8,
           freq_stripes_num=2)

        self.bn0 = nn.BatchNorm2d(mel_bins)
        
        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        
        input_dim = 512
        
        self.pooling = pooling

        if encoder_type == "Transformer":
            self.encoder_kwargs = {
            'adim': 144,
            'aheads': 4,
            'dropout_rate': 0.1,
            'elayers': 3,
            'eunits': 576,
            'kernel_size': 7,
            'transformer_input_layer': 'conv2d',
            'transformer_attn_dropout_rate': 0.0,
            'after_conv': False}
            adim = self.encoder_kwargs["adim"]
            self.encoder = TransformerEncoder(input_dim, **self.encoder_kwargs)
        elif encoder_type == "Conformer":
            self.encoder_kwargs = {
            'adim': 144,
            'aheads': 4,
            'dropout_rate': 0.1,
            'elayers': 3,
            'eunits': 576,
            'kernel_size': 7}
            adim = self.encoder_kwargs["adim"]
            self.encoder = ConformerEncoder(input_dim, **self.encoder_kwargs)
        else:
            raise ValueError("Choose encoder_type in ['Transformer', 'Conformer']")
        
        self.att_block = AttBlock(n_in=144, n_out=25, activation='sigmoid')
        
        self.classifier = torch.nn.Linear(adim, classes_num)

        if self.pooling == "attention":
            self.dense = torch.nn.Linear(adim, classes_num)
            self.sigmoid = torch.sigmoid
            self.softmax = torch.nn.Softmax(dim=-1)

        elif self.pooling == "token":
            self.linear_emb = torch.nn.Linear(1, input_dim)

        self.reset_parameters(layer_init)
       
    def preprocess(self, input, mixup_lambda=None, timeshift=False, spec_augment=True):
        # t1 = time.time()
        x = self.spectrogram_extractor(input)  # (batch_size, 1, time_steps, freq_bins)
        x = self.logmel_extractor(x)  # (batch_size, 1, time_steps, mel_bins)

        frames_num = x.shape[2]

        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        if self.training and spec_augment:
            x = self.spec_augmenter(x)

        # Mixup on spectrogram
        if self.training and mixup_lambda is not None:
            if not timeshift:
                x = do_mixup(x, mixup_lambda)
            elif timeshift:
                x = do_mixup_timeshift(x, mixup_lambda)
                
        if self.training and mixup_lambda is None and timeshift:
            x = do_timeshift(x)
            
        return x, frames_num
    
    def forward(self, x, mixup_lambda=None, timeshift=False, spec_augment=True, mask=None):

        # input
        #x = torch.reshape(x, (x.size()[0], x.size()[-1]))
        
        interpolate_ratio = 8
        x, frames_num = self.preprocess(x, mixup_lambda=mixup_lambda, timeshift=timeshift, spec_augment=spec_augment)
        
        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block4(x, pool_size=(1, 1), pool_type='avg')
#        x = torch.reshape(x, (x.size()[0], x.size()[1], x.size()[2]*x.size()[3]))
#        x = x.squeeze(-1).permute(0, 2, 1)
        
#        if self.pooling == "token":
#            tag_token = self.linear_emb(torch.ones(x.size(0), 1, 1).cuda())
#            x = torch.cat([tag_token, x], dim=1)

        #x, _ = self.encoder(x, mask)
        
        x = torch.mean(x, dim=3)
        x = x.transpose(1, 2)   # (batch_size, time_steps, channels)
        x, _ = self.encoder(x, mask)
        x = x.transpose(1, 2)
        embedding = x   # (batch_size, feature_maps, time_steps)

        (clipwise_output, norm_att, cla) = self.att_block(x)
        """cla: (batch_size, classes_num, time_stpes)"""

        # Framewise output
        framewise_output = cla.transpose(1, 2)
        framewise_output = interpolate(framewise_output, interpolate_ratio)
        #framewise_output = pad_framewise_output(framewise_output, 1000)
        if framewise_output.size()[1] != 1000:
            framewise_output = pad_framewise_output(framewise_output, roundup(framewise_output.size()[1]))
        
        output_dict = {
            'framewise_output': framewise_output,
            'clipwise_output': clipwise_output,
            'embedding': embedding}
            
        return output_dict
#        if self.pooling == "attention":
#            strong = self.classifier(x)
#            sof = self.dense(x)  # [bs, frames, nclass]
#            sof = self.softmax(sof)
#            sof = torch.clamp(sof, min=1e-7, max=1)
#            weak = (torch.sigmoid(strong) * sof).sum(1) / sof.sum(1)  # [bs, nclass]
#            # Convert to logit to calculate loss with bcelosswithlogits
#            weak = torch.log(weak / (1 - weak))
#        elif self.pooling == "token":
#            x = self.classifier(x)
#            weak = x[:, 0, :]
#            strong = x[:, 1:, :]
#        elif self.pooling == "auto":
#            strong = self.classifier(x)
#            weak = self.autopool(strong)
            
#        return {"framewise_output": strong, "clipwise_output": weak}

    def reset_parameters(self, initialization: str = "pytorch"):
        if initialization.lower() == "pytorch":
            return
        # weight init
        for p in self.parameters():
            if p.dim() > 1:
                if initialization.lower() == "xavier_uniform":
                    torch.nn.init.xavier_uniform_(p.data)
                elif initialization.lower() == "xavier_normal":
                    torch.nn.init.xavier_normal_(p.data)
                elif initialization.lower() == "kaiming_uniform":
                    torch.nn.init.kaiming_uniform_(p.data, nonlinearity="relu")
                elif initialization.lower() == "kaiming_normal":
                    torch.nn.init.kaiming_normal_(p.data, nonlinearity="relu")
                else:
                    raise ValueError(f"Unknown initialization: {initialization}")
        # bias init
        for p in self.parameters():
            if p.dim() == 1:
                p.data.zero_()
        # reset some modules with default init
        for m in self.modules():
            if isinstance(m, (torch.nn.Embedding, LayerNorm)):
                m.reset_parameters()
                

class Cnn_9layers_Conformer_FrameAvg(torch.nn.Module):
    def __init__(
        self,
        sample_rate: int, window_size: int, hop_size: int,
        mel_bins: int, fmin: int, fmax: int, classes_num,
        cnn_kwargs=None,
        encoder_kwargs=None,
        encoder_type="Conformer",
        pooling="token",
        layer_init="pytorch",
    ):
        super(Cnn_9layers_Conformer_FrameAvg, self).__init__()
        
        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None
        #self.interpolate_ratio = 32  # Downsampled ratio
        
        self.cnn_kwargs = {
            'activation': "Relu",
            'conv_dropout': 0.1,
            'kernel_size': [3, 3, 3, 3, 3, 3, 3],
            'padding': [1, 1, 1, 1, 1, 1, 1],
            'stride': [1, 1, 1, 1, 1, 1, 1],
            'nb_filters': [16, 32, 64, 128, 128, 128, 128],
            'pooling': [[2, 2], [2, 2], [2, 2], [1, 2], [1, 2], [1, 2], [1, 1]]}
            
        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(
           n_fft=window_size,
           hop_length=hop_size,
           win_length=window_size,
           window=window,
           center=center,
           pad_mode=pad_mode,
           freeze_parameters=True)

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(
           sr=sample_rate,
           n_fft=window_size,
           n_mels=mel_bins,
           fmin=fmin,
           fmax=fmax,
           ref=ref,
           amin=amin,
           top_db=top_db,
           freeze_parameters=True)

        # Spec augmenter
        self.spec_augmenter = SpecAugmentation(
           time_drop_width=64,
           time_stripes_num=2,
           freq_drop_width=8,
           freq_stripes_num=2)

        self.bn0 = nn.BatchNorm2d(mel_bins)
        
        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        
        input_dim = 512
        
        self.pooling = pooling

        if encoder_type == "Transformer":
            self.encoder_kwargs = {
            'adim': 144,
            'aheads': 4,
            'dropout_rate': 0.1,
            'elayers': 3,
            'eunits': 576,
            'kernel_size': 7,
            'transformer_input_layer': 'conv2d',
            'transformer_attn_dropout_rate': 0.0,
            'after_conv': False}
            adim = self.encoder_kwargs["adim"]
            self.encoder = TransformerEncoder(input_dim, **self.encoder_kwargs)
        elif encoder_type == "Conformer":
            self.encoder_kwargs = {
            'adim': 144,
            'aheads': 4,
            'dropout_rate': 0.1,
            'elayers': 3,
            'eunits': 576,
            'kernel_size': 7}
            adim = self.encoder_kwargs["adim"]
            self.encoder = ConformerEncoder(input_dim, **self.encoder_kwargs)
        else:
            raise ValueError("Choose encoder_type in ['Transformer', 'Conformer']")
        
        self.fc = nn.Linear(144, classes_num, bias=True)
        
        self.classifier = torch.nn.Linear(adim, classes_num)

        if self.pooling == "attention":
            self.dense = torch.nn.Linear(adim, classes_num)
            self.sigmoid = torch.sigmoid
            self.softmax = torch.nn.Softmax(dim=-1)

        elif self.pooling == "token":
            self.linear_emb = torch.nn.Linear(1, input_dim)

        self.reset_parameters(layer_init)
       
    def preprocess(self, input, mixup_lambda=None, timeshift=False, spec_augment=True):
        # t1 = time.time()
        x = self.spectrogram_extractor(input)  # (batch_size, 1, time_steps, freq_bins)
        x = self.logmel_extractor(x)  # (batch_size, 1, time_steps, mel_bins)

        frames_num = x.shape[2]

        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        if self.training and spec_augment:
            x = self.spec_augmenter(x)

        # Mixup on spectrogram
        if self.training and mixup_lambda is not None:
            if not timeshift:
                x = do_mixup(x, mixup_lambda)
            elif timeshift:
                x = do_mixup_timeshift(x, mixup_lambda)
                
        if self.training and mixup_lambda is None and timeshift:
            x = do_timeshift(x)
            
        return x, frames_num
    
    def forward(self, x, mixup_lambda=None, timeshift=False, spec_augment=True, mask=None):

        # input
        #x = torch.reshape(x, (x.size()[0], x.size()[-1]))
        
        interpolate_ratio = 8
        x, frames_num = self.preprocess(x, mixup_lambda=mixup_lambda, timeshift=timeshift, spec_augment=spec_augment)
        
        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block4(x, pool_size=(1, 1), pool_type='avg')
        
        x = torch.mean(x, dim=3)
        x = x.transpose(1, 2)   # (batch_size, time_steps, channels)
        x, _ = self.encoder(x, mask)
        x = x.transpose(1, 2)
        embedding = x   # (batch_size, feature_maps, time_steps)

        # Framewise output
        x = x.transpose(1, 2)
        framewise_output = torch.sigmoid(self.fc(x))
        framewise_output = interpolate(framewise_output, interpolate_ratio)
        if framewise_output.size()[1] != 1000:
            framewise_output = pad_framewise_output(framewise_output, roundup(framewise_output.size()[1]))
        
        # Clipwise output
        clipwise_output = torch.mean(framewise_output, dim=1)
        
        output_dict = {
            'framewise_output': framewise_output,
            'clipwise_output': clipwise_output,
            'embedding': embedding}
            
        return output_dict
#        if self.pooling == "attention":
#            strong = self.classifier(x)
#            sof = self.dense(x)  # [bs, frames, nclass]
#            sof = self.softmax(sof)
#            sof = torch.clamp(sof, min=1e-7, max=1)
#            weak = (torch.sigmoid(strong) * sof).sum(1) / sof.sum(1)  # [bs, nclass]
#            # Convert to logit to calculate loss with bcelosswithlogits
#            weak = torch.log(weak / (1 - weak))
#        elif self.pooling == "token":
#            x = self.classifier(x)
#            weak = x[:, 0, :]
#            strong = x[:, 1:, :]
#        elif self.pooling == "auto":
#            strong = self.classifier(x)
#            weak = self.autopool(strong)
            
#        return {"framewise_output": strong, "clipwise_output": weak}

    def reset_parameters(self, initialization: str = "pytorch"):
        if initialization.lower() == "pytorch":
            return
        # weight init
        for p in self.parameters():
            if p.dim() > 1:
                if initialization.lower() == "xavier_uniform":
                    torch.nn.init.xavier_uniform_(p.data)
                elif initialization.lower() == "xavier_normal":
                    torch.nn.init.xavier_normal_(p.data)
                elif initialization.lower() == "kaiming_uniform":
                    torch.nn.init.kaiming_uniform_(p.data, nonlinearity="relu")
                elif initialization.lower() == "kaiming_normal":
                    torch.nn.init.kaiming_normal_(p.data, nonlinearity="relu")
                else:
                    raise ValueError(f"Unknown initialization: {initialization}")
        # bias init
        for p in self.parameters():
            if p.dim() == 1:
                p.data.zero_()
        # reset some modules with default init
        for m in self.modules():
            if isinstance(m, (torch.nn.Embedding, LayerNorm)):
                m.reset_parameters()


class Cnn_14layers_Conformer_FrameAtt(torch.nn.Module):
    def __init__(
        self,
        sample_rate: int, window_size: int, hop_size: int,
        mel_bins: int, fmin: int, fmax: int, classes_num, feature_type,
        cnn_kwargs=None,
        encoder_kwargs=None,
        encoder_type="Conformer",
        pooling="token",
        layer_init="pytorch",
    ):
        super(Cnn_14layers_Conformer_FrameAtt, self).__init__()
        
        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None
        #self.interpolate_ratio = 32  # Downsampled ratio
        
        self.cnn_kwargs = {
            'activation': "Relu",
            'conv_dropout': 0.1,
            'kernel_size': [3, 3, 3, 3, 3, 3, 3],
            'padding': [1, 1, 1, 1, 1, 1, 1],
            'stride': [1, 1, 1, 1, 1, 1, 1],
            'nb_filters': [16, 32, 64, 128, 128, 128, 128],
            'pooling': [[2, 2], [2, 2], [2, 2], [1, 2], [1, 2], [1, 2], [1, 1]]}
            
        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(
           n_fft=window_size,
           hop_length=hop_size,
           win_length=window_size,
           window=window,
           center=center,
           pad_mode=pad_mode,
           freeze_parameters=True)

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(
           sr=sample_rate,
           n_fft=window_size,
           n_mels=mel_bins,
           fmin=fmin,
           fmax=fmax,
           ref=ref,
           amin=amin,
           top_db=top_db,
           freeze_parameters=True)

        # Spec augmenter
        self.spec_augmenter = SpecAugmentation(
           time_drop_width=64,
           time_stripes_num=2,
           freq_drop_width=8,
           freq_stripes_num=2)

        self.bn0 = nn.BatchNorm2d(mel_bins)
        
        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.conv_block5 = ConvBlock(in_channels=512, out_channels=1024)
        self.conv_block6 = ConvBlock(in_channels=1024, out_channels=2048)
        
        input_dim = 2048
        
        self.pooling = pooling

        if encoder_type == "Transformer":
            self.encoder_kwargs = {
            'adim': 144,
            'aheads': 4,
            'dropout_rate': 0.1,
            'elayers': 3,
            'eunits': 576,
            'kernel_size': 7,
            'transformer_input_layer': 'conv2d',
            'transformer_attn_dropout_rate': 0.0,
            'after_conv': False}
            adim = self.encoder_kwargs["adim"]
            self.encoder = TransformerEncoder(input_dim, **self.encoder_kwargs)
        elif encoder_type == "Conformer":
            self.encoder_kwargs = {
            'adim': 144,
            'aheads': 4,
            'dropout_rate': 0.1,
            'elayers': 3,
            'eunits': 576,
            'kernel_size': 7}
            adim = self.encoder_kwargs["adim"]
            self.encoder = ConformerEncoder(input_dim, **self.encoder_kwargs)
        else:
            raise ValueError("Choose encoder_type in ['Transformer', 'Conformer']")
        
        self.att_block = AttBlock(n_in=144, n_out=25, activation='sigmoid')
        
        self.classifier = torch.nn.Linear(adim, classes_num)

        if self.pooling == "attention":
            self.dense = torch.nn.Linear(adim, classes_num)
            self.sigmoid = torch.sigmoid
            self.softmax = torch.nn.Softmax(dim=-1)

        elif self.pooling == "token":
            self.linear_emb = torch.nn.Linear(1, input_dim)

        self.reset_parameters(layer_init)
       
    def preprocess(self, input, mixup_lambda=None, timeshift=False, spec_augment=True):
        # t1 = time.time()
        x = self.spectrogram_extractor(input)  # (batch_size, 1, time_steps, freq_bins)
        x = self.logmel_extractor(x)  # (batch_size, 1, time_steps, mel_bins)

        frames_num = x.shape[2]

        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        if self.training and spec_augment:
            x = self.spec_augmenter(x)

        # Mixup on spectrogram
        if self.training and mixup_lambda is not None:
            if not timeshift:
                x = do_mixup(x, mixup_lambda)
            elif timeshift:
                x = do_mixup_timeshift(x, mixup_lambda)
                
        if self.training and mixup_lambda is None and timeshift:
            x = do_timeshift(x)
            
        return x, frames_num
    
    def forward(self, x, mixup_lambda=None, timeshift=False, spec_augment=True, mask=None):

        # input
        #x = torch.reshape(x, (x.size()[0], x.size()[-1]))
        
        interpolate_ratio = 8
        x, frames_num = self.preprocess(x, mixup_lambda=mixup_lambda, timeshift=timeshift, spec_augment=spec_augment)
        
        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block5(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block6(x, pool_size=(1, 1), pool_type='avg')
        
        x = torch.mean(x, dim=3)
        x = x.transpose(1, 2)   # (batch_size, time_steps, channels)
        x, _ = self.encoder(x, mask)
        x = x.transpose(1, 2)
        embedding = x   # (batch_size, feature_maps, time_steps)

        (clipwise_output, norm_att, cla) = self.att_block(x)
        """cla: (batch_size, classes_num, time_stpes)"""

        # Framewise output
        framewise_output = cla.transpose(1, 2)
        interpolate_ratio = 1000 // framewise_output.size()[1]
        framewise_output = interpolate(framewise_output, interpolate_ratio)
        if framewise_output.size()[1] != 1000:
            framewise_output = pad_framewise_output(framewise_output, roundup(framewise_output.size()[1]))
        
        output_dict = {
            'framewise_output': framewise_output,
            'clipwise_output': clipwise_output,
            'embedding': embedding}
            
        return output_dict

    def reset_parameters(self, initialization: str = "pytorch"):
        if initialization.lower() == "pytorch":
            return
        # weight init
        for p in self.parameters():
            if p.dim() > 1:
                if initialization.lower() == "xavier_uniform":
                    torch.nn.init.xavier_uniform_(p.data)
                elif initialization.lower() == "xavier_normal":
                    torch.nn.init.xavier_normal_(p.data)
                elif initialization.lower() == "kaiming_uniform":
                    torch.nn.init.kaiming_uniform_(p.data, nonlinearity="relu")
                elif initialization.lower() == "kaiming_normal":
                    torch.nn.init.kaiming_normal_(p.data, nonlinearity="relu")
                else:
                    raise ValueError(f"Unknown initialization: {initialization}")
        # bias init
        for p in self.parameters():
            if p.dim() == 1:
                p.data.zero_()
        # reset some modules with default init
        for m in self.modules():
            if isinstance(m, (torch.nn.Embedding, LayerNorm)):
                m.reset_parameters()


class Cnn_7layers_Conformer(torch.nn.Module):
    def __init__(
        self,
        sample_rate: int, window_size: int, hop_size: int,
        mel_bins: int, fmin: int, fmax: int, classes_num,
        cnn_kwargs=None,
        encoder_kwargs=None,
        encoder_type="Conformer",
        pooling="token",
        layer_init="pytorch",
    ):
        super(Cnn_7layers_Conformer, self).__init__()
        
        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None
        #self.interpolate_ratio = 32  # Downsampled ratio
        
        self.cnn_kwargs = {
            'activation': "Relu",
            'conv_dropout': 0.1,
            'kernel_size': [3, 3, 3, 3, 3, 3, 3],
            'padding': [1, 1, 1, 1, 1, 1, 1],
            'stride': [1, 1, 1, 1, 1, 1, 1],
            'nb_filters': [16, 32, 64, 128, 128, 128, 128],
            'pooling': [[2, 2], [2, 2], [2, 2], [1, 2], [1, 2], [1, 2], [1, 1]]}
            
        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(
           n_fft=window_size,
           hop_length=hop_size,
           win_length=window_size,
           window=window,
           center=center,
           pad_mode=pad_mode,
           freeze_parameters=True)

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(
           sr=sample_rate,
           n_fft=window_size,
           n_mels=mel_bins,
           fmin=fmin,
           fmax=fmax,
           ref=ref,
           amin=amin,
           top_db=top_db,
           freeze_parameters=True)

        # Spec augmenter
        self.spec_augmenter = SpecAugmentation(
           time_drop_width=64,
           time_stripes_num=2,
           freq_drop_width=8,
           freq_stripes_num=2)

        self.bn0 = nn.BatchNorm2d(mel_bins)

        self.cnn = CNN(n_in_channel=1, **self.cnn_kwargs)
        input_dim = self.cnn.nb_filters[-1]
        
        self.pooling = pooling

        if encoder_type == "Transformer":
            self.encoder_kwargs = {
            'adim': 144,
            'aheads': 4,
            'dropout_rate': 0.1,
            'elayers': 3,
            'eunits': 576,
            'kernel_size': 7,
            'transformer_input_layer': 'conv2d',
            'transformer_attn_dropout_rate': 0.0,
            'after_conv': False}
            adim = self.encoder_kwargs["adim"]
            self.encoder = TransformerEncoder(input_dim, **self.encoder_kwargs)
        elif encoder_type == "Conformer":
            self.encoder_kwargs = {
            'adim': 144,
            'aheads': 4,
            'dropout_rate': 0.1,
            'elayers': 3,
            'eunits': 576,
            'kernel_size': 7}
            adim = self.encoder_kwargs["adim"]
            self.encoder = ConformerEncoder(input_dim, **self.encoder_kwargs)
        else:
            raise ValueError("Choose encoder_type in ['Transformer', 'Conformer']")

        self.classifier = torch.nn.Linear(adim, classes_num)

        if self.pooling == "attention":
            self.dense = torch.nn.Linear(adim, classes_num)
            self.sigmoid = torch.sigmoid
            self.softmax = torch.nn.Softmax(dim=-1)

        elif self.pooling == "token":
            self.linear_emb = torch.nn.Linear(1, input_dim)

        self.reset_parameters(layer_init)
       
    def preprocess(self, input, mixup_lambda=None, timeshift=False):
        # t1 = time.time()
        x = self.spectrogram_extractor(input)  # (batch_size, 1, time_steps, freq_bins)
        x = self.logmel_extractor(x)  # (batch_size, 1, time_steps, mel_bins)

        frames_num = x.shape[2]

        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

#        if self.training:
#            x = self.spec_augmenter(x)
       
        # Mixup on spectrogram
        if self.training and mixup_lambda is not None and not timeshift:
            x = do_mixup(x, mixup_lambda)
            
        if self.training and mixup_lambda is not None and timeshift:
            x = do_mixup_timeshift(x, mixup_lambda)
        return x, frames_num
    
    def forward(self, x, mixup_lambda=None, timeshift=False, spec_augment=True, mask=None):

        # input
        #x = torch.reshape(x, (x.size()[0], x.size()[-1]))
        
        x, frames_num = self.preprocess(x, mixup_lambda=mixup_lambda, timeshift=timeshift, spec_augment=spec_augment)
        x = self.cnn(x)
        x = x.squeeze(-1).permute(0, 2, 1)

        if self.pooling == "token":
            tag_token = self.linear_emb(torch.ones(x.size(0), 1, 1).cuda())
            x = torch.cat([tag_token, x], dim=1)
        
        x, _ = self.encoder(x, mask)

        if self.pooling == "attention":
            strong = self.classifier(x)
            sof = self.dense(x)  # [bs, frames, nclass]
            sof = self.softmax(sof)
            sof = torch.clamp(sof, min=1e-7, max=1)
            weak = (torch.sigmoid(strong) * sof).sum(1) / sof.sum(1)  # [bs, nclass]
            # Convert to logit to calculate loss with bcelosswithlogits
            weak = torch.log(weak / (1 - weak))
        elif self.pooling == "token":
            x = self.classifier(x)
            weak = x[:, 0, :]
            strong = x[:, 1:, :]
        elif self.pooling == "auto":
            strong = self.classifier(x)
            weak = self.autopool(strong)
            
        interpolate_ratio = 8
        strong = interpolate(strong, interpolate_ratio)
        #strong = pad_framewise_output(strong, 1000)
#        print('FRAMEWISE:', strong.size())
#        print('CLIPWISE:', weak.size(), weak)
        return {"framewise_output": strong, "clipwise_output": weak}

    def reset_parameters(self, initialization: str = "pytorch"):
        if initialization.lower() == "pytorch":
            return
        # weight init
        for p in self.parameters():
            if p.dim() > 1:
                if initialization.lower() == "xavier_uniform":
                    torch.nn.init.xavier_uniform_(p.data)
                elif initialization.lower() == "xavier_normal":
                    torch.nn.init.xavier_normal_(p.data)
                elif initialization.lower() == "kaiming_uniform":
                    torch.nn.init.kaiming_uniform_(p.data, nonlinearity="relu")
                elif initialization.lower() == "kaiming_normal":
                    torch.nn.init.kaiming_normal_(p.data, nonlinearity="relu")
                else:
                    raise ValueError(f"Unknown initialization: {initialization}")
        # bias init
        for p in self.parameters():
            if p.dim() == 1:
                p.data.zero_()
        # reset some modules with default init
        for m in self.modules():
            if isinstance(m, (torch.nn.Embedding, LayerNorm)):
                m.reset_parameters()


class Cnn_9layers_Conformer(torch.nn.Module):
    def __init__(
        self,
        sample_rate: int, window_size: int, hop_size: int,
        mel_bins: int, fmin: int, fmax: int, classes_num,
        cnn_kwargs=None,
        encoder_kwargs=None,
        encoder_type="Conformer",
        pooling="token",
        layer_init="pytorch",
    ):
        super(Cnn_9layers_Conformer, self).__init__()
        
        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None
        #self.interpolate_ratio = 32  # Downsampled ratio
        
        self.cnn_kwargs = {
            'activation': "Relu",
            'conv_dropout': 0.1,
            'kernel_size': [3, 3, 3, 3, 3, 3, 3],
            'padding': [1, 1, 1, 1, 1, 1, 1],
            'stride': [1, 1, 1, 1, 1, 1, 1],
            'nb_filters': [16, 32, 64, 128, 128, 128, 128],
            'pooling': [[2, 2], [2, 2], [2, 2], [1, 2], [1, 2], [1, 2], [1, 1]]}
            
        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(
           n_fft=window_size,
           hop_length=hop_size,
           win_length=window_size,
           window=window,
           center=center,
           pad_mode=pad_mode,
           freeze_parameters=True)

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(
           sr=sample_rate,
           n_fft=window_size,
           n_mels=mel_bins,
           fmin=fmin,
           fmax=fmax,
           ref=ref,
           amin=amin,
           top_db=top_db,
           freeze_parameters=True)

        # Spec augmenter
        self.spec_augmenter = SpecAugmentation(
           time_drop_width=64,
           time_stripes_num=2,
           freq_drop_width=8,
           freq_stripes_num=2)

        self.bn0 = nn.BatchNorm2d(mel_bins)
        
        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        
        input_dim = 512
        
        self.pooling = pooling

        if encoder_type == "Transformer":
            self.encoder_kwargs = {
            'adim': 144,
            'aheads': 4,
            'dropout_rate': 0.1,
            'elayers': 3,
            'eunits': 576,
            'kernel_size': 7,
            'transformer_input_layer': 'conv2d',
            'transformer_attn_dropout_rate': 0.0,
            'after_conv': False}
            adim = self.encoder_kwargs["adim"]
            self.encoder = TransformerEncoder(input_dim, **self.encoder_kwargs)
        elif encoder_type == "Conformer":
            self.encoder_kwargs = {
            'adim': 144,
            'aheads': 4,
            'dropout_rate': 0.1,
            'elayers': 3,
            'eunits': 576,
            'kernel_size': 7}
            adim = self.encoder_kwargs["adim"]
            self.encoder = ConformerEncoder(input_dim, **self.encoder_kwargs)
        else:
            raise ValueError("Choose encoder_type in ['Transformer', 'Conformer']")

        self.classifier = torch.nn.Linear(adim, classes_num)

        if self.pooling == "attention":
            self.dense = torch.nn.Linear(adim, classes_num)
            self.sigmoid = torch.sigmoid
            self.softmax = torch.nn.Softmax(dim=-1)

        elif self.pooling == "token":
            self.linear_emb = torch.nn.Linear(1, input_dim)

        self.reset_parameters(layer_init)
       
    def preprocess(self, input, mixup_lambda=None, timeshift=False):
        # t1 = time.time()
        x = self.spectrogram_extractor(input)  # (batch_size, 1, time_steps, freq_bins)
        x = self.logmel_extractor(x)  # (batch_size, 1, time_steps, mel_bins)

        frames_num = x.shape[2]

        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        if self.training:
            x = self.spec_augmenter(x)
       
        # Mixup on spectrogram
        if self.training and mixup_lambda is not None and not timeshift:
            x = do_mixup(x, mixup_lambda)
            
        if self.training and mixup_lambda is not None and timeshift:
            x = do_mixup_timeshift(x, mixup_lambda)
        return x, frames_num
    
    def forward(self, x, mixup_lambda=None, timeshift=False, mask=None):

        # input
        #x = torch.reshape(x, (x.size()[0], x.size()[-1]))
        
        x, frames_num = self.preprocess(x, mixup_lambda=mixup_lambda, timeshift=timeshift)
        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block4(x, pool_size=(1, 1), pool_type='avg')
        x = torch.reshape(x, (x.size()[0], x.size()[1], x.size()[2]*x.size()[3]))
        x = x.squeeze(-1).permute(0, 2, 1)
        
        if self.pooling == "token":
            tag_token = self.linear_emb(torch.ones(x.size(0), 1, 1).cuda())
            x = torch.cat([tag_token, x], dim=1)

        x, _ = self.encoder(x, mask)

        if self.pooling == "attention":
            strong = self.classifier(x)
            sof = self.dense(x)  # [bs, frames, nclass]
            sof = self.softmax(sof)
            sof = torch.clamp(sof, min=1e-7, max=1)
            weak = (torch.sigmoid(strong) * sof).sum(1) / sof.sum(1)  # [bs, nclass]
            # Convert to logit to calculate loss with bcelosswithlogits
            weak = torch.log(weak / (1 - weak))
        elif self.pooling == "token":
            x = self.classifier(x)
            weak = x[:, 0, :]
            strong = x[:, 1:, :]
        elif self.pooling == "auto":
            strong = self.classifier(x)
            weak = self.autopool(strong)
            
        #interpolate_ratio = 8
        #strong = interpolate(strong, interpolate_ratio)
        #strong = pad_framewise_output(strong, 1000)
#        print('FRAMEWISE:', strong.size())
#        print('CLIPWISE:', weak.size(), weak)
        return {"framewise_output": strong, "clipwise_output": weak}

    def reset_parameters(self, initialization: str = "pytorch"):
        if initialization.lower() == "pytorch":
            return
        # weight init
        for p in self.parameters():
            if p.dim() > 1:
                if initialization.lower() == "xavier_uniform":
                    torch.nn.init.xavier_uniform_(p.data)
                elif initialization.lower() == "xavier_normal":
                    torch.nn.init.xavier_normal_(p.data)
                elif initialization.lower() == "kaiming_uniform":
                    torch.nn.init.kaiming_uniform_(p.data, nonlinearity="relu")
                elif initialization.lower() == "kaiming_normal":
                    torch.nn.init.kaiming_normal_(p.data, nonlinearity="relu")
                else:
                    raise ValueError(f"Unknown initialization: {initialization}")
        # bias init
        for p in self.parameters():
            if p.dim() == 1:
                p.data.zero_()
        # reset some modules with default init
        for m in self.modules():
            if isinstance(m, (torch.nn.Embedding, LayerNorm)):
                m.reset_parameters()


# Transfer Learning Models

class VGGish(nn.Module):
    """
    PyTorch implementation of the VGGish model.
    Adapted from: https://github.com/harritaylor/torch-vggish
    The following modifications were made: (i) correction for the missing ReLU layers, (ii) correction for the
    improperly formatted data when transitioning from NHWC --> NCHW in the fully-connected layers, and (iii)
    correction for flattening in the fully-connected layers.
    """

    def __init__(self):
        super(VGGish, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(128, 256, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(256, 512, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2)
        )
        self.fc = nn.Sequential(
            nn.Linear(512 * 24, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 128),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        print(x.size())
        x = self.features(x).permute(0, 2, 3, 1).contiguous()
        print(x.size())
        x = x.view(x.size(0), -1)
        print(x.size())
        x = self.fc(x)
        return x

def frame(data, window_length, hop_length):
    num_samples = data.size()[2]
    print('NUM_SAMPLES', num_samples)
    num_frames = 1 + int(np.floor((num_samples - window_length) / hop_length))
    print('NUM_FRAMES', num_frames)
    shape = (data.size()[0], 1, num_frames * window_length) + data.shape[3:]
    print('SHAPE', shape)
    strides = (data.stride()[2] * hop_length,) + data.stride()[1:]
    print('STRIDES', strides)
    waves = torch.as_strided(data, size=shape, stride=strides)
    print(waves.size())
    
    return waves


class VGGish_FrameAtt(nn.Module):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin,
        fmax, classes_num, feature_type, checkpoint_path, freeze=False):
        
        super(VGGish_FrameAtt, self).__init__()

        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None
        self.checkpoint_path = checkpoint_path
        self.freeze = freeze
        feature_sample_rate = 1.0 / (hop_size/sample_rate)
        self.example_window_length = int(round(0.96 * feature_sample_rate))
        self.example_hop_length = int(round(0.96 * feature_sample_rate))
        
        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(n_fft=window_size, hop_length=hop_size,
            win_length=window_size, window=window, center=center, pad_mode=pad_mode,
            freeze_parameters=True)

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=window_size,
            n_mels=mel_bins, fmin=fmin, fmax=fmax, ref=ref, amin=amin, top_db=top_db,
            freeze_parameters=True)
        
        # Spec augmenter
        self.spec_augmenter = SpecAugmentation(time_drop_width=64, time_stripes_num=2,
            freq_drop_width=8, freq_stripes_num=2)

        self.bn0 = nn.BatchNorm2d(64)
        
        self.gru = nn.GRU(input_size=512, hidden_size=256, num_layers=1,
            bias=True, batch_first=True, bidirectional=True)
        
        self.vggish = VGGish()

        self.att_block = AttBlock(n_in=512, n_out=25, activation='sigmoid')

        self.init_weights()
    
    def init_weights(self):
        init_bn(self.bn0)
        checkpoint = torch.load(self.checkpoint_path)
        self.vggish.load_state_dict(checkpoint)
        # Remove last layers
        self.vggish = nn.Sequential(*list(self.vggish.children())[:-1])
        # Freeze layers
        if self.freeze:
            for param in self.vggish.parameters():
                param.requires_grad = False
        #print(self.vggish)

    def forward(self, input, mixup_lambda=None, timeshift=False, spec_augment=True):
        """Input: (batch_size, times_steps, freq_bins)"""
        
        interpolate_ratio = 12
        
        x = self.spectrogram_extractor(input)   # (batch_size, 1, time_steps, freq_bins)
        x = self.logmel_extractor(x)    # (batch_size, 1, time_steps, mel_bins)
        
#        x = x.transpose(1, 3)
#        x = self.bn0(x)
#        x = x.transpose(1, 3)
        
        if self.training and spec_augment:
            x = self.spec_augmenter(x)
        
        # Mixup on spectrogram
        if self.training and mixup_lambda is not None:
            if not timeshift:
                x = do_mixup(x, mixup_lambda)
            elif timeshift:
                x = do_mixup_timeshift(x, mixup_lambda)
                
        if self.training and mixup_lambda is None and timeshift:
            x = do_timeshift(x)
            
        x = self.vggish(x)
        x = torch.mean(x, dim=3)
#        x = x.transpose(1, 2)   # (batch_size, time_steps, channels)
#        (x, _) = self.gru(x)
#        x = x.transpose(1, 2)

        (clipwise_output, norm_att, cla) = self.att_block(x)
        """cla: (batch_size, classes_num, time_stpes)"""

        # Framewise output
        framewise_output = cla.transpose(1, 2)
        framewise_output = interpolate(framewise_output, interpolate_ratio)
        framewise_output = pad_framewise_output(framewise_output, 1000)
        
        output_dict = {
            'framewise_output': framewise_output,
            'clipwise_output': clipwise_output,
            'embedding': cla}
            
        return output_dict


class VGGish_Gru_FrameAtt(nn.Module):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin,
        fmax, classes_num, feature_type, checkpoint_path, freeze=False):
        
        super(VGGish_Gru_FrameAtt, self).__init__()

        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None
        self.checkpoint_path = checkpoint_path
        self.freeze = freeze
        feature_sample_rate = 1.0 / (hop_size/sample_rate)
        self.example_window_length = int(round(0.96 * feature_sample_rate))
        self.example_hop_length = int(round(0.96 * feature_sample_rate))
        
        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(n_fft=window_size, hop_length=hop_size,
            win_length=window_size, window=window, center=center, pad_mode=pad_mode,
            freeze_parameters=True)

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=window_size,
            n_mels=mel_bins, fmin=fmin, fmax=fmax, ref=ref, amin=amin, top_db=top_db,
            freeze_parameters=True)
        
        # Spec augmenter
        self.spec_augmenter = SpecAugmentation(time_drop_width=64, time_stripes_num=2,
            freq_drop_width=8, freq_stripes_num=2)

        self.bn0 = nn.BatchNorm2d(64)
        
        self.gru = nn.GRU(input_size=512, hidden_size=256, num_layers=1,
            bias=True, batch_first=True, bidirectional=True)
        
        self.vggish = VGGish()

        self.att_block = AttBlock(n_in=512, n_out=25, activation='sigmoid')

        self.init_weights()
    
    def init_weights(self):
        init_bn(self.bn0)
        checkpoint = torch.load(self.checkpoint_path)
        self.vggish.load_state_dict(checkpoint)
        # Remove last layers
        self.vggish = nn.Sequential(*list(self.vggish.children())[:-1])
        # Freeze layers
        if self.freeze:
            for param in self.vggish.parameters():
                param.requires_grad = False
        #print(self.vggish)

    def forward(self, input, mixup_lambda=None, timeshift=False, spec_augment=True):
        """Input: (batch_size, times_steps, freq_bins)"""
        
        interpolate_ratio = 12
        
        x = self.spectrogram_extractor(input)   # (batch_size, 1, time_steps, freq_bins)
        x = self.logmel_extractor(x)    # (batch_size, 1, time_steps, mel_bins)
        
#        x = x.transpose(1, 3)
#        x = self.bn0(x)
#        x = x.transpose(1, 3)
        
        if self.training and spec_augment:
            x = self.spec_augmenter(x)
        
        # Mixup on spectrogram
        if self.training and mixup_lambda is not None:
            if not timeshift:
                x = do_mixup(x, mixup_lambda)
            elif timeshift:
                x = do_mixup_timeshift(x, mixup_lambda)
                
        if self.training and mixup_lambda is None and timeshift:
            x = do_timeshift(x)
            
        x = self.vggish(x)
        x = torch.mean(x, dim=3)
        x = x.transpose(1, 2)   # (batch_size, time_steps, channels)
        (x, _) = self.gru(x)
        x = x.transpose(1, 2)

        (clipwise_output, norm_att, cla) = self.att_block(x)
        """cla: (batch_size, classes_num, time_stpes)"""

        # Framewise output
        framewise_output = cla.transpose(1, 2)
        framewise_output = interpolate(framewise_output, interpolate_ratio)
        framewise_output = pad_framewise_output(framewise_output, 1000)
        
        output_dict = {
            'framewise_output': framewise_output,
            'clipwise_output': clipwise_output,
            'embedding': cla}
            
        return output_dict

class VGGish_FrameAvg(nn.Module):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin,
        fmax, classes_num, feature_type, checkpoint_path, freeze=False):
        
        super(VGGish_FrameAvg, self).__init__()

        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None
        self.checkpoint_path = checkpoint_path
        self.freeze = freeze
        feature_sample_rate = 1.0 / (hop_size/sample_rate)
        self.example_window_length = int(round(0.96 * feature_sample_rate))
        self.example_hop_length = int(round(0.96 * feature_sample_rate))
        
        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(n_fft=window_size, hop_length=hop_size,
            win_length=window_size, window=window, center=center, pad_mode=pad_mode,
            freeze_parameters=True)

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=window_size,
            n_mels=mel_bins, fmin=fmin, fmax=fmax, ref=ref, amin=amin, top_db=top_db,
            freeze_parameters=True)
        
        # Spec augmenter
        self.spec_augmenter = SpecAugmentation(time_drop_width=64, time_stripes_num=2,
            freq_drop_width=8, freq_stripes_num=2)

        self.bn0 = nn.BatchNorm2d(64)
        
        self.gru = nn.GRU(input_size=512, hidden_size=256, num_layers=1,
            bias=True, batch_first=True, bidirectional=True)
        
        self.vggish = VGGish()

        self.fc = nn.Linear(512, 25, bias=True)

        self.init_weights()
    
    def init_weights(self):
        init_bn(self.bn0)
        checkpoint = torch.load(self.checkpoint_path)
        self.vggish.load_state_dict(checkpoint)
        # Remove last layers
        self.vggish = nn.Sequential(*list(self.vggish.children())[:-1])
        # Freeze layers
        if self.freeze:
            for param in self.vggish.parameters():
                param.requires_grad = False
        #print(self.vggish)

    def forward(self, input, mixup_lambda=None, timeshift=False, spec_augment=True):
        """Input: (batch_size, times_steps, freq_bins)"""
        
        #interpolate_ratio = 12
        
        x = self.spectrogram_extractor(input)   # (batch_size, 1, time_steps, freq_bins)
        x = self.logmel_extractor(x)    # (batch_size, 1, time_steps, mel_bins)
        
#        x = x.transpose(1, 3)
#        x = self.bn0(x)
#        x = x.transpose(1, 3)
        
        if self.training and spec_augment:
            x = self.spec_augmenter(x)
        
        # Mixup on spectrogram
        if self.training and mixup_lambda is not None:
            if not timeshift:
                x = do_mixup(x, mixup_lambda)
            elif timeshift:
                x = do_mixup_timeshift(x, mixup_lambda)
                
        if self.training and mixup_lambda is None and timeshift:
            x = do_timeshift(x)
            
        x = self.vggish(x)
        x = torch.mean(x, dim=3)
#        x = x.transpose(1, 2)   # (batch_size, time_steps, channels)
#        (x, _) = self.gru(x)
#        x = x.transpose(1, 2)

        embedding = x   # (batch_size, feature_maps, time_steps)

        # Framewise output
        x = x.transpose(1, 2)
        framewise_output = torch.sigmoid(self.fc(x))
        interpolate_ratio = 1000 // framewise_output.size()[1]
        framewise_output = interpolate(framewise_output, interpolate_ratio)
        if framewise_output.size()[1] != 1000:
            framewise_output = pad_framewise_output(framewise_output, roundup(framewise_output.size()[1]))
        framewise_output = pad_framewise_output(framewise_output, 1000)
        
        # Clipwise output
        clipwise_output = torch.mean(framewise_output, dim=1)
        
        output_dict = {
            'framewise_output': framewise_output,
            'clipwise_output': clipwise_output,
            'embedding': embedding}
            
        return output_dict


#class VGGish_FrameAtt(nn.Module):
#    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin,
#        fmax, classes_num, feature_type):
#
#        super(VGGish_FrameAtt, self).__init__()
#
#        window = 'hann'
#        center = True
#        pad_mode = 'reflect'
#        ref = 1.0
#        amin = 1e-10
#        top_db = None
#        self.checkpoint_path = '../../../../../storage/leey0204/fsd50k_audioset/audioset/checkpoints/vggish/pytorch_vggish.pth'
#
#        # Spectrogram extractor
#        self.spectrogram_extractor = Spectrogram(n_fft=window_size, hop_length=hop_size,
#            win_length=window_size, window=window, center=center, pad_mode=pad_mode,
#            freeze_parameters=True)
#
#        # Logmel feature extractor
#        self.logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=window_size,
#            n_mels=mel_bins, fmin=fmin, fmax=fmax, ref=ref, amin=amin, top_db=top_db,
#            freeze_parameters=True)
#
#        # Spec augmenter
#        self.spec_augmenter = SpecAugmentation(time_drop_width=64, time_stripes_num=2,
#            freq_drop_width=8, freq_stripes_num=2)
#
#        self.bn0 = nn.BatchNorm2d(64)
#
#        self.vggish = VGGish()
#
#        self.att_block = AttBlock(n_in=512, n_out=25, activation='sigmoid')
#
#        self.init_weights()
#
#    def init_weights(self):
#        init_bn(self.bn0)
#        checkpoint = torch.load(self.checkpoint_path)
#        self.vggish.load_state_dict(checkpoint)
#        # Remove last layers
#        self.vggish = nn.Sequential(*list(self.vggish.children())[:-1])
#        # Freeze layers
##        if self.freeze:
##            for param in self.vggish.parameters():
##                param.requires_grad = False
#        #print(self.vggish)
#
#    def forward(self, input, mixup_lambda=None, timeshift=False, spec_augment=True):
#        """Input: (batch_size, times_steps, freq_bins)"""
#
#        interpolate_ratio = 12
#
#        x = self.spectrogram_extractor(input)   # (batch_size, 1, time_steps, freq_bins)
#        x = self.logmel_extractor(x)    # (batch_size, 1, time_steps, mel_bins)
#
#        x = x.transpose(1, 3)
#        x = self.bn0(x)
#        x = x.transpose(1, 3)
#
#        if self.training and spec_augment:
#            x = self.spec_augmenter(x)
#
#        # Mixup on spectrogram
#        if self.training and mixup_lambda is not None and not timeshift:
#            x = do_mixup(x, mixup_lambda)
#
#        if self.training and mixup_lambda is not None and timeshift:
#            x = do_mixup_timeshift(x, mixup_lambda)
#
#        x = self.vggish(x)
#
#        x = torch.mean(x, dim=3)
#
#        (clipwise_output, norm_att, cla) = self.att_block(x)
#        """cla: (batch_size, classes_num, time_stpes)"""
#
#        # Framewise output
#        framewise_output = cla.transpose(1, 2)
#        framewise_output = interpolate(framewise_output, interpolate_ratio)
#        framewise_output = pad_framewise_output(framewise_output, 1000)
#
#        output_dict = {
#            'framewise_output': framewise_output,
#            'clipwise_output': clipwise_output,
#            'embedding': cla}
#
#        return output_dict


class Cnn14_DecisionLevelAtt(nn.Module):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin,
        fmax, classes_num):
        
        super(Cnn14_DecisionLevelAtt, self).__init__()

        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None
        #self.interpolate_ratio = 32     # Downsampled ratio

        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(n_fft=window_size, hop_length=hop_size,
            win_length=window_size, window=window, center=center, pad_mode=pad_mode,
            freeze_parameters=True)

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=window_size,
            n_mels=mel_bins, fmin=fmin, fmax=fmax, ref=ref, amin=amin, top_db=top_db,
            freeze_parameters=True)

        # Spec augmenter
        self.spec_augmenter = SpecAugmentation(time_drop_width=64, time_stripes_num=2,
            freq_drop_width=8, freq_stripes_num=2)

        self.bn0 = nn.BatchNorm2d(64)

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.conv_block5 = ConvBlock(in_channels=512, out_channels=1024)
        self.conv_block6 = ConvBlock(in_channels=1024, out_channels=2048)

        self.fc1 = nn.Linear(2048, 2048, bias=True)
        self.att_block = AttBlock(2048, classes_num, activation='sigmoid')
        
        self.init_weight()

    def init_weight(self):
        init_bn(self.bn0)
        init_layer(self.fc1)

    def forward(self, input, mixup_lambda=None):
        """
        Input: (batch_size, data_length)"""
        interpolate_ratio = 32
        x = self.spectrogram_extractor(input)   # (batch_size, 1, time_steps, freq_bins)
        x = self.logmel_extractor(x)    # (batch_size, 1, time_steps, mel_bins)

        frames_num = x.shape[2]-1
        
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        
        if self.training:
            x = self.spec_augmenter(x)

        # Mixup on spectrogram
        if self.training and mixup_lambda is not None:
            x = do_mixup(x, mixup_lambda)
        
        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block5(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block6(x, pool_size=(1, 1), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = torch.mean(x, dim=3)
        
        x1 = F.max_pool1d(x, kernel_size=3, stride=1, padding=1)
        x2 = F.avg_pool1d(x, kernel_size=3, stride=1, padding=1)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        x = x.transpose(1, 2)
        x = F.relu_(self.fc1(x))
        x = x.transpose(1, 2)
        x = F.dropout(x, p=0.5, training=self.training)
        (clipwise_output, _, segmentwise_output) = self.att_block(x)
        segmentwise_output = segmentwise_output.transpose(1, 2)
        
        # Get framewise output
        framewise_output = interpolate(segmentwise_output, interpolate_ratio)
        framewise_output = pad_framewise_output(framewise_output, frames_num)

        output_dict = {'framewise_output': framewise_output,
            'clipwise_output': clipwise_output}

        return output_dict


# Others

class Cnn_9layers_Gru_Reg(nn.Module):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin,
        fmax, classes_num, feature_type):
        
        super(Cnn_9layers_Gru_Reg, self).__init__()

        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None
        num_bins = 80
        self.feature_type = feature_type
        
        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(n_fft=window_size, hop_length=hop_size,
            win_length=window_size, window=window, center=center, pad_mode=pad_mode,
            freeze_parameters=True)

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=window_size,
            n_mels=mel_bins, fmin=fmin, fmax=fmax, ref=ref, amin=amin, top_db=top_db,
            freeze_parameters=True)
        
        # Spec augmenter
        self.spec_augmenter = SpecAugmentation(time_drop_width=64, time_stripes_num=2,
            freq_drop_width=8, freq_stripes_num=2)

        self.bn0 = nn.BatchNorm2d(64)

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)

        self.gru = nn.GRU(input_size=512, hidden_size=256, num_layers=1,
            bias=True, batch_first=True, bidirectional=True)

        self.att_block = AttBlock(n_in=512, n_out=25, activation='sigmoid')

        self.init_weights()

    def init_weights(self):
        init_bn(self.bn0)
        init_gru(self.gru)

    def forward(self, input, mixup_lambda=None, timeshift=False):
        """Input: (batch_size, times_steps, freq_bins)"""
        
        interpolate_ratio = 8
        
        if self.feature_type == 'logmel':
            x = self.spectrogram_extractor(input)   # (batch_size, 1, time_steps, freq_bins)
            x = self.logmel_extractor(x)    # (batch_size, 1, time_steps, mel_bins)
        elif self.feature_type == 'cqt':
            x = self.stft_extractor(input)
            x = self.cqt_extractor(x)
        elif self.feature_type == 'gamma':
            #x = self.gamma_extractor(input)
            x = torch.unsqueeze(input, 1)
            x = x.transpose(2,3)
            x = x.to('cuda')
        
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        
        if self.training:
            x = self.spec_augmenter(x)
        
        # Mixup on spectrogram
        if self.training and mixup_lambda is not None and not timeshift:
            x = do_mixup(x, mixup_lambda)
            
        if self.training and mixup_lambda is not None and timeshift:
            x = do_mixup_timeshift(x, mixup_lambda)
            
        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block4(x, pool_size=(1, 1), pool_type='avg')

        x = torch.mean(x, dim=3)
        x = x.transpose(1, 2)   # (batch_size, time_steps, channels)
        (x, _) = self.gru(x)
        x = x.transpose(1, 2)

        (clipwise_output, norm_att, cla) = self.att_block(x)
        """cla: (batch_size, classes_num, time_stpes)"""

        # Framewise output
        framewise_output = cla.transpose(1, 2)
        framewise_output = interpolate(framewise_output, interpolate_ratio)
        #framewise_output = pad_framewise_output(framewise_output, 1000)
        
        output_dict = {
            'framewise_output': framewise_output,
            'clipwise_output': clipwise_output,
            'embedding': cla}
            
        return output_dict
