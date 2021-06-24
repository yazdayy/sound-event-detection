import torch
import torch.nn.functional as F

def clip_bce(output_dict, target_dict):
    """Binary cross entropy loss.

    Args:
      output_dict: {'clipwise_output': (N, classes_num)}
      target_dict: {'target': (N, classes_num)}
    """
    return F.binary_cross_entropy(output_dict['clipwise_output'], target_dict['target'])

def clip_bce_logits(output_dict, target_dict):
    return F.binary_cross_entropy_with_logits(output_dict['clipwise_output'], target_dict['target'])

def frame_bce(output_dict, target_dict):
    '''Strongly labelled loss. The output and target have shape of:
    (batch_size, frames_num, classes_num)
    '''
    output = output_dict['framewise_output']
    target = target_dict['strong_target']

    # To let output and target to have the same time steps
    N = min(output.shape[1], target.shape[1])

    return F.binary_cross_entropy(
        output[:, 0 : N, :],
        target[:, 0 : N, :])

def get_loss_func(loss_type):
    if loss_type == 'clip_bce':
        return clip_bce
    
    elif loss_type == 'clip_bce_logits':
        return clip_bce_logits
        
    elif loss_type == 'frame_bce':
        return frame_bce
