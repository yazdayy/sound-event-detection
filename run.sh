#!/bin/bash
# You need to modify this path to your downloaded dataset directory
DATASET_DIR="--dataset_dir=../../../../../storage/leey0204/fsd50k_audioset/audioset/dataset"

# You need to modify this path to your workspace to store features and models
WORKSPACE="--dataset_dir=../../../../../storage/leey0204/fsd50k_audioset/audioset"

# Set model type
MODEL_TYPE="Cnn_9layers_Gru_FrameAtt"

# ------ Run pre-trained system ------
python pytorch/predict.py predict --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE --filename='main_strong' --holdout_fold 1 --model_type=$MODEL_TYPE --loss_type='clip_bce' --augmentation='mixup' --batch_size=32 --feature_type='logmel' --cuda --sample_duration=5 --overlap --overlap_value=1 --sed_thresholds

# ------ Preparation of data for training ------
# Pack waveforms to hdf5
python utils/features.py pack_audio_files_to_hdf5 --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE --feature_type='logmel' --data_type='testing'
# If only doing weak training
python utils/features.py pack_audio_files_to_hdf5 --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE --feature_type='logmel' --data_type='training'
# If doing combined weak and strong training
python utils/features.py pack_audio_files_to_hdf5 --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE --feature_type='logmel' --data_type='weak_training'
python utils/features.py pack_audio_files_to_hdf5 --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE --feature_type='logmel' --data_type='strong_training'

# ------ Train ------
# Weak training
python pytorch/main.py train --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE --holdout_fold=1 --model_type=$MODEL_TYPE --loss_type='clip_bce' --augmentation='mixup' --learning_rate=1e-3 --batch_size=32 --resume_iteration=0 --stop_iteration=50000 --feature_type='logmel' --cuda

# Combined weak and strong training
python pytorch/main_strong.py train --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE --holdout_fold=1 --model_type=$MODEL_TYPE --loss_type='clip_bce' --augmentation='mixup' --learning_rate=1e-3 --batch_size=32 --resume_iteration=0 --stop_iteration=50000 --feature_type='logmel' --cuda

# ------ Optimize thresholds ------
# Optimize sound event detection thresholds
python utils/optimize_thresholds.py optimize_sed_thresholds  --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE --filename='main_strong' --holdout_fold=1 --model_type=$MODEL_TYPE --loss_type='clip_bce' --augmentation='mixup' --feature_type='logmel' --batch_size=32

# ------ Evaluate and Calculate metrics ------
# If using optimized thresholds
python pytorch/main_strong.py inference_prob_overlap --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE --holdout_fold=1 --model_type=$MODEL_TYPE --loss_type='clip_bce' --augmentation='mixup' --batch_size=32 --feature_type='logmel' --cuda --sed_thresholds

# If not using optimized thresholds
python pytorch/main_strong.py inference_prob_overlap --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE --holdout_fold=1 --model_type=$MODEL_TYPE --loss_type='clip_bce' --augmentation='mixup' --batch_size=32 --feature_type='logmel' --cuda
