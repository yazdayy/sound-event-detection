# Sound Event Detection with Human and Emergency Sounds

## Dataset
The dataset is a subset of the AudioSet dataset, consisting of human and emergency sounds. There are 25,834 audio clips in the weakly-labelled train set, 2,975 clips in the strongly-labelled set, and 747 clips in the test set. There are a total of 25 classes, as listed below.

Human sounds:
1. Applause, 2. Breathing, 3. Chatter, 4. Child_speech_kid_speaking, 5. Cheering, 6. Clapping, 7. Conversation, 8. Cough, 9. Crowd, 10. Crying_sobbing, 11. Female_speech_woman_speaking, 12. Laughter, 13. Male_speech_man_speaking, 14. Run, 15. Screaming, 16. Shout, 17. Sneeze, 18. Walk_footsteps, 19. Whispering

Emergency sounds:
1. Air_horn_truck_horn, 2. Car_alarm, 3. Emergency_vehicle, 4. Explosion, 5. Gunshot_gunfire, 6. Siren

If you are interested in downloading this dataset, please use youtube_dl to download the files indicated in metadata/training_set.csv and metadata/testing_set.csv, and store them in dataset/training and dataset/testing respectively.

## Pre-trained Models
The pre-trained models can be found in the 'checkpoints' directory and stored according to the parameters they were trained on. 

For example, if the pre-trained model was trained with both strongly and weakly-labelled sets on the Cnn-9layers-Gru-FrameAtt model, with clipwise binary crossentropy loss and mixup applied, and with a batch size of 32, it can be found in the following directory:
```
checkpoints/main_strong/holdout_fold=1/model_type=Cnn_9layers_Gru_FrameAtt/loss_type=clip_bce/augmentation=mixup/batch_size=32
```

### Performance of the Pre-trained Models
The pre-trained models were trained on both the weakly-labelled and strongly-labelled train sets, and evaluated on the strongly-labelled test set. The table below shows the performance of the models:

| Model | Data Augmentation | Optimised Thresholds | Stride (s) | Segment Length (s) | Segment-based Error Rate | Segment-based F1-score |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Cnn-9 + Gru + Attention | Timeshift + Mixup | No | 0 | 10 | 0.584 | 0.591 |
| Cnn-9 + Gru + Attention | Mixup | No | 0 | 10 | 0.566 | 0.600 |
| Cnn-9 + Gru + Attention | Mixup | Yes | 0 | 10 | 0.557 | 0.609 |
| Cnn-9 + Gru + Attention | Mixup | Yes | 1 | 2 | 0.616 | 0.619 |
| Cnn-9 + Gru + Attention | Mixup | Yes | 1 | 3 | 0.593 | 0.628 |
| Cnn-9 + Gru + Attention | Mixup | Yes | 1 | 5 | 0.567 | 0.640 |
| Cnn-9 + Gru + Attention | Mixup | Yes | 0.5 | 6 | 0.560 | 0.644 |
| Cnn-9 + Gru + Attention | Mixup | Yes | 0.5 | 7 | 0.557 | 0.641 |

## Predicition System
Instructions (more details can be found in run.sh):

1. Upload the audio clips you would like to process in the 'long_predict' folder

2. Run the following command:
    ```
    python pytorch/predict.py predict --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE --filename='main_strong' --holdout_fold 1 --model_type=$MODEL_TYPE --loss_type='clip_bce' --augmentation='mixup' --batch_size=32 --feature_type='logmel' --cuda
    ```
    
3. The prediction output is saved in the 'long_predict_results' directory in the following xml format:
    ![xml_output_example](https://user-images.githubusercontent.com/56859670/123733914-f5955800-d8ce-11eb-8c4b-11dd3c7de29b.png)

Note:
- Integration with Automatic Speech Recognition (ASR)
    - If you would like to run the system integrated with DeepSpeech ASR, use predict_asr.py instead of predict.py
    - In this system, ASR is activated whenever Male_speech_man_speaking, Female_speech_woman_speaking and Child_speech_kid_speaking events are detected.

- Weak vs Combined System
    - If you would like to run the purely weakly-labelled system, use --filename='main'.
    - If you would like to run the combined weakly-labelled and strongly-labelled system, use --filename='main_strong'.
    
- Data Augmentation Types
    - If you would like to run the system with only mixup applied, use --augmentation='mixup'.
    - If you would like to run the system with both timeshift and mixup applied, use --augmentation='timeshift_mixup'.

## Training and Evaluation
Instructions (more details can be found in run.sh):

1. Prepare data for training by packing the waveforms to hdf5:
    ```
    python utils/features.py pack_audio_files_to_hdf5 --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE --feature_type='logmel' --data_type='testing'
    ```
    
    If only doing weak training:
    ```
    python utils/features.py pack_audio_files_to_hdf5 --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE --feature_type='logmel' --data_type='training'
    ```
    If doing combined weak and strong training:
    ```
    python utils/features.py pack_audio_files_to_hdf5 --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE --feature_type='logmel' --data_type='weak_training'
    ```
    ```
    python utils/features.py pack_audio_files_to_hdf5 --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE --feature_type='logmel' --data_type='strong_training'
    ```
    
2. Commence training

    If only doing weak training:
    ```
    python pytorch/main.py train --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE --holdout_fold=1 --model_type=$MODEL_TYPE --loss_type='clip_bce' --augmentation='mixup' --learning_rate=1e-3 --batch_size=32 --resume_iteration=0 --stop_iteration=50000 --feature_type='logmel' --cuda
    ```
    If doing combined weak and strong training:
    ```
    python pytorch/main_strong.py train --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE --holdout_fold=1 --model_type=$MODEL_TYPE --loss_type='clip_bce' --augmentation='mixup' --learning_rate=1e-3 --batch_size=32 --resume_iteration=0 --stop_iteration=50000 --feature_type='logmel' --cuda
    ```

3. Inference and dump predicted probabilities:
    ```
    python pytorch/main_strong.py inference_prob --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE --holdout_fold=1 --model_type=$MODEL_TYPE --loss_type='clip_bce' --augmentation='mixup' --batch_size=32 --feature_type='logmel' --cuda
    ```

4. Optimize thresholds (OPTIONAL)
    ```
    python utils/optimize_thresholds.py optimize_sed_thresholds  --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE --filename='main_strong' --holdout_fold=1 --model_type=$MODEL_TYPE --loss_type='clip_bce' --augmentation='mixup' --feature_type='logmel' --batch_size=32
    ```
    
5. Calculate metrics
    
    If not using optimized thresholds:
    ```
    python utils/calculate_metrics.py calculate_metrics --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE --filename='main_strong' --holdout_fold=1 --model_type=$MODEL_TYPE --loss_type='clip_bce' --augmentation='mixup' --batch_size=32 --feature_type='logmel' --data_type='test'
    ```
    
    If using optimized thresholds:
    ```
    python utils/calculate_metrics.py calculate_metrics --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE --filename='main_strong' --holdout_fold=1 --model_type=$MODEL_TYPE --loss_type='clip_bce' --augmentation='mixup' --batch_size=32 --feature_type='logmel' --data_type='test' --sed_thresholds
    ```

Note:
- Weak vs Combined System
    - If you would like to train a purely weakly-labelled system, use --filename='main'.
    - If you would like to train a combined weakly-labelled and strongly-labelled system, use --filename='main_strong'.
    
- Data Augmentation Types
    - If you would like to train a system with only mixup applied, use --augmentation='mixup'.
    - If you would like to train a system with both timeshift and mixup applied, use --augmentation='timeshift_mixup'.
