# Sound Event Detection with Human and Emergency Sounds

## Dataset
The dataset is a subset of the AudioSet dataset, consisting of human and emergency sounds. There are 25,834 audio clips in the weakly-labelled train set, 2,380 clips in the strongly-labelled train set, 595 clips in the validation set and 747 clips in the test set. There are a total of 25 classes, as listed below.

Human sounds:
1. Applause, 2. Breathing, 3. Chatter, 4. Child_speech_kid_speaking, 5. Cheering, 6. Clapping, 7. Conversation, 8. Cough, 9. Crowd, 10. Crying_sobbing, 11. Female_speech_woman_speaking, 12. Laughter, 13. Male_speech_man_speaking, 14. Run, 15. Screaming, 16. Shout, 17. Sneeze, 18. Walk_footsteps, 19. Whispering

Emergency sounds:
1. Air_horn_truck_horn, 2. Car_alarm, 3. Emergency_vehicle, 4. Explosion, 5. Gunshot_gunfire, 6. Siren

If you are interested in downloading this dataset, please use youtube_dl to download the files indicated in metadata/training_set.csv and metadata/testing_set.csv, and store them in dataset/training and dataset/testing respectively:
```
python download_audioset.py --workspace=$WORKSPACE --data_type=training
```
```
python download_audioset.py --workspace=$WORKSPACE --data_type=testing
```

## Pre-trained Models
The pre-trained models can be found in the 'checkpoints' directory and stored according to the parameters they were trained on. 

For example, if the pre-trained model was trained with both strongly and weakly-labelled sets on the Cnn-9layers-Gru-FrameAtt model, with clipwise binary crossentropy loss and mixup applied, and with a batch size of 32, it can be found in the following directory:
```
checkpoints/main_strong/holdout_fold=1/model_type=Cnn_9layers_Gru_FrameAtt/loss_type=clip_bce/augmentation=mixup/batch_size=32
```

 The models are named best_<feature_type>_<audio_quality>.pth, where the feature type may either be 'logmel' or 'gammatone', and the audio_quality may be '8k', '16k' or '32k'.

### Performance of the Pre-trained Models
The pre-trained models were trained on both the weakly-labelled and strongly-labelled 16k train sets, and evaluated on the strongly-labelled test set. The table below shows the performance of the models:

| Model | Data Augmentation | Processing Type  | Threshold Type | Segment-based Error Rate | Segment-based F1-score |
| :---: | :---: | :---: | :---: | :---: | :---: |
| Cnn-9 + Gru + Attention | SpecAugment + Mixup | None | Non-optimised | 0.555 | 0.624 |
| Cnn-9 + Gru + Attention | SpecAugment + Mixup | None | Optimised | 0.601 | 0.635 |
| Cnn-9 + Gru + Attention | SpecAugment + Mixup | Frame-wise averaging | Non-optimised | 0.557 | 0.618 |
| Cnn-9 + Gru + Attention | SpecAugment + Mixup | Frame-wise averaging | Optimised | 0.574 | 0.639 |
| Cnn-9 + Transformer + Attention | SpecAugment + Mixup | None | Non-optimised | 0.552 | 0.628 |
| Cnn-9 + Transformer + Attention | SpecAugment + Mixup | None | Optimised | 0.561 | 0.648 |
| Cnn-9 + Transformer + Attention | SpecAugment + Mixup | Frame-wise averaging | Non-optimised | 0.549 | 0.622 |
| Cnn-9 + Transformer + Attention | SpecAugment + Mixup | Frame-wise averaging | Optimised | **0.550** | **0.647** |


If you would like to test the performance of the pre-trained model on the test set yourself, please follow the instructions in the **Dataset** section to download the test set and then run the following commands:
```
python pytorch/main_strong.py inference_prob_overlap --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE --holdout_fold=1 --model_type=$MODEL_TYPE --loss_type='clip_bce' --augmentation='mixup' --batch_size=32 --feature_type='logmel' --cuda --sed_thresholds --audio_16k
```

## Predicition System
Instructions (more details can be found in run.sh):

1. Upload the audio clips you would like to process to a folder
    - The path to this directory will be your $INPUT_DIR
    - If your audio files are not in .wav format, the prediction system will automatically convert them from their current format to .wav

2. Run the following command:
    ```
    python pytorch/predict.py predict --input_dir=$INPUT_DIR --workspace=$WORKSPACE --filename='main_strong' --holdout_fold 1 --model_type=$MODEL_TYPE --loss_type='clip_bce' --augmentation='mixup' --batch_size=32 --feature_type='logmel' --cuda --sample_duration=5 --overlap --overlap_value=1 --sed_thresholds
    ```
    
3. The prediction output is saved in the 'predict_results' directory in the following xml format:

    ![xml_output_example](https://user-images.githubusercontent.com/56859670/123733914-f5955800-d8ce-11eb-8c4b-11dd3c7de29b.png)

Note:
- Integration with Automatic Speech Recognition (ASR)
    - If you would like to run the system integrated with DeepSpeech ASR, please run the following command:
    ```
    python pytorch/predict.py predict_asr --input_dir=$INPUT_DIR --workspace=$WORKSPACE --filename='main_strong' --holdout_fold=1 --model_type=$MODEL_TYPE --loss_type='clip_bce' --augmentation='mixup' --batch_size=32 --feature_type='logmel' --cuda --sample_duration=5 --overlap --overlap_value=1 --sed_thresholds --language='eng'
    ```
    - In this system, ASR is activated whenever Male_speech_man_speaking, Female_speech_woman_speaking and Child_speech_kid_speaking events are detected.

- Weak vs Combined System
    - If you would like to run the purely weakly-labelled system, use --filename='main'.
    - If you would like to run the combined weakly-labelled and strongly-labelled system, use --filename='main_strong'.
    
- Data Augmentation Types
    - If you would like to run the system with only mixup applied, use --augmentation='mixup'.
    - If you would like to run the system with both timeshift and mixup applied, use --augmentation='timeshift_mixup'.
    - The following list indicates the possible augmentation techniques you may apply: 
    `['none', 'spec_augment', 'timeshift', 'mixup', 'timeshift_mixup', 'specaugment_timeshift_mixup', 'specaugment_mixup', 'specaugment_timeshift']`
    
- Audio Quality Selection
    - If you would like to run the system trained on the 8k or 16k dataset, add a '--audio_8k' or 'audio_16k' tag, respectively.
    - If you would like to run the system trained on the 32k dataset, no additional tag is required.

## Training and Evaluation
Instructions (more details can be found in run.sh):
- Do include '--audio_16k' or '--audio_8k' if you would like to train on the 16k or 8k dataset, respectively.
- No additional tag is needed if you would like to train on the 32k dataset.

1. Prepare data for training by packing the waveforms to hdf5
    
    Pack test set:
    ```
    python utils/features.py pack_audio_files_to_hdf5 --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE --feature_type='logmel' --data_type='testing' --audio_16k
    ```
    
    Pack validation set:
    ```
    python utils/features.py pack_audio_files_to_hdf5 --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE --feature_type='logmel' --data_type='strong_validation' --audio_16k
    ```
    
    If only doing weak training:
    ```
    python utils/features.py pack_audio_files_to_hdf5 --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE --feature_type='logmel' --data_type='training' --audio_16k
    ```
    If doing combined weak and strong training:
    ```
    python utils/features.py pack_audio_files_to_hdf5 --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE --feature_type='logmel' --data_type='weak_training' --audio_16k
    ```
    ```
    python utils/features.py pack_audio_files_to_hdf5 --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE --feature_type='logmel' --data_type='strong_training' --audio_16k
    ```
    
2. Commence training

    If only doing weak training:
    ```
    python pytorch/main.py train --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE --holdout_fold=1 --model_type=$MODEL_TYPE --loss_type='clip_bce' --augmentation='specaugment_mixup' --learning_rate=1e-3 --batch_size=32 --resume_iteration=0 --stop_iteration=50000 --feature_type='logmel' --cuda
    ```
    If doing combined weak and strong training:
    ```
    python pytorch/main_strong.py train --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE --holdout_fold=1 --model_type=$MODEL_TYPE --loss_type='clip_bce' --augmentation='specaugment_mixup' --learning_rate=1e-3 --batch_size=32 --resume_iteration=0 --stop_iteration=50000 --feature_type='logmel' --cuda --audio_16k
    ```

3. Optimize thresholds (optional but strongly recommended)
    
    ```
    python utils/optimize_thresholds.py optimize_sed_thresholds  --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE --filename='main_strong' --holdout_fold=1 --model_type=$MODEL_TYPE --loss_type='clip_bce' --augmentation='specaugment_mixup' --feature_type='logmel' --batch_size=32 --audio_16k
    ```

4. Evaluate and Calculate metrics:

    If using optimized thresholds:
    ```
    python pytorch/main_strong.py inference_prob_overlap --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE --holdout_fold=1 --model_type=$MODEL_TYPE --loss_type='clip_bce' --augmentation='specaugment_mixup' --batch_size=32 --feature_type='logmel' --cuda --audio_16k --sed_thresholds
    ```
    If not using optimized thresholds:
    ```
    python pytorch/main_strong.py inference_prob_overlap --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE --holdout_fold=1 --model_type=$MODEL_TYPE --loss_type='clip_bce' --augmentation='specaugment_mixup' --batch_size=32 --feature_type='logmel' --cuda --audio_16k
    ```

