# Sound Event Detection

## Predicition System

1. Upload the audio clips you would like to process to a folder
    - The path to this directory will be your $INPUT_DIR
    - If your audio files are not in .wav format, the prediction system will automatically convert them from their current format to .wav

2. Run the following command:
    Original:
    ```
    python pytorch/predict_new.py predict_asr --dataset_dir=$INPUT_DIR --workspace=workspace --holdout_fold=1 --model_type=Cnn_9layers_Gru_FrameAtt --loss_type=clip_bce --augmentation=specaugment_mixup --feature_type=logmel --batch_size=8 --cuda --audio_16k --sed_thresholds --filename=main_strong --overlap
    ```
    With dynamic segmentation:
    ```
    python pytorch/predict_new.py predict_silent_asr --dataset_dir=$INPUT_DIR --workspace=workspace --holdout_fold=1 --model_type=Cnn_9layers_Gru_FrameAtt --loss_type=clip_bce --augmentation=specaugment_mixup --feature_type=logmel --batch_size=8 --cuda --audio_16k --sed_thresholds --filename=main_strong
    ```
    
3. The prediction output is saved in the 'workspace/predict_results' directory in the following xml format:

    ![xml_output_example](https://user-images.githubusercontent.com/56859670/123733914-f5955800-d8ce-11eb-8c4b-11dd3c7de29b.png)
