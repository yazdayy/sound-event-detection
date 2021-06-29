# Sound Event Detection with Human and Emergency Sounds

## Dataset
The dataset is a subset of the AudioSet dataset, consisting of human and emergency sounds. There are 25,834 audio clips in the weakly-labelled train set, 2,975 clips in the strongly-labelled set, and 747 clips in the test set. There are a total of 25 classes, as listed below.

Human sounds:
1. Applause, 2. Breathing, 3. Chatter, 4. Child_speech_kid_speaking, 5. Cheering, 6. Clapping, 7. Conversation, 8. Cough, 9. Crowd, 10. Crying_sobbing, 11. Female_speech_woman_speaking, 12. Laughter, 13. Male_speech_man_speaking, 14. Run, 15. Screaming, 16. Shout, 17. Sneeze, 18. Walk_footsteps, 19. Whispering

Emergency sounds:
1. Air_horn_truck_horn, 2. Car_alarm, 3. Emergency_vehicle, 4. Explosion, 5. Gunshot_gunfire, 6. Siren

## Pre-trained models
The pre-trained models can be found in the 'checkpoints' directory.

## Run
Instructions to run or train the system can be found in run.sh.
Note:
- Please use main.py / --filename='main' if you would like to train or run the purely weakly-labelled system, and main_strong.py / --filename='main_strong' if you would like to train or run the combined weakly-labelled and strongly-labelled system. 

- Please change to --augmentation='mixup' if you would like to train or run the system with mixup applied only.

- Please upload the audio clips you would like to predict on in the 'long_predict' folder.

- The prediction output is saved in the long_predict_results directory in the following xml format:
    ![xml_output_example](https://user-images.githubusercontent.com/56859670/123733914-f5955800-d8ce-11eb-8c4b-11dd3c7de29b.png)


