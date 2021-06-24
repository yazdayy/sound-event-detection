sample_rate = 32000
audio_duration = 10     # Audio clips have durations of 10 seconds
audio_samples = sample_rate * audio_duration

# Hyper-parameters follow [1] Kong, Q., Cao, Y., Iqbal, T., Wang, 
# Y., Wang, W. and Plumbley, M. D., 2019. PANNs: Large-Scale Pretrained Audio 
# Neural Networks for Audio Pattern Recognition. arXiv preprint arXiv:1912.10211.
mel_bins = 64
fmin = 50
fmax = 14000
window_size = 1024
hop_size = 320
frames_per_second = sample_rate // hop_size
window = 'hann'
pad_mode = 'reflect'
center = True
device = 'cuda'
ref = 1.0
amin = 1e-10
top_db = None

# ID of classes
ids = ['/m/028ght', '/m/0lyf6', '/m/07rkbfh', '/m/053hz1', '/m/0ytgt', '/m/0l15bq', '/m/01h8n0', '/m/01b_21', '/m/03qtwd', '/m/0463cq4', '/m/02zsn', '/m/01j3sz', '/m/05zppz', '/m/06h7j', '/m/03qc9zr', '/m/07p6fty', '/m/01hsr_', '/m/07pbtc8', '/m/02rtxlg', '/m/05x_td', '/m/02mfyn', '/m/03j1ly', '/m/014zdl', '/m/032s66', '/m/03kmc9']

# Name of classes
labels = ['Applause', 'Breathing', 'Chatter', 'Cheering', 'Child_speech_kid_speaking', 'Clapping', 'Conversation', 'Cough', 'Crowd', 'Crying_sobbing', 'Female_speech_woman_speaking', 'Laughter', 'Male_speech_man_speaking', 'Run', 'Screaming', 'Shout', 'Sneeze', 'Walk_footsteps', 'Whispering', 'Air_horn_truck_horn', 'Car_alarm', 'Emergency_vehicle', 'Explosion', 'Gunshot_gunfire', 'Siren']
       
# Number of training samples of sound classes
samples_num = [441, 407, 273, 337, 624, 2399, 2399, 1506, 744, 2020, 1617, 
    25744, 3724, 3745, 7090, 3291, 2301]
    
classes_num = len(labels)
lb_to_idx = {lb: idx for idx, lb in enumerate(labels)}
idx_to_lb = {idx: lb for idx, lb in enumerate(labels)}
