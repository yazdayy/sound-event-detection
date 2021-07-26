import os
import json
import shutil
import argparse
import subprocess
import pandas as pd
from glob import glob
from tqdm import tqdm
from youtube_dl import YoutubeDL

def main(args):
    
    data_type = args.data_type
    workspace = args.workspace
    
    data_path = os.join(workspace, 'dataset', data_type)
    os.makedirs(data_path, exist_ok=True)
    
    csv_path = os.join(workspace, 'metadata', '{}_set.csv'.format(data_type))
    df = pd.read_csv(csv_path, header=None)
    distinct_files = df[0].unique()
    distinct_set = [(x, df[1].loc[df[0] == x].unique()[0]) for x in distinct_files]
    print(len(distinct_set))
    
     # Create directories
    root = os.getcwd()
    in_dir = '{}/raw_audioset'.format(workspace)
    os.makedirs(in_dir, exist_ok=True)
    try:
        # Change the current working Directory
        os.chdir('{}/raw_audioset'.format(workspace))
        print("Directory changed")
    except OSError:
        print("Can't change the current working directory")

    duration = str(10)
    
   
    # Extract videos from YouTube
    audio_downloader = YoutubeDL({'format':'bestaudio'})
    error_count = 0
    
    for file in tqdm(distinct_set):
        try:
            URL = 'https://www.youtube.com/watch?v={}'.format(file[0])
            command = 'youtube-dl ' + '-f "bestaudio/best" ' + '--extract-audio ' + '--audio-format wav ' + '--audio-quality 0 ' + URL +  ' --external-downloader ' + 'ffmpeg ' + '--external-downloader-args ' + '"-ss ' + str(int(file[-1])) + ' -t ' + duration + ' {}/Y{}.wav"'.format(data_path, file[0])
            print('Command: ', command)
            p = subprocess.Popen(command, universal_newlines=True, shell=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            text = p.stdout.read()
            retcode = p.wait()
            print(text)
            for filename in os.listdir(os.getcwd()):
                filepath = os.path.join(os.getcwd(), filename)
                try:
                    shutil.rmtree(filepath)
                except OSError:
                    os.remove(filepath)
        except Exception:
            error_count += 1
            print("Couldn\'t download the audio")
    print('Number of files that could not be downloaded:', error_count)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract AudioSet')
    parser.add_argument('--workspace', type=str, required=True, help='Directory of your workspace.')
    parser.add_argument('--data_type', type=str, required=True, choices=['training', 'testing'])
    args = parser.parse_args()

    main(args)
