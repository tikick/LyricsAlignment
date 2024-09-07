import DALI as dali_code
import os
from pytube import YouTube

import config


def download_dali_audio():
    dali_data = dali_code.get_the_DALI_dataset(config.dali_annotations, skip=[], keep=[])

    base_url = 'https://www.youtube.com/watch?v='
    lang = None if config.dali_multilingual else 'english'

    if not os.path.isdir(config.dali_audio):
        os.makedirs(config.dali_audio)

    num_downloads = 0
    num_fails = 0

    for dali_id in dali_data:

        if lang is not None and dali_data[dali_id].info['metadata']['language'] != lang:
            continue

        url = base_url + dali_data[dali_id].info['audio']['url']

        try:
            video = YouTube(url)
            stream = video.streams.filter(only_audio=True).first()
            stream.download(output_path=config.dali_audio, filename=dali_id + '.wav')  # librosa.load() has problems with .mp3 
            num_downloads += 1
        except Exception as e:
            num_fails += 1
            print(f'Failed to download dali_id = {dali_id}, url = {url}: {repr(e)}')

    print(f'Successfully donwloaded {num_downloads} / {num_downloads + num_fails} songs')


if __name__ == '__main__':
    print('Running download.py')
    download_dali_audio()