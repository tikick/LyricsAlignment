import DALI as dali_code
import bisect
import os
from tqdm import tqdm
import pickle
import csv
import pandas as pd
from math import floor
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from audiomentations import TimeStretch, PitchShift, BandPassFilter, LowPassFilter, HighPassFilter

import config
from utils import *


def get_dali():

    lang = None if config.dali_multilingual else 'english'

    dali_data = dali_code.get_the_DALI_dataset(config.dali_annotations, skip=[], keep=[])
    if config.use_dali_remarks:
        remarks = get_dali_remarks()

    songs = []
    
    def unique(l):
        last = object()
        for item in l:
            if item == last:
                continue
            yield item
            last = item

    audio_files = os.listdir(config.dali_audio)  # only get songs for which we have audio files
    for file in tqdm(audio_files):
        id = file[:-4]
        annot = dali_data[id].annotations['annot']
        metadata = dali_data[id].info['metadata']

        if lang is not None and metadata['language'] != lang:
            continue
        
        offset = 0
        cutoff = 1e10
        if config.use_dali_remarks and id in remarks:
            if remarks[id]['corrupt from'] == 0 or remarks[id]['noisy'] or remarks[id]['offset'] == 'linear' or remarks[id]['non-english']:
                continue
            offset = float(remarks[id]['offset'])
            cutoff = remarks[id]['corrupt from']

        times = [d['time'] for d in annot['words']]
        words = [d['text'] for d in annot['words']]
        words, times = normalize_dali(words, times, cutoff, offset)
        words, phowords, times = words2phowords(words, times)  #[d['text'] for d in annot['phonemes']]

        if not monotonically_increasing_starts(times):
            # sort and remove duplicates
            times, words, phowords = (list(t) for t in zip(*unique(sorted((zip(times, words, phowords))))))

        assert len(words) == len(phowords) and len(phowords) == len(times), id
        if len(words) == 0:
            continue

        song = {'id': id,
                'url': dali_data[id].info['audio']['url'],
                'audio_path': os.path.join(config.dali_audio, file),
                'words': words,
                'phowords': phowords,
                'times': times}

        songs.append(song)

    return songs


def get_jamendo(lang='English'):
    songs = []

    with open(config.jamendo_metadata, 'r') as f:
        reader = csv.DictReader(f, delimiter=',')
        for row in reader:
            if row['Language'] != lang:
                continue

            audio_file = row['Filepath']
            with open(os.path.join(config.jamendo_lyrics, audio_file[:-4] + '.txt'), 'r') as f:
                lines = f.read().splitlines()
            lines = normalize_jamendo(lines)
            words = ' '.join(lines).split()
            times = read_jamendo_times(os.path.join(config.jamendo_annotations, audio_file[:-4] + '.csv'))
            _, phowords, _ = words2phowords(words, times)
            assert len(phowords) == len(words)
            pholines = lines2pholines(lines)
            
            song = {'id': audio_file[:-4],
                    'audio_path': os.path.join(config.jamendo_audio, audio_file),
                    'words': words,
                    'phowords': phowords,
                    'lines': lines,
                    'pholines': pholines,
                    'times': times}
            
            songs.append(song)
    
    return songs

def get_jamendopp():
    songs = []

    with open(config.jamendopp_metadata, 'r') as f:
        reader = csv.DictReader(f, delimiter=',')
        for row in reader:
            audio_file = row['Filepath']
            with open(os.path.join(config.jamendopp_lyrics, audio_file[:-4] + '.txt'), 'r') as f:
                lines = f.read().splitlines()
            lines = normalize_jamendo(lines)
            words = ' '.join(lines).split()
            times = read_jamendo_times(os.path.join(config.jamendopp_annotations, audio_file[:-4] + '.csv'))
            _, phowords, _ = words2phowords(words, times)
            assert len(phowords) == len(words)
            pholines = lines2pholines(lines)
            
            song = {'id': audio_file[:-4],
                    'audio_path': os.path.join(config.jamendopp_audio, audio_file),
                    'words': words,
                    'phowords': phowords,
                    'lines': lines,
                    'pholines': pholines,
                    'times': times}
            
            songs.append(song)
    
    return songs


def get_jamendoshorts(lang='English'):
    songs = []

    with open(config.jamendoshorts_metadata, 'r') as f:
        reader = csv.DictReader(f, delimiter=',')
        for row in reader:
            if row['Language'] != lang:
                continue

            audio_file = row['Filepath']
            with open(os.path.join(config.jamendoshorts_lyrics, audio_file[:-4] + '.txt'), 'r') as f:
                lines = f.read().splitlines()
            lines = normalize_jamendo(lines)
            words = ' '.join(lines).split()
            times = read_jamendo_times(os.path.join(config.jamendoshorts_annotations, audio_file[:-4] + '.csv'))
            _, phowords, _ = words2phowords(words, times)
            assert len(phowords) == len(words)
            pholines = lines2pholines(lines)
            
            song = {'id': audio_file[:-4],
                    'audio_path': os.path.join(config.jamendoshorts_audio, audio_file),
                    'words': words,
                    'phowords': phowords,
                    'lines': lines,
                    'pholines': pholines,
                    'times': times}
            
            songs.append(song)
    
    return songs


def get_georg():
    # num_unk_chars = 39330, num_total_chars = 19920586, alignment_nones = 755
    songs = []
    
    for i in range(20):  # for folders from 0 to 19
        parq_file = os.path.join(config.georg_annotations, str(i), 'alignment.parq')

        df = pd.read_parquet(parq_file, engine='pyarrow')
        for _, row in df.iterrows():

            if row['alignment'] is None:
                continue

            audio_path = os.path.join(config.georg_audio, row['ytid'] + '.mp3')
            if not os.path.exists(audio_path):
                continue

            token_starts = row['alignment']['starts']
            token_ends = row['alignment']['ends']
            tokens_per_word = list(row['alignment']['tokens_per_word'])
            token_offsets = np.cumsum([0] + tokens_per_word)
            assert token_offsets[-1] == len(token_starts)

            word_starts = []
            word_ends = []
            for token_offset in token_offsets[:-1]:
                word_starts.append(token_starts[token_offset])
            for token_offset in token_offsets[1:]:
                word_ends.append(token_ends[token_offset - 1])

            times = list(zip(word_starts, word_ends))
            words = row['alignment']['words']
            words, times = normalize_georg(words, times)
            words, phowords, times = words2phowords(words, times)
            
            song = {'id': row['ytid'],
                    'audio_path': audio_path,
                    'words': words,
                    'phowords': phowords,
                    'times': times}
            
            #if georg_song_is_corrupt(song):
            #    continue
            
            songs.append(song)
    
    return songs


def eval_collate(song):
    waveform = load(song['audio_path'], sr=config.sr)
    song['duration'] = len(waveform) / config.sr
    spec = wav2spec(waveform)
    spectrogram, all_tokens, _, _, _ = collate(data=[(spec, song['words'], song['phowords'], song['times'])])
    return spectrogram, all_tokens


def collate(data):
    spectrograms = []
    all_tokens = []
    all_times = []
    all_is_duplicate = []
    tokens_per_spectrogram = []

    for spec, words, phowords, times in data:
        spectrograms.append(spec)

        if config.use_chars:
            tokens, token_times, is_duplicate = encode_words(words, times)
        else:
            tokens, token_times, is_duplicate = encode_phowords(phowords, times)
        assert len(is_duplicate) == len(tokens) and len(tokens) == len(token_times)

        tokens_per_spectrogram.append(len(tokens))
        all_tokens += tokens
        all_times += token_times
        all_is_duplicate += is_duplicate

    # Creating a tensor from a list of numpy.ndarrays is extremely slow. Convert the list to a single numpy.ndarray with numpy.array() before converting to a tensor.
    spectrograms = torch.Tensor(np.array(spectrograms))
    all_tokens = torch.IntTensor(all_tokens)

    return spectrograms, all_tokens, all_times, all_is_duplicate, tokens_per_spectrogram


class LA_Dataset(Dataset):
    def __init__(self, dataset, partition):
        super(LA_Dataset, self).__init__()

        dataset_name = 'dali' if config.use_dali else 'georg'
        if config.use_dali and config.use_dali_remarks:
            dataset_name += 'Clean'
        if config.use_dali and config.dali_multilingual:
            dataset_name += 'Multilingual'
        if config.augment_data:
            dataset_name += 'Augm'
        if config.use_IPA:
            dataset_name += 'IPA' 

        file_name = f'{dataset_name}_{partition}'
        pickle_file = os.path.join(config.pickle_dir, file_name + '.pkl')

        if not os.path.exists(pickle_file):
            if not os.path.exists(config.pickle_dir):
                os.makedirs(config.pickle_dir)

            print(f'Creating {file_name} samples')
            samples = []
            for song in tqdm(dataset):
                waveform = load(song['audio_path'], sr=config.sr)

                word_starts = [start for (start, _) in song['times']]
                word_ends = [end for (_, end) in song['times']]

                max_num_samples = floor(((len(waveform) / config.sr - config.segment_length) / config.hop_size) + 1)
                for i in range(max_num_samples):
                    sample_start = i * config.hop_size
                    sample_end = sample_start + config.segment_length
                    assert sample_end <= len(waveform) / config.sr
                    waveform_slice = waveform[int(sample_start * config.sr):int(sample_end * config.sr)]

                    # find the lyrics within (sample_start, sample_end)
                    idx_first_word = bisect.bisect_left(word_starts, sample_start)
                    idx_past_last_word = idx_first_word  #bisect.bisect_left(word_ends, sample_end)
                    while idx_past_last_word < len(word_ends) and word_ends[idx_past_last_word] < sample_end:
                        idx_past_last_word += 1

                    if idx_first_word >= idx_past_last_word:  # no words (fully contained) in this sample, skip
                        continue
                    
                    # sample spectrogram, words, phowords and relative times withing sample
                    spec = wav2spec(waveform_slice)
                    words = song['words'][idx_first_word:idx_past_last_word]
                    phowords = song['phowords'][idx_first_word:idx_past_last_word]
                    times = song['times'][idx_first_word:idx_past_last_word]
                    offset = sample_start
                    times = [(start - offset, end - offset) for (start, end) in times]
                    for (start, end) in times:
                        assert 0 <= start < end < config.segment_length, f'id={song["id"]}, i={i}, sample_start={sample_start}, offset={offset} start={start}, end={end}'
                    sample = (spec, words, phowords, times)
                    samples.append(sample)

                    if config.augment_data:
                        transforms = ['pitch_shift', 'freq_filter', 'time_stretch']
                        choice = np.random.choice(transforms)
                        if choice == 'pitch_shift':
                            transform = PitchShift(p=1)
                        elif choice == 'freq_filter':
                            transform = np.random.choice([BandPassFilter(p=1), LowPassFilter(p=1), HighPassFilter(p=1)])
                        elif choice == 'time_stretch':
                            rate = np.random.choice([0.8, 1.25])
                            transform = TimeStretch(min_rate=rate, max_rate=rate, p=1)
                            times = [(start/rate, end/rate) for (start, end) in times if end/rate < config.segment_length]
                            if len(times) == 0:
                                continue
                            words = words[:len(times)]
                            phowords = phowords[:len(times)]
                        else:
                            raise NotImplemented()
                        
                        transformed_waveform_slice = transform(waveform_slice, sample_rate=config.sr)
                        transformed_spec = wav2spec(transformed_waveform_slice)
                        sample = (transformed_spec, words, phowords, times)
                        samples.append(sample)


            with open(pickle_file, 'wb') as f:
                print(f'Writing {file_name} samples')
                pickle.dump(samples, f)

        with open(pickle_file, 'rb') as f:
            print(f'Loading {file_name} samples')
            self.samples = pickle.load(f)

    def __getitem__(self, index):
        return self.samples[index]  # (spec, words, phowords, times)

    def __len__(self):
        return len(self.samples)


class NegativeSampler:
    def __init__(self, dataset):
        # do not store frequencies in a file, they depend on mutable config fields, e.g., use_chars, context

        print('Computing negative sampling probabilities')

        assert config.context <= 1  # very memory intensive to store the distribution otherwise
        self.frequencies = np.zeros((pow(config.vocab_size, 1 + 2 * config.context),), dtype=int)

        for song in tqdm(dataset):
            if config.use_chars:
                tokens, _, _ = encode_words(song['words'], song['times'])
            else:
                tokens, _, _ = encode_phowords(song['phowords'], song['times'])

            for token in tokens:
                idx = self._token2idx(token)
                self.frequencies[idx] += 1
        

    def sample(self, num_samples, positives, positives_per_scpetrogram):
        # to avoid sampling positives set their frequencies to 0

        negatives = []
        cumsum = np.cumsum([0] + positives_per_scpetrogram)

        for i in range(len(positives_per_scpetrogram)):
            j, k = cumsum[i], cumsum[i + 1]

            # set frequencies of positives to 0
            mutable_frequencies = self.frequencies.copy()
            for l in range(j, k):
                token = positives[l]
                idx = self._token2idx(token)
                mutable_frequencies[idx] = 0
            
            # sample negatives
            prob = mutable_frequencies / np.sum(mutable_frequencies)
            indices = np.random.choice(len(prob), size=num_samples, p=prob)
            negatives += [self._idx2token(idx) for idx in indices]

        return negatives
    
    def fast_sample(self, num_samples, positives, positives_per_scpetrogram):
        # to avoid sampling positives set their frequencies to 0
        
        negatives = []
        cumsum = np.cumsum([0] + positives_per_scpetrogram)

        for i in range(len(positives_per_scpetrogram)):
            j, k = cumsum[i], cumsum[i + 1]

            # set frequencies of positives to 0
            original_idx_freq_pairs = []
            for l in range(j, k):
                token = positives[l]
                idx = self._token2idx(token)
                original_idx_freq_pairs.append((idx, self.frequencies[idx]))
                self.frequencies[idx] = 0

            # sample negatives
            prob = self.frequencies / np.sum(self.frequencies)
            indices = np.random.choice(len(prob), size=num_samples, p=prob)
            negatives += [self._idx2token(idx) for idx in indices]

            # restore original frequencies
            for l in reversed(range(j, k)):  
                # reversed necessary, without, second appearence of same token overwrites true freq with 0 (second appearence gets 0 freq in original_idx_freq_pairs)
                idx, freq = original_idx_freq_pairs[l - j]
                self.frequencies[idx] = freq

        return negatives

    @staticmethod
    def _token2idx(token):
        idx = 0
        for t in token:
            idx *= config.vocab_size
            idx += t
        return idx

    @staticmethod
    def _idx2token(idx):
        token = []
        for _ in range(1 + 2 * config.context):
            token.append(idx % config.vocab_size)
            idx = idx // config.vocab_size
        token = list(reversed(token))
        return token
    


if __name__ == '__main__':
    print('Running data.py')

    cfg = {'num_RCBs': config.num_RCBs,
           'channels': config.channels,
           'context': config.context,
           'use_chars': config.use_chars,
           'embedding_dim': config.embedding_dim,
           'num_epochs': config.num_epochs,
           'lr': config.lr,
           'batch_size': config.batch_size,
           'num_negative_samples': config.num_negative_samples,
           'loss': config.loss,
           'box_slack': config.box_slack,
           'use_dali': config.use_dali,
           'use_dali_remarks': config.use_dali_remarks,
           'dali_multilingual': config.dali_multilingual,
           'use_IPA': config.use_IPA,
           'augment_data': config.augment_data,
           'val_size': config.val_size,
           'masked': config.masked,
           'load_epoch': config.load_epoch,
           'load_dir': config.load_dir}
    
    print(cfg)
    
    #georg = get_georg()
    #print('Size of Georg:', len(georg))
    dali = get_dali()
    print('Size of DALI:', len(dali))
    train, val = train_test_split(dali, test_size=config.val_size, random_state=97)

    train_data = LA_Dataset(train, 'train')
    val_data = LA_Dataset(val, 'val')
    print('Num training samples:', len(train_data))
    print('Num validation samples:', len(val_data))
