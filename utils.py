import torch
import numpy as np
from prettytable import PrettyTable
import torchaudio
import warnings
import librosa
from torch import nn
import string
from g2p_en import G2p
import eng_to_ipa as ipa
from phonemizer.backend import EspeakBackend
import re
import csv
import random
import pickle

import config


char_dict = [' ', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', "'"]
char2int = {char_dict[i]: i for i in range(len(char_dict))}
int2char = {i: char_dict[i] for i in range(len(char_dict))}

phonemizer_phoneme_dict = [' ', 'a', 'b', 'd', 'e', 'f', 'h', 'i', 'iː', 'j', 'k', 'l', 'm', 'n', 'o', 'oː', 'p', 'r', 's', 't', 'uː', 'v', 'w', 'x', 'z', 'æ', 'æː', 'ð', 'ŋ', 'ɐ', 'ɐː', 'ɑ', 'ɑː', 'ɔ', 'ɔː', 'ə', 'ɚ', 'ɛ', 'ɜː', 'ɡ', 'ɪ', 'ɫ', 'ɬ', 'ɹ', 'ɾ', 'ʃ', 'ʊ', 'ʌ', 'ʒ', 'ʔ', '̃', 'θ', 'ᵻ']
g2p_phoneme_dict = [' ', 'AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'B', 'CH', 'D', 'DH', 'EH', 'ER', 'EY', 'F', 'G', 'HH', 'IH', 'IY', 'JH', 'K', 'L', 'M', 'N', 'NG', 'OW', 'OY', 'P', 'R', 'S', 'SH', 'T', 'TH', 'UH', 'UW', 'V', 'W', 'Y', 'Z', 'ZH']
phoneme_dict = phonemizer_phoneme_dict if config.use_IPA else g2p_phoneme_dict
phoneme2int = {phoneme_dict[i]: i for i in range(len(phoneme_dict))}
int2phoneme = {i: phoneme_dict[i] for i in range(len(phoneme_dict))}



g2p = G2p()
espeak_backend = EspeakBackend('en-us')  # BACKEND SET TO ENGLISH; CHANGE IF TRAIN OR EVAL WITH OTHER LANGUAGES

def fix_seeds(seed=97):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    #torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    #torch.use_deterministic_algorithms(True)

def display_module_parameters(model):
    table = PrettyTable(['Modules', 'Parameters'])
    total_num_params = 0
    for name, params in model.named_parameters():
        if not params.requires_grad:
            continue
        num_params = params.numel()
        table.add_row([name, num_params])
        total_num_params += num_params
    print(table)
    print(f'Total Trainable Params: {total_num_params}')


def split_IPA_string(s):
    # splits the string into its chars, concats the Long symbol 'ː' with its root
    pattern = re.compile(r'.ː?')
    return pattern.findall(s)

def words2phowords(words, times):

    ret_words = []
    phowords = []
    ret_times = []
    for word, time in zip(words, times):

        if config.use_IPA:
            phoword = espeak_backend.phonemize([word], strip=True)[0]

            if phoword.__contains__('('):  # non-english word, e.g., phonemize('dieu') = '(fr)djø(enus)'
                continue
            if phoword.startswith('ɹoʊmən '):  # ii and others interpreted as roman numbers
                continue
            
            phoword = split_IPA_string(phoword)
            phoword = [c for c in phoword if c not in ['̩', 'ː']]  # remove Syllabic and remaining Long symbols
       
        else:
            word = word.strip("'")  # g2p does not remove leading and trailing '
            phoword = g2p(word)
            phoword = [p[:-1] if p[-1] in string.digits else p for p in phoword]
        
        assert len(phoword) > 0
        ret_words.append(word)
        phowords.append(phoword)
        ret_times.append(time)

    for word, phoword in zip(words, phowords):
        for p in phoword:
            if p not in phoneme_dict[1:]:
                raise NotImplementedError(f'Unknown phoneme "{p}" in word "{word}"')

    return ret_words, phowords, ret_times     

def lines2pholines(lines):
    pholines = []
    for line in lines:
        words = line.split()
        _, phowords, _ = words2phowords(words, times=[(0, 0)] * len(words))  # dummy times
        assert len(phowords) == len(words)
        pholine = []
        for phoword in phowords:
            pholine += phoword + [' ']
        pholine = pholine[:-1]  # remove last space
        pholines.append(pholine)
    return pholines


def get_dali_remarks():
    remarks = dict()
    with open(config.dali_remarks, 'r') as f:
        reader = csv.DictReader(f, delimiter=';')
        for row in reader:
            remark = {'PCO': float(row['PCO']),
                      'AAE': float(row['AAE']),
                      'corrupt from': float(row['corrupt from']),
                      'noisy': row['noisy'] == 'True',
                      'offset': str(row['offset']),
                      'non-english': row['non-english'] == 'True',
                      'vocalizations': row['vocalizations'] == 'True',
                      'repeated words/lines': row['repeated words/lines'] == 'True',
                      'multiple singers': row['multiple singers'] == 'True',
                      'split words': row['split words'] == 'True',
                      'missing words': row['missing words'] == 'True'}
            remarks[row['id']] = remark
    return remarks


def monotonically_increasing_times(times):
    raise NotImplemented('do not use')
    flat_times = [t for time in times for t in time]
    if all(flat_times[i] <= flat_times[i + 1] for i in range(len(flat_times) - 1)):
        return True
    return False

def monotonically_increasing_starts(times):    
    flat_times = [time[0] for time in times]
    if all(flat_times[i] <= flat_times[i + 1] for i in range(len(flat_times) - 1)):
        return True
    return False

def monotonically_increasing_ends(times):    
    flat_times = [time[1] for time in times]
    if all(flat_times[i] <= flat_times[i + 1] for i in range(len(flat_times) - 1)):
        return True
    return False


def normalize_dali(words, times, cutoff, offset):
    ret_words = []
    ret_times = []
    for raw_word, time in zip(words, times):
        if time[0] >= cutoff:
            break

        if time[0] >= time[1]:
            continue

        word = raw_word.lower()
        word = ''.join([c for c in word if c in char_dict[1:]])
        word = word.strip("'")  # e.g. filter('89) = ', not a word
        if len(word) == 0 or len(word) >= 12:  # if len(word) >= 12, then word is likely the concat of multiple words
            continue
        ret_words.append(word)
        ret_times.append(time)

    ret_times = [(start + offset, end + offset) for (start, end) in ret_times]

    return ret_words, ret_times

def normalize_georg(raw_words, raw_times):
    return normalize_dali(raw_words, raw_times, cutoff=1e10, offset=0)

def normalize_jamendo(raw_lines):
    lines = [l for l in raw_lines if len(l) > 0]  # remove empty lines between paragraphs
    lines = [' '.join([word.strip("'") for word in line.split()]) for line in lines]
    return lines


def encode_words(words, times):  # could merge with encode_phowords, same code
    raise NotImplementedError('no duplicate checking')
    chars = ''
    char_times = []
    for word, time, next_time in zip(words[:-1], times[:-1], times[1:]):
        chars += word + ' '
        end, next_start = time[1], next_time[0]
        char_times += [time] * len(word) + [(end, next_start)]
        #start, next_end = time[0], next_time[1]
        #char_times += [time] * len(word) + [(start, next_end)]
    chars += words[-1]
    char_times += [times[-1]] * len(words[-1])

    chars = ' ' * config.context + chars + ' ' * config.context
    
    enc_chars = []
    for c in chars:
        idx = char2int[c]
        enc_chars.append(idx)

    tokens = [enc_chars[i:i + (1 + 2 * config.context)] for i in range(len(enc_chars) - 2 * config.context)]  # token: enc_char and context
    assert len(tokens) > 0

    return tokens, char_times

def encode_phowords(phowords, times):
    phonemes = []
    phoneme_times = []
    for phoword, time, next_time in zip(phowords[:-1], times[:-1], times[1:]):
        phonemes += phoword + [' ']
        end, next_start = time[1], next_time[0]
        phoneme_times += [time] * len(phoword) + [(end, next_start)]
        #start, next_end = time[0], next_time[1]
        #phoneme_times += [time] * len(phoword) + [(start, next_end)]
    phonemes += phowords[-1]
    phoneme_times += [times[-1]] * len(phowords[-1])
    
    phoneme_freq = [0] * len(phoneme_dict)
    for p in phonemes:
        phoneme_freq[phoneme2int[p]] += 1
    phoneme_freq[phoneme2int[' ']] = 2  # it's better to always consider space as duplicate (there's probably space/silence at the start and end of the segment)

    phonemes = [' '] * config.context + phonemes + [' '] * config.context

    enc_phonemes = []
    is_duplicate = [False] * len(phonemes)  # is_duplicate does not consider context
    for i, p in enumerate(phonemes):
        idx = phoneme2int[p]
        enc_phonemes.append(idx)
        is_duplicate[i] = phoneme_freq[idx] > 1
        assert phoneme_freq[idx] >= 1

    tokens = [enc_phonemes[i:i + (1 + 2 * config.context)] for i in range(len(enc_phonemes) - 2 * config.context)]  # token: enc_phoneme and context
    assert len(tokens) > 0

    return tokens, phoneme_times, is_duplicate[config.context:-config.context]


def read_jamendo_times(times_file):
    times = []
    with open(times_file, 'r') as f:
        reader = csv.DictReader(f, delimiter=',')
        for row in reader:
            times.append((float(row['word_start']), float(row['word_end'])))
    return times


def load(path: str, sr: int) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        y, _ = librosa.load(path, sr=sr, res_type='kaiser_fast')

    if len(y.shape) != 1:
        raise ValueError('Waveform has multiple dimensions')

    return y

class LogSpectrogram(nn.Module):
    def __init__(self):
        super(LogSpectrogram, self).__init__()
        self.spectrogram = torchaudio.transforms.Spectrogram(n_fft=config.n_fft, power=1)

    def forward(self, waveform):
        spec = self.spectrogram(waveform)
        return torch.log(1 + spec)
    
def wav2spec(waveform: np.ndarray) -> np.ndarray:
    waveform = torch.Tensor(waveform)
    log_spec = LogSpectrogram()(waveform)
    return log_spec.numpy()
