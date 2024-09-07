import os


# paths
base_path = None #'/itet-stor/tikick/net_scratch/LyricsAlign'

checkpoint_dir = os.path.join(base_path, 'checkpoints')
pickle_dir = os.path.join(base_path, 'pickles')

dali_base = os.path.join(base_path, 'DALI_v2.0')
dali_annotations = os.path.join(dali_base, 'annot')
dali_audio = os.path.join(dali_base, 'wav')
dali_remarks = os.path.join(dali_base, 'dali_remarks.csv')

jamendo_base = os.path.join(base_path, 'jamendolyrics')
jamendo_metadata = os.path.join(jamendo_base, 'JamendoLyrics.csv')
jamendo_annotations = os.path.join(jamendo_base, 'annotations/words')
jamendo_lyrics = os.path.join(jamendo_base, 'lyrics')
jamendo_audio = os.path.join(jamendo_base, 'mp3')

jamendopp_base = os.path.join(base_path, 'jamendolyrics++')
jamendopp_metadata = os.path.join(jamendopp_base, 'JamendoLyrics++.csv')
jamendopp_annotations = os.path.join(jamendopp_base, 'annotations/words')
jamendopp_lyrics = os.path.join(jamendopp_base, 'lyrics')
jamendopp_audio = os.path.join(jamendopp_base, 'mp3')

jamendoshorts_base = os.path.join(base_path, 'jamendoshorts')
jamendoshorts_metadata = os.path.join(jamendoshorts_base, 'JamendoShorts.csv')
jamendoshorts_annotations = os.path.join(jamendoshorts_base, 'annotations/words')
jamendoshorts_lyrics = os.path.join(jamendoshorts_base, 'lyrics')
jamendoshorts_audio = os.path.join(jamendoshorts_base, 'mp3')

georg_base = os.path.join(base_path, 'Georg')
georg_annotations = os.path.join(georg_base, 'ttv')
georg_audio = os.path.join(georg_base, 'data/audio')


# dataset
use_dali = True  # if false uses georg
use_dali_remarks = False
dali_multilingual = False  # eng_to_ipa does not support multilingual
use_IPA = True  # if false uses english phonemes from g2p
augment_data = False

# audio encoder
num_RCBs = 10
channels = 64

# text encoder
context = 1  # when context = 0, for some audio segments the set of negatives might be empty -> NegativeSampler.sample() fails
use_chars = False  # if false uses phonemes
vocab_size = 28 if use_chars else (53 if use_IPA else 40)  # len(char_dict) if use_chars else len(phoneme_dict)

# audio and text encoder
embedding_dim = 64

# others
num_epochs = 16
lr = 0.0001
batch_size = 8
num_negative_samples = 1_000

# loss
loss = 'neg_box_loss'  # [box_loss, neg_box_loss, contrastive_loss]
box_slack = 1  # seconds

# alignment
masked = True

# load model
load_epoch = 3
load_dir = None


# WARNING: if you change the following parameters remember to delete the sample files or the new samples will not be computed

# dataset
val_size = 0.02

# waveform
sr = 11025
segment_length = 5  # seconds
hop_size = segment_length / 2  # seconds

# spectrogram
n_fft = 512
freq_bins = n_fft // 2 + 1
