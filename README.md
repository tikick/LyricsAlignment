# LyricsAlignment

Code for the paper "Contrastive Lyrics Alignment with a Timestamp-Informed Loss"

## Dependencies

To install the required python packages, run

```
pip install -r requirements.txt
```

Refer to [phonemizer](https://github.com/bootphon/phonemizer) on how to install the required background for IPA phonemes extraction.

## Data

The [DALI](https://github.com/gabolsgabs/DALI) dataset is used for training.

The [jamendolyrics](https://github.com/f90/jamendolyrics) evaluation dataset is used for evaluation.

When you train for the first time, pickle files containing the training samples will be generated.

## Training

Run `python train.py` to train the model. Set the configurations in the config.py file.
