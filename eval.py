import os
import torch
import wandb
from statistics import median

import config
from data import get_jamendo, get_jamendoshorts, eval_collate, get_jamendopp
from models import SimilarityModel
from utils import fix_seeds
from decode import get_alignment
from media import log_plots


def evaluate(model, device, eval_dataset, offset=-0.1, log=False):
    model.eval()
    AAE = 0.
    MedAE = 0.
    PCO = 0.

    with torch.no_grad():
        for song in eval_dataset:
            spectrogram, positives = eval_collate(song)
            spectrogram, positives = spectrogram.to(device), positives.to(device)

            S = model(spectrogram, positives)
            S = S.cpu().numpy()

            _, word_alignment = get_alignment(S, song, time_measure='seconds')
            word_alignment = [(start + offset, end + offset) for (start, end) in word_alignment]  # correct tiny delay
            AAE_song, MedAE_song, PCO_song = compute_metrics(word_alignment, song['times'])
            AAE += AAE_song
            MedAE += MedAE_song
            PCO += PCO_song

            # log plots for inspection
            if log:
                token_alignment, word_alignment = get_alignment(S, song, time_measure='frames')
                log_plots(S, song, token_alignment, word_alignment)

    return AAE / len(eval_dataset), MedAE / len(eval_dataset), PCO / len(eval_dataset)


def compute_metrics(alignment, gt_alignment, PCO_tol=0.3):
    assert len(alignment) == len(gt_alignment)
    deviations = [abs(time[0] - gt_time[0]) for (time, gt_time) in zip(alignment, gt_alignment)]
    AAE = sum(deviations) / len(deviations)
    MedAE = median(deviations)
    PCO = sum(1 for dev in deviations if dev <= PCO_tol) / len(deviations)
    return AAE, MedAE, PCO


if __name__ == '__main__':
    print('Running eval.py')

    fix_seeds()

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
    os.environ["WANDB__SERVICE_WAIT"] = '600'
    wandb.init(project='Results', config=cfg)

    device = torch.device('cuda')  # torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimilarityModel().to(device)
    #jamendo = get_jamendo()
    #jamendopp = get_jamendopp()
    jamendoshorts = get_jamendoshorts()

    epoch = 15
    checkpoint_path = os.path.join(config.checkpoint_dir, '08-29,15:04:41', str(epoch))
    model.load_state_dict(torch.load(checkpoint_path))

    evaluate(model, device, jamendoshorts, log=True)
    #AAE_jamendo, MedAE_jamendo, PCO_jamendo = evaluate(model, device, jamendo, log=False)
    #print(epoch)
    #print(AAE_jamendo, MedAE_jamendo, PCO_jamendo)
