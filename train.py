import os
import torch
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import wandb
from datetime import datetime

import config
from data import get_dali, get_georg, get_jamendo, get_jamendoshorts, LA_Dataset, NegativeSampler, collate
from models import SimilarityModel, contrastive_loss, box_loss, neg_box_loss
from utils import fix_seeds, display_module_parameters, int2char, int2phoneme
from eval import evaluate


def train(model, device, train_loader, negative_sampler, criterion, optimizer):
    model.train()
    num_batches = len(train_loader.dataset) // config.batch_size
    train_loss = 0.
    batch_loss = 0.

    for idx, batch in enumerate(tqdm(train_loader)):
        spectrograms, positives, times, is_duplicate, positives_per_spectrogram = batch
        negatives = negative_sampler.sample(config.num_negative_samples, positives, positives_per_spectrogram)
        negatives = torch.IntTensor(negatives)
        spectrograms, positives, negatives = spectrograms.to(device), positives.to(device), negatives.to(device)

        PA, NA = model(spectrograms, positives, positives_per_spectrogram, negatives)
        loss = criterion(PA, NA, times, is_duplicate)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        batch_loss += loss.item()

        log_freq = 100  # log every 100 batches
        if (idx + 1) % log_freq == 0:
            wandb.log({'train/batch_loss': batch_loss / log_freq, 'train/batch_idx': idx + 1})
            batch_loss = 0.

    return train_loss / num_batches


def validate(model, device, val_loader, negative_sampler, criterion, epoch):
    model.eval()
    num_batches = len(val_loader.dataset) // config.batch_size
    val_loss = 0.

    with torch.no_grad():
        for idx, batch in enumerate(tqdm(val_loader)):
            spectrograms, positives, times, is_duplicate, positives_per_spectrogram = batch
            negatives = negative_sampler.sample(config.num_negative_samples, positives, positives_per_spectrogram)
            negatives = torch.IntTensor(negatives)
            spectrograms, positives, negatives = spectrograms.to(device), positives.to(device), negatives.to(device)

            PA, NA = model(spectrograms, positives, positives_per_spectrogram, negatives)

            loss = criterion(PA, NA, times, is_duplicate)
            val_loss += loss.item()

    return val_loss / num_batches


def main():
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
    wandb.init(project='New-Align', config=cfg)

    run_start_time = datetime.now().strftime('%m-%d,%H:%M:%S')
    run_checkpoint_dir = os.path.join(config.checkpoint_dir, run_start_time)
    os.makedirs(run_checkpoint_dir)

    device = torch.device('cuda')  # torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)

    model = SimilarityModel().to(device)
    # display_module_parameters(model)

    if config.use_dali:
        dataset = get_dali()
        print('Size of DALI:', len(dataset))
    else:
        dataset = get_georg()
        print('Size of Georg:', len(dataset))
    train_split, val_split = train_test_split(dataset, test_size=config.val_size, random_state=97)

    negative_sampler = NegativeSampler(dataset)
    train_data = LA_Dataset(train_split, 'train')
    val_data = LA_Dataset(val_split, 'val')
    print('Num training samples:', len(train_data))
    print('Num validation samples:', len(val_data))

    train_loader = DataLoader(dataset=train_data, batch_size=config.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(dataset=val_data, batch_size=config.batch_size, shuffle=False, collate_fn=collate)

    jamendo = get_jamendo()
    jamendoshorts = get_jamendoshorts()

    optimizer = optim.Adam(model.parameters(), config.lr)

    if config.loss == 'contrastive_loss':
        criterion = contrastive_loss
    elif config.loss == 'box_loss':
        criterion = box_loss
    elif config.loss == 'neg_box_loss':
        criterion = neg_box_loss
    else:
        raise NotImplemented()

    epoch = -1
    if config.load_dir is not None:
        epoch = config.load_epoch
        model.load_state_dict(torch.load(config.load_dir))
    epoch += 1
    
    while epoch < config.num_epochs:
        print('Epoch:', epoch)

        train_loss = train(model, device, train_loader, negative_sampler, criterion, optimizer)
        wandb.log({'train/train_loss': train_loss, 'train/epoch': epoch})

        # save checkpoint
        torch.save(model.state_dict(), os.path.join(run_checkpoint_dir, str(epoch)))

        if not config.masked:
            AAE_val, MedAE_val, PCO_val = evaluate(model, device, val_split, log=False)
            wandb.log({'metric/AAE_val': AAE_val, 'metric/epoch': epoch})
            wandb.log({'metric/MedAE_val': MedAE_val, 'metric/epoch': epoch})
            wandb.log({'metric/PCO_val': PCO_val, 'metric/epoch': epoch})

        val_loss = validate(model, device, val_loader, negative_sampler, criterion, epoch)
        wandb.log({'val/val_loss': val_loss, 'val/epoch': epoch})

        epoch += 1

    wandb.finish()


if __name__ == '__main__':
    print('Running train.py')
    main()
