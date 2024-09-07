import wandb
import numpy as np
import matplotlib.pyplot as plt

import config

def log_plots(S, song, token_alignment, word_alignment):

    num_tokens, num_frames = S.shape

    token_alignment_image = np.zeros_like(S)
    for token, frames in enumerate(token_alignment):
        token_alignment_image[token, frames[0]:frames[1]] = 1

    words = song['words']
    word_alignment_image = np.zeros(shape=(len(words), num_frames))
    gt_word_alignment_image = np.zeros(shape=(len(words), num_frames))

    for word, frames in enumerate(word_alignment):
        word_alignment_image[word, frames[0]:frames[1]] = 1

    fps = S.shape[1] / song['duration']
    for word, time in enumerate(song['times']):
        frames = (int(time[0] * fps), int(time[1] * fps))
        gt_word_alignment_image[word, frames[0]:frames[1]] = 1

    tokens = chars(song['words']) if config.use_chars else phonemes(song['phowords'])    
    #heights = [len(tokens)] * 2 + [len(song['words'])] * 2
    heights = [len(tokens), len(song['words'])]
    fig, axs = plt.subplots(2, 1, height_ratios=heights, 
                            figsize=(min(num_frames // 14, 100), min((sum(heights) + 20 * len(heights)) // 12, 100)))
    
    show_plot(S, axs[0], 'S', tokens)
    alignment_cmap = 'Blues'
    #show_plot(token_alignment_image, axs[1], 'token alignment', tokens, alignment_cmap)
    #show_plot(word_alignment_image, axs[2], 'word alignment', song['words'], alignment_cmap)
    show_plot(gt_word_alignment_image, axs[1], 'ground truth word alignment', song['words'], alignment_cmap)
    #show_plot(DP, axs[5], 'DP', tokens)

    fig.tight_layout()

    wandb.log({'media/' + song['id']: plt})
    plt.close()


def show_plot(data, ax, title, ytick_labels, cmap='hot'):
    im = ax.imshow(data, cmap=cmap, aspect='auto', interpolation='none')
    ax.figure.colorbar(im, ax=ax)
    ax.set_title(title)
    ax.set_yticks(ticks=np.arange(data.shape[0]), labels=ytick_labels)

    # Set x-axis ticks to time values (instead of column indices)
    ax.set_xticks(np.linspace(0, data.shape[1] - 1, num=6))  # Set 6 ticks from 0 to 5 seconds
    ax.set_xticklabels(np.linspace(0, 5, num=6))  # Corresponding time labels in seconds
    ax.set_xlabel("Seconds")

    ax.tick_params(axis='both', labelsize=9)


def chars(words):
    lyrics = ' '.join(words)
    return [c for c in lyrics]

def phonemes(phowords):
    phonemes = []
    for phoword in phowords:
        phonemes += phoword + [' ']
    phonemes = phonemes[:-1]
    return phonemes