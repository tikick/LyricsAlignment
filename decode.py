import numpy as np

import config


def diagonal_align(S, song):
    # finds monotonic path maximizing the cumulative log similarity score, without horizontal score accumulation

    assert np.all((S >= 0) & (S <= 1))

    S = np.log(S)  # RuntimeWarning: divide by zero encountered in log
    num_tokens, num_frames = S.shape

    # add begin of sentence token and frame for convinience
    DP = -np.inf * np.ones((num_tokens + 1, num_frames + 1), dtype=np.float32)
    DP[0, :] = 0
    parent_is_prev_token = np.zeros_like(DP, dtype=bool)

    for frame in range(num_frames):
        #for i in range(1, num_tokens + 1):  # can vectorize
        stay = DP[1:, frame]
        move = DP[:-1, frame] + S[:, frame]
        DP[1:, frame + 1] = np.maximum(stay, move)
        parent_is_prev_token[1:, frame + 1] = stay < move
    
    token_alignment = []
    token = num_tokens - 1
    end_frame = num_frames - 1
    for frame in reversed(range(num_frames)):
        if parent_is_prev_token[token + 1, frame + 1]:
            token_alignment.append((frame, end_frame))
            end_frame = frame - 1
            token -= 1

    token_alignment = list(reversed(token_alignment))
    
    words = song['words'] if config.use_chars else song['phowords']
    word_alignment = []
    first_word_token = last_word_token = 0
    for word in words:
        num_word_tokens = len(word)
        last_word_token = first_word_token + num_word_tokens - 1
        word_start = token_alignment[first_word_token][0]
        word_end = token_alignment[last_word_token][1]
        word_alignment.append((word_start, word_end))
        first_word_token = last_word_token + 2  # +1 space between words
    
    assert len(word_alignment) == len(song['times'])
    
    return token_alignment, word_alignment


def get_alignment(S, song, time_measure='seconds'):
    assert time_measure in ['seconds', 'frames']

    if config.masked:
        token_alignment, _ = diagonal_align(S, song)
        mask = compute_line_mask(S, song, token_alignment)
        S = S * mask
    token_alignment, word_alignment = diagonal_align(S, song)

    if time_measure == 'seconds':
        fps = S.shape[1] / song['duration']
        return convert_frames_to_seconds(token_alignment, fps), convert_frames_to_seconds(word_alignment, fps)
    else:
        return token_alignment, word_alignment
    

def compute_line_mask(S, song, token_alignment):    
    token_duration = 9 if config.use_chars else 17  # duration in frames (0.2 * fps and 0.4 * fps)
    tol_window_length = 108  # 2.5 * fps

    mask = np.zeros_like(S)
    num_tokens, num_frames = S.shape

    lines = song['lines'] if config.use_chars else song['pholines']
    first_line_token = past_last_line_token = 0
    for i, line in enumerate(lines):
        num_line_tokens = len(line) + (1 if i + 1 < len(lines) else 0)  # +1 space at the end of the line (see paper image)
        past_last_line_token = first_line_token + num_line_tokens
        middle_token = first_line_token + num_line_tokens // 2
        line_center = token_alignment[middle_token][0]
        line_start = max(line_center - (num_line_tokens - 1) * token_duration // 2, 0)
        line_end = min(line_center + (num_line_tokens + 1) * token_duration // 2 + 1, num_frames)  # +1 to make non-inclusive

        mask[first_line_token:past_last_line_token, line_start:line_end] = 1
        # add linear tolerance window
        # left tolerance window
        window_start = max(line_start - tol_window_length, 0)
        window_end = line_start
        mask[first_line_token:past_last_line_token, window_start:window_end] = \
            np.linspace(0, 1, tol_window_length)[tol_window_length - (window_end - window_start):]
        # right tolerance window
        window_start = line_end
        window_end = min(line_end + tol_window_length, num_frames)
        mask[first_line_token:past_last_line_token, window_start:window_end] = \
            np.linspace(1, 0, tol_window_length)[:window_end - window_start]

        first_line_token = past_last_line_token  # +1 space between lines already added in num_line_tokens
    
    return mask


def convert_frames_to_seconds(alignment, fps):
    # convert (start, end) from spec frames to seconds
    return [(start / fps, end / fps) for (start, end) in alignment]


if __name__ == '__main__':
    pass