import numpy as np
from scipy.io.wavfile import read

def read_wav_np(path):
    sr, wav = read(path)

    print(wav.shape)

    if len(wav.shape) == 2:
        wav = wav[:, 0]

    print(wav[30000])

    if wav.dtype == np.int16:
        wav = wav / 32768.0
    elif wav.dtype == np.int32:
        wav = wav / 2147483648.0
    elif wav.dtype == np.uint8:
        wav = (wav - 128) / 128.0

    wav = wav.astype(np.float32)

    return sr, wav