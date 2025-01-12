from typing import Optional

from torch import nn
import torchaudio

def transform(sample_rate: int = 16000,
              n_fft: int = 512,
              win_length: int = 400,
              hop_length: int = 160,
              n_mels: int = 80,
              augmentation: Optional[bool] = False,
              *args, **kwargs):
    transform = nn.Sequential(torchaudio.transforms.MelSpectrogram(sample_rate = sample_rate,
                                                                   n_fft = n_fft,
                                                                   win_length = win_length,
                                                                   hop_length = hop_length,
                                                                   n_mels = n_mels),
                              torchaudio.transforms.AmplitudeToDB())
    if augmentation:
        augment = torchaudio.transforms.SpecAugment(*args, **kwargs)
    else:
        augment = None
    return transform, augment
