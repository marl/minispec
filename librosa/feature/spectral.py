#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Spectral feature extraction"""

import numpy as np
from .. import filters
from ..core.spectrum import _spectrogram


__all__ = ['melspectrogram']


# -- Spectral features -- #

def melspectrogram(y=None, sr=22050, S=None, n_fft=2048, hop_length=512,
                   power=2.0, **kwargs):
    """Compute a mel-scaled spectrogram.

    If a spectrogram input `S` is provided, then it is mapped directly onto
    the mel basis `mel_f` by `mel_f.dot(S)`.

    If a time-series input `y, sr` is provided, then its magnitude spectrogram
    `S` is first computed, and then mapped onto the mel scale by
    `mel_f.dot(S**power)`.  By default, `power=2` operates on a power spectrum.

    Parameters
    ----------
    y : np.ndarray [shape=(n,)] or None
        audio time-series

    sr : number > 0 [scalar]
        sampling rate of `y`

    S : np.ndarray [shape=(d, t)]
        spectrogram

    n_fft : int > 0 [scalar]
        length of the FFT window

    hop_length : int > 0 [scalar]
        number of samples between successive frames.
        See `minispec.core.stft`

    power : float > 0 [scalar]
        Exponent for the magnitude melspectrogram.
        e.g., 1 for energy, 2 for power, etc.

    kwargs : additional keyword arguments
      Mel filter bank parameters.
      See `minispec.filters.mel` for details.

    Returns
    -------
    S : np.ndarray [shape=(n_mels, t)]
        Mel spectrogram

    See Also
    --------
    minispec.filters.mel
        Mel filter bank construction

    minispec.core.stft
        Short-time Fourier Transform


    Examples
    --------
    >>> y, sr = minispec.load(minispec.util.example_audio_file())
    >>> minispec.feature.melspectrogram(y=y, sr=sr)
    array([[  2.891e-07,   2.548e-03, ...,   8.116e-09,   5.633e-09],
           [  1.986e-07,   1.162e-02, ...,   9.332e-08,   6.716e-09],
           ...,
           [  3.668e-09,   2.029e-08, ...,   3.208e-09,   2.864e-09],
           [  2.561e-10,   2.096e-09, ...,   7.543e-10,   6.101e-10]])

    Using a pre-computed power spectrogram

    >>> D = np.abs(minispec.stft(y))**2
    >>> S = minispec.feature.melspectrogram(S=D)

    >>> # Passing through arguments to the Mel filters
    >>> S = minispec.feature.melspectrogram(y=y, sr=sr, n_mels=128,
    ...                                     fmax=8000)

    >>> import matplotlib.pyplot as plt
    >>> plt.figure(figsize=(10, 4))
    >>> minispec.display.specshow(minispec.power_to_db(S,
    ...                                              ref=np.max),
    ...                          y_axis='mel', fmax=8000,
    ...                          x_axis='time')
    >>> plt.colorbar(format='%+2.0f dB')
    >>> plt.title('Mel spectrogram')
    >>> plt.tight_layout()


    """

    S, n_fft = _spectrogram(y=y, S=S, n_fft=n_fft, hop_length=hop_length,
                            power=power)

    # Build a Mel filter
    mel_basis = filters.mel(sr, n_fft, **kwargs)

    return np.dot(mel_basis, S)
