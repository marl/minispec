#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DSP
===============

Spectral representations
------------------------
.. autosummary::
    :toctree: generated/

    stft
    istft

    get_fftlib
    set_fftlib

Magnitude scaling
-----------------
.. autosummary::
    :toctree: generated/

    amplitude_to_db
    db_to_amplitude
    power_to_db
    db_to_power

Time and frequency conversion
-----------------------------
.. autosummary::
    :toctree: generated/

    frames_to_samples
    frames_to_time
    samples_to_frames
    samples_to_time
    time_to_frames
    time_to_samples

    hz_to_mel
    mel_to_hz

    fft_frequencies
    mel_frequencies
"""

from .time_frequency import *  # pylint: disable=wildcard-import
from .spectrum import *  # pylint: disable=wildcard-import
<<<<<<< HEAD
=======
from .pitch import *  # pylint: disable=wildcard-import
from .constantq import *  # pylint: disable=wildcard-import
from .harmonic import *  # pylint: disable=wildcard-import
from .fft import *  # pylint: disable=wildcard-import

from ..util.decorators import moved as _moved
from ..util import fill_off_diagonal as _fod
from ..sequence import dtw as _dtw

dtw = _moved('librosa.sequence.dtw', '0.6.1', '0.7')(_dtw)
fill_off_diagonal = _moved('librosa.util.fill_off_diagonal', '0.6.1', '0.7')(_fod)
>>>>>>> acd4e6fbd5b3b361e8c2e89797cfe3eb0b1a3aba

__all__ = [_ for _ in dir() if not _.startswith('_')]
