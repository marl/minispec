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

__all__ = [_ for _ in dir() if not _.startswith('_')]
