#!/usr/bin/env python
# CREATED:2014-12-29 10:52:23 by Brian McFee <brian.mcfee@nyu.edu>
# unit tests for ill-formed inputs

# Disable cache
import os
try:
    os.environ.pop('minispec_CACHE_DIR')
except:
    pass

import numpy as np
import minispec
import pytest


@pytest.mark.xfail(raises=minispec.ParameterError)
def test_mono_valid_stereo():
    '''valid_audio: mono=True,  y.ndim==2'''
    y = np.zeros((2, 1000))
    minispec.util.valid_audio(y, mono=True)


@pytest.mark.xfail(raises=minispec.ParameterError)
def test_valid_audio_int():
    y = np.zeros(10, dtype=np.int)
    minispec.util.valid_audio(y)


@pytest.mark.xfail(raises=minispec.ParameterError)
def test_valid_audio_scalar():
    y = np.array(0.0)
    minispec.util.valid_audio(y)


def test_valid_stereo_or_mono():
    '''valid_audio: mono=False, y.ndim==1'''
    y = np.zeros(1000)
    minispec.util.valid_audio(y, mono=False)


def test_valid_mono():
    '''valid_audio: mono=True,  y.ndim==1'''
    y = np.zeros(1000)
    minispec.util.valid_audio(y, mono=True)


def test_valid_stereo():
    '''valid_audio: mono=False, y.ndim==2'''
    y = np.zeros((2, 1000))
    minispec.util.valid_audio(y, mono=False)


@pytest.mark.xfail(raises=minispec.ParameterError)
def test_valid_audio_type():
    '''valid_audio: list input'''
    y = list(np.zeros(1000))
    minispec.util.valid_audio(y)


@pytest.mark.xfail(raises=minispec.ParameterError)
def test_valid_audio_nan():
    '''valid_audio: NaN'''
    y = np.zeros(1000)
    y[10] = np.NaN
    minispec.util.valid_audio(y)


@pytest.mark.xfail(raises=minispec.ParameterError)
def test_valid_audio_inf():
    '''valid_audio: Inf'''
    y = np.zeros(1000)
    y[10] = np.inf
    minispec.util.valid_audio(y)


def test_valid_audio_ndim():
    '''valid_audio: y.ndim > 2'''

    y = np.zeros((3, 10, 10))

    @pytest.mark.xfail(raises=minispec.ParameterError)
    def __test(mono):
        minispec.util.valid_audio(y, mono=mono)

    for mono in [False, True]:
        yield __test, mono


@pytest.mark.xfail(raises=minispec.ParameterError)
def test_frame_hop():
    '''frame: hop_length=0'''
    y = np.zeros(128)
    minispec.util.frame(y, frame_length=10, hop_length=0)


@pytest.mark.xfail(raises=minispec.ParameterError)
def test_frame_discontiguous():
    '''frame: discontiguous input'''
    y = np.zeros((128, 2)).T
    minispec.util.frame(y[0], frame_length=64, hop_length=64)


def test_frame_contiguous():
    '''frame: discontiguous input'''
    y = np.zeros((2, 128))
    minispec.util.frame(y[0], frame_length=64, hop_length=64)


@pytest.mark.xfail(raises=minispec.ParameterError)
def test_frame_size():
    # frame: len(y) == 128, frame_length==256, hop_length=128
    y = np.zeros(64)
    minispec.util.frame(y, frame_length=256, hop_length=128)


@pytest.mark.xfail(raises=minispec.ParameterError)
def test_frame_size_difference():
    # In response to issue #385
    # https://github.com/minispec/minispec/issues/385
    # frame: len(y) == 129, frame_length==256, hop_length=128
    y = np.zeros(129)
    minispec.util.frame(y, frame_length=256, hop_length=128)


@pytest.mark.xfail(raises=minispec.ParameterError)
def test_stft_bad_window():

    y = np.zeros(22050 * 5)

    n_fft = 2048
    window = np.ones(n_fft // 2)

    minispec.stft(y, n_fft=n_fft, window=window)


@pytest.mark.xfail(raises=minispec.ParameterError)
def test_istft_bad_window():

    D = np.zeros((1025, 10), dtype=np.complex64)

    n_fft = 2 * (D.shape[0] - 1)

    window = np.ones(n_fft // 2)

    minispec.istft(D, window=window)
