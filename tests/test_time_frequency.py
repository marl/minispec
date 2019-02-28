#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# CREATED:2015-02-14 19:13:49 by Brian McFee <brian.mcfee@nyu.edu>
'''Unit tests for time and frequency conversion'''
import os
try:
    os.environ.pop('MINISPEC_CACHE_DIR')
except KeyError:
    pass

import minispec
import numpy as np


def test_frames_to_samples():

    def __test(x, y, hop_length, n_fft):
        y_test = minispec.frames_to_samples(x,
                                           hop_length=hop_length,
                                           n_fft=n_fft)
        assert np.allclose(y, y_test)
        y = np.asanyarray(y)
        assert y.shape == y_test.shape
        assert y.ndim == y_test.ndim

    for x in [100, np.arange(10.5)]:
        for hop_length in [512, 1024]:
            for n_fft in [None, 1024]:
                y = x * hop_length
                if n_fft is not None:
                    y += n_fft // 2
                yield __test, x, y, hop_length, n_fft


def test_samples_to_frames():

    def __test(x, y, hop_length, n_fft):
        y_test = minispec.samples_to_frames(x,
                                           hop_length=hop_length,
                                           n_fft=n_fft)
        assert np.allclose(y, y_test)
        y = np.asanyarray(y)
        assert y.shape == y_test.shape
        assert y.ndim == y_test.ndim

    for x in [100, np.arange(10.5)]:
        for hop_length in [512, 1024]:
            for n_fft in [None, 1024]:
                y = x * hop_length
                if n_fft is not None:
                    y += n_fft // 2
                yield __test, y, x, hop_length, n_fft


def test_frames_to_time():

    def __test(sr, hop_length, n_fft):

        # Generate frames at times 0s, 1s, 2s
        frames = np.arange(3) * sr // hop_length

        if n_fft:
            frames -= n_fft // (2 * hop_length)

        times = minispec.frames_to_time(frames,
                                       sr=sr,
                                       hop_length=hop_length,
                                       n_fft=n_fft)

        # we need to be within one frame
        assert np.all(np.abs(times - np.asarray([0, 1, 2])) * sr
                      < hop_length)

    for sr in [22050, 44100]:
        for hop_length in [256, 512]:
            for n_fft in [None, 2048]:
                yield __test, sr, hop_length, n_fft


def test_time_to_samples():

    def __test(sr):
        assert np.allclose(minispec.time_to_samples([0, 1, 2], sr=sr),
                           [0, sr, 2 * sr])

    for sr in [22050, 44100]:
        yield __test, sr


def test_samples_to_time():

    def __test(sr):
        assert np.allclose(minispec.samples_to_time([0, sr, 2 * sr], sr=sr),
                           [0, 1, 2])

    for sr in [22050, 44100]:
        yield __test, sr


def test_time_to_frames():

    def __test(sr, hop_length, n_fft):

        # Generate frames at times 0s, 1s, 2s
        times = np.arange(3)

        frames = minispec.time_to_frames(times,
                                        sr=sr,
                                        hop_length=hop_length,
                                        n_fft=n_fft)

        if n_fft:
            frames -= n_fft // (2 * hop_length)

        # we need to be within one frame
        assert np.all(np.abs(times - np.asarray([0, 1, 2])) * sr
                      < hop_length)

    for sr in [22050, 44100]:
        for hop_length in [256, 512]:
            for n_fft in [None, 2048]:
                yield __test, sr, hop_length, n_fft


def test_fft_frequencies():

    def __test(sr, n_fft):
        freqs = minispec.fft_frequencies(sr=sr, n_fft=n_fft)

        # DC
        assert freqs[0] == 0

        # Nyquist, positive here for more convenient display purposes
        assert freqs[-1] == sr / 2.0

        # Ensure that the frequencies increase linearly
        dels = np.diff(freqs)
        assert np.allclose(dels, dels[0])

    for n_fft in [1024, 2048]:
        for sr in [8000, 22050]:
            yield __test, sr, n_fft

